from glob import glob
import os
from os.path import join as opj
import argparse
import json
import numpy as np
import time
import psutil
import cv2
import math
import subprocess
import shutil

from tqdm import tqdm
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from cleanfid import fid
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# -------------------------------
# Genel device seçimi
# -------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def nvidia_smi_available():
    """nvidia-smi erişilebilir mi?"""
    return shutil.which("nvidia-smi") is not None

def nvidia_smi_used_mb_snapshot():
    """GPU:0 için nvidia-smi 'memory.used' MB anlık snapshot (çok GPU varsa max'ı alır)."""
    if not nvidia_smi_available():
        return float('nan')
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"],
            stderr=subprocess.STDOUT
        ).decode().strip().splitlines()
        vals = [float(x) for x in out if x.strip()]
        return max(vals) if vals else float('nan')
    except Exception:
        return float('nan')

def bytes_to_gb(x: int) -> float:
    return float(x) / (1024.0 ** 3)

def mb_to_gb(x: float) -> float:
    return float(x) / 1024.0

class PairedDataset(Dataset):
    def __init__(self, pred_ps, gt_ps, img_h, img_w):
        self.pred_ps = pred_ps
        self.gt_ps = gt_ps
        self.transform = T.Compose([
            T.Resize((img_h, img_w)),
            T.ToTensor(),
        ])
        assert len(self.pred_ps) == len(self.gt_ps), f"pred and gt lengths don't match: {len(self.pred_ps)} vs {len(self.gt_ps)}"
    
    def __len__(self):
        return len(self.pred_ps)
    
    def __getitem__(self, idx):
        pred_img = self.transform(Image.open(self.pred_ps[idx]).convert("RGB"))
        gt_img = self.transform(Image.open(self.gt_ps[idx]).convert("RGB"))
        return pred_img, gt_img, self.pred_ps[idx]

class PerformanceTracker:
    def __init__(self):
        # zaman
        self.inference_times_batch = []
        self.inference_times_per_image = []
        # CPU
        self.cpu_abs_mb = []
        self.cpu_diff_mb = []
        self.initial_cpu_used_mb = None
        # PyTorch GPU (tensör)
        self.gpu_allocated_mb = []
        self.gpu_reserved_mb = []
        self.pytorch_peak_allocated_b = 0
        self.pytorch_peak_reserved_b = 0
        # nvidia-smi toplam
        self.nvsmi_used_mb_snapshots = []
        # dosya boyutu
        self.output_file_sizes = []
        # kontrol
        self.start_time = None

    def start_tracking(self):
        self.start_time = time.time()
        try:
            self.initial_cpu_used_mb = psutil.virtual_memory().used / 1024.0 / 1024.0
        except Exception:
            self.initial_cpu_used_mb = None

        # PyTorch peak sayaçlarını sıfırla
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

    def track_inference_time(self, start_time, end_time, n_items_in_batch: int = 1):
        """Batch süresi ve per-image süresi (batch_size ile normalize)"""
        dt = end_time - start_time
        self.inference_times_batch.append(dt)
        per_image = dt / max(1, n_items_in_batch)
        self.inference_times_per_image.append(per_image)
        return dt, per_image

    def track_memory_usage(self):
        """CPU abs/diff ve GPU (PyTorch+nvidia-smi) snapshot"""
        try:
            cpu_used = psutil.virtual_memory().used / 1024.0 / 1024.0
        except Exception:
            cpu_used = float('nan')
        self.cpu_abs_mb.append(cpu_used)
        if self.initial_cpu_used_mb is not None:
            self.cpu_diff_mb.append(cpu_used - self.initial_cpu_used_mb)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            alloc_mb = torch.cuda.memory_allocated() / (1024.0 * 1024.0)
            reserv_mb = torch.cuda.memory_reserved() / (1024.0 * 1024.0)
            self.gpu_allocated_mb.append(alloc_mb)
            self.gpu_reserved_mb.append(reserv_mb)
            # PyTorch peak byte
            self.pytorch_peak_allocated_b = max(self.pytorch_peak_allocated_b, torch.cuda.max_memory_allocated())
            self.pytorch_peak_reserved_b  = max(self.pytorch_peak_reserved_b,  torch.cuda.max_memory_reserved())
        # nvidia-smi snapshot
        nvsmi = nvidia_smi_used_mb_snapshot()
        if not np.isnan(nvsmi):
            self.nvsmi_used_mb_snapshots.append(nvsmi)

    def track_output_file_size(self, file_path):
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path) / (1024.0 * 1024.0)
            self.output_file_sizes.append(file_size)
            return file_size
        return 0.0

    def get_model_size(self, model_path):
        if model_path and os.path.exists(model_path):
            return os.path.getsize(model_path) / (1024.0 * 1024.0)
        return 0.0

    @staticmethod
    def _p95(arr):
        if not arr:
            return 0.0
        return float(np.percentile(np.array(arr), 95))

    def calculate_performance_metrics(self):
        # Inference zamanları
        it_batch = np.array(self.inference_times_batch) if self.inference_times_batch else np.array([])
        it_img   = np.array(self.inference_times_per_image) if self.inference_times_per_image else np.array([])

        perf = {
            "inference_time_batch": {
                "avg": float(it_batch.mean()) if it_batch.size else 0.0,
                "std": float(it_batch.std()) if it_batch.size else 0.0,
                "min": float(it_batch.min()) if it_batch.size else 0.0,
                "max": float(it_batch.max()) if it_batch.size else 0.0,
                "total": float(it_batch.sum()) if it_batch.size else 0.0,
                "median": float(np.median(it_batch)) if it_batch.size else 0.0,
                "p95": self._p95(it_batch.tolist()) if it_batch.size else 0.0,
                "fps_equivalent": (len(it_batch) / it_batch.sum()) if it_batch.size and it_batch.sum() > 0 else 0.0
            },
            "inference_time_per_image": {
                "avg": float(it_img.mean()) if it_img.size else 0.0,
                "std": float(it_img.std()) if it_img.size else 0.0,
                "min": float(it_img.min()) if it_img.size else 0.0,
                "max": float(it_img.max()) if it_img.size else 0.0,
                "total": float(it_img.sum()) if it_img.size else 0.0,
                "median": float(np.median(it_img)) if it_img.size else 0.0,
                "p95": self._p95(it_img.tolist()) if it_img.size else 0.0,
                "fps_equivalent": (len(it_img) / it_img.sum()) if it_img.size and it_img.sum() > 0 else 0.0
            },
            "memory_usage": {
                "cpu_abs_mb": {
                    "avg_mb": float(np.mean(self.cpu_abs_mb)) if self.cpu_abs_mb else 0.0,
                    "max_mb": float(np.max(self.cpu_abs_mb)) if self.cpu_abs_mb else 0.0,
                    "std_mb": float(np.std(self.cpu_abs_mb)) if self.cpu_abs_mb else 0.0,
                    "min_mb": float(np.min(self.cpu_abs_mb)) if self.cpu_abs_mb else 0.0
                },
                "cpu_diff_mb": {
                    "avg_mb": float(np.mean(self.cpu_diff_mb)) if self.cpu_diff_mb else 0.0,
                    "max_mb": float(np.max(self.cpu_diff_mb)) if self.cpu_diff_mb else 0.0,
                    "std_mb": float(np.std(self.cpu_diff_mb)) if self.cpu_diff_mb else 0.0,
                    "min_mb": float(np.min(self.cpu_diff_mb)) if self.cpu_diff_mb else 0.0
                },
                "pytorch_gpu_mb": {
                    "avg_allocated_mb": float(np.mean(self.gpu_allocated_mb)) if self.gpu_allocated_mb else 0.0,
                    "max_allocated_mb": float(np.max(self.gpu_allocated_mb)) if self.gpu_allocated_mb else 0.0,
                    "avg_reserved_mb": float(np.mean(self.gpu_reserved_mb)) if self.gpu_reserved_mb else 0.0,
                    "max_reserved_mb": float(np.max(self.gpu_reserved_mb)) if self.gpu_reserved_mb else 0.0,
                    "peak_allocated_gb": round(bytes_to_gb(self.pytorch_peak_allocated_b), 3) if self.pytorch_peak_allocated_b else 0.0,
                    "peak_reserved_gb":  round(bytes_to_gb(self.pytorch_peak_reserved_b), 3)  if self.pytorch_peak_reserved_b else 0.0
                },
                "nvidia_smi_used": {
                    "avg_used_gb": round(mb_to_gb(float(np.mean(self.nvsmi_used_mb_snapshots))), 3) if self.nvsmi_used_mb_snapshots else 0.0,
                    "peak_used_gb": round(mb_to_gb(float(np.max(self.nvsmi_used_mb_snapshots))), 3) if self.nvsmi_used_mb_snapshots else 0.0,
                    "min_used_gb":  round(mb_to_gb(float(np.min(self.nvsmi_used_mb_snapshots))), 3) if self.nvsmi_used_mb_snapshots else 0.0
                }
            },
            "output_file_size": {
                "avg_mb": float(np.mean(self.output_file_sizes)) if self.output_file_sizes else 0.0,
                "total_mb": float(np.sum(self.output_file_sizes)) if self.output_file_sizes else 0.0,
                "std_mb": float(np.std(self.output_file_sizes)) if self.output_file_sizes else 0.0,
                "min_mb": float(np.min(self.output_file_sizes)) if self.output_file_sizes else 0.0,
                "max_mb": float(np.max(self.output_file_sizes)) if self.output_file_sizes else 0.0
            }
        }
        return perf

def calculate_psnr(pred, gt, max_val=1.0):
    mse = F.mse_loss(pred, gt)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(max_val / math.sqrt(mse.item()))

def calculate_mae(pred, gt):
    return F.l1_loss(pred, gt).item()

def calculate_mse(pred, gt):
    return F.mse_loss(pred, gt).item()

def calculate_rmse(pred, gt):
    return math.sqrt(F.mse_loss(pred, gt).item())

def calculate_image_sharpness(image_tensor):
    # Tensor -> numpy -> grayscale
    image_np = image_tensor.detach().cpu().numpy().transpose(1, 2, 0)
    gray = cv2.cvtColor((image_np * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return laplacian.var()

def calculate_edge_similarity(pred, gt):
    """Sobel edge haritaları üzerinden kosinüs benzerliği"""
    dev = pred.device
    pred_gray = torch.mean(pred, dim=1, keepdim=True)
    gt_gray   = torch.mean(gt,   dim=1, keepdim=True)

    sobel_x = torch.tensor([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]], dtype=torch.float32, device=dev).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1],
                            [ 0,  0,  0],
                            [ 1,  2,  1]], dtype=torch.float32, device=dev).view(1, 1, 3, 3)

    pred_edges_x = F.conv2d(pred_gray, sobel_x, padding=1)
    pred_edges_y = F.conv2d(pred_gray, sobel_y, padding=1)
    pred_edges   = torch.sqrt(pred_edges_x**2 + pred_edges_y**2)

    gt_edges_x = F.conv2d(gt_gray, sobel_x, padding=1)
    gt_edges_y = F.conv2d(gt_gray, sobel_y, padding=1)
    gt_edges   = torch.sqrt(gt_edges_x**2 + gt_edges_y**2)

    return F.cosine_similarity(pred_edges.flatten(), gt_edges.flatten(), dim=0).item()

def calculate_histogram_similarity(pred, gt):
    """RGB kanal histogram korelasyonu"""
    pred_np = pred.detach().cpu().numpy()
    gt_np   = gt.detach().cpu().numpy()
    similarities = []
    for i in range(3):
        pred_hist = np.histogram(pred_np[i], bins=256, range=(0, 1))[0]
        gt_hist   = np.histogram(gt_np[i],   bins=256, range=(0, 1))[0]
        pred_hist = pred_hist / max(1, pred_hist.sum())
        gt_hist   = gt_hist   / max(1, gt_hist.sum())
        corr = np.corrcoef(pred_hist, gt_hist)[0, 1]
        if not np.isnan(corr):
            similarities.append(corr)
    return float(np.mean(similarities)) if similarities else 0.0

class EnhancedPromptDresserEvaluation:
    def __init__(self, text_pairs_json_path="test_gpt4o.json"):
        possible_paths = [
            text_pairs_json_path,
            f"./DATA/zalando-hd-resized/{text_pairs_json_path}",
            f"./DATA/zalando-hd-resized/test_gpt4o.json",
            f"./DATA/{text_pairs_json_path}",
            f"./{text_pairs_json_path}"
        ]
        self.text_pairs_data = {}
        for path in possible_paths:
            try:
                if os.path.exists(path):
                    with open(path, 'r') as f:
                        self.text_pairs_data = json.load(f)
                    print(f"✅ Found text pairs JSON at: {path}")
                    break
            except Exception:
                continue
        if not self.text_pairs_data:
            print(f"⚠️  Warning: Could not find {text_pairs_json_path} in any expected location. Text alignment evaluation disabled.")

        self.sample_categories = self.define_sample_categories()
        self.performance_tracker = PerformanceTracker()
    
    def define_sample_categories(self):
        categories = {}
        # BASE CLOTH - Plain
        base_cloth_pairs = [
            "00017_00__00006_00", "00055_00__00017_00", "00057_00__00064_00", 
            "00067_00__00084_00", "00069_00__00096_00", "00071_00__00121_00", 
            "00075_00__00145_00", "00084_00__00339_00", "00095_00__00401_00", 
            "00096_00__00428_00", "00135_00__00475_00", "00151_00__00503_00", 
            "00158_00__00599_00", "00176_00__00664_00", "00254_00__00690_00", 
            "00259_00__00698_00", "00260_00__00725_00", "00261_00__00737_00", 
            "00272_00__00877_00", "00273_00__00884_00"
        ]
        # LOGO
        logo_cloth_pairs = [
            "00017_00__00055_00", "00055_00__00057_00", "00057_00__00075_00", 
            "00067_00__00126_00", "00069_00__00260_00", "00071_00__00278_00", 
            "00075_00__00345_00", "00084_00__00462_00", "00095_00__00484_00", 
            "00096_00__00502_00", "00135_00__00641_00", "00151_00__00824_00", 
            "00158_00__01229_00", "00176_00__01382_00", "00254_00__01449_00", 
            "00259_00__01625_00", "00260_00__01967_00", "00261_00__02268_00", 
            "00272_00__02500_00", "00273_00__02871_00", "00287_00__02991_00", 
            "00291_00__03838_00", "00311_00__03857_00", "00330_00__04094_00", 
            "00339_00__04191_00"
        ]
        # DESIGN - Patterned
        design_cloth_pairs = [
            "00017_00__01048_00", "00055_00__01268_00", "00057_00__01287_00", 
            "00067_00__01341_00", "00069_00__01430_00", "00071_00__01503_00", 
            "00075_00__01518_00", "00084_00__01689_00", "00095_00__01874_00", 
            "00096_00__01881_00", "00135_00__01893_00", "00151_00__01969_00", 
            "00158_00__02060_00", "00176_00__02245_00", "00254_00__02459_00", 
            "00259_00__02532_00", "00260_00__02579_00", "00261_00__02653_00", 
            "00272_00__02665_00", "00273_00__02682_00", "00287_00__02726_00", 
            "00291_00__02757_00", "00311_00__02783_00", "00330_00__02912_00", 
            "00339_00__02942_00", "00345_00__03006_00", "00348_00__03032_00", 
            "00349_00__03075_00", "00373_00__03158_00", "00396_00__03192_00", 
            "00400_00__03244_00", "00428_00__03601_00", "00458_00__03745_00", 
            "00462_00__03751_00", "00499_00__03881_00", "00504_00__03900_00"
        ]
        # EXTREME
        extreme_cloth_pairs = [
            "00017_00__01215_00", "00055_00__01248_00", "00057_00__01641_00", 
            "00067_00__01814_00", "00069_00__01900_00", "00071_00__02016_00", 
            "00075_00__02305_00", "00084_00__02364_00", "00095_00__02390_00", 
            "00096_00__02653_00", "00135_00__02743_00", "00151_00__02771_00", 
            "00158_00__02786_00", "00176_00__03067_00", "00254_00__03158_00", 
            "00259_00__03199_00", "00260_00__03284_00", "00261_00__03291_00", 
            "00272_00__03392_00", "00273_00__03413_00", "00287_00__03458_00", 
            "00291_00__04131_00", "00311_00__04215_00", "00330_00__04584_00", 
            "00339_00__04632_00", "00345_00__04825_00", "00348_00__05112_00", 
            "00349_00__05115_00", "00373_00__05244_00", "00396_00__05830_00"
        ]
        for pair in base_cloth_pairs:
            categories[pair] = "base_cloth"
        for pair in logo_cloth_pairs:
            categories[pair] = "logo_cloth"
        for pair in design_cloth_pairs:
            categories[pair] = "design_cloth"
        for pair in extreme_cloth_pairs:
            categories[pair] = "extreme_cloth"
        return categories
    
    def get_sample_category(self, filename):
        sample_id = os.path.splitext(os.path.basename(filename))[0]
        return self.sample_categories.get(sample_id, "unknown")
    
    def analyze_by_category(self, detailed_scores, file_paths):
        category_results = {}
        for i, file_path in enumerate(file_paths):
            category = self.get_sample_category(file_path)
            if category not in category_results:
                category_results[category] = {
                    'ssim_scores': [], 'lpips_scores': [], 'psnr_scores': [],
                    'mae_scores': [], 'mse_scores': [], 'rmse_scores': [],
                    'edge_similarity_scores': [], 'histogram_similarity_scores': [],
                    'sharpness_scores': [], 'count': 0
                }
            # puanlar
            if i < len(detailed_scores.get('ssim_scores', [])):
                category_results[category]['ssim_scores'].append(detailed_scores['ssim_scores'][i])
            if i < len(detailed_scores.get('lpips_scores', [])):
                category_results[category]['lpips_scores'].append(detailed_scores['lpips_scores'][i])
            if i < len(detailed_scores.get('psnr_scores', [])):
                category_results[category]['psnr_scores'].append(detailed_scores['psnr_scores'][i])
            if i < len(detailed_scores.get('mae_scores', [])):
                category_results[category]['mae_scores'].append(detailed_scores['mae_scores'][i])
            if i < len(detailed_scores.get('mse_scores', [])):
                category_results[category]['mse_scores'].append(detailed_scores['mse_scores'][i])
            if i < len(detailed_scores.get('rmse_scores', [])):
                category_results[category]['rmse_scores'].append(detailed_scores['rmse_scores'][i])
            if i < len(detailed_scores.get('edge_similarity_scores', [])):
                category_results[category]['edge_similarity_scores'].append(detailed_scores['edge_similarity_scores'][i])
            if i < len(detailed_scores.get('histogram_similarity_scores', [])):
                category_results[category]['histogram_similarity_scores'].append(detailed_scores['histogram_similarity_scores'][i])
            if i < len(detailed_scores.get('sharpness_scores', [])):
                category_results[category]['sharpness_scores'].append(detailed_scores['sharpness_scores'][i])
            category_results[category]['count'] += 1
        
        for category in category_results:
            for metric in ['ssim_scores','lpips_scores','psnr_scores','mae_scores',
                           'mse_scores','rmse_scores','edge_similarity_scores',
                           'histogram_similarity_scores','sharpness_scores']:
                scores = category_results[category][metric]
                key = metric.replace('_scores','')
                if scores:
                    category_results[category][f'avg_{key}'] = float(np.mean(scores))
                    category_results[category][f'std_{key}'] = float(np.std(scores))
                else:
                    category_results[category][f'avg_{key}'] = 0.0
                    category_results[category][f'std_{key}'] = 0.0
        return category_results

def get_matching_gt_files(pred_ps, gt_dir):
    matched_gt_ps = []
    matched_pred_ps = []
    for pred_path in pred_ps:
        pred_filename = os.path.basename(pred_path)
        possible_gt_names = [
            pred_filename,
            pred_filename.replace('.jpg', '.png'),
            pred_filename.replace('.png', '.jpg'),
        ]
        if "__" in pred_filename:
            person_id = pred_filename.split("__")[0]
            possible_gt_names.extend([
                f"{person_id}.jpg", f"{person_id}.png",
                f"{person_id}_00.jpg", f"{person_id}_00.png"
            ])
        gt_found = False
        for gt_name in possible_gt_names:
            gt_path = opj(gt_dir, gt_name)
            if os.path.exists(gt_path):
                matched_gt_ps.append(gt_path)
                matched_pred_ps.append(pred_path)
                gt_found = True
                break
        if not gt_found:
            print(f" No GT found for: {pred_filename}")
    print(f" Matched {len(matched_pred_ps)} pred files with GT files")
    return matched_pred_ps, matched_gt_ps

@torch.no_grad()
def get_comprehensive_metrics(pred_dir, gt_dir, img_h, img_w, is_unpaired, evaluator=None, model_path=None, batch_size=8, num_workers=2):
    pred_ps = sorted(glob(opj(pred_dir, "*.jpg"))) + sorted(glob(opj(pred_dir, "*.png"))) + sorted(glob(opj(pred_dir, "*.jpeg")))
    print(f"Found {len(pred_ps)} prediction files")
    if not is_unpaired:
        matched_pred_ps, gt_ps = get_matching_gt_files(pred_ps, gt_dir)
        pred_ps = matched_pred_ps
        if len(pred_ps) == 0:
            print("No matching GT files found!")
            return {}, {}
        print(f"Using {len(pred_ps)} matched pairs for evaluation")
    else:
        gt_ps = sorted(glob(opj(gt_dir, "*.jpg"))) + sorted(glob(opj(gt_dir, "*.png"))) + sorted(glob(opj(gt_dir, "*.jpeg")))
    
    if evaluator:
        evaluator.performance_tracker.start_tracking()
    
    if is_unpaired:
        detailed_scores = {}
        avg_metrics = {}
    else:
        # metrics objelerini DEVICE'e göre oluştur
        ssim_metric  = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)
        lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(DEVICE)

        paired_dataset = PairedDataset(pred_ps, gt_ps, img_h, img_w)
        paired_loader  = DataLoader(paired_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())

        detailed_scores = {
            'ssim_scores': [], 'lpips_scores': [], 'psnr_scores': [],
            'mae_scores': [], 'mse_scores': [], 'rmse_scores': [],
            'edge_similarity_scores': [], 'histogram_similarity_scores': [],
            'sharpness_pred_scores': [], 'sharpness_gt_scores': []
        }
        file_paths = []

        print("Calculating comprehensive metrics...")
        for batch_idx, (pred, gt, paths) in enumerate(tqdm(paired_loader, desc="Processing batches")):
            batch_start = time.time()
            pred = pred.to(DEVICE, non_blocking=True)
            gt   = gt.to(DEVICE,   non_blocking=True)

            if evaluator:
                evaluator.performance_tracker.track_memory_usage()

            # batch metrikleri
            _ = ssim_metric(pred, gt)
            _ = lpips_metric(pred, gt)

            # tekil örnek metrikleri
            for i in range(pred.shape[0]):
                single_pred = pred[i:i+1]
                single_gt   = gt[i:i+1]

                single_ssim  = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)(single_pred, single_gt)
                single_lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(DEVICE)(single_pred, single_gt)

                psnr_v  = calculate_psnr(single_pred, single_gt)
                mae_v   = calculate_mae(single_pred, single_gt)
                mse_v   = calculate_mse(single_pred, single_gt)
                rmse_v  = calculate_rmse(single_pred, single_gt)
                edge_v  = calculate_edge_similarity(single_pred, single_gt)
                hist_v  = calculate_histogram_similarity(single_pred[0], single_gt[0])
                sharp_p = calculate_image_sharpness(single_pred[0])
                sharp_g = calculate_image_sharpness(single_gt[0])

                detailed_scores['ssim_scores'].append(float(single_ssim.detach().cpu().item()))
                detailed_scores['lpips_scores'].append(float(single_lpips.detach().cpu().item()))
                detailed_scores['psnr_scores'].append(float(psnr_v))
                detailed_scores['mae_scores'].append(float(mae_v))
                detailed_scores['mse_scores'].append(float(mse_v))
                detailed_scores['rmse_scores'].append(float(rmse_v))
                detailed_scores['edge_similarity_scores'].append(float(edge_v))
                detailed_scores['histogram_similarity_scores'].append(float(hist_v))
                detailed_scores['sharpness_pred_scores'].append(float(sharp_p))
                detailed_scores['sharpness_gt_scores'].append(float(sharp_g))

                file_paths.append(paths[i])

            batch_end = time.time()
            if evaluator:
                evaluator.performance_tracker.track_inference_time(batch_start, batch_end, n_items_in_batch=pred.shape[0])

        # ortalamalar
        avg_metrics = {
            'avg_ssim': float(ssim_metric.compute().item()),
            'avg_lpips': float(lpips_metric.compute().item()),
            'avg_psnr': float(np.mean(detailed_scores['psnr_scores'])),
            'avg_mae': float(np.mean(detailed_scores['mae_scores'])),
            'avg_mse': float(np.mean(detailed_scores['mse_scores'])),
            'avg_rmse': float(np.mean(detailed_scores['rmse_scores'])),
            'avg_edge_similarity': float(np.mean(detailed_scores['edge_similarity_scores'])),
            'avg_histogram_similarity': float(np.mean(detailed_scores['histogram_similarity_scores'])),
            'avg_sharpness_pred': float(np.mean(detailed_scores['sharpness_pred_scores'])),
            'avg_sharpness_gt': float(np.mean(detailed_scores['sharpness_gt_scores'])),
            'sharpness_ratio': (float(np.mean(detailed_scores['sharpness_pred_scores'])) /
                                float(np.mean(detailed_scores['sharpness_gt_scores'])) if np.mean(detailed_scores['sharpness_gt_scores']) > 0 else 0.0)
        }

    # output dosya boyutları
    if evaluator:
        for pred_file in pred_ps:
            evaluator.performance_tracker.track_output_file_size(pred_file)

    print("Calculating FID and KID...")
    fid_start = time.time()
    try:
        fid_score = fid.compute_fid(pred_dir, gt_dir, mode="clean", use_dataparallel=False, dataset_split="custom")
        kid_score = fid.compute_kid(pred_dir, gt_dir, mode="clean", use_dataparallel=False, dataset_split="custom")
    except Exception as e:
        print(f"FID/KID calculation failed: {e}")
        fid_score = 0.0
        kid_score = 0.0
    fid_end = time.time()

    avg_metrics.update({
        'fid_score': float(fid_score),
        'kid_score': float(kid_score)
    })

    enhanced_results = {}
    if evaluator:
        performance_metrics = evaluator.performance_tracker.calculate_performance_metrics()

        if model_path:
            performance_metrics["model_size_mb"] = evaluator.performance_tracker.get_model_size(model_path)

        performance_metrics["fid_kid_time"] = float(fid_end - fid_start)

        category_analysis = {}
        if not is_unpaired and detailed_scores:
            category_analysis = evaluator.analyze_by_category(detailed_scores, file_paths)

        enhanced_results = {
            "performance_metrics": performance_metrics,
            "detailed_scores": detailed_scores if detailed_scores else {},
            "category_analysis": category_analysis,
            "file_info": {
                "total_pred_files": len(pred_ps),
                "total_gt_files": len(gt_ps) if 'gt_ps' in locals() else 0,
                "matched_pairs": len(pred_ps) if not is_unpaired else 0
            }
        }
    return avg_metrics, enhanced_results

def save_comprehensive_results(avg_metrics, enhanced_results, save_dir, pair_type, img_h):
    save_path = opj(save_dir, f"{pair_type}_results_{img_h}.txt")
    with open(save_path, "w") as f:
        f.write(f"ssim : {avg_metrics.get('avg_ssim', 0.0)}\n")
        f.write(f"lpips : {avg_metrics.get('avg_lpips', 0.0)}\n")
        f.write(f"psnr : {avg_metrics.get('avg_psnr', 0.0)}\n")
        f.write(f"mae : {avg_metrics.get('avg_mae', 0.0)}\n")
        f.write(f"mse : {avg_metrics.get('avg_mse', 0.0)}\n")
        f.write(f"rmse : {avg_metrics.get('avg_rmse', 0.0)}\n")
        f.write(f"edge_similarity : {avg_metrics.get('avg_edge_similarity', 0.0)}\n")
        f.write(f"histogram_similarity : {avg_metrics.get('avg_histogram_similarity', 0.0)}\n")
        f.write(f"sharpness_ratio : {avg_metrics.get('sharpness_ratio', 0.0)}\n")
        f.write(f"fid : {avg_metrics.get('fid_score', 0.0)}\n")
        f.write(f"kid_score : {avg_metrics.get('kid_score', 0.0)}")
    print(f"Basic results saved to: {save_path}")
    
    if enhanced_results:
        all_results = {
            "overall_metrics": avg_metrics,
            **enhanced_results
        }
        enhanced_save_path = opj(save_dir, f"{pair_type}_comprehensive_results_{img_h}.json")
        with open(enhanced_save_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"Comprehensive results saved to: {enhanced_save_path}")

        print("\n" + "="*80)
        print("COMPREHENSIVE PROMPTDRESSER EVALUATION RESULTS")
        print("="*80)

        print("\nOVERALL VISUAL QUALITY METRICS:")
        print("-"*50)
        print(f"SSIM: {avg_metrics.get('avg_ssim', 0.0):.4f}")
        print(f"LPIPS: {avg_metrics.get('avg_lpips', 0.0):.4f}")
        print(f"PSNR: {avg_metrics.get('avg_psnr', 0.0):.4f} dB")
        print(f"MAE: {avg_metrics.get('avg_mae', 0.0):.6f}")
        print(f"MSE: {avg_metrics.get('avg_mse', 0.0):.6f}")
        print(f"RMSE: {avg_metrics.get('avg_rmse', 0.0):.6f}")
        print(f"Edge Similarity: {avg_metrics.get('avg_edge_similarity', 0.0):.4f}")
        print(f"Histogram Similarity: {avg_metrics.get('avg_histogram_similarity', 0.0):.4f}")
        print(f"Sharpness Ratio (Pred/GT): {avg_metrics.get('sharpness_ratio', 0.0):.4f}")
        print(f"FID: {avg_metrics.get('fid_score', 0.0):.4f}")
        print(f"KID: {avg_metrics.get('kid_score', 0.0):.4f}")

        if "performance_metrics" in enhanced_results:
            perf = enhanced_results["performance_metrics"]
            print("\nPERFORMANCE METRICS:")
            print("-"*50)

            # per-image latency (e-commerce için daha anlamlı)
            pit = perf.get('inference_time_per_image', {})
            print(f"Per-Image Inference Time: avg={pit.get('avg',0):.4f}s, median={pit.get('median',0):.4f}s, p95={pit.get('p95',0):.4f}s")

            bit = perf.get('inference_time_batch', {})
            print(f"Per-Batch Inference Time: avg={bit.get('avg',0):.4f}s, median={bit.get('median',0):.4f}s, p95={bit.get('p95',0):.4f}s, total={bit.get('total',0):.2f}s")

            mem = perf.get('memory_usage', {})
            cpu_abs = mem.get('cpu_abs_mb', {})
            cpu_dif = mem.get('cpu_diff_mb', {})
            pyt = mem.get('pytorch_gpu_mb', {})
            nvs = mem.get('nvidia_smi_used', {})

            print(f"CPU Mem (abs): avg={cpu_abs.get('avg_mb',0):.2f}MB, peak={cpu_abs.get('max_mb',0):.2f}MB")
            print(f"CPU Mem (diff): avg={cpu_dif.get('avg_mb',0):.2f}MB, peak={cpu_dif.get('max_mb',0):.2f}MB")

            print(f"PyTorch GPU: avg_alloc={pyt.get('avg_allocated_mb',0):.2f}MB, max_alloc={pyt.get('max_allocated_mb',0):.2f}MB, "
                  f"avg_reserved={pyt.get('avg_reserved_mb',0):.2f}MB, max_reserved={pyt.get('max_reserved_mb',0):.2f}MB, "
                  f"peak_allocated={pyt.get('peak_allocated_gb',0):.3f}GB, peak_reserved={pyt.get('peak_reserved_gb',0):.3f}GB")

            print(f"nvidia-smi (TOTAL): avg_used={nvs.get('avg_used_gb',0):.3f}GB, peak_used={nvs.get('peak_used_gb',0):.3f}GB, min_used={nvs.get('min_used_gb',0):.3f}GB")

            ofs = perf.get('output_file_size', {})
            print(f"Output File Size: avg={ofs.get('avg_mb',0):.3f}MB, total={ofs.get('total_mb',0):.3f}MB")

            if "model_size_mb" in perf:
                print(f"Model Size: {perf['model_size_mb']:.2f}MB")

            print(f"FID/KID Calculation Time: {perf.get('fid_kid_time',0):.2f}s")

        if "category_analysis" in enhanced_results and enhanced_results["category_analysis"]:
            print("\nCATEGORY-WISE ANALYSIS:")
            print("-"*50)
            for category, metrics in enhanced_results["category_analysis"].items():
                if metrics['count'] > 0:
                    print(f"\n{category.upper()} ({metrics['count']} samples):")
                    print(f"  SSIM: {metrics.get('avg_ssim', 0.0):.4f} ± {metrics.get('std_ssim', 0.0):.4f}")
                    print(f"  LPIPS: {metrics.get('avg_lpips', 0.0):.4f} ± {metrics.get('std_lpips', 0.0):.4f}")
                    print(f"  PSNR: {metrics.get('avg_psnr', 0.0):.4f} ± {metrics.get('std_psnr', 0.0):.4f}")
                    print(f"  Edge Sim: {metrics.get('avg_edge_similarity', 0.0):.4f} ± {metrics.get('std_edge_similarity', 0.0):.4f}")

        if "file_info" in enhanced_results:
            file_info = enhanced_results["file_info"]
            print("\nFILE MATCHING INFO:")
            print("-"*50)
            print(f"Total prediction files: {file_info['total_pred_files']}")
            print(f"Total GT files: {file_info['total_gt_files']}")
            print(f"Successfully matched pairs: {file_info['matched_pairs']}")
            if file_info['matched_pairs'] < file_info['total_pred_files']:
                print(f"{file_info['total_pred_files'] - file_info['matched_pairs']} files could not be matched")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Comprehensive PromptDresser Evaluation (GPU-aware)")
    parser.add_argument("--pred_dir", type=str, required=True, help="Directory containing predicted images")
    parser.add_argument("--img_h", type=int, default=1024, help="Image height for resizing")
    parser.add_argument("--img_w", type=int, default=768, help="Image width for resizing")
    parser.add_argument("--category", default=None, help="Category name")
    parser.add_argument("--gt_dir", type=str, default=None, help="Directory containing ground truth images")
    parser.add_argument("--pair_type", default=None, help="Pair type (paired/unpaired)")
    parser.add_argument("--text_pairs_json", type=str, default="test_gpt4o.json", help="Path to text pairs JSON file")
    parser.add_argument("--model_path", type=str, default=None, help="Path to model file for size calculation")
    parser.add_argument("--enhanced", action="store_true", help="Enable enhanced evaluation with comprehensive metrics")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for evaluation")
    parser.add_argument("--num_workers", type=int, default=2, help="Dataloader workers")
    args = parser.parse_args()

    args.pred_dir = args.pred_dir.rstrip("/")
    if args.pair_type is None:
        pair_type = args.pred_dir.split("/")[-1]
    else:
        pair_type = args.pair_type
    if args.category is None:
        category = args.pred_dir.split("/")[-2]
    else:
        category = args.category
    
    if args.gt_dir is None:
        gt_dir = "./DATA/zalando-hd-resized/test_fine/image"
    else:
        gt_dir = args.gt_dir

    evaluator = None
    if args.enhanced:
        evaluator = EnhancedPromptDresserEvaluation(args.text_pairs_json)
        print("Enhanced comprehensive evaluation enabled")

    avg_metrics, enhanced_results = get_comprehensive_metrics(
        args.pred_dir, gt_dir,
        args.img_h, args.img_w,
        is_unpaired=(pair_type == "unpaired"),
        evaluator=evaluator,
        model_path=args.model_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    print(f"\nBASIC RESULTS:")
    print(f"SSIM: {avg_metrics.get('avg_ssim', 0.0):.4f}")
    print(f"LPIPS: {avg_metrics.get('avg_lpips', 0.0):.4f}")
    print(f"PSNR: {avg_metrics.get('avg_psnr', 0.0):.4f}")
    print(f"FID: {avg_metrics.get('fid_score', 0.0):.4f}")
    print(f"KID: {avg_metrics.get('kid_score', 0.0):.4f}")

    save_dir = os.path.dirname(args.pred_dir)
    if args.enhanced:
        save_comprehensive_results(avg_metrics, enhanced_results, save_dir, pair_type, args.img_h)
    else:
        save_path = opj(save_dir, f"{pair_type}_results_{args.img_h}.txt")
        with open(save_path, "w") as f:
            f.write(f"ssim : {avg_metrics.get('avg_ssim', 0.0)}\n")
            f.write(f"lpips : {avg_metrics.get('avg_lpips', 0.0)}\n")
            f.write(f"fid : {avg_metrics.get('fid_score', 0.0)}\n")
            f.write(f"kid_score : {avg_metrics.get('kid_score', 0.0)}")
        print(f"Results saved to: {save_path}")
    
    print(f"\nEvaluation completed successfully!")
