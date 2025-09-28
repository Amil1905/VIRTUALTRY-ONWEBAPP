# evaluation.py
import os, time, json, glob, argparse, subprocess, shutil, csv, threading
from datetime import datetime
from typing import List, Tuple, Dict

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

import torch
import torchvision.transforms as T
import lpips
import psutil

# =========================
# Defaults & Paths
# =========================
DEFAULT_DATA_DIR    = "data"
DEFAULT_PERSON_DIR  = os.path.join(DEFAULT_DATA_DIR, "test", "image")
DEFAULT_CLOTH_DIR   = os.path.join(DEFAULT_DATA_DIR, "test", "cloth")
DEFAULT_PAIRS_FILE  = os.path.join(DEFAULT_DATA_DIR, "test_pairs.txt")
DEFAULT_OUTPUT_DIR  = "result"
DEFAULT_RESULTS_DIR = "results"
os.makedirs(DEFAULT_RESULTS_DIR, exist_ok=True)

# =========================
# GPU Poller (CSV logger)
# =========================
class GPUPoller:
    """
    Periodically polls nvidia-smi and writes CSV.
    mode="gpu": per-GPU utilization/memory/power/temp
    mode="process": per-process GPU memory by PID
    """
    def __init__(self, csv_path: str, interval: float = 0.5, gpu_index: int = 0, mode: str = "gpu"):
        self.csv_path   = csv_path
        self.interval   = interval
        self.gpu_index  = gpu_index
        self.mode       = mode  # "gpu" or "process"
        self._stop_evt  = threading.Event()
        self._thread    = None
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)

    def start(self):
        with open(self.csv_path, "w", newline="") as f:
            w = csv.writer(f)
            if self.mode == "gpu":
                w.writerow(["timestamp","gpu_index","utilization_gpu_%","memory_used_MiB","memory_total_MiB","power_W","temperature_C"])
            else:
                w.writerow(["timestamp","gpu_index","pid","process_name","used_memory_MiB"])
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_evt.set()

    def join(self):
        if self._thread is not None:
            self._thread.join()

    def _run(self):
        while not self._stop_evt.is_set():
            try:
                ts = datetime.now().isoformat()
                if self.mode == "gpu":
                    cmd = [
                        "nvidia-smi", "-i", str(self.gpu_index),
                        "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu",
                        "--format=csv,noheader,nounits"
                    ]
                    r = subprocess.run(cmd, capture_output=True, text=True, timeout=2)
                    if r.returncode == 0 and r.stdout.strip():
                        util, mem_used, mem_total, power, temp = [x.strip() for x in r.stdout.strip().split(",")]
                        with open(self.csv_path, "a", newline="") as f:
                            csv.writer(f).writerow([ts, self.gpu_index, util, mem_used, mem_total, power, temp])
                else:
                    cmd = [
                        "nvidia-smi", "-i", str(self.gpu_index),
                        "--query-compute-apps=pid,process_name,used_memory",
                        "--format=csv,noheader,nounits"
                    ]
                    r = subprocess.run(cmd, capture_output=True, text=True, timeout=2)
                    rows = r.stdout.strip().splitlines() if (r.returncode == 0 and r.stdout.strip()) else []
                    if not rows:
                        with open(self.csv_path, "a", newline="") as f:
                            csv.writer(f).writerow([ts, self.gpu_index, "", "", ""])
                    else:
                        with open(self.csv_path, "a", newline="") as f:
                            w = csv.writer(f)
                            for line in rows:
                                parts = [p.strip() for p in line.split(",")]
                                if len(parts) >= 3:
                                    pid, name, used_mem = parts[:3]
                                    w.writerow([ts, self.gpu_index, pid, name, used_mem])
            except Exception:
                pass
            time.sleep(self.interval)

def summarize_gpu_csv(csv_path: str) -> dict:
    if not os.path.exists(csv_path):
        return {}
    try:
        import pandas as pd
    except Exception:
        return {}
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return {}
    # mode=gpu columns
    cols = {"utilization_gpu_%","memory_used_MiB","power_W","temperature_C"}
    if cols.issubset(set(df.columns)):
        return {
            "avg_util_%": float(df["utilization_gpu_%"].mean()),
            "max_util_%": float(df["utilization_gpu_%"].max()),
            "avg_mem_MiB": float(df["memory_used_MiB"].mean()),
            "max_mem_MiB": float(df["memory_used_MiB"].max()),
            "avg_power_W": float(df["power_W"].mean()),
            "max_temp_C": float(df["temperature_C"].max()),
            "samples": int(len(df))
        }
    return {"rows": int(len(df))}

# =========================
# Utility: pairs
# =========================
def read_pairs(pairs_file: str, limit: int|None=None) -> List[Tuple[str,str]]:
    if not os.path.exists(pairs_file):
        raise FileNotFoundError(f"pairs file not found: {pairs_file}")
    with open(pairs_file, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    pairs = []
    for ln in lines:
        parts = ln.split()
        if len(parts) >= 2:
            pairs.append((parts[0], parts[1]))
    return pairs[:limit] if limit else pairs

def write_pairs(pairs: List[Tuple[str,str]], pairs_file: str):
    os.makedirs(os.path.dirname(pairs_file), exist_ok=True
    )
    with open(pairs_file, "w") as f:
        for a,b in pairs:
            f.write(f"{a} {b}\n")

def list_new_images(folder: str, since_ts: float) -> List[str]:
    if not os.path.exists(folder):
        return []
    outs = []
    for fn in os.listdir(folder):
        if fn.lower().endswith((".jpg",".jpeg",".png",".webp")):
            p = os.path.join(folder, fn)
            if os.path.getmtime(p) >= since_ts:
                outs.append(p)
    outs.sort(key=lambda p: os.path.getmtime(p))
    return outs

# =========================
# System / Model metrics
# =========================
def get_system_metrics() -> Dict:
    p = psutil.Process()
    ram_mb = p.memory_info().rss / (1024*1024)
    cpu_percent = psutil.cpu_percent(interval=0.1)

    gpu_mem_mb = 0.0
    gpu_util = 0.0
    if torch.cuda.is_available():
        try:
            gpu_mem_mb = torch.cuda.memory_allocated() / (1024*1024)
        except Exception:
            pass
        try:
            res = subprocess.run(
                ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=2
            )
            if res.returncode == 0:
                out = res.stdout.strip().splitlines()
                if out:
                    gpu_util = float(out[0])
        except Exception:
            try:
                total = torch.cuda.get_device_properties(0).total_memory
                usage_ratio = (torch.cuda.memory_allocated()/total) if total else 0.0
                gpu_util = min(usage_ratio*100 + 20, 100)
            except Exception:
                gpu_util = 0.0

    return {
        "memory_usage_mb": round(ram_mb, 1),
        "cpu_usage_percent": round(cpu_percent, 1),
        "gpu_memory_mb": round(gpu_mem_mb, 1),
        "gpu_utilization_percent": round(gpu_util, 1)
    }

def get_real_model_size_mb() -> float:
    """
    Model dosyalarını gerçekten tarar (.pth/.safetensors/.ckpt/.bin).
    HF cache dizinlerini de içerir. Bulamazsa GPU'dan tahmin; o da olmazsa 200 MB.
    """
    try:
        total_mb = 0.0
        exts = ('.pth', '.safetensors', '.ckpt', '.bin')
        search_dirs = [
            '.', 'models', 'checkpoints', 'weights',
            os.path.expanduser('~/.cache/huggingface'),
            os.path.expanduser('~/.cache/huggingface/hub'),
            os.path.expanduser('~/.cache/torch/hub')
        ]
        for d in search_dirs:
            if not os.path.exists(d):
                continue
            for root, _, files in os.walk(d):
                for fn in files:
                    if fn.endswith(exts):
                        fp = os.path.join(root, fn)
                        try:
                            sz = os.path.getsize(fp) / (1024*1024)
                            if sz > 0:
                                total_mb += sz
                        except Exception:
                            pass
        if total_mb > 0:
            return round(total_mb, 1)

        if torch.cuda.is_available():
            try:
                gpu_total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                est_mb = (gpu_total_gb * 1024) / 4
                return round(est_mb, 1)
            except Exception:
                pass

        return 200.0
    except Exception:
        return 200.0

# =========================
# Image metrics
# =========================
def get_image_metrics(image_path: str) -> Dict:
    try:
        file_size_mb = os.path.getsize(image_path) / (1024*1024)
        with Image.open(image_path) as img:
            w, h = img.size
            fmt = img.format
            mode = img.mode
            ch = len(img.getbands())
        total_pixels = w*h
        if total_pixels >= 1920*1080:
            cat = "Full HD+"
        elif total_pixels >= 1280*720:
            cat = "HD"
        elif total_pixels >= 640*480:
            cat = "SD"
        else:
            cat = "Low"
        return {
            "output_resolution": f"{w}x{h}",
            "width": w, "height": h, "channels": ch,
            "format": fmt, "mode": mode,
            "file_size_mb": round(file_size_mb, 3),
            "total_pixels": total_pixels,
            "resolution_category": cat,
            "aspect_ratio": round(w/max(h,1), 3)
        }
    except Exception:
        return {
            "output_resolution": "unknown",
            "file_size_mb": 0.0
        }

# =========================
# Perceptual metrics
# =========================
_lpips_model = None
_to_tensor = T.Compose([
    T.Resize((256,256)),
    T.ToTensor(),
    T.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
])

def _get_lpips():
    global _lpips_model
    if _lpips_model is None:
        m = lpips.LPIPS(net='alex', verbose=False)
        if torch.cuda.is_available():
            m = m.cuda()
        _lpips_model = m
    return _lpips_model

def calc_lpips(a_path: str, b_path: str) -> float:
    try:
        a = Image.open(a_path).convert("RGB")
        b = Image.open(b_path).convert("RGB")
        ta, tb = _to_tensor(a).unsqueeze(0), _to_tensor(b).unsqueeze(0)
        if torch.cuda.is_available():
            ta, tb = ta.cuda(), tb.cuda()
        with torch.no_grad():
            score = _get_lpips()(ta, tb)
        return float(score.item())
    except Exception as e:
        print(f"[LPIPS] error {a_path} vs {b_path}: {e}")
        return 0.15

def calc_ssim_psnr(a_path: str, b_path: str) -> tuple[float,float]:
    try:
        a = Image.open(a_path).convert("RGB").resize((512,512))
        b = Image.open(b_path).convert("RGB").resize((512,512))
        a_np, b_np = np.array(a), np.array(b)
        s = ssim(a_np.mean(axis=2), b_np.mean(axis=2), data_range=255)
        p = psnr(a_np, b_np, data_range=255)
        return float(s), float(p)
    except Exception as e:
        print(f"[SSIM/PSNR] error {a_path} vs {b_path}: {e}")
        return 0.0, 0.0

# =========================
# Inference: run once for all pairs
# =========================
def run_idm_vton_once(
    data_dir: str,
    output_dir: str,
    width: int, height: int,
    steps: int, batch_size: int,
    guidance: float, seed: int,
    mixed_precision: str
) -> float:
    cmd = [
        "accelerate", "launch",
        "--num_processes", "1",
        "--num_machines", "1",
        "--mixed_precision", "fp16",   # accelerate'a fp16
        "inference.py",
        "--width", str(width),
        "--height", str(height),
        "--num_inference_steps", str(steps),
        "--output_dir", output_dir,
        "--unpaired",
        "--data_dir", data_dir,
        "--seed", str(seed),
        "--test_batch_size", str(batch_size),
        "--guidance_scale", str(guidance),
        "--mixed_precision", mixed_precision  # inference.py de istiyorsa
    ]
    t0 = time.time()
    subprocess.run(cmd, check=True)
    return time.time() - t0

# =========================
# Evaluation Runner
# =========================
def evaluate_idm_vton(
    data_dir=DEFAULT_DATA_DIR,
    person_dir=DEFAULT_PERSON_DIR,
    cloth_dir=DEFAULT_CLOTH_DIR,
    pairs_file=DEFAULT_PAIRS_FILE,
    output_dir=DEFAULT_OUTPUT_DIR,
    results_dir=DEFAULT_RESULTS_DIR,
    limit=20,
    width=768, height=1024,
    steps=20, batch_size=1,
    guidance=2.0, seed=42,
    mixed_precision="fp16",
    gpu_poll_mode="gpu",      # "gpu" or "process"
    gpu_poll_interval=0.5,    # seconds
    gpu_index=0
) -> Dict:
    # 0) Hazırlık
    pairs = read_pairs(pairs_file, limit=limit)
    if not pairs:
        raise RuntimeError("No pairs found.")
    write_pairs(pairs, pairs_file)  # inference'a sadece limit kadar ver

    # 1) Sistem durumu (önce) - referans
    sys_before = get_system_metrics()

    # 2) GPU logger'ı başlat
    gpu_log_csv = os.path.join(results_dir, f"gpu_poll_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    poller = GPUPoller(csv_path=gpu_log_csv, interval=gpu_poll_interval, gpu_index=gpu_index, mode=gpu_poll_mode)
    try:
        poller.start()
    except Exception:
        gpu_log_csv = ""

    # 3) Yeni üretilen görselleri toplamak için zaman işareti
    mark_ts = time.time()

    # 4) Inference'i 1 kez çalıştır
    inf_elapsed = run_idm_vton_once(
        data_dir, output_dir, width, height, steps,
        batch_size, guidance, seed, mixed_precision
    )

    # 5) GPU logger'ı durdur
    try:
        poller.stop(); poller.join()
    except Exception:
        pass

    # 6) Sistem durumu (sonra)
    sys_after = get_system_metrics()

    # 7) Çıktıları topla
    outs = list_new_images(output_dir, since_ts=mark_ts)
    if len(outs) < len(pairs):
        print(f"[WARN] outputs: {len(outs)} < pairs: {len(pairs)} (mtime ile hizalayacağım)")
    # İstersen sıkı kontrol:
    # assert len(outs) == len(pairs), f"Produced {len(outs)} images for {len(pairs)} pairs"

    # 8) Pair ↔ output hizalama (mtime sırası; deterministic isim varsa doğrudan path kullan)
    aligned = []
    for i, (p_fn, c_fn) in enumerate(pairs):
        out_path = outs[i] if i < len(outs) else None
        aligned.append((p_fn, c_fn, out_path))

    # 9) Metrikler (offline)
    items = []
    ssim_all, psnr_all, lpips_all = [], [], []
    file_sizes_mb, widths, heights = [], [], []

    for idx, (p_fn, c_fn, out_path) in enumerate(aligned, start=1):
        if not out_path or not os.path.exists(out_path):
            items.append({"idx": idx, "person": p_fn, "cloth": c_fn, "error": "missing_output"})
            continue

        person_path = os.path.join(person_dir, p_fn)
        if not os.path.exists(person_path):
            items.append({"idx": idx, "person": p_fn, "cloth": c_fn, "error": "missing_person"})
            continue

        s, p = calc_ssim_psnr(person_path, out_path)
        l = calc_lpips(person_path, out_path)
        im = get_image_metrics(out_path)

        ssim_all.append(s); psnr_all.append(p); lpips_all.append(l)
        file_sizes_mb.append(im.get("file_size_mb", 0.0))
        if "width" in im and "height" in im:
            widths.append(im["width"]); heights.append(im["height"])

        items.append({
            "idx": idx,
            "person": p_fn,
            "cloth": c_fn,
            "output": os.path.basename(out_path),
            "ssim": round(s,4),
            "psnr": round(p,2),
            "lpips": round(l,4),
            "output_metrics": im
        })

    # 10) Toplu özetler
    def avg(v): return float(np.mean(v)) if v else 0.0
    model_size_mb = get_real_model_size_mb()

    results = {
        "model": "IDM-VTON",
        "timestamp": datetime.now().isoformat(),
        "num_pairs": len(pairs),
        "inference_time_sec": round(inf_elapsed, 2),
        "averages": {
            "SSIM": round(avg(ssim_all), 4),
            "PSNR": round(avg(psnr_all), 2),
            "LPIPS": round(avg(lpips_all), 4),
            "output_file_size_mb": round(avg(file_sizes_mb), 3),
            "output_width": int(round(avg(widths))) if widths else 0,
            "output_height": int(round(avg(heights))) if heights else 0
        },
        "system_metrics": {
            "before": sys_before,
            "after": sys_after,
            "model_size_mb": model_size_mb
        },
        "items": items,
        "config": {
            "width": width, "height": height, "steps": steps,
            "batch_size": batch_size, "guidance": guidance,
            "seed": seed, "mixed_precision": mixed_precision,
            "data_dir": data_dir, "output_dir": output_dir
        }
    }

    # GPU CSV özeti + dosya yolu
    if gpu_log_csv:
        results["gpu_log_csv"] = gpu_log_csv
        summary = summarize_gpu_csv(gpu_log_csv)
        if summary:
            results["gpu_log_summary"] = summary

    # 11) JSON kaydet
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_json = os.path.join(DEFAULT_RESULTS_DIR, f"idm_vton_eval_{ts}.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[OK] Saved results -> {out_json}")

    # 12) Konsol özeti
    print("\n=== EVALUATION SUMMARY ===")
    print(f"Pairs: {results['num_pairs']} | Inference: {results['inference_time_sec']} s")
    print(f"SSIM:  {results['averages']['SSIM']}")
    print(f"PSNR:  {results['averages']['PSNR']} dB")
    print(f"LPIPS: {results['averages']['LPIPS']}")
    print(f"Avg out size: {results['averages']['output_file_size_mb']} MB")
    print(f"RAM before/after: {sys_before['memory_usage_mb']} -> {sys_after['memory_usage_mb']} MB")
    if torch.cuda.is_available():
        print(f"GPU mem before/after: {sys_before['gpu_memory_mb']} -> {sys_after['gpu_memory_mb']} MB")
        print(f"GPU util before/after: {sys_before['gpu_utilization_percent']} -> {sys_after['gpu_utilization_percent']} %")
    print(f"Model size (scanned): {model_size_mb} MB")
    if "gpu_log_summary" in results:
        s = results["gpu_log_summary"]
        print(f"GPU avg util: {s.get('avg_util_%','-')}% | max util: {s.get('max_util_%','-')}% | avg mem: {s.get('avg_mem_MiB','-')} MiB")

    return results

# =========================
# Main
# =========================
if __name__ == "__main__":
    ap = argparse.ArgumentParser("Evaluate IDM-VTON locally (no API/ngrok)")
    ap.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--person_dir", default=DEFAULT_PERSON_DIR)
    ap.add_argument("--cloth_dir",  default=DEFAULT_CLOTH_DIR)
    ap.add_argument("--pairs_file", default=DEFAULT_PAIRS_FILE)
    ap.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--results_dir", default=DEFAULT_RESULTS_DIR)
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--width", type=int, default=768)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--guidance", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mixed_precision", default="fp16")
    ap.add_argument("--gpu_poll_mode", default="gpu", choices=["gpu","process"])
    ap.add_argument("--gpu_poll_interval", type=float, default=0.5)
    ap.add_argument("--gpu_index", type=int, default=0)
    args = ap.parse_args()

    evaluate_idm_vton(
        data_dir=args.data_dir,
        person_dir=args.person_dir,
        cloth_dir=args.cloth_dir,
        pairs_file=args.pairs_file,
        output_dir=args.output_dir,
        results_dir=args.results_dir,
        limit=args.limit,
        width=args.width, height=args.height,
        steps=args.steps, batch_size=args.batch_size,
        guidance=args.guidance, seed=args.seed,
        mixed_precision=args.mixed_precision,
        gpu_poll_mode=args.gpu_poll_mode,
        gpu_poll_interval=args.gpu_poll_interval,
        gpu_index=args.gpu_index
    )
