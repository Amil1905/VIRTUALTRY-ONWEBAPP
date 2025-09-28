import os
import json
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class PromptDresserStyleDataset(Dataset):
    """
    MINIMAL DATASET (NO STYLE, NO AUTO-FILTER)
    - Hiçbir otomatik desen/keyword/JSON filtresi YOK.
    - Senin verdiğin pairs dosyasını (örn. test_pairs.txt) AYNEN okur.
    - Prompt varsa JSON'dan alır; yoksa boş string döner.
    """
    def __init__(self, data_dir, mode='fine', image_size=512, return_pose=False, pairs_file='test_pairs.txt', prompts_json=None):
        self.data_dir = data_dir
        self.mode = mode
        self.image_size = image_size
        self.return_pose = return_pose
        self.pairs_file = pairs_file
        self.prompts = self._load_prompts(prompts_json)

        self.pairs = self._load_pairs()

        # FIX: Use Image.Resampling instead of InterpolationMode
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=Image.Resampling.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    # -----------------------------
    # I/O
    # -----------------------------
    def _load_prompts(self, prompts_json):
        if not prompts_json:
            return {}
        p = prompts_json if os.path.isabs(prompts_json) else os.path.join(self.data_dir, prompts_json)
        if not os.path.exists(p):
            print(f"[WARN] prompt json not found: {p}")
            return {}
        try:
            with open(p, 'r') as f:
                data = json.load(f)
            print(f"[INFO] Loaded prompts from: {os.path.basename(p)}")
            return data
        except Exception as e:
            print(f"[WARN] failed to load prompts: {e}")
            return {}

    def _load_pairs(self):
        pairs_path = os.path.join(self.data_dir, self.pairs_file)
        pairs = []
        if not os.path.exists(pairs_path):
            print(f"[WARN] pairs file not found: {pairs_path}")
            return pairs
        with open(pairs_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                toks = line.split()
                if len(toks) < 2:
                    continue
                person_img, cloth_img = toks[0], toks[1]
                pairs.append((person_img, cloth_img))
        print(f"[INFO] Loaded {len(pairs)} pairs from {self.pairs_file}")
        return pairs

    # -----------------------------
    # Loaders
    # -----------------------------
    def _load_image(self, path):
        try:
            img = Image.open(path).convert('RGB')
        except Exception as e:
            print(f"[WARN] Error loading image {path}: {e}; using black dummy.")
            img = Image.new('RGB', (self.image_size, self.image_size), color='black')
        return self.transform(img)

    def _load_mask(self, path):
        try:
            m = Image.open(path).convert('L')
            # FIX: Use Image.Resampling.NEAREST instead of IM.NEAREST
            m = m.resize((self.image_size, self.image_size), resample=Image.Resampling.NEAREST)
            arr = np.array(m, dtype=np.float32) / 255.0
            return torch.from_numpy(arr).unsqueeze(0)
        except Exception as e:
            print(f"[WARN] Error loading mask {path}: {e}; using 0.5 dummy.")
            return torch.full((1, self.image_size, self.image_size), 0.5, dtype=torch.float32)

    def _find_mask_path(self, person_id):
        tries = [
            os.path.join(self.data_dir, 'test_coarse', 'agnostic-mask', f'{person_id}_mask.png'),
            os.path.join(self.data_dir, f'test_{self.mode}', 'agnostic-mask', f'{person_id}_mask.png'),
            os.path.join(self.data_dir, 'test_coarse', 'agnostic-mask', f'{person_id}.png'),
            os.path.join(self.data_dir, f'test_{self.mode}', 'agnostic-mask', f'{person_id}.png'),
        ]
        for p in tries:
            if os.path.exists(p):
                return p
        return None

    def _find_pose_path(self, person_id):
        tries = [
            os.path.join(self.data_dir, f'test_{self.mode}', 'image-densepose', f'{person_id}.png'),
            os.path.join(self.data_dir, f'test_{self.mode}', 'image-densepose', f'{person_id}.jpg'),
            os.path.join(self.data_dir, f'test_{self.mode}', 'openpose_img', f'{person_id}_rendered.png'),
        ]
        for p in tries:
            if os.path.exists(p):
                return p
        return None

    # -----------------------------
    # PyTorch API
    # -----------------------------
    def __getitem__(self, idx):
        person_img_name, cloth_img_name = self.pairs[idx]
        person_id = os.path.splitext(person_img_name)[0]
        cloth_id  = os.path.splitext(cloth_img_name)[0]

        person_path = os.path.join(self.data_dir, f'test_{self.mode}', 'image', person_img_name)
        cloth_path  = os.path.join(self.data_dir, f'test_{self.mode}', 'cloth', cloth_img_name)

        person_img = self._load_image(person_path)
        cloth_img  = self._load_image(cloth_path)
        target_img = person_img  # clone yok — daha az RAM/VRAM

        mask_path = self._find_mask_path(person_id)
        mask = self._load_mask(mask_path) if mask_path else torch.full(
            (1, self.image_size, self.image_size), 0.5, dtype=torch.float32
        )

        if self.return_pose:
            pose_path = self._find_pose_path(person_id)
            pose_img = self._load_image(pose_path) if pose_path else person_img
        else:
            pose_img = torch.zeros(1, self.image_size, self.image_size, dtype=torch.float32)

        # Prompt: varsa JSON'dan; yoksa boş string
        prompt = ""
        if self.prompts:
            data = self.prompts.get(person_id) or self.prompts.get(cloth_id)
            if isinstance(data, dict):
                prompt = data.get('prompt') or data.get('caption') or ""
            elif isinstance(data, str):
                prompt = data

        return {
            'person': person_img,
            'cloth': cloth_img,
            'target': target_img,
            'mask': mask,
            'pose': pose_img,
            'prompt': prompt,
            'person_id': person_id,
            'cloth_id': cloth_id
        }

    def __len__(self):
        return len(self.pairs)

if __name__ == "__main__":
    ds = PromptDresserStyleDataset(
        data_dir="./DATA/zalando-hd-resized",
        mode="fine",
        image_size=512,
        return_pose=False,
        pairs_file='test_pairs.txt',
        prompts_json=None
    )
    print(f"Dataset size: {len(ds)}")
    for i in range(min(3, len(ds))):
        s = ds[i]
        print(f"[{i}] {s['person_id']} -> {s['cloth_id']} | prompt[:60]={s['prompt'][:60]!r}")
