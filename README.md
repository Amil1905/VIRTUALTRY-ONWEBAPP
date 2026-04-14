# 🧥 Virtual Try-On Models vs E-Commerce

> MSc Artificial Intelligence Dissertation — Brunel University London (2025)  
> **Grade: A+ (Distinction)**

A full-stack web application that embeds and compares three state-of-the-art Virtual Try-On (VTON) AI models — **VITON-HD (GAN)**, **IDM-VTON (Diffusion)**, and **PromptDresser (Prompt-based)** — inside a simulated e-commerce platform, evaluating them not just on image quality but on real-world deployment metrics: inference time, GPU memory, scalability, and user experience.

📹 **[Watch the demo video](https://youtu.be/nkdvynezQco)**

---

## 🔍 Research Question

> *"Which virtual try-on model offers the best trade-off between quality, performance, and scalability for e-commerce integration?"*

Most VTON research compares models only on image quality metrics (SSIM, PSNR, LPIPS). This project goes further — embedding all three models into a working e-commerce prototype to evaluate them as a retailer would: latency, cost, scalability, and what users actually see.

---

## 🏗️ System Architecture

```
User (Browser)
    │
    ▼
React-Vite Frontend
    │  POST /try-on
    ▼
FastAPI Backend (Google Colab + ngrok tunnel)
    ├──▶ VITON-HD     → Generated image
    ├──▶ IDM-VTON     → Generated image  
    └──▶ PromptDresser → Generated image (+ 3 style variants)
    │
    ▼
Frontend renders results side-by-side
User selects preferred model / style
```

**Tech Stack:**
- **Frontend:** React + Vite + Tailwind CSS
- **Backend:** Python + FastAPI
- **ML Models:** PyTorch, HuggingFace Diffusers, NumPy
- **Infrastructure:** Google Colab Pro+ (A100/T4 GPU), ngrok tunnelling, Google Drive
- **Fine-tuning:** LoRA (applied to PromptDresser on textured garments)
- **Dataset:** High-Resolution VITON-Zalando Dataset (Kaggle)

---

## 🤖 Models Compared

| Model | Paradigm | Resolution | Key Innovation |
|-------|----------|------------|----------------|
| **VITON-HD** | GAN-based | 1024×768 | ALIAS Normalization for misalignment correction |
| **IDM-VTON** | Diffusion-based | 768×1024 | Dual UNet (TryonNet + GarmentNet) + CLIP IP-Adapter |
| **PromptDresser** | Prompt/LMM-driven | 768×1024 | Text prompt control for style editing (casual/smart/formal) |
| **PromptDresser + LoRA** | Fine-tuned | 768×1024 | Parameter-efficient fine-tuning on textured garments |

---

## 📊 Results Summary

### Quality Metrics (400 test cases across 4 garment categories)

| Model | SSIM ↑ | PSNR ↑ | LPIPS ↓ |
|-------|--------|--------|---------|
| VITON-HD | **0.811** | 14.47 | 0.222 |
| IDM-VTON | 0.805 | **14.55** | **0.218** |
| PromptDresser | 0.799 | 14.19 | 0.222 |
| PromptDresser + LoRA | 0.762 | 14.29 | 0.240 |

> ⚠️ **Key finding:** Quantitative metrics don't tell the whole story. VITON-HD scores highest on SSIM but produces blurry, oversmoothed outputs in practice. PromptDresser scores lowest yet preserves logos and textures better visually — a significant metric–perception gap.

### Performance & Scalability

| Model | Inference Time | Model Size | GPU Memory |
|-------|---------------|------------|------------|
| **VITON-HD** | **~0.87s** ✅ | **1.1 GB** ✅ | 0.6 GB |
| IDM-VTON | ~6.5s | 53 GB ❌ | **0.4 GB** ✅ |
| PromptDresser | ~5.7s | 9.7 GB | 3.3 GB |

> Research shows that latency above 2–3 seconds increases user abandonment on e-commerce platforms (Basalla et al., 2021). Only VITON-HD clears this threshold.

### Key Takeaway — No Single Winner

| Use Case | Recommended Model |
|----------|------------------|
| Free tier / instant preview | VITON-HD (fast, lightweight) |
| Premium quality output | IDM-VTON (best PSNR/LPIPS) |
| Style customisation / engagement | PromptDresser (prompt control) |

**→ The dissertation proposes a hybrid deployment strategy:** use VITON-HD as a free fast preview, with IDM-VTON and PromptDresser as premium tiers — mirroring subscription models used by SaaS products.

---

## 🖥️ Web Application

The frontend simulates an Amazon-style e-commerce platform. Users can:
- Browse a garment catalogue
- Upload their own photo
- Select any of the three VTON models
- View results side-by-side
- Use PromptDresser's style feature to see the same garment in **Casual, Smart Casual, and Formal** styles

Screenshots from the live app:

> Main product catalogue → product detail page → model selection → try-on result → style comparison

---

## 📁 Repository Structure

```
├── src/
│   └── components/
│       └── TestComparison.jsx   # Main web app frontend
├── public/
├── appendix_2449223/            # Backend, evaluation & LoRA fine-tuning code
├── CS5500_2449223.pdf           # Full dissertation
├── test_pairsamil.txt           # Test pair definitions
└── README.md
```

> **Note:** Model checkpoints were removed after project completion due to
> cloud hosting costs. The full frontend code, backend architecture, LoRA
> fine-tuning scripts, and evaluation pipeline are available in this repo.
> A live demo recording is available on YouTube (link above).

---


## 🧪 Evaluation Setup

- **Dataset:** High-Resolution VITON-Zalando (Kaggle)
- **Test cases:** 400 person–garment pairs
- **Garment categories:** Plain, Logo-heavy, Textured, Extreme (tank tops/hoodies)
- **Quality metrics:** SSIM, PSNR, LPIPS
- **Performance metrics:** Inference time, GPU memory, model size, output file size
- **Statistical testing:** Wilcoxon signed-rank test for LoRA improvements
- **Hardware:** Google Colab Pro+ (A100 / T4 GPU)

---

## 📄 LoRA Fine-Tuning

LoRA was applied to PromptDresser on the **textured garment** category (worst-performing). Only mid and upper UNet attention layers (k, q, v projections) were tuned due to VRAM constraints.

Results: modest but consistent improvement — SSIM improved in **46%** of test cases, LPIPS in **41%**. Wilcoxon test: not statistically significant (p > 0.05), but shows promise for targeted garment adaptation.

---

## 📚 Citation

```
Kazimoglu, A. (2025). Virtual Try-On Models vs E-Commerce.
MSc Artificial Intelligence Dissertation, Brunel University London.
```

---

## 👤 Author

**Amil Kazimoglu**  
MSc Artificial Intelligence (Distinction) — Brunel University London  
BSc Computer Science — Sabancı University  
[LinkedIn](https://linkedin.com/in/amil-kazımoğlu-74b12a269) · [GitHub](https://github.com/Amil1905)
