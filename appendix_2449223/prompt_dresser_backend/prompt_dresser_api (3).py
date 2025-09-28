from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os, shutil, time, base64, glob, gc, json
from pyngrok import ngrok
from PIL import Image
import numpy as np
import torch
from typing import Optional, Dict, Any
from omegaconf import OmegaConf
from accelerate import Accelerator
from diffusers import AutoencoderKL, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer, CLIPTextModelWithProjection
from fastapi import Form

# PromptDresser imports
from promptdresser.utils import zero_rank_print_, get_inputs, load_file
from promptdresser.models.unet import UNet2DConditionModel
from promptdresser.models.cloth_encoder import ClothEncoder
from promptdresser.models.mutual_self_attention import ReferenceAttentionControl
from promptdresser.pipelines.sdxl import PromptDresser as PromptDresserPipeline

app = FastAPI(title="PromptDresser API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Global model components
pipeline = None
cloth_encoder = None
unet = None
accelerator = None
config = None

# Style data
style_data = {}
test_pairs = []

# Paths - Klasör yapından
CONFIG_PATH = "./configs/VITONHD.yaml"
SAVE_ROOT_DIR = "./sampled_images"
SAVE_NAME = "api_demo"
SAVE_DIR = os.path.join(SAVE_ROOT_DIR, SAVE_NAME)

# Model paths - Klasör yapından
INIT_MODEL_PATH = "./pretrained_models/stable-diffusion-xl-1.0-inpainting-0.1"
INIT_VAE_PATH = "./pretrained_models/sdxl-vae-fp16-fix"  
INIT_CLOTH_ENCODER_PATH = "./pretrained_models/stable-diffusion-xl-base-1.0"

# Style data paths
STYLE_JSON_PATH = "./DATA/zalando-hd-resized/test_gpt4o.json"  # JSON dosyası path'i
TEST_PAIRS_PATH = "./DATA/zalando-hd-resized/test_pairs.txt"         # Test pairs dosyası

os.makedirs(SAVE_DIR, exist_ok=True)

def load_style_data():
    """Style JSON ve test pairs dosyalarını yükle"""
    global style_data, test_pairs
    
    # Style JSON yükle
    if os.path.exists(STYLE_JSON_PATH):
        with open(STYLE_JSON_PATH, 'r', encoding='utf-8') as f:
            style_data = json.load(f)
        print(f"Style data loaded: {len(style_data)} entries")
    else:
        print(f"Warning: Style JSON not found at {STYLE_JSON_PATH}")
        style_data = {}
    
    # Test pairs yükle (opsiyonel)
    if os.path.exists(TEST_PAIRS_PATH):
        with open(TEST_PAIRS_PATH, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    person_file, cloth_file = parts
                    test_pairs.append((person_file, cloth_file))
        print(f"Test pairs loaded: {len(test_pairs)} pairs")
    else:
        print(f"Test pairs not found - using fallback styles")

def find_cloth_style_data(cloth_filename: str) -> Dict[str, Any]:
    """Cloth filename'e göre style data bul"""
    cloth_id = os.path.splitext(cloth_filename)[0]
    
    # Direct match - JSON'da direkt bu ID var mı?
    if cloth_id in style_data:
        return style_data[cloth_id]
    
    # Test pairs'de ara
    for person_file, cloth_file in test_pairs:
        if cloth_file == cloth_filename:
            person_id = os.path.splitext(person_file)[0]
            if person_id in style_data:
                return style_data[person_id]
    
    # Fallback - default style data
    print(f"No style data found for {cloth_id}, using default style")
    return {
        "person": {
            "body shape": "slim",
            "fit of upper cloth": "relaxed",
            "tucking style": "untucked", 
            "sleeve rolling style": "short sleeve",
            "gender": "woman",
            "hair length": "medium",
            "pose": "standing straight, looking at camera",
            "hand pose": "hands relaxed at sides"
        },
        "clothing": {
            "upper cloth category": "t-shirt",
            "material": "cotton", 
            "neckline": "crew",
            "sleeve": "short"
        }
    }

def create_style_prompt(person_data: Dict, clothing_data: Dict, style_overrides: Dict = None) -> tuple:
    """Style data'dan prompt oluştur"""
    
    # Override'ları uygula
    if style_overrides:
        person_data = person_data.copy()
        clothing_data = clothing_data.copy()
        
        for key, value in style_overrides.items():
            if key in person_data:
                person_data[key] = value
            elif key in clothing_data:
                clothing_data[key] = value
    
    # Main prompt template (PromptDresser paper'ındaki gibi)
    main_prompt = (
        f"a {person_data.get('body shape', 'slender')} {person_data.get('gender', 'woman')} "
        f"wears {person_data.get('fit of upper cloth', 'relaxed')}, "
        f"{clothing_data.get('upper cloth category', 't-shirt')} "
        f"({clothing_data.get('material', 'cotton')}), "
        f"{clothing_data.get('neckline', 'crew')}, "
        f"{person_data.get('sleeve rolling style', 'short sleeve')}, "
        f"{person_data.get('tucking style', 'untucked')}. "
        f"With {person_data.get('hair length', 'medium')} hair, "
        f"{person_data.get('pose', 'standing straight')} "
        f"with hands {person_data.get('hand pose', 'relaxed at sides')}"
    )
    
    # Reference prompt (clothing only)
    reference_prompt = (
        f"a {clothing_data.get('upper cloth category', 't-shirt')}, "
        f"{clothing_data.get('material', 'cotton')}, "
        f"with {clothing_data.get('sleeve', 'short')}, "
        f"{clothing_data.get('neckline', 'crew')}"
    )
    
    return main_prompt, reference_prompt

def initialize_model():
    """Model yükleme - tam senin inference kodun gibi"""
    global pipeline, cloth_encoder, unet, accelerator, config
    
    if pipeline is not None:
        return "Already loaded"
    
    print("Loading PromptDresser...")
    
    # Config yükle
    if not os.path.exists(CONFIG_PATH):
        print(f"Config file not found: {CONFIG_PATH}")
        return "Config file not found"
    
    config = OmegaConf.load(CONFIG_PATH)
    print(f"Config loaded - using test_folder_name: {config.dataset.test_folder_name}")
    
    accelerator = Accelerator(mixed_precision="fp16")
    weight_dtype = torch.float16

    # Model bileşenlerini yükle
    noise_scheduler = DDPMScheduler.from_pretrained(
        INIT_MODEL_PATH, subfolder="scheduler", 
        rescale_betas_zero_snr=True, 
        timestep_spacing="leading"
    )
    tokenizer = CLIPTokenizer.from_pretrained(INIT_MODEL_PATH, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(INIT_MODEL_PATH, subfolder="text_encoder")
    tokenizer_2 = CLIPTokenizer.from_pretrained(INIT_MODEL_PATH, subfolder="tokenizer_2")
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(INIT_MODEL_PATH, subfolder="text_encoder_2")
    vae = AutoencoderKL.from_pretrained(INIT_VAE_PATH)
    unet = UNet2DConditionModel.from_pretrained(INIT_MODEL_PATH, subfolder="unet")
    cloth_encoder = ClothEncoder.from_pretrained(INIT_CLOTH_ENCODER_PATH, subfolder="unet")

    # Ayarlar
    unet.add_clothing_text = False
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    cloth_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)

    # Device'a taşı
    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    text_encoder_2.to(accelerator.device, dtype=weight_dtype)
    cloth_encoder.to(accelerator.device, dtype=weight_dtype)

    # Reference attention control - senin kodun gibi
    if not config.get("detach_cloth_encoder", False):
        reference_control_writer = ReferenceAttentionControl(
            cloth_encoder, 
            do_classifier_free_guidance=True, 
            mode="write", fusion_blocks="full",
            batch_size=1, 
            is_train=True,
            is_second_stage=False, 
            use_jointcond=config.get("use_jointcond", False)
        )
        reference_control_reader = ReferenceAttentionControl(
            unet, 
            do_classifier_free_guidance=True, 
            mode="read", 
            fusion_blocks="full", 
            batch_size=1, 
            is_train=True, 
            is_second_stage=False, 
            use_jointcond=config.get("use_jointcond", False)
        )

    # Pretrained model yükle - senin komutunda var
    pretrained_unet_path = "./checkpoints/VITONHD/model/pytorch_model.bin"
    if os.path.exists(pretrained_unet_path):
        unet.load_state_dict(load_file(pretrained_unet_path))
        print(f"UNet loaded from {pretrained_unet_path}")
    else:
        print(f"Warning: {pretrained_unet_path} not found")

    # Pipeline oluştur
    pipeline = PromptDresserPipeline(
        vae=vae,
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        unet=unet,
        scheduler=noise_scheduler,
    ).to(accelerator.device, dtype=weight_dtype)

    pipeline.set_progress_bar_config(leave=False)
    
    print("PromptDresser loaded!")
    return "Model loaded successfully"

@app.on_event("startup")
async def startup():
    load_style_data()
    initialize_model()

@app.get("/")
def root():
    return {"status": "PromptDresser API Running", "model": "PromptDresser"}

@app.post("/tryon")
async def tryon(
    person_image: UploadFile = File(...),
    cloth_image: UploadFile = File(...),
    prompt: str = "A person wearing the garment",
    cloth_prompt: str = "A garment",
    style: str = "casual",
    tucking_style: str = "untucked",
    sleeve_rolling: str = "a long-sleeved with the sleeves down", 
    fit_style: str = "regular fit",
    num_inference_steps: int = 30,
    guidance_scale: float = 2.0,
    guidance_scale_img: float = 4.5,
    guidance_scale_text: float = 7.5,
    img_h: int = 1024,
    img_w: int = 768
):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = time.time()
    
    try:
        person_filename = person_image.filename
        cloth_filename = cloth_image.filename
        
        print(f"Processing: {person_filename} + {cloth_filename}")
        
        # Dosyaların dataset'te var olduğunu kontrol et
        dataset_root = "./DATA/zalando-hd-resized"
        person_path = os.path.join(dataset_root, "test_coarse", "image", person_filename)
        cloth_path = os.path.join(dataset_root, "test_coarse", "cloth", cloth_filename)
        
        if not os.path.exists(person_path):
            raise HTTPException(status_code=404, detail=f"Person image not found: {person_filename}")
        if not os.path.exists(cloth_path):
            raise HTTPException(status_code=404, detail=f"Cloth image not found: {cloth_filename}")
        
        print(f"Found files: {person_path}, {cloth_path}")

        # Inference - TAM senin kodun gibi
        img_fn = os.path.splitext(person_filename)[0]
        c_fn = os.path.splitext(cloth_filename)[0]
        result_path = os.path.join(SAVE_DIR, "paired", f"{img_fn}__{c_fn}.jpg")
        os.makedirs(os.path.dirname(result_path), exist_ok=True)

        # get_inputs ile preprocessing - config ile
        person, mask, pose, cloth = get_inputs(
            root_dir=config.dataset.data_root_dir,
            data_type="test",  # get_inputs internally uses test_folder_name
            pose_type=config.dataset.pose_type,
            img_bn=person_filename,
            c_bn=cloth_filename,
            img_h=img_h,
            img_w=img_w,
            train_folder_name=config.dataset.get("train_folder_name", None),
            test_folder_name=config.dataset.get("test_folder_name", None),
            category=config.dataset.get("category", None),
            pad_type=None,
            use_dc_cloth=config.dataset.get("use_dc_cloth", False),
        )

        # Style prompt ekle - basit versiyon
        full_txt = prompt
        cloth_txt = cloth_prompt
        
        # Style parametrelerini prompt'a ekle
        if tucking_style != "untucked":
            cloth_txt += f", {tucking_style}"
        if "sleeves down" not in sleeve_rolling:
            cloth_txt += f", {sleeve_rolling}"  
        if fit_style != "regular fit":
            full_txt += f" with {fit_style}"
            
        if config.get("use_style_prompt", True) and style:
            full_txt += f" This is a {style} outfit."

        print(f"Final prompt: {full_txt}")
        print(f"Final cloth prompt: {cloth_txt}")

        # Inference
        inference_start = time.time()
        with torch.autocast("cuda"):
            sample = pipeline(
                image=person, 
                mask_image=mask,
                pose_image=pose,
                cloth_encoder=cloth_encoder,
                cloth_encoder_image=cloth,
                prompt=full_txt,
                prompt_clothing=cloth_prompt,
                style=style,
                height=img_h, 
                width=img_w,
                guidance_scale=guidance_scale,
                guidance_scale_img=guidance_scale_img,
                guidance_scale_text=guidance_scale_text,
                num_inference_steps=num_inference_steps,
                use_jointcond=config.get("use_jointcond", False),
                interm_cloth_start_ratio=config.get("interm_cloth_start_ratio", 0.5),
                detach_cloth_encoder=config.get("detach_cloth_encoder", False),
                strength=1.0,
                category=config.dataset.get("category", None),
                generator=None,
            ).images[0]
            
            sample.save(result_path)

        inference_time = time.time() - inference_start
        total_time = time.time() - start_time
        
        # Memory cleanup
        torch.cuda.empty_cache()
        gc.collect()
        
        print(f"Completed in {total_time:.1f}s")
        
        headers = {
            "X-Processing-Time": str(round(total_time, 1)),
            "X-Inference-Time": str(round(inference_time, 1)),
            "X-Model": "PromptDresser"
        }
        
        return FileResponse(result_path, media_type="image/jpeg", headers=headers)
        
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# YENİ ENDPOINT: Style-aware try-on
@app.post("/tryon-style")
async def tryon_with_style(
    person_image: UploadFile = File(...),
    cloth_image: UploadFile = File(...),
    style_overrides: str = Form(default="{}"),  # JSON string olarak style override'ları
    num_inference_steps: int = 30,
    guidance_scale: float = 2.0,
    guidance_scale_img: float = 4.5,
    guidance_scale_text: float = 7.5,
    img_h: int = 1024,
    img_w: int = 768
):
    """Style JSON data'sını kullanarak try-on yap"""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = time.time()
    
    try:
        person_filename = person_image.filename
        cloth_filename = cloth_image.filename
        
        print(f"Processing with style: {person_filename} + {cloth_filename}")
        
        # Style overrides parse et
        print(f"Raw style_overrides parameter: {style_overrides}")
        print(f"Type: {type(style_overrides)}")

        try:
          style_overrides_dict = json.loads(style_overrides)
          print(f"Parsed successfully: {style_overrides_dict}")
        except json.JSONDecodeError as e:
          print(f"JSON parse error: {e}")
          style_overrides_dict = {}

        print(f"Final style overrides: {style_overrides_dict}")
        
        # Style data bul
        cloth_style_data = find_cloth_style_data(cloth_filename)
        person_data = cloth_style_data.get("person", {})
        clothing_data = cloth_style_data.get("clothing", {})
        
        # Dosyaların dataset'te var olduğunu kontrol et
        dataset_root = "./DATA/zalando-hd-resized"
        person_path = os.path.join(dataset_root, "test_coarse", "image", person_filename)
        cloth_path = os.path.join(dataset_root, "test_coarse", "cloth", cloth_filename)
        
        if not os.path.exists(person_path):
            raise HTTPException(status_code=404, detail=f"Person image not found: {person_filename}")
        if not os.path.exists(cloth_path):
            raise HTTPException(status_code=404, detail=f"Cloth image not found: {cloth_filename}")
        
        # Style prompt oluştur
        main_prompt, reference_prompt = create_style_prompt(
            person_data, clothing_data, style_overrides_dict
        )
        
        print(f"Main prompt: {main_prompt}")
        print(f"Reference prompt: {reference_prompt}")
        
        # Dosya yolları
        img_fn = os.path.splitext(person_filename)[0]
        c_fn = os.path.splitext(cloth_filename)[0]
        
        # Override string'i dosya adına ekle
        override_suffix = ""
        if style_overrides_dict:
            override_keys = "_".join(f"{k}-{v}" for k, v in style_overrides_dict.items())
            override_suffix = f"__{override_keys.replace(' ', '-')}"
        
        result_path = os.path.join(SAVE_DIR, "styled", f"{img_fn}__{c_fn}{override_suffix}.jpg")
        os.makedirs(os.path.dirname(result_path), exist_ok=True)

        # get_inputs ile preprocessing
        person, mask, pose, cloth = get_inputs(
            root_dir=config.dataset.data_root_dir,
            data_type="test",
            pose_type=config.dataset.pose_type,
            img_bn=person_filename,
            c_bn=cloth_filename,
            img_h=img_h,
            img_w=img_w,
            train_folder_name=config.dataset.get("train_folder_name", None),
            test_folder_name=config.dataset.get("test_folder_name", None),
            category=config.dataset.get("category", None),
            pad_type=None,
            use_dc_cloth=config.dataset.get("use_dc_cloth", False),
        )

        # Inference
        inference_start = time.time()
        with torch.autocast("cuda"):
            sample = pipeline(
                image=person, 
                mask_image=mask,
                pose_image=pose,
                cloth_encoder=cloth_encoder,
                cloth_encoder_image=cloth,
                prompt=main_prompt,
                prompt_clothing=reference_prompt,
                height=img_h, 
                width=img_w,
                guidance_scale=guidance_scale,
                guidance_scale_img=guidance_scale_img,
                guidance_scale_text=guidance_scale_text,
                num_inference_steps=num_inference_steps,
                use_jointcond=config.get("use_jointcond", False),
                interm_cloth_start_ratio=config.get("interm_cloth_start_ratio", 0.5),
                detach_cloth_encoder=config.get("detach_cloth_encoder", False),
                strength=1.0,
                category=config.dataset.get("category", None),
                generator=None,
            ).images[0]
            
            sample.save(result_path)

        inference_time = time.time() - inference_start
        total_time = time.time() - start_time
        
        # Memory cleanup
        torch.cuda.empty_cache()
        gc.collect()
        
        print(f"Completed in {total_time:.1f}s")
        
        headers = {
            "X-Processing-Time": str(round(total_time, 1)),
            "X-Inference-Time": str(round(inference_time, 1)),
            "X-Model": "PromptDresser",
            "X-Style-Data": json.dumps({"person": person_data, "clothing": clothing_data}),
            "X-Style-Overrides": style_overrides,
            "X-Main-Prompt": main_prompt,
            "X-Reference-Prompt": reference_prompt
        }
        
        return FileResponse(result_path, media_type="image/jpeg", headers=headers)
        
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# YENİ ENDPOINT: Style prompts al
@app.get("/get-prompts/{cloth_filename}")
async def get_cloth_prompts(cloth_filename: str):
    """Cloth için mevcut style variation'ları döndür"""
    try:
        cloth_style_data = find_cloth_style_data(cloth_filename)
        person_data = cloth_style_data.get("person", {})
        clothing_data = cloth_style_data.get("clothing", {})
        
        # 3 farklı style variation oluştur
        variations = [
            {
                "name": "Casual & Relaxed",
                "description": "Comfortable everyday look",
                "tucking": "untucked",
                "fit": "relaxed", 
                "sleeve_rolling": person_data.get("sleeve rolling style", "short sleeve")
            },
            {
                "name": "Smart Casual",
                "description": "Polished but comfortable",
                "tucking": "french tucked",
                "fit": "regular fit",
                "sleeve_rolling": "a long-sleeved with the sleeves down"
            },
            {
                "name": "Formal & Fitted",
                "description": "Sharp and professional",
                "tucking": "fully tucked in",
                "fit": "tight fit",
                "sleeve_rolling": "a long-sleeved with the sleeves down"
            }
        ]
        
        return {
            "clothing_description": clothing_data,
            "style_options": {
                "material": clothing_data.get("material", "cotton"),
                "neckline": clothing_data.get("neckline", "crew"),
                "sleeve": clothing_data.get("sleeve", "short"),
                "fit": person_data.get("fit of upper cloth", "relaxed")
            },
            "style_variations": variations
        }
        
    except Exception as e:
        return {"error": str(e)}

@app.get("/health")
def health():
    model_status = "loaded" if pipeline is not None else "not_loaded"
    return {
        "status": "healthy",
        "model": "PromptDresser",
        "model_status": model_status,
        "style_data_loaded": len(style_data),
        "test_pairs_loaded": len(test_pairs),
        "config_path": CONFIG_PATH,
        "save_dir": SAVE_DIR
    }

@app.get("/results")
def list_results():
    """Sonuçları listele"""
    try:
        files = glob.glob(os.path.join(SAVE_DIR, "**", "*.jpg"), recursive=True)
        
        results = []
        for f in sorted(files, key=os.path.getmtime, reverse=True)[:10]:
            results.append({
                "filename": os.path.basename(f),
                "path": f,
                "size_mb": round(os.path.getsize(f) / (1024*1024), 2),
                "modified": time.ctime(os.path.getmtime(f))
            })
        
        return {"results": results, "total": len(files)}
        
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    public_url = ngrok.connect(8003)
    print(f"PromptDresser API: {public_url}")
    print(f"Docs: {public_url}/docs")
    print(f"Style Test: curl -X POST '{public_url}/tryon-style' -F 'person_image=@person.jpg' -F 'cloth_image=@cloth.jpg' -F 'style_overrides=...'")
    
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)