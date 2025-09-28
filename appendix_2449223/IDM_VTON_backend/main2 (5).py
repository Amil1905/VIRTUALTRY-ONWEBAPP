from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
import subprocess
import os
import shutil
import time
from pyngrok import ngrok
import uvicorn
from threading import Thread
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

DATA_DIR = "data"
TEST_PAIRS_FILE = os.path.join(DATA_DIR, "test_pairs.txt")
PERSON_DIR = os.path.join(DATA_DIR, "test", "image")
CLOTH_DIR = os.path.join(DATA_DIR, "test", "cloth")

INFERENCE_CMD = [
    "accelerate", "launch", "inference.py",
    "--width", "768",
    "--height", "1024",
    "--num_inference_steps", "20",
    "--output_dir", "result",
    "--unpaired",
    "--data_dir", DATA_DIR,
    "--seed", "42",
    "--test_batch_size", "1",
    "--guidance_scale", "2.0",
    "--mixed_precision", "fp16"
]

@app.get("/")
async def root():
    return {
        "message": "IDM-VTON API Ready",
        "model": "IDM-VTON",
        "version": "1.0.0"
    }

@app.post("/try-on")
async def try_on(
    person_image: UploadFile = File(...),
    cloth_image: UploadFile = File(...)
):
    """Simple IDM-VTON try-on without metrics"""
    try:
        # Get filenames
        person_filename = person_image.filename
        cloth_filename = cloth_image.filename
        
        print(f"Processing: {person_filename} + {cloth_filename}")
        
        # Save files
        person_path = os.path.join(PERSON_DIR, person_filename)
        cloth_path = os.path.join(CLOTH_DIR, cloth_filename)
        
        with open(person_path, "wb") as buffer:
            shutil.copyfileobj(person_image.file, buffer)
            
        with open(cloth_path, "wb") as buffer:
            shutil.copyfileobj(cloth_image.file, buffer)
        
        # Write test_pairs.txt
        with open(TEST_PAIRS_FILE, "w") as f:
            f.write(f"{person_filename} {cloth_filename}\n")
        
        # Run inference
        print("Starting inference...")
        result = subprocess.run(INFERENCE_CMD, check=True, capture_output=True, text=True)
        
        # Find latest result
        result_dir = "result"
        if os.path.exists(result_dir):
            result_files = [f for f in os.listdir(result_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
            
            if result_files:
                # Get newest file
                result_files_with_time = []
                for f in result_files:
                    file_path = os.path.join(result_dir, f)
                    mtime = os.path.getmtime(file_path)
                    result_files_with_time.append((f, mtime))
                
                latest_file = max(result_files_with_time, key=lambda x: x[1])[0]
                latest_path = os.path.join(result_dir, latest_file)
                
                print(f"Result ready: {latest_file}")
                return FileResponse(latest_path, media_type="image/jpeg")
            else:
                return JSONResponse(status_code=500, content={"error": "No result files found"})
        else:
            return JSONResponse(status_code=500, content={"error": "Result directory not found"})
            
    except subprocess.CalledProcessError as e:
        return JSONResponse(
            status_code=500, 
            content={"error": f"Inference failed: {e}"}
        )
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/view-result/{filename}")
async def view_result(filename: str):
    """View a result image"""
    try:
        file_path = os.path.join("result", filename)
        if os.path.exists(file_path):
            return FileResponse(file_path, media_type="image/jpeg")
        else:
            return JSONResponse(status_code=404, content={"error": "File not found"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/list-results")
async def list_results():
    """List all results"""
    try:
        result_dir = "result"
        if os.path.exists(result_dir):
            files = [f for f in os.listdir(result_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
            
            files_with_time = []
            for f in files:
                file_path = os.path.join(result_dir, f)
                mtime = os.path.getmtime(file_path)
                files_with_time.append({
                    "filename": f,
                    "modified_time": mtime,
                    "url": f"/view-result/{f}"
                })
            
            files_with_time.sort(key=lambda x: x["modified_time"], reverse=True)
            
            return {
                "result_files": files_with_time,
                "count": len(files_with_time)
            }
        else:
            return {"result_files": [], "message": "result folder not found"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/health")
async def health_check():
    """Health check"""
    return {
        "status": "healthy",
        "model": "IDM-VTON",
        "version": "1.0.0"
    }

if __name__ == "__main__":
    port = 8012
    public_url = ngrok.connect(port)
    print(f"Public URL: {public_url}")
    print(f"Docs: {public_url}/docs")

    def start_server():
        uvicorn.run("main2:app", host="0.0.0.0", port=port, reload=False)

    thread = Thread(target=start_server, daemon=True)
    thread.start()

    input("Server running. Press Enter to exit...\n")