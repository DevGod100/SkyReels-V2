# app.py
from fastapi import FastAPI, File, UploadFile
import shutil, os
from generate_video import main  # we’ll refactor generate_video.py in a sec

app = FastAPI()

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    prompt: str = "A cinematic scene",
    length: int = 5
):
    # save upload
    os.makedirs("uploads", exist_ok=True)
    upload_path = os.path.join("uploads", file.filename)
    with open(upload_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # call your existing CLI logic
    # we’ll refactor generate_video.py so that main() takes these args
    out_video = main(
      image=upload_path,
      prompt=prompt,
      length=length
    )
    return {"video": out_video}
