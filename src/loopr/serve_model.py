import requests
from ray import serve
import ray
from fastapi import FastAPI, File, UploadFile, Form
from fastapi import FastAPI, File, Form, UploadFile, Request
import requests
from starlette.requests import Request
from typing import Dict
from time import sleep
from fastapi import FastAPI
from fastapi import FastAPI, File, UploadFile, Form
from pydantic import BaseModel

class ImageRequest(BaseModel):  # Used for request validation and generating API documentation
    image: list[list[float]] | list[list[list[float]]]  # 2D or 3D array
    
app = FastAPI()


from fastapi import HTTPException
from PIL import Image

@app.post("/upload")
def upload(file: UploadFile = File()):
    try:        
        im = Image.open(file.file)
        if im.mode in ("RGBA", "P"): 
            im = im.convert("RGB")
        im.save('out.jpg', 'JPEG', quality=50) 
    except Exception:
        raise HTTPException(status_code=500, detail='Something went wrong')
    finally:
        file.file.close()
        im.close()
        
@app.get("/")
async def root():
    return {"message": "Hello World"}
