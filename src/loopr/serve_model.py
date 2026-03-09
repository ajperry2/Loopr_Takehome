import requests
import os
from fastapi import FastAPI, File, UploadFile, Form
from fastapi import FastAPI, File, Form, UploadFile, Request
import requests
from starlette.requests import Request
from typing import Dict
from time import sleep
from fastapi import FastAPI
from fastapi import FastAPI, File, UploadFile, Form
from pydantic import BaseModel
from fastapi import FastAPI, Header
from typing import Annotated
from fastapi import FastAPI, File, UploadFile
import numpy as np
import torch

from loopr.models.total import Model
from loopr.data.utils import tile_image as tiler
from loopr.data.contrastive import get_valid_transforms
app = FastAPI()

model = Model()


from fastapi import HTTPException
from PIL import Image
import io
transforms = get_valid_transforms()


def class_index_to_name(index):
    if index == 0:
        return "No Defect"
    elif index < len(TrainingNNConfig.kept_classes)+1:
        return TrainingNNConfig.class_label_to_name[TrainingNNConfig.kept_classes[index-1]]
    else:
        return TrainingNNConfig.class_label_to_name[TrainingNNConfig.unkept_classes[index-len(TrainingNNConfig.kept_classes)-1]]
        
@app.post("/upload")
def upload(file: Annotated[bytes, File()], password: str | None = None):
    
    if not password == os.environ["PASSWORD"]:
        raise ValueError("Invalid Password")
    try:        
        im = Image.open(io.BytesIO(file))
        im=im.convert('RGB')
        pix = np.array(im)
        resized_image = transforms(image=pix)["image"]
        mat = torch.tensor(resized_image).cuda()
        image_tiles = tiler(mat)
        output = model(image_tiles)
        return {
            "index": int(output["prediction"]), 
            "class": class_index_to_name(output["prediction"]), 
            "logits": output["logits"]}
    except Exception as e:
        print(e, flush=True)
        raise HTTPException(status_code=500, detail='Something went wrong')
    finally:
        if hasattr(file, "close"): file.close()
        
        
@app.get("/")
async def root():
    return {"message": "Hello World"}
