#server api
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

#base64 handling
import base64
from io import BytesIO
from PIL import Image

#model imports
import re
from transformers import DonutProcessor, VisionEncoderDecoderModel
import torch


#models
processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# prepare decoder inputs
task_prompt = "<s_cord-v2>"
decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids


#Server Stuff
class ImageBase64(BaseModel):
    image: bytes

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Donut OCR prediction"}

@app.get("/predict")
async def predict(img: ImageBase64):
    im_bytes = base64.b64decode(img.image)
    im_file = BytesIO(im_bytes)
    img_pil = Image.open(im_file)

    #generate pixel values
    pixel_values = processor(img_pil, return_tensors="pt").pixel_values
    #predict
    outputs = model.generate(
        pixel_values.to(device),
        decoder_input_ids=decoder_input_ids.to(device),
        max_length=model.decoder.config.max_position_embeddings,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        use_cache=True,
        bad_words_ids=[[processor.tokenizer.unk_token_id]],
        return_dict_in_generate=True,
    )

    sequence = processor.batch_decode(outputs.sequences)[0]
    sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
    sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()
    return {"value": processor.token2json(sequence)}