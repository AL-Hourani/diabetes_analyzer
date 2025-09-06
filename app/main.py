from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import shutil
import uuid
import os
from app.ocr import extract_text
from app.helper.parser import clean_text
from app.helper.parser import extract_medical_values
from app.helper.utils import normalize_keys_fuzzy
from app.helper.mapping import STANDARD_MAPPING
from app.helper.converting import analyze_results



app = FastAPI()

@app.post("/ocr")
async def ocr_image(file: UploadFile = File(...)):
    file_ext = file.filename.split(".")[-1]
    temp_filename = f"temp_{uuid.uuid4()}.{file_ext}"

    with open(temp_filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
       
        text_data = extract_text(temp_filename)
        if isinstance(text_data, list):
           text_data = " ".join(text_data)
        tokens = clean_text(text_data)  # clean text 
        results = extract_medical_values(tokens) 
        
        normalized = normalize_keys_fuzzy(results , STANDARD_MAPPING)
        
        final = analyze_results(normalized)
         
        return JSONResponse(content={"analyzed_results":final})
    finally:
        os.remove(temp_filename)
