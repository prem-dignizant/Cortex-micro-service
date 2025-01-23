from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from service import get_s3_data , pdf_to_image
import os, shutil
from pdf_process_model import get_segment , process_segmentation_masks , process_masks_to_xfdf
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time
from datetime import datetime
# Initialize FastAPI app
app = FastAPI()

folder_path = "input_files"
executor = ThreadPoolExecutor(max_workers=4)  
# Input schema
class PDFRequest(BaseModel):
    s3_url: str


# A helper function to run the GPU task in a thread
async def run_in_thread_pool(func, *args):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, func, *args)

def ml_process(s3_url):
    pdf_file = get_s3_data(s3_url,folder_path)
    all_images = pdf_to_image(pdf_file,folder_path)
    # print(f'************ done', datetime.now())
    # for image in all_images:
    #     sam_result = get_segment(image)
    #     annotations = process_segmentation_masks(sam_result)
    #     xfdf_content = process_masks_to_xfdf(sam_result, 'output.xfdf')

@app.post("/process-pdf")
async def process_pdf(request: PDFRequest):
    os.makedirs(folder_path, exist_ok=True)
    try:
        ml_process(request.s3_url)
        return {"message": 'success'}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to download PDF: {e}")


@app.post("/multi_process_pdf")
async def multi_process_pdf(request: PDFRequest,background_tasks: BackgroundTasks):
    os.makedirs(folder_path, exist_ok=True)
    try:
        result = background_tasks.add_task(lambda: executor.submit(ml_process,request.s3_url))
        return {"message": 'success'}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to download PDF: {e}")

