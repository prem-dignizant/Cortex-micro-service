from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel
from service import get_s3_data , pdf_to_image
import os , random
from pdf_process_model import get_segment , process_segmentation_masks , process_masks_to_xfdf
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time
from datetime import datetime
import zipfile
# Initialize FastAPI app
app = FastAPI()

folder_path = "input_files"
output_path = "output_files"
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
    print(f'************ done', datetime.now())
    xfdf_files = []
    for image in all_images:
        sam_result = get_segment(image)
        annotations = process_segmentation_masks(sam_result)
        xfdf_file= process_masks_to_xfdf(sam_result, 'output.xfdf')
        xfdf_files.append(xfdf_file)
    while True:
        zip_file_path = os.path.join(output_path, f"xfdf_content_{random.randint(0, 10000)}.zip")
        if not os.path.exists(zip_file_path):
            break
    with zipfile.ZipFile(zip_file_path, "w") as zipf:
        for file_path in xfdf_files:
            zipf.write(file_path, arcname=os.path.basename(file_path))  

    return  zip_file_path
    
@app.post("/process-pdf")
async def process_pdf(request: PDFRequest):
    os.makedirs(folder_path, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)    
    try:
        zip_file_path = ml_process(request.s3_url)
        
        return FileResponse(
            zip_file_path,
            media_type="application/zip",
            filename="output_files.zip",
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to download PDF: {e}")


@app.post("/multi_process_pdf")
async def multi_process_pdf(request: PDFRequest,background_tasks: BackgroundTasks):
    os.makedirs(folder_path, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)  
    try:
        result = background_tasks.add_task(lambda: executor.submit(ml_process,request.s3_url))
        return {"message": 'success'}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to download PDF: {e}")

