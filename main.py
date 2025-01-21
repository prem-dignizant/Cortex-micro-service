from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from service import get_s3_data , pdf_to_image
import os, shutil
from check_model import get_segment

# Initialize FastAPI app
app = FastAPI()

folder_path = "input_files"

# Input schema
class PDFRequest(BaseModel):
    s3_url: str


@app.post("/process-pdf")
async def process_pdf(request: PDFRequest):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)

    # Recreate the folder
    os.makedirs(folder_path, exist_ok=True)
    s3_url = request.s3_url
    
    try:
        data = []
        file_name = get_s3_data(s3_url)
        all_images = pdf_to_image(file_name)
        for image in all_images:
            segment_data = get_segment(image)
            data.append(segment_data)
        return {"file_name": data}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to download PDF: {e}")


