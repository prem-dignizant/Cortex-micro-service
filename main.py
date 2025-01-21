from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from service import get_s3_data , pdf_to_image
import os
# Initialize FastAPI app
app = FastAPI()

os.makedirs('input_files', exist_ok=True)

# Input schema
class PDFRequest(BaseModel):
    s3_url: str


@app.post("/process-pdf")
async def process_pdf(request: PDFRequest):
    s3_url = request.s3_url
    
    try:
        file_name = get_s3_data(s3_url)
        pdf_to_image(file_name)

        return {"file_name": file_name}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to download PDF: {e}")


