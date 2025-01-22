from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

import os, shutil
from pdf_process_model import get_segment , process_segmentation_masks , process_masks_to_xfdf

# Initialize FastAPI app
app = FastAPI()

folder_path = "input_files"

# Input schema
class PDFRequest(BaseModel):
    s3_url: str


@app.get("/process-pdf")
async def process_pdf():
    # if os.path.exists(folder_path):
    #     shutil.rmtree(folder_path)

    # Recreate the folder
    # os.makedirs(folder_path, exist_ok=True)
    # s3_url = request.s3_url
    
    try:
        # file_name = get_s3_data(s3_url)
        # all_images = pdf_to_image(file_name)
        all_images = ['input_files\sample.png','input_files\sample_2.png']
        for image in all_images:
            sam_result = get_segment(image)
            print(sam_result[0])
            annotations = process_segmentation_masks(sam_result)
            # xfdf_content = process_masks_to_xfdf(sam_result, 'output.xfdf')

            print('*****************')




        return {"file_name": 'data'}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to download PDF: {e}")


