from pydantic import BaseModel

# Input schema
class PDFRequest(BaseModel):
    s3_url: str
    page_num: int 

class MultiPDFRequest(PDFRequest):
    client_id: str