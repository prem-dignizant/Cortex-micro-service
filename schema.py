from pydantic import BaseModel

# Input schema
class PDFRequest(BaseModel):
    client_id: str
    s3_url: str