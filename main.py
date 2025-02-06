from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os 
import asyncio
from concurrent.futures import ThreadPoolExecutor
import uuid
from service import get_s3_data , pdf_to_image
from pdf_process_model import convert_json_to_xfdf, get_segment , process_segmentation_masks
from schema import PDFRequest , MultiPDFRequest
from apscheduler.schedulers.background import BackgroundScheduler
from contextlib import asynccontextmanager

# Store active WebSocket connections
active_connections = {}

BASE_URL = os.getenv("BASE_URL")
print(BASE_URL)
folder_path = "input_files"
output_path = "output_files"
os.makedirs(folder_path, exist_ok=True)
# os.makedirs(output_path, exist_ok=True)  

executor = ThreadPoolExecutor(max_workers=4)    # A helper function to run the GPU task in a thread
scheduler = BackgroundScheduler()

# Initialize the FastAPI app with lifecycle management
app = FastAPI()

origins = ['*']

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allow only specified origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],  # Allow all headers
)


# Helper function to send data via WebSocket
async def notify_client(websocket: WebSocket, task_id: str, xfdf_content: list,page_num:int):
    message = {
        "task_id": task_id,
        "status": "completed",
        "xfdf_content": xfdf_content,
        "page_num": page_num
    }
    await websocket.send_json(message)


@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await websocket.accept()
    print(f"Client {client_id} connected")
    active_connections[client_id] = websocket
    try:
        while True:
            await websocket.receive_text()  # Keep connection alive
    except WebSocketDisconnect:
        print(f"Client {client_id} disconnected")
        del active_connections[client_id]


async def run_in_thread_pool(func, *args):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, func, *args)

def ml_process(s3_url,page_num):
    try:
        pdf_file = get_s3_data(s3_url,folder_path)
        if not pdf_file:
            raise HTTPException(status_code=400, detail="Failed to download PDF")
        image = pdf_to_image(pdf_file,folder_path,page_num)

        sam_result = get_segment(image)
        coco_format = process_segmentation_masks(sam_result)
        xfdf_content = convert_json_to_xfdf(coco_format,page_num)  
        os.remove(pdf_file)
        os.remove(image)
        return xfdf_content
    except Exception as e:
        try:
            os.remove(pdf_file)
            os.remove(image)
        except:
            pass
        raise HTTPException(status_code=400, detail=f"Failed to ml_process: {e}")
    
    
    
@app.post("/process-pdf")
async def process_pdf(request: PDFRequest):
    try:
        s3_url = request.s3_url
        page_num = request.page_num 
        xfdf_content = await run_in_thread_pool(ml_process, s3_url,page_num)
        return {"xfdf_content": xfdf_content}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to download PDF: {e}")


@app.post("/multi_process_pdf")
async def multi_process_pdf(request: MultiPDFRequest,background_tasks: BackgroundTasks):
    print('************')
    s3_url = request.s3_url
    client_id  = request.client_id 
    page_num = request.page_num
    if client_id not in active_connections:
        print('******WebSocket connection not found******')
        raise HTTPException(status_code=400, detail="WebSocket connection not found")
    print('************')
    print(active_connections)
    task_id = str(uuid.uuid4())

    def task_wrapper():
        try:
            xfdf_content = ml_process(s3_url,page_num)
            asyncio.run(notify_client(active_connections[client_id], task_id, xfdf_content,page_num))
        except Exception as e:
            error_message = {"task_id": task_id, "status": "failed", "error": str(e)}
            asyncio.run(active_connections[client_id].send_json(error_message))


    background_tasks.add_task(lambda: executor.submit(task_wrapper))
    return {"task_id": task_id, "message": "Task submitted"}

@app.post("/test")
async def root():
    return {"message": "Hello World"}