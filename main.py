from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse , Response
from fastapi.middleware.cors import CORSMiddleware
import os , random
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time
from datetime import datetime
import zipfile
import uuid
from service import get_s3_data , pdf_to_image , random_file_name , delete_old_files
from pdf_process_model import get_segment , process_segmentation_masks , process_masks_to_xfdf
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

# def start_scheduler():
#     """
#     Starts the APScheduler and schedules the `delete_old_files` job.
#     """
#     scheduler.add_job(delete_old_files, "interval", days=1, kwargs={"output_path": output_path})
#     scheduler.start()


# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     print("Server starting: Running initial cleanup and scheduler")
#     delete_old_files(output_path)  # Run the cleanup task at startup
#     start_scheduler()  # Start the scheduler for periodic cleanup

#     yield  # Wait here until the application shuts down

#     print("Server shutting down: Stopping scheduler")
#     scheduler.shutdown()



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
async def notify_client(websocket: WebSocket, task_id: str, xfdf_content: list):
    message = {
        "task_id": task_id,
        "status": "completed",
        "xfdf_content": xfdf_content
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


# @app.get("/download/{file_name}")
# async def download_file(file_name: str):
#     file_path = os.path.join(output_path, file_name)     # MANAGE FILE PATH 
#     if not os.path.exists(file_path):
#         raise HTTPException(status_code=404, detail="File not found")
#     return FileResponse(file_path, media_type="application/zip", filename=file_name)


async def run_in_thread_pool(func, *args):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, func, *args)

def ml_process(s3_url):
    try:
        pdf_file = get_s3_data(s3_url,folder_path)
        if not pdf_file:
            raise HTTPException(status_code=400, detail="Failed to download PDF")
        all_images = pdf_to_image(pdf_file,folder_path)
        # return all_images[0]
        xfdf_list = []
        for image in all_images:
            sam_result = get_segment(image)
            # annotations = process_segmentation_masks(sam_result)
            file_name_prefix = f"page_{all_images.index(image)}_xfdf"
            xfdf_content= process_masks_to_xfdf(sam_result)
            data = {}
            data[f'page_{all_images.index(image)}'] = xfdf_content
            # xfdf_files.append(xfdf_file)
            xfdf_list.append(data)

            os.remove(image)
        return xfdf_list
        # zip_file_path =  random_file_name(output_path , "xfdf_folder" , "zip")
        # with zipfile.ZipFile(zip_file_path, "w") as zipf:
        #     for file_path in xfdf_files:
        #         zipf.write(file_path, arcname=os.path.basename(file_path))  
        #         os.remove(file_path)
        # return  zip_file_path
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to ml_process: {e}")
    
@app.post("/process-pdf")
async def process_pdf(request: PDFRequest):
    try:
        s3_url = request.s3_url
        xfdf_content = await run_in_thread_pool(ml_process, s3_url)
        # file_url = f"{BASE_URL}/download/{os.path.basename(zip_file_path)}"
        return {"xfdf_content": xfdf_content}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to download PDF: {e}")


@app.post("/multi_process_pdf")
async def multi_process_pdf(request: MultiPDFRequest,background_tasks: BackgroundTasks):
    print('************')
    s3_url = request.s3_url
    client_id  = request.client_id 
    if client_id not in active_connections:
        print('******WebSocket connection not found******')
        raise HTTPException(status_code=400, detail="WebSocket connection not found")
    print('************')
    print(active_connections)
    task_id = str(uuid.uuid4())

    def task_wrapper():
        try:
            xfdf_content = ml_process(s3_url)
            # file_url = f"{BASE_URL}/download/{os.path.basename(zip_file_path)}"
            asyncio.run(notify_client(active_connections[client_id], task_id, xfdf_content))
        except Exception as e:
            error_message = {"task_id": task_id, "status": "failed", "error": str(e)}
            asyncio.run(active_connections[client_id].send_json(error_message))


    background_tasks.add_task(lambda: executor.submit(task_wrapper))
    return {"task_id": task_id, "message": "Task submitted"}

@app.post("/test")
async def root():
    return {"message": "Hello World"}