import unittest
from fastapi.testclient import TestClient
from httpx import AsyncClient
from main import app
import os
client = TestClient(app)
BASE_URL = os.getenv("BASE_URL")
print(BASE_URL)
class TestAPI(unittest.IsolatedAsyncioTestCase):

    async def test_websocket_endpoint(self):
        async with AsyncClient(app=app, base_url=BASE_URL) as ac:
            async with ac.websocket_connect("/ws/test_client") as websocket:
                await websocket.send_text("Hello")
                response = await websocket.receive_text()
                self.assertEqual(response, "Hello")

    # def test_download_file(self):
    #     response = client.get("/download/non_existent_file.zip")
    #     self.assertEqual(response.status_code, 404)
    #     self.assertEqual(response.json(), {"detail": "File not found"})

    async def test_process_pdf(self):
        response = await client.post("/process-pdf", json={"s3_url": "https://geometra4-dev.s3.eu-west-1.amazonaws.com/1182117"})
        self.assertEqual(response.status_code, 200)
        self.assertIn("file_url", response.json())        

    async def test_multi_process_pdf(self):
        response = await client.post("/multi_process_pdf", json={"s3_url": "https://geometra4-dev.s3.eu-west-1.amazonaws.com/1182117", "client_id": "test_client"})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json(), {"detail": "WebSocket connection not found"})

if __name__ == '__main__':
    unittest.main()