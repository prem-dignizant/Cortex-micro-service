import unittest
from fastapi.testclient import TestClient
from main import app
from schema import MultiPDFRequest
import websockets
client = TestClient(app)

BASE_URL = "ws://localhost:8080/ws/test_client"
class TestAPI(unittest.IsolatedAsyncioTestCase):


    async def test_websocket_and_multi_process_pdf(self):
        uri = "ws://localhost:8000/ws/test_client"
        
        async with websockets.connect(uri) as websocket:
            # Send a message to keep the connection alive
            await websocket.send("Hello")
            response = await websocket.recv()
            self.assertEqual(response, "Hello")

            # Send a request to the /multi_process_pdf endpoint
            request_data = MultiPDFRequest(s3_url="https://geometra4-dev.s3.eu-west-1.amazonaws.com/1182117", client_id="test_client")
            response = client.post("/multi_process_pdf", json=request_data.dict())
            self.assertEqual(response.status_code, 200)
            self.assertIn("task_id", response.json())

            # Wait for the response from the WebSocket
            response = await websocket.recv()
            self.assertIn("file_url", response)

    def test_root_endpoint(self):
        response = client.post("/test")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"message": "Hello World"})


    # async def test_process_pdf(self):
    #     response = await client.post("/process-pdf", json={"s3_url": "https://geometra4-dev.s3.eu-west-1.amazonaws.com/1182117"})
    #     self.assertEqual(response.status_code, 200)
    #     self.assertIn("file_url", response.json())        

    


print(__name__)
if __name__ == '__main__':
    print('*********************')
    unittest.main()