import pytest
from httpx import AsyncClient
import websockets,asyncio
from main import app  

base_url = "localhost:8080"

@pytest.mark.asyncio
async def test_multiple_websocket_clients():
    client_ids = ["client_1","client_2"]
    websocket_uris = [f"ws://{base_url}/ws/{client_id}" for client_id in client_ids]

    async def connect_websocket(client_id, uri):
        async with websockets.connect(uri) as websocket:
            await asyncio.sleep(2)  # Ensure WebSocket is registered
            
            async with AsyncClient(base_url=f"http://{base_url}") as ac:
                response = await ac.post("/multi_process_pdf", json={"s3_url": "https://geometra4-dev.s3.eu-west-1.amazonaws.com/1182117", "client_id": client_id})
                assert response.status_code == 200
                assert "task_id" in response.json()

            # Wait for WebSocket response
            response_data = await websocket.recv()
            assert "xfdf_content" in response_data
            print(f"Client {client_id} received data: {response_data}")

    # Run all WebSocket clients concurrently
    await asyncio.gather(*(connect_websocket(client_id, uri) for client_id, uri in zip(client_ids, websocket_uris)))

@pytest.mark.asyncio
async def test_process_pdf_success():
    async with AsyncClient(base_url=f"http://{base_url}") as ac:
        # Test data
        request_data = {"s3_url": "https://example.com/sample.pdf"}
        
        response = await ac.post("/process-pdf", json=request_data)
        
        # Assertions
        assert response.status_code == 200
        assert "xfdf_content" in response.json()
        assert response.json()["xfdf_content"] is not None

@pytest.mark.asyncio
async def test_process_pdf_failure():
    async with AsyncClient(base_url=f"http://{base_url}") as ac:
        # Invalid URL to trigger exception
        request_data = {"s3_url": "invalid-url"}
        
        response = await ac.post("/process-pdf", json=request_data)
        
        # Assertions
        assert response.status_code == 400
        assert "detail" in response.json()
        assert "Failed to download PDF" in response.json()["detail"]