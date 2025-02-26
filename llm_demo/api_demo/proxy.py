from fastapi import FastAPI, Request
import httpx
from fastapi.responses import JSONResponse
from sse_starlette import EventSourceResponse
import requests

app = FastAPI()

MODEL_TO_URL = {
    "sdu": "http://10.1.2.163:8000/v1/chat/completions",
    "mingcha": "http://10.1.2.163:8001/v1/chat/completions",
}

@app.post("/v1/chat/completions")
async def proxy_openai_api(request: Request):
    body = await request.json()
    model = body.get("model")

    if model not in MODEL_TO_URL:
        return JSONResponse(content={"error": "Invalid model"}, status_code=400)

    target_url = MODEL_TO_URL[model]
    headers = dict(request.headers)
    headers.pop("content-length", None)  # 删除 Content-Length，交由 httpx 计算
    if body.get("stream", False):
        response=requests.post(target_url, headers=headers, json=body,stream=True)             
        return EventSourceResponse(response, media_type="text/event-stream")
    else :
        response = requests.post(target_url, headers=headers, json=body)
        return JSONResponse(content=response.json(), status_code=response.status_code)
