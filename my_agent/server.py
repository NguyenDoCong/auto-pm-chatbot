from http.client import HTTPException
import os
from fastapi.responses import RedirectResponse
from dotenv import load_dotenv
from fastapi import FastAPI, Depends
from fastapi.security import APIKeyHeader
from my_agent import State
from my_agent import task_graph
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field
from fastapi import FastAPI, Request, Depends, Body
import json
import requests
from fastapi.responses import JSONResponse
 
# Load .env vars
load_dotenv()
ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")

# Initialize FastAPI app
app = FastAPI()

class Message(BaseModel):
    message: str = ""
    sender_id: str
    role: str
    message_id: str
    
# Generated docs endpoint
@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")

async def generate_route(message: str, thread_id: str):
    graph = task_graph.compile()
    graph.name = "LangGraphDeployDemo"
    thread_id = "3"
    config = {"configurable": {"thread_id": thread_id}}
    try:
        result = await graph.ainvoke({"messages": [{"role": "user", "content": {message}}]}, config)
        message = result["messages"][-1]
        return message.content
    except Exception as e:
        print(e)
        return "Error occurred, please try again later."
        # raise HTTPException(status_code=500, detail=str(e))
        
@app.post("/api/sendMessage")
# @HTTPException() 
async def sendMessage(inputs: Message = Body(...)):
    print(f"----message: {inputs.message}")
    print(f"----sender_id: {inputs.sender_id}")
    print(f"----role: {inputs.role}")
    print(f"----message_id: {inputs.message_id}")
    print(f"----ACCESS_TOKEN: {ACCESS_TOKEN}")
    message = await generate_route(inputs.message, inputs.sender_id)
 
    url = "http://192.168.6.189:3333/api/zalo/send-message"
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f"Bearer {ACCESS_TOKEN}"
    }
    payload = json.dumps({
        "toUserId": inputs.sender_id,
        "message": message, 
    })        
        
    res = requests.request("POST", url, headers=headers, data=payload)
    print(res.json())
    return JSONResponse(status_code=200, content=str(f"Message {message} delivered to {inputs.sender_id}"))

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=9100)
