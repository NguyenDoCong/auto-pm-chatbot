import requests
from dotenv import load_dotenv
# from fastapi.security import APIKeyHeader
import json
load_dotenv()
# X_API_KEY = APIKeyHeader(name="X_API_KEY")

url = 'http://192.168.6.194:9100/sendMessage'
myobj = {
    "messages":[
        {
            "type": "human",
            # "content": "list task with highest priority assigned to Cong"
            "content": "Hello"
        }
    ]
}

x = requests.post(url, json = myobj)
result = x.json()

print(x.text)
# print(json.dumps(result, indent=2))
print(result["result"]["content"])