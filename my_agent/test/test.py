import requests
import pprint

url = "http://192.168.6.88:8000/api/users?skip=0&limit=20"

response = requests.get(url)

# pprint.pp(response.json())

result = response.json()
pprint.pprint(result)