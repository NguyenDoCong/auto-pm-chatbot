import requests

url = "https://api.plane.so/api/v1/workspaces/congnguyendo/projects/e0361220-8104-4600-a463-ee5c2572eb2b/issues/"

payload = { "name": "test-assign" , "assignees": ['507817ec-d907-461e-a86a-be44fde519d3']}
headers = {
    "x-api-key": "plane_api_b343a356f3d1480ab568697a162150dd",
    "Content-Type": "application/json"
}

response = requests.post(url, json=payload, headers=headers)

print(response.json())