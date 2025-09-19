import requests
import pprint

url = "https://api.plane.so/api/v1/workspaces/congnguyendo/members/"
# url = "https://api.plane.so/api/v1/workspaces/congnguyendo/projects/e0361220-8104-4600-a463-ee5c2572eb2b/issues/?expand=assignees&fields=id,name,assignees"
# url = "https://api.plane.so/api/v1/workspaces/congnguyendo/projects/e0361220-8104-4600-a463-ee5c2572eb2b/issues/?fields=id,name,assignees,description_stripped"
# url = "https://api.plane.so/api/v1/workspaces/congnguyendo/projects/e0361220-8104-4600-a463-ee5c2572eb2b/issues/?assignees=['507817ec-d907-461e-a86a-be44fde519d3']"
url = "https://api.plane.so/api/v1/workspaces/congnguyendo/projects/e0361220-8104-4600-a463-ee5c2572eb2b/issues"
# url = "https://api.plane.so/api/v1/workspaces/congnguyendo/projects/e0361220-8104-4600-a463-ee5c2572eb2b/states/"
url = "https://api.plane.so/api/v1/workspaces/congnguyendo/projects/e0361220-8104-4600-a463-ee5c2572eb2b/issues/e43aa54e-7d33-4c7e-988f-f1e207fac1c5/comments/"

headers = {"x-api-key": "plane_api_b343a356f3d1480ab568697a162150dd"}

response = requests.get(url, headers=headers)

# pprint.pp(response.json())

result = response.json()["results"]
pprint.pprint(result)