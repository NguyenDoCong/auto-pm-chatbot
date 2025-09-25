import requests
import pprint
from langchain.schema import Document

def get_data_from_api(url: str, table_name: str = None):
    headers = {"x-api-key": "plane_api_b343a356f3d1480ab568697a162150dd"}

    response = requests.get(url, headers=headers)
    if table_name:
        return response.json()
        
    return response.json()["results"]


if __name__ == "__main__":
    assignees = get_data_from_api('https://api.plane.so/api/v1/workspaces/congnguyendo/members/','members')
    issues = get_data_from_api('https://api.plane.so/api/v1/workspaces/congnguyendo/projects/e0361220-8104-4600-a463-ee5c2572eb2b/issues')
    states = get_data_from_api('https://api.plane.so/api/v1/workspaces/congnguyendo/projects/e0361220-8104-4600-a463-ee5c2572eb2b/states')
    # pprint.pprint(assignees)
    # print("Issues:", issues)

    # Tạo index theo id (id là string)
    assignee_index = {a["id"]: a["display_name"] for a in assignees}
    state_index = {s["id"]: s["group"] for s in states}
    
    # pprint.pprint(state_index)

    result = []
    for issue in issues:
        comments = get_data_from_api(f'https://api.plane.so/api/v1/workspaces/congnguyendo/projects/e0361220-8104-4600-a463-ee5c2572eb2b/issues/{str(issue["id"])}/comments/')
        if not comments:
            comment_texts = []
        else:
            comment_texts = [c["comment_stripped"] for c in comments]
        enriched = {
            "id": issue["id"],
            "name": issue["name"],
            "priority": issue["priority"],
            "assignees": [assignee_index[aid] for aid in issue["assignees"]],
            "state": state_index.get(issue["state"]),
            "comments": comment_texts,
            }
        result.append(enriched)


    results = []
    for response in result:
        assignees = " and ".join(response["assignees"])
        comments  = " and ".join(response["comments"])
        content = f"This is a work item that has name: {response['name']}, priority: {response['priority']}, assignees: {assignees}, state: {response['state']}, 'comments': {comments}"
        metadata = {}
        document = Document(page_content=content, metadata=metadata)
        results.append(document)

    pprint.pprint(results)

