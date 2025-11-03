import requests
import pprint
from langchain.schema import Document
from datetime import datetime
import os

script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in

def get_short_term_memory():
    rel_path = "short_term.txt"
    abs_file_path = os.path.join(script_dir, rel_path)
    with open(abs_file_path, 'r') as f:
        raw_data = f.read()
        result = raw_data.replace('{','').replace('}','')
    return result
        # return Document(page_content=raw_data, metadata={})

def get_long_term_memory():
    rel_path = "long_term.txt"
    abs_file_path = os.path.join(script_dir, rel_path)
    with open(abs_file_path, 'r') as f:
        raw_data = f.read()
        result = raw_data.replace('{','').replace('}','')
    return result
        # return Document(page_content=raw_data, metadata={})

def get_data_from_api(url: str, table_name: str = None):
    headers = {"x-api-key": "plane_api_b343a356f3d1480ab568697a162150dd"}

    response = requests.get(url, headers=headers)
    if table_name:
        return response.json()

    return response.json()["results"]

def unite_tables():
    projects = get_data_from_api(
        "https://api.plane.so/api/v1/workspaces/congnguyendo/projects/"
    )
    result = []

    for project in projects:
        project_id = project["id"]
        project_name = project["name"]

        assignees = get_data_from_api(
            "https://api.plane.so/api/v1/workspaces/congnguyendo/members", table_name="members"
        )
        # pprint.pprint(assignees)
        issues = get_data_from_api(
            f"https://api.plane.so/api/v1/workspaces/congnguyendo/projects/{project_id}/issues"
        )
        states = get_data_from_api(
            f"https://api.plane.so/api/v1/workspaces/congnguyendo/projects/{project_id}/states"
        )

        # pprint.pprint(assignees)
        # print("Issues:", issues)

        # Tạo index theo id (id là string)
        assignee_index = {a["id"]: a["display_name"] for a in assignees}
        # pprint.pprint(assignee_index)
        state_index = {s["id"]: s["name"] for s in states}

        for issue in issues:
            comments = get_data_from_api(f'https://api.plane.so/api/v1/workspaces/congnguyendo/projects/e0361220-8104-4600-a463-ee5c2572eb2b/issues/{str(issue["id"])}/comments/')
            if not comments:
                comment_texts = []
            else:
                comment_texts = [c["comment_stripped"] for c in comments]
            enriched = {
                "project_name": project_name,
                "id": issue["id"],
                "name": issue["name"],
                "priority": issue["priority"],
                "start_date": issue["start_date"],
                "target_date": issue["target_date"],
                "completed_at": issue["completed_at"],
                "description": issue["description_stripped"],
                "assignees": [assignee_index[aid] for aid in issue["assignees"]],
                "state": state_index.get(issue["state"]),
                "comments": comment_texts,
                }
            result.append(enriched)

    return result

def get_all_tasks():
    """Get all tasks."""

    responses = unite_tables()
    results = []
    for response in responses:
        assignees = " and ".join(response["assignees"])
        comments  = " and ".join(response["comments"])
        try:
            date_obj = datetime.strptime(response['start_date'], "%Y-%m-%d")
            start_date = date_obj.timestamp()
        except Exception as e:
            # print(e)
            start_date = None
        try:
            date_obj = datetime.strptime(response['target_date'], "%Y-%m-%d")
            target_date = date_obj.timestamp()
        except Exception as e:
            # print(e)
            target_date = None
        try:
            date_obj = datetime.strptime(response['complete_at'], "%Y-%m-%d")
            completed_at = date_obj.timestamp()
        except Exception as e:
            # print(e)
            completed_at = None
        content = f"This is a task that has name: {response['name']}, assignees: {assignees}, \
            'comments': {comments}, 'start date': {response['start_date']}, 'target date': {response['target_date']}, \
            'description': {response['description']}, \
            'priority': {response['priority']}, 'state': {response['state']}."
        metadata = {"task id": str(response['id']),
                    "priority": str(response['priority']), 
                    "state": str(response['state']), 
                    "start date": start_date, 
                    "target date": target_date, 
                    "completed at": completed_at,
                    "project name": str(response['project_name'])
                    }
        document = Document(page_content=content, metadata=metadata)
        results.append(document)

    # pprint.pprint(results)
    return results

if __name__ == "__main__":
    result = get_all_tasks()
    # projects = get_data_from_api(
    #     "https://api.plane.so/api/v1/workspaces/congnguyendo/projects/"
    # )

    # for project in projects:
    #     print(f"Project: {project['name']}")

    pprint.pprint(result)
