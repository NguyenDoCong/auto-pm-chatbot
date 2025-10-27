import requests
import pprint
from langchain.schema import Document


def get_data_from_api(url: str):
    response = requests.get(url)

    return response.json()

def unite_tables():
    assignees = get_data_from_api("https://zalooawebhook.demo.mqsolutions.vn/api/users?skip=0&limit=20")
    # pprint.pprint(assignees["users"])
    issues = get_data_from_api("https://zalooawebhook.demo.mqsolutions.vn/api/tasks?skip=0&limit=20")
    # pprint.pprint(issues["tasks"])
    assignments = get_data_from_api(
        "https://zalooawebhook.demo.mqsolutions.vn/api/assignments?skip=0&limit=20"
    )
    # pprint.pprint(assignments["assignments"])

    # Tạo index theo id (id là string)
    assignee_index = {a["id"]: a["name"] for a in assignees["users"]}

    # pprint.pprint(state_index)

    result = []
    for issue in issues["tasks"]:
        comments = get_data_from_api(f'https://zalooawebhook.demo.mqsolutions.vn/api/comments?skip=0&limit=50&task_id={issue["id"]}')["comments"]
        if not comments:
            comment_texts = []
        else:
            comment_texts = [c["content"] for c in comments]
        for assignment in assignments["assignments"]:
            if assignment["task_id"] == issue["id"]:
                aid = assignment["user_id"]
                enriched = {
                    "id": issue["id"],
                    "name": issue["title"],
                    "priority": issue["priority"],
                    "assignees": [assignee_index[aid]],
                    "start_date": issue["created_at"],
                    "target_date": issue["deadline"],
                    # "completed_at": issue["completed_at"],
                    "description": issue["description"],
                    "state": issue["status"],
                    "comments": comment_texts,
                }
                result.append(enriched)
    return result

if __name__ == "__main__":
    result = unite_tables()

    results = []
    for response in result:
        assignees = " and ".join(response["assignees"])
        comments  = " and ".join(response["comments"])
        content = f"""This is a work item that has name: {response["name"]}, 
        id: {response["id"]},
        assignees: {", ".join(response["assignees"])},
        priority: {response["priority"]}, 
        start_date: {response["start_date"]},
        target_date: {response["target_date"]},
        description: {response["description"]},
        state: {response["state"]},
        comment: {comments}.
        """
        metadata = {}
        document = Document(page_content=content, metadata=metadata)
        results.append(document)

    pprint.pprint(results)
