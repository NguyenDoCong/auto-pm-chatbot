def add_report(id, report):
    """Add report by task id."""

    url = f"https://api.plane.so/api/v1/workspaces/congnguyendo/projects/e0361220-8104-4600-a463-ee5c2572eb2b/issues/{id}/comments/"

    payload = {"comment_html": report}
    headers = {
        "x-api-key": "plane_api_29c429f9822146aba6782cba5d3c1a4a",
        "Content-Type": "application/json",
    }

    response = requests.post(url, json=payload, headers=headers)

    # print(response.json())
    return {"result": response.json()}