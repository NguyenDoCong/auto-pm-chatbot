
def update_status(id):
    """Update status by task id."""
    for t in tasks:
        if t["id"] == id:
            t["status"] = "complete"
            return {"result": t}
    return {"error": "Task not found"}


def summarize_reports(id: Optional[int] = None):
    """Summarize reports by task id."""
    task = None
    for t in tasks:
        if t["id"] == id:
            task = t

    if not task:
        return {"error": "Task not found"}

    daily_report = None
    daily_report = task["daily_report"]
    if not daily_report:
        return {"error": "No reports to summarize"}

    summary_message = f"Create a short summary of all reports of the task {task['name']} by using the following reports: {daily_report}"

    response = model.invoke(summary_message)

    task["daily_report_summary"] = response.content

    return {"summary": response.content}
