import datetime
import json
import os

STORE = os.path.join(r"C:\ai_project_hub", "store", "project_summary.json")
CHANGELOG = os.path.join(r"C:\ai_project_hub", "store", "changelog.md")


def load_summary():
    with open(STORE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_summary(data):
    with open(STORE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def log(msg):
    with open(CHANGELOG, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.datetime.utcnow().isoformat()}Z] {msg}\n")


if __name__ == "__main__":
    import sys

    task = " ".join(sys.argv[1:]) or "no-op"
    s = load_summary()
    s["last_orchestrator_task"] = {
        "when": datetime.datetime.utcnow().isoformat() + "Z",
        "task": task,
    }
    save_summary(s)
    log(f"orchestrator: {task}")
    print("OK:", task)
