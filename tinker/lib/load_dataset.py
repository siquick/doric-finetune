import json


def load_dataset(dataset_path: str):
    """Load a dataset from a jsonl file and return a list of messages"""
    messages = []
    with open(dataset_path, "r") as f:
        for line in f:
            data = json.loads(line)
            data.pop("meta", None)
            messages.append(data)
    print(len(messages))
    return messages
