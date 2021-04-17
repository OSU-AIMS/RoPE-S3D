import json
JSON_PATH = r'data/paths.json'

class Paths:
    def __init__(self):
        self._load()

    def _load(self):
        with open(JSON_PATH,'r') as f:
            data = json.load(f)

        for key in data:
            exec(f"self.{key}=r'{data[key]}'")