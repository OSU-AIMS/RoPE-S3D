import json
import os
JSON_PATH = r'data/paths.json'

class Paths:
    def __init__(self):
        self._load()

    def _load(self):
        """Load each item in the path JSON as an attribute of Paths()
        """
        with open(JSON_PATH,'r') as f:
            data = json.load(f)

        for key in data:
            exec(f"self.{key}=r'{data[key]}'")

    def create(self):
        """Create any folders in the paths JSON that do not yet exist
        """
        for item in self.__dict__.values():
            if os.path.basename(item) == "":
                if not os.path.isdir(item):
                    os.makedirs(item)