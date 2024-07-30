from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi


class DatabaseConnection:
    def __init__(self, uri) -> None:
        self.uri = uri

    def connect(self):
        try:
            client = MongoClient(self.uri, server_api=ServerApi("1"))
            client.admin.command("ping")
            return True
        except Exception as e:
            return False
