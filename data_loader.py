from services.azure_blob_service import azure_blob

class DataLoader:
    def __init__(self, blob_name="vietnam_travel_dataset.json"):
        self.blob_name = blob_name

    def load_data(self):
        print(f"Loading dataset from Azure Blob: {self.blob_name}")
        data = azure_blob.download_json(self.blob_name)
        print(f"Loaded {len(data)} nodes from blob!")
        return data
