import json
from typing import Optional, Any
from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import ResourceNotFoundError
import config

class AzureBlob:
    """Minimal Azure Blob helper for uploading/downloading files."""

    def __init__(
        self,
        connection_string: Optional[str] = None,
        container_name: str = config.AZURE_CONTAINER_NAME
    ):
        self.container_name = container_name
        self.connection_string = connection_string or config.AZURE_STORAGE_CONNECTION_STRING
        
        if not self.connection_string:
            raise RuntimeError("Azure Storage connection string is missing.")
        
        self.client = BlobServiceClient.from_connection_string(self.connection_string)
        self._ensure_container()

    def _ensure_container(self):
        """Create container if it doesn't exist."""
        container = self.client.get_container_client(self.container_name)
        try:
            container.get_container_properties()
        except ResourceNotFoundError:
            container.create_container()

    def upload_text(self, blob_name: str, text: str):
        blob = self.client.get_blob_client(self.container_name, blob_name)
        blob.upload_blob(text, overwrite=True, content_type="text/plain")
        return True

    def upload_json(self, blob_name: str, data: Any):
        blob = self.client.get_blob_client(self.container_name, blob_name)
        blob.upload_blob(
            json.dumps(data, ensure_ascii=False, indent=2),
            overwrite=True,
            content_type="application/json"
        )
        return True

    def download_text(self, blob_name: str) -> str:
        blob = self.client.get_blob_client(self.container_name, blob_name)
        data = blob.download_blob().readall()
        return data.decode("utf-8")

    def download_json(self, blob_name: str) -> Any:
        text = self.download_text(blob_name)
        return json.loads(text)

    def list_blobs(self):
        container = self.client.get_container_client(self.container_name)
        return [b.name for b in container.list_blobs()]

    def delete_blob(self, blob_name: str):
        blob = self.client.get_blob_client(self.container_name, blob_name)
        try:
            blob.delete_blob()
            return True
        except ResourceNotFoundError:
            return False

# Global instance
azure_blob = AzureBlob()
