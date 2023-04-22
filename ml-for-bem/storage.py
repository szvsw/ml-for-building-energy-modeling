from google.cloud import storage

storage_client = storage.Client.from_service_account_json(
    "../credentials/bucket-key.json"
)
bucket = storage_client.get_bucket("ml-for-bem-data")


def upload_to_bucket(blob_name, file_name):
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(file_name)

def download_from_bucket(blob_name, file_name):
    blob = bucket.blob(blob_name)
    blob.download_to_filename(file_name)