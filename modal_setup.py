import modal
from transforms import radon
import torch
app = modal.App("upload-files")
volume = modal.Volume.from_name("train-multiflow-v1")


@app.local_entrypoint()
def main():
    with volume.batch_upload() as batch:
        batch.put_file("data/shepp-logan-multiflow-v1-1-24-128.pt", "data/shepp-logan-multiflow-v1-1-24-128.pt")
        batch.put_file("data/shepp-logan-multiflow-v1-2-24-128.pt", "data/shepp-logan-multiflow-v1-2-24-128.pt")
        batch.put_file("data/shepp-logan-multiflow-v1-3-24-128.pt", "data/shepp-logan-multiflow-v1-3-24-128.pt")
        batch.put_file("data/shepp-logan-multiflow-v1-4-24-128.pt", "data/shepp-logan-multiflow-v1-4-24-128.pt")