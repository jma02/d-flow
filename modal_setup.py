import modal
app = modal.App("upload-files")
volume = modal.Volume.from_name("eit-spsolve", create_if_missing=True)


@app.local_entrypoint()
def main():
    with volume.batch_upload() as batch:
        # batch.put_file("data/ct-shepp-logan-multiflow-1-24-128.pt", "data/ct-shepp-logan-multiflow-1-24-128.pt")
        # batch.put_file("data/ct-shepp-logan-multiflow-2-24-128.pt", "data/ct-shepp-logan-multiflow-2-24-128.pt")
        # batch.put_file("data/ct-shepp-logan-multiflow-3-24-128.pt", "data/ct-shepp-logan-multiflow-3-24-128.pt")
        # batch.put_file("data/ct-shepp-logan-multiflow-4-24-128.pt", "data/ct-shepp-logan-multiflow-4-24-128.pt")
        # batch.put_directory("./forward_solvers", "/forward_solvers")
        batch.put_file("dflow_eit.ipynb", "dflow_eit.ipynb")
