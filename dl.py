from huggingface_hub import snapshot_download
import os
DIR = '.'
REPO_ID = "lllyasviel/ControlNet"
snapshot_download(repo_id=REPO_ID, local_dir=DIR)
