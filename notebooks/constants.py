import os

# Change the HOME_ROOT default to your own default HOME path
HOME_ROOT = os.environ.get("HOME", "")
# NOTE: MOUNT_ROOT only necessary if you're working in a filestore
MOUNT_ROOT = "/mnt/ml-data"
HF_MODELS_ROOT = f"{MOUNT_ROOT}/huggingface"
LOCAL_MODELS_ROOT = f"{HOME_ROOT}/huggingface"
DATASETS_ROOT = "../../data"


MATH_DATASET_DIR = f"{HOME_ROOT}/preemo-llms-in-prod-2/datasets"
