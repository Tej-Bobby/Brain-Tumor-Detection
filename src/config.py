from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODEL_DIR = BASE_DIR / "models"

MODEL_PATH = MODEL_DIR / "best_model.pth"
CLASS_NAMES_PATH = MODEL_DIR / "class_names.json"

IMAGE_SIZE = 224
NUM_CLASSES = 4

BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 0.0001

DEVICE = "cuda"