import os

DATASET_PATH = os.path.join("dataset", "semantic segmentation dataset")

IMAGE_DATASET_PATH = os.path.join(DATASET_PATH, "images")
MASKS_DATASET_PATH = os.path.join(DATASET_PATH, "masks")

INPUT_IMAGE_WIDTH = 256
INPUT_IMAGE_HEIGHT = 256

NUM_CHANNELS = 3
NUM_CLASSES = 6
BATCH_SIZE = 16
NUM_EPOCHS = 100

TEST_SPLIT = 0.25

TEST_IMAGE_PLOT = True

# Put as -1 if you don't want to break the image into tiles
PATCH_SIZE_X = 256
PATCH_SIZE_Y = 256

FILTERS = [64, 128, 256, 512, 1024]
USE_BN = True

# Threshold to filter weak predictions
THRESHOLD = 0.5

BASE_OUTPUT = "output"

MODEL_PATH = os.path.join(BASE_OUTPUT, "unet.model")
MID_MODEL_PATH = os.path.join(BASE_OUTPUT, "unet_checkpoint_{epoch:02d}.model")
PLOT_PATH = os.path.join(BASE_OUTPUT, "plot.png")
JSON_PATH = os.path.join(BASE_OUTPUT, "plot.json")
TEST_PATHS = os.path.join(BASE_OUTPUT, "test_paths.txt")

MODEL_SUMMARY = True
