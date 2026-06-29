# Dataset Configuration
IMAGE_SIZE = 224
NUM_CLASSES = 4
CLASS_NAMES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]

# DataLoader Configuration
BATCH_SIZE = 64
NUM_WORKERS = 0


# Dataset Statistics
DATASET_MEAN = [
    0.1794,
    0.1794,
    0.1794
]
DATASET_STD = [
    0.1868,
    0.1868,
    0.1868
]

# Dataset Directory
TRAIN_DIR = "../Dataset/Training"
TEST_DIR = "../Dataset/Testing"

# Training Configuration
LEARNING_RATE = 0.001
NUM_EPOCHS = 20