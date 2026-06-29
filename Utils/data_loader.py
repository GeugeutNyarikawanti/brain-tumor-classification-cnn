from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from settings.config import *

def get_train_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(DATASET_MEAN, DATASET_STD)
    ])

def get_test_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(DATASET_MEAN, DATASET_STD)
    ])

def create_dataloaders():
    train_dataset = datasets.ImageFolder(
    root=TRAIN_DIR,
    transform=get_train_transform()
)
    
    test_dataset = datasets.ImageFolder(
    root=TEST_DIR,
    transform=get_test_transform()
)
    train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS
)
    
    test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)   
    
    return train_loader, test_loader