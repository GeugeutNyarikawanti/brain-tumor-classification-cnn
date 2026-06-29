import torch
import torch.nn as nn
import settings.config as config

# Brain Tumor CNN
class BrainTumorCNN(nn.Module):
    def __init__(self):
        super().__init__()

        #Feature Extraction
        self.features = nn.Sequential(    
            #Block 1
            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),

             # Block 2
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1
            ),

            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),

            # Block 3
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1
            ),

            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),
        )

        # Classifier
        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                in_features=128 * 28 * 28,
                out_features=256
            ),

            nn.ReLU(),

            nn.Dropout(0.5),

            nn.Linear(
                in_features=256,
                out_features=config.NUM_CLASSES
            )
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)

        return x