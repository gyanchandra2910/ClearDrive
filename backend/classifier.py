import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import MobileNet_V2_Weights  # Modern weights API (replaces deprecated pretrained=True)
from PIL import Image
import cv2

class WeatherClassifier:
    def __init__(self):
        # Detect available hardware — use GPU (CUDA) if available, otherwise fall back to CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"WeatherClassifier initialized on: {self.device}")

        # Load MobileNetV2 with ImageNet pre-trained weights via the modern torchvision weights API
        self.model = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)

        # Replace the final classification head to output 3 classes: CLEAR, FOGGY, NIGHT
        self.model.classifier[1] = nn.Linear(self.model.last_channel, 3)

        # Transfer model to the target computation device and set to inference mode
        self.model = self.model.to(self.device)
        self.model.eval()

        # Define the preprocessing pipeline: resize, normalize using ImageNet mean/std
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # Class index mapping
        self.classes = ["CLEAR", "FOGGY", "NIGHT"]

    def predict(self, frame):
        # Preprocess the incoming BGR frame into a normalized tensor and move to device
        input_tensor = self.transform(frame).unsqueeze(0).to(self.device)

        # Run inference without gradient tracking for maximum efficiency
        with torch.no_grad():
            outputs = self.model(input_tensor)
            _, predicted = torch.max(outputs, 1)
            return self.classes[predicted.item()]