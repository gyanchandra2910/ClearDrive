import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import MobileNet_V2_Weights # Naya standard
from PIL import Image
import cv2

class WeatherClassifier:
    def __init__(self):
        # RTX 3050 Logic: Check for CUDA
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"AI Brain using: {self.device}")

        # Chapter 12: MobileNetV2 with latest Weights API
        # 'pretrained=True' ki jagah hum 'weights' use kar rahe hain (Warning fix)
        self.model = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        
        # Output layers fix for 3 classes: CLEAR, FOGGY, NIGHT
        self.model.classifier[1] = nn.Linear(self.model.last_channel, 3)
        
        # Model ko GPU par bhejo
        self.model = self.model.to(self.device)
        self.model.eval()

        # Chapter 2: Sampling & Normalization pipeline
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.classes = ["CLEAR", "FOGGY", "NIGHT"]

    def predict(self, frame):
        # Frame ko GPU compatible banao
        input_tensor = self.transform(frame).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(input_tensor)
            _, predicted = torch.max(outputs, 1)
            return self.classes[predicted.item()]