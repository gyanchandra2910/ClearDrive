import torch
import cv2
import numpy as np
from torchvision import models
from torchvision.models.segmentation import LRASPP_MobileNet_V3_Large_Weights

class RoadSegmentor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Pre-trained Segmentation Model load karo (Chapter 12)
        self.model = models.segmentation.lraspp_mobilenet_v3_large(weights=LRASPP_MobileNet_V3_Large_Weights.DEFAULT)
        self.model = self.model.to(self.device).eval()

    # segmentation.py mein ye change karo
    def get_road_mask(self, frame):
        input_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_tensor = torch.from_numpy(input_img).permute(2, 0, 1).float().div(255).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)['out'][0]
        
        # Argmax se prediction lo
        output_predictions = output.argmax(0).byte().cpu().numpy()
        
        # ROAD DETECTION FIX:
        # Aksar COCO dataset mein Road = 0 (Background/Flat) ya 7 hota hai.
        # Hum 'background' aur 'road' dono ko check karenge.
        road_mask = np.zeros_like(frame)
        
        # Class 0 (Background/Road) ko highlight karo
        # Note: LRASPP model mein background aksar road area hi cover karta hai highway par
        road_mask[(output_predictions == 0) | (output_predictions == 7)] = [0, 255, 0] 
        
        # Lower half masking (Chapter 10 Logic): 
        # Sirf image ke niche wale hisse (road) par mask rakho
        height = frame.shape[0]
        road_mask[0:int(height*0.6), :] = 0 # Top 60% clean kar do
        
        return road_mask