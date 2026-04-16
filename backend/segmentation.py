import torch
import cv2
import numpy as np
from torchvision import models
from torchvision.models.segmentation import LRASPP_MobileNet_V3_Large_Weights

class RoadSegmentor:
    def __init__(self):
        # Detect available hardware — use GPU (CUDA) if available, otherwise fall back to CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load LR-ASPP MobileNetV3-Large with COCO pre-trained weights for semantic segmentation
        self.model = models.segmentation.lraspp_mobilenet_v3_large(
            weights=LRASPP_MobileNet_V3_Large_Weights.DEFAULT
        )
        self.model = self.model.to(self.device).eval()

    def get_road_mask(self, frame):
        # Convert BGR (OpenCV default) to RGB format required by the PyTorch model
        input_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Build a normalized float tensor: HWC -> CHW, scale to [0, 1], add batch dimension
        input_tensor = (
            torch.from_numpy(input_img)
            .permute(2, 0, 1)
            .float()
            .div(255)
            .unsqueeze(0)
            .to(self.device)
        )

        # Run segmentation inference without gradient computation
        with torch.no_grad():
            output = self.model(input_tensor)['out'][0]

        # Take the argmax across class channels to get per-pixel class predictions
        output_predictions = output.argmax(0).byte().cpu().numpy()

        # Build a blank green-channel mask canvas
        road_mask = np.zeros_like(frame)

        # In the COCO dataset, class 0 = background (often covers flat road surface on highways)
        # and class 7 = train/road — both are valid drivable surface indicators
        road_mask[(output_predictions == 0) | (output_predictions == 7)] = [0, 255, 0]

        # Ego-vehicle masking: zero out the top 60% of the mask to eliminate
        # sky, bonnet, and dashboard false positives outside the drivable zone
        height = frame.shape[0]
        road_mask[0:int(height * 0.6), :] = 0

        return road_mask