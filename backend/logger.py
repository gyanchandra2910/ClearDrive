import csv
import time

class PerformanceLogger:
    def __init__(self, filename="performance_results.csv"):
        self.filename = filename
        # Excel ke headers create karo
        with open(self.filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Timestamp", "Mode", "FPS", "Contrast_Gain", "Visibility_Score"])

    def log(self, mode, fps, contrast, visibility):
        with open(self.filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([time.strftime("%H:%M:%S"), mode, round(fps, 2), contrast, visibility])