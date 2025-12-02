import numpy as np
import time
from .camera_interface import CameraInterface

class DummyCamera(CameraInterface):
    def __init__(self, width=640, height=480):
        self.width = width
        self.height = height
        self.connected = False
        self.is_live = False

    def connect(self) -> bool:
        self.connected = True
        print("Dummy Camera Connected")
        return True

    def disconnect(self):
        self.connected = False
        print("Dummy Camera Disconnected")

    def snap(self) -> np.ndarray:
        if not self.connected:
            raise RuntimeError("Camera not connected")
        
        # Generate noise image
        noise = np.random.normal(0, 10, (self.height, self.width))
        
        # Generate some pattern
        y, x = np.mgrid[0:self.height, 0:self.width]
        pattern = 100 * (np.sin(x/20) + 1)
        
        img = np.clip(pattern + noise, 0, 255).astype(np.uint8)
        return img

    def start_live(self):
        self.is_live = True

    def stop_live(self):
        self.is_live = False
