from abc import ABC, abstractmethod
import numpy as np

class CameraInterface(ABC):
    @abstractmethod
    def connect(self) -> bool:
        pass

    @abstractmethod
    def disconnect(self):
        pass

    @abstractmethod
    def snap(self) -> np.ndarray:
        """Capture a single frame."""
        pass

    @abstractmethod
    def start_live(self):
        pass

    @abstractmethod
    def stop_live(self):
        pass
