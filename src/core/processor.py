from abc import ABC, abstractmethod
import numpy as np

class WavefrontProcessor(ABC):
    """
    Abstract base class for wavefront analysis algorithms.
    """

    def __init__(self, name="Generic Processor"):
        self.name = name
        self.params = {}

    @abstractmethod
    def process(self, image: np.ndarray, params: dict = None) -> dict:
        """
        Process the input image and return wavefront data.
        
        Args:
            image (np.ndarray): The input image from the sensor (2D array).
            params (dict): Analysis parameters from the GUI.
            
        Returns:
            dict: A dictionary containing results.
        """
        pass

    def propagate_focus(self, phase_map: np.ndarray, params: dict) -> dict:
        """
        Perform numerical propagation to find focal properties.
        
        Args:
            phase_map (np.ndarray): The wavefront phase.
            params (dict): Focus analysis parameters (distance, method, etc.)
            
        Returns:
            dict: Focus fields (2D, 3D, cuts) and statistics.
        """
        pass

    def set_reference(self, image: np.ndarray):
        """Set the reference image for relative analysis."""
        self.reference_image = image
        """Update a processing parameter."""
        self.params[key] = value

    def get_parameters(self):
        """Return current parameters."""
        return self.params
