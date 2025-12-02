from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                             QDoubleSpinBox, QPushButton, QWidget)
from PyQt6.QtCore import Qt
try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
except ImportError:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import scipy.ndimage as snd

class MaskPreviewDialog(QDialog):
    def __init__(self, image, pitch_px, initial_threshold=0.2, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Mask Preview")
        self.resize(800, 600)
        
        self.image = image
        self.pitch_px = pitch_px
        self.threshold = initial_threshold
        self.mask = None
        self.accepted_mask = False
        
        # Pre-calculate envelope for speed
        self.img_env = self._calculate_envelope()
        
        self.layout = QVBoxLayout(self)
        
        # --- Controls ---
        controls_layout = QHBoxLayout()
        
        controls_layout.addWidget(QLabel("Mask Threshold:"))
        self.spin_thresh = QDoubleSpinBox()
        self.spin_thresh.setRange(0.0, 1.0)
        self.spin_thresh.setSingleStep(0.05)
        self.spin_thresh.setValue(self.threshold)
        self.spin_thresh.valueChanged.connect(self.update_mask)
        controls_layout.addWidget(self.spin_thresh)
        
        controls_layout.addStretch()
        
        self.btn_confirm = QPushButton("Confirm & Continue")
        self.btn_confirm.clicked.connect(self.confirm)
        controls_layout.addWidget(self.btn_confirm)
        
        self.layout.addLayout(controls_layout)
        
        # --- Plot ---
        self.fig = Figure(figsize=(5, 4))
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.layout.addWidget(self.canvas)
        
        self.update_mask()

    def _calculate_envelope(self):
        # Same logic as HartmannProcessor
        # 1. Preprocessing (Gaussian Filter)
        # In HartmannProcessor: sigma = int(hole_size / p_x / 4)
        # We don't have hole_size here easily, but let's approximate or use a small sigma
        # Assuming hole_size ~ pitch / 5 for typical Hartmann
        sigma = max(1, int(self.pitch_px / 20)) 
        img_filter = snd.gaussian_filter(self.image, sigma=sigma)
        
        # 2. Flat field correction
        # In HartmannProcessor: sigma_flat = int(pitch / p_x)
        sigma_flat = int(self.pitch_px)
        flat = snd.gaussian_filter(img_filter, sigma_flat)
        img_flat = img_filter / (flat + 1e-6)
        
        # 3. Envelope Calculation
        # Maximum filter then Gaussian filter
        img_max = snd.maximum_filter(img_flat, size=int(self.pitch_px))
        img_env = snd.gaussian_filter(img_max, sigma=int(self.pitch_px))
        
        # Store min/range for thresholding
        self.env_min = img_env.min()
        self.env_range = img_env.max() - self.env_min
        
        return img_env

    def update_mask(self):
        self.threshold = self.spin_thresh.value()
        
        # Correct formula: (val - min) > thresh * range
        mask_val = (self.img_env - self.env_min) > (self.threshold * self.env_range)
        self.mask = mask_val.astype(float)
        
        self.ax.clear()
        self.ax.imshow(self.image, cmap='gray')
        # Overlay mask (red, semi-transparent)
        # Create RGBA mask
        h, w = self.mask.shape
        overlay = np.zeros((h, w, 4))
        overlay[..., 0] = 1.0 # Red
        overlay[..., 3] = self.mask * 0.3 # Alpha
        
        self.ax.imshow(overlay)
        self.ax.set_title(f"Mask Threshold: {self.threshold:.2f}")
        self.ax.axis('off')
        self.canvas.draw()

    def confirm(self):
        self.accepted_mask = True
        self.accept()
