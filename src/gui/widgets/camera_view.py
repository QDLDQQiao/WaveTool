from PyQt6.QtWidgets import QWidget, QVBoxLayout
import numpy as np
import pyqtgraph as pg

class CameraView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        
        # Use pyqtgraph ImageView for advanced display (zoom, contrast, colormap)
        self.image_view = pg.ImageView(view=pg.PlotItem())
        self.image_view.ui.roiBtn.hide() # Hide ROI button by default
        self.image_view.ui.menuBtn.hide() # Hide Menu button by default
        
        # Set default colormap to something scientific (e.g., 'viridis' equivalent in pyqtgraph is 'bipolar' or custom)
        # pyqtgraph handles colormaps via its histogram LUT widget
        
        self.layout.addWidget(self.image_view)
        
        # Overlay item for Beam Envelope
        self.overlay_item = pg.ImageItem()
        self.overlay_item.setOpacity(1) # Semi-transparent overlay
        # Add to the ViewBox inside ImageView
        self.image_view.getView().addItem(self.overlay_item)
        self.overlay_item.hide()
        
        self.current_image = None

    def update_image(self, image_data: np.ndarray, crop_rect: tuple = None):
        """
        Update the display with a numpy array.
        
        Args:
            image_data: 2D numpy array
            crop_rect: tuple (left, top, right, bottom) or None. 
                       If provided and not all zeros, crops the display.
        """
        if image_data is None:
            return

        display_data = image_data
        
        # Apply crop if valid
        if crop_rect and any(crop_rect):
            l, t, r, b = map(int, crop_rect)
            h, w = image_data.shape
            # Ensure bounds
            l = max(0, l)
            t = max(0, t)
            r = min(w, r) if r > 0 else w
            b = min(h, b) if b > 0 else h
            
            if r > l and b > t:
                display_data = image_data[t:b, l:r]

        self.current_image = image_data # Store raw full image
        # autoRange=False keeps the zoom level if user changed it
        # autoLevels=False keeps the contrast levels if user changed it
        # We only auto-scale on the first frame or if explicitly requested
        self.image_view.setImage(display_data.T, autoRange=False, autoLevels=False) # Transpose for correct orientation in pyqtgraph

    def show_envelope(self, envelope_data: np.ndarray):
        """Display the envelope as an overlay."""
        if envelope_data is None:
            return
        # Transpose to match the main image orientation
        self.overlay_item.setImage(envelope_data.T)
        self.overlay_item.show()

    def hide_envelope(self):
        """Hide the envelope overlay."""
        self.overlay_item.hide()

    def get_image(self):
        return self.current_image
