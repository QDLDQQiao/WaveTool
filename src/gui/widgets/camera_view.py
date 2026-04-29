from PyQt6.QtWidgets import QWidget, QVBoxLayout
from PyQt6.QtCore import pyqtSignal
import numpy as np
import pyqtgraph as pg

class CameraView(QWidget):
    clicked = pyqtSignal()

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
        self._has_valid_view = False

        # Catch clicks from pyqtgraph scene (works with PlotItem-based ImageView).
        self.image_view.getView().scene().sigMouseClicked.connect(self._on_scene_clicked)

    def update_image(self, image_data: np.ndarray, crop_rect: tuple = None, force_autorange: bool = False):
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
            h, w = image_data.shape
            (x0, x1), (y0, y1) = self._resolve_crop_bounds(crop_rect, h, w)
            if x1 > x0 and y1 > y0:
                display_data = image_data[y0:y1, x0:x1]

        self.current_image = image_data # Store raw full image
        # autoRange=False keeps the zoom level if user changed it
        # autoLevels=False keeps the contrast levels if user changed it
        # Auto-scale on first frame, after hidden ref view becomes active, or if explicitly requested.
        should_autorange = force_autorange or (not self._has_valid_view)
        self.image_view.setImage(display_data.T, autoRange=should_autorange, autoLevels=should_autorange) # Transpose for correct orientation in pyqtgraph
        self._has_valid_view = True

    def _on_scene_clicked(self, event):
        if self.image_view.getView().sceneBoundingRect().contains(event.scenePos()):
            self.clicked.emit()

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

    def _resolve_crop_bounds(self, crop_rect, h, w):
        l = int(crop_rect[0]) if len(crop_rect) > 0 else 0
        t = int(crop_rect[1]) if len(crop_rect) > 1 else 0
        r = int(crop_rect[2]) if len(crop_rect) > 2 else 0
        b = int(crop_rect[3]) if len(crop_rect) > 3 else 0

        l = max(0, l)
        t = max(0, t)
        r = max(0, r)
        b = max(0, b)

        # Absolute mode
        x0_abs = min(max(l, 0), w)
        x1_abs = min(max(r, 0), w) if r > 0 else w
        y0_abs = min(max(t, 0), h)
        y1_abs = min(max(b, 0), h) if b > 0 else h
        abs_valid = (x1_abs > x0_abs) and (y1_abs > y0_abs)

        # Margin mode
        x0_m = min(max(l, 0), w)
        x1_m = max(x0_m, w - r if r > 0 else w)
        y0_m = min(max(t, 0), h)
        y1_m = max(y0_m, h - b if b > 0 else h)
        margin_valid = (x1_m > x0_m) and (y1_m > y0_m)

        if (not abs_valid and margin_valid) or (r <= l and b <= t and margin_valid):
            return (x0_m, x1_m), (y0_m, y1_m)
        if abs_valid:
            return (x0_abs, x1_abs), (y0_abs, y1_abs)
        return (0, w), (0, h)
