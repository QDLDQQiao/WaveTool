from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QTabWidget, QHBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np

class ResultsDisplay(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        
        # Metrics Label
        self.metrics_label = QLabel("No Data")
        self.layout.addWidget(self.metrics_label)

        # Tabs for different views
        self.tabs = QTabWidget()
        self.layout.addWidget(self.tabs)

        # Tab 1: Phase & Zernike
        self.tab_phase = QWidget()
        self.setup_phase_tab()
        self.tabs.addTab(self.tab_phase, "Phase & Zernike")

        # Tab 2: Focus Analysis
        self.tab_focus = QWidget()
        self.setup_focus_tab()
        self.tabs.addTab(self.tab_focus, "Focus Analysis")

    def setup_phase_tab(self):
        layout = QVBoxLayout(self.tab_phase)
        self.fig_phase = Figure(figsize=(5, 8))
        self.canvas_phase = FigureCanvas(self.fig_phase)
        layout.addWidget(self.canvas_phase)
        
        # Subplots: Top for Phase, Bottom for Zernike
        self.ax_phase = self.fig_phase.add_subplot(211)
        self.ax_zernike = self.fig_phase.add_subplot(212)
        self.fig_phase.tight_layout()

    def setup_focus_tab(self):
        layout = QHBoxLayout(self.tab_focus)
        
        # Left: 2D Spot + Cuts
        self.fig_focus_2d = Figure(figsize=(5, 5))
        self.canvas_focus_2d = FigureCanvas(self.fig_focus_2d)
        layout.addWidget(self.canvas_focus_2d)
        
        # Right: 3D Surface
        self.fig_focus_3d = Figure(figsize=(5, 5))
        self.canvas_focus_3d = FigureCanvas(self.fig_focus_3d)
        layout.addWidget(self.canvas_focus_3d)

        # Setup axes
        self.ax_focus_2d = self.fig_focus_2d.add_subplot(221) # Main spot
        self.ax_cut_x = self.fig_focus_2d.add_subplot(223)    # X cut
        self.ax_cut_y = self.fig_focus_2d.add_subplot(222)    # Y cut
        # 224 is empty or text
        
        self.ax_focus_3d = self.fig_focus_3d.add_subplot(111, projection='3d')

    def update_results(self, results: dict):
        # 1. Update Metrics
        metrics_text = f"FWHM: {results.get('fwhm', 0):.2f} px | Sigma: {results.get('sigma', 0):.2f} px | Residual: {results.get('zernike_residual', 0):.4f}"
        self.metrics_label.setText(metrics_text)

        # 2. Phase Map
        self.ax_phase.clear()
        if 'phase_map' in results:
            im = self.ax_phase.imshow(results['phase_map'], cmap='jet')
            self.ax_phase.set_title("Wavefront Phase")
            # self.fig_phase.colorbar(im, ax=self.ax_phase) # Optional: add colorbar logic

        # 3. Zernike Coefficients
        self.ax_zernike.clear()
        if 'zernike_coeffs' in results:
            coeffs = results['zernike_coeffs']
            self.ax_zernike.bar(range(len(coeffs)), coeffs)
            self.ax_zernike.set_title("Zernike Coefficients")
            self.ax_zernike.set_xlabel("Mode Index")
            self.ax_zernike.set_ylabel("Amplitude")

        self.canvas_phase.draw()

        # 4. Focus Analysis
        if 'focus_field' in results:
            field = results['focus_field']
            h, w = field.shape
            cy, cx = h // 2, w // 2
            
            # Crop for better visualization if needed, or show full
            # Let's show a ROI around center
            roi_size = 50
            y1, y2 = max(0, cy-roi_size), min(h, cy+roi_size)
            x1, x2 = max(0, cx-roi_size), min(w, cx+roi_size)
            roi = field[y1:y2, x1:x2]
            
            # 2D Plot
            self.ax_focus_2d.clear()
            self.ax_focus_2d.imshow(roi, cmap='hot')
            self.ax_focus_2d.set_title("Focal Spot (ROI)")
            
            # Cuts
            mid_y, mid_x = roi.shape[0]//2, roi.shape[1]//2
            cut_x = roi[mid_y, :]
            cut_y = roi[:, mid_x]
            
            self.ax_cut_x.clear()
            self.ax_cut_x.plot(cut_x)
            self.ax_cut_x.set_title("X-Cut")
            
            self.ax_cut_y.clear()
            self.ax_cut_y.plot(cut_y, range(len(cut_y))) # Plot vertically
            self.ax_cut_y.invert_yaxis()
            self.ax_cut_y.set_title("Y-Cut")

            self.canvas_focus_2d.draw()

            # 3D Plot
            self.ax_focus_3d.clear()
            X, Y = np.meshgrid(np.arange(roi.shape[1]), np.arange(roi.shape[0]))
            self.ax_focus_3d.plot_surface(X, Y, roi, cmap='viridis')
            self.ax_focus_3d.set_title("3D Intensity Profile")
            
            self.canvas_focus_3d.draw()
