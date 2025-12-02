from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSpinBox, 
                             QPushButton, QGridLayout, QScrollArea)
try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT
except ImportError:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT
from matplotlib.figure import Figure
import numpy as np

from ...core.calculations import fit_zernike
from .focus_window import FocusAnalysisWindow

class AnalysisResultWindow(QWidget):
    def __init__(self, processor, results, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Analysis Results")
        self.resize(1400, 900)
        self.processor = processor
        self.results = results
        
        self.layout = QVBoxLayout(self)
        
        # --- Top Bar: Stats & Inputs ---
        top_layout = QHBoxLayout()
        
        roc_x = results.get('roc_x', float('inf'))
        roc_y = results.get('roc_y', float('inf'))
        
        stats_text = (f"PV: {results.get('pv_value', 0):.4f} rad | "
                      f"RMS: {results.get('rms_value', 0):.4f} rad | "
                      f"Rx: {roc_x:.4f} m | Ry: {roc_y:.4f} m")
        
        if 'rotation_angle' in results:
            angle = results['rotation_angle']
            period = results['period_real']
            stats_text += f" | Rot: {angle:.2f}° | P: {period[0]:.2f}x{period[1]:.2f} um"
                      
        self.lbl_stats = QLabel(stats_text)
        self.lbl_stats.setStyleSheet("font-size: 14px; font-weight: bold;")
        top_layout.addWidget(self.lbl_stats)
        
        top_layout.addStretch()
        
        top_layout.addWidget(QLabel("Zernike Order:"))
        self.spin_zernike = QSpinBox()
        self.spin_zernike.setRange(1, 100)
        self.spin_zernike.setValue(10)
        top_layout.addWidget(self.spin_zernike)
        
        self.btn_recalc_zernike = QPushButton("Update Zernike")
        self.btn_recalc_zernike.clicked.connect(self.update_zernike)
        top_layout.addWidget(self.btn_recalc_zernike)
        
        self.layout.addLayout(top_layout)
        
        # --- Center: Figures Grid ---
        scroll = QScrollArea()
        self.layout.addWidget(scroll)
        
        content_widget = QWidget()
        scroll.setWidget(content_widget)
        scroll.setWidgetResizable(True)
        
        self.grid = QGridLayout(content_widget)
        
        # 1. Displacement X
        self.add_plot(0, 0, "Displacement X", results.get('displacement_x'))
        # 2. Displacement Y
        self.add_plot(0, 1, "Displacement Y", results.get('displacement_y'))
        # 3. Transmission
        self.add_plot(0, 2, "Transmission", results.get('transmission'), cmap='gray')
        
        # 4. Integrated Phase
        self.add_plot(1, 0, "Integrated Phase", results.get('phase_map'), cmap='jet')
        # 5. Residual (2nd Order)
        self.add_plot(1, 1, "Residual (2nd Order Removed)", results.get('phase_residual_2nd'), cmap='jet')
        
        # 6. Zernike Coeffs
        self.add_bar_plot(1, 2, "Zernike Coefficients", results.get('zernike_coeffs'))
        
        # 7. Zernike Fitted
        self.add_plot(2, 0, "Zernike Fitted Phase", results.get('zernike_fitted'), cmap='jet')
        # 8. Zernike Residual
        self.add_plot(2, 1, "Zernike Residual", results.get('zernike_residual'), cmap='jet')
        # 9. Mask
        self.add_plot(2, 2, "Analysis Mask", results.get('mask'), cmap='gray')
        
        # --- Bottom: Focus Button ---
        self.btn_focus = QPushButton("Focus Analysis")
        self.btn_focus.setStyleSheet("font-size: 16px; padding: 10px;")
        self.btn_focus.clicked.connect(self.open_focus_window)
        self.layout.addWidget(self.btn_focus)

    def add_plot(self, row, col, title, data, cmap='viridis'):
        if data is None: return
        
        fig = Figure(figsize=(4, 3))
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        im = ax.imshow(data, cmap=cmap, aspect='auto')
        fig.colorbar(im, ax=ax)
        ax.set_title(title)
        fig.tight_layout()
        
        # Enable interaction (zoom/pan)
        # Matplotlib toolbar is one way, but just enabling mouse interaction works too
        # For simple zoom/pan, we can use the built-in navigation toolbar or just rely on default behavior if enabled.
        # However, FigureCanvasQTAgg doesn't enable interaction by default without a toolbar.
        # Let's add a simple toolbar or just use pyqtgraph if we want better interaction.
        # Sticking to matplotlib as requested, let's add the toolbar.
        
        # Create a container widget to hold canvas and toolbar
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(canvas)
        
        toolbar = NavigationToolbar2QT(canvas, container)
        # toolbar.hide() # Show toolbar for interaction
        layout.addWidget(toolbar)
        
        self.grid.addWidget(container, row, col)

    def add_bar_plot(self, row, col, title, data):
        if data is None: return
        
        fig = Figure(figsize=(4, 3))
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        ax.bar(range(len(data)), data)
        ax.set_title(title)
        fig.tight_layout()
        
        # Add toolbar for interaction
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(canvas)
        toolbar = NavigationToolbar2QT(canvas, container)
        layout.addWidget(toolbar)
        
        self.grid.addWidget(container, row, col)

    def open_focus_window(self):
        if 'phase_map' in self.results:
            self.focus_win = FocusAnalysisWindow(self.processor, 
                                               self.results['phase_map'],
                                               self.results.get('transmission'))
            self.focus_win.show()

    def clear_plot(self, row, col):
        item = self.grid.itemAtPosition(row, col)
        if item:
            widget = item.widget()
            if widget:
                self.grid.removeWidget(widget)
                widget.deleteLater()

    def update_zernike(self):
        phase = self.results.get('phase_map')
        if phase is None: return
        
        order = self.spin_zernike.value()
        coeffs, fitted, residual = fit_zernike(phase, n_terms=order)
        
        self.results['zernike_coeffs'] = coeffs
        self.results['zernike_fitted'] = fitted
        self.results['zernike_residual'] = residual

        # Update plots
        self.clear_plot(1, 2)
        self.add_bar_plot(1, 2, "Zernike Coefficients", coeffs)
        
        self.clear_plot(2, 0)
        self.add_plot(2, 0, "Zernike Fitted Phase", fitted, cmap='jet')
        
        self.clear_plot(2, 1)
        self.add_plot(2, 1, "Zernike Residual", residual, cmap='jet')
