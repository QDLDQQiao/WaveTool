from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QDoubleSpinBox, 
                             QCheckBox, QComboBox, QPushButton, QGroupBox, QTabWidget, QSlider, QProgressBar, QApplication, QGridLayout)
from PyQt6.QtCore import Qt, QRectF
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import pyqtgraph as pg
import os

class FocusAnalysisWindow(QWidget):
    def __init__(self, processor, phase_map, transmission=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Focus Analysis")
        self.resize(1200, 800)
        self.processor = processor
        self.phase_map = phase_map
        self.transmission = transmission
        
        self.main_layout = QHBoxLayout(self)
        self.results = None # Initialize results
        
        # --- Left: Parameters ---
        self.param_panel = QGroupBox("Parameters")
        self.param_layout = QVBoxLayout(self.param_panel)
        self.main_layout.addWidget(self.param_panel, stretch=1)
        
        self.distance = QDoubleSpinBox()
        self.distance.setRange(-1e9, 1e9)
        self.distance.setValue(100)
        self.distance.setSuffix(" mm")
        self.param_layout.addWidget(QLabel("Distance:"))
        self.param_layout.addWidget(self.distance)
        
        self.dist_range = QDoubleSpinBox()
        self.dist_range.setRange(0, 1e4)
        self.dist_range.setValue(10)
        self.dist_range.setSuffix(" mm")
        self.param_layout.addWidget(QLabel("Distance Range:"))
        self.param_layout.addWidget(self.dist_range)

        self.dist_step = QDoubleSpinBox()
        self.dist_step.setRange(0, 1000)
        self.dist_step.setValue(1)
        self.dist_step.setSuffix(" mm")
        self.param_layout.addWidget(QLabel("Distance step:"))
        self.param_layout.addWidget(self.dist_step)
        
        self.real_intensity = QCheckBox("Real Beam Intensity")
        self.param_layout.addWidget(self.real_intensity)
        
        self.upsampling = QDoubleSpinBox()
        self.upsampling.setRange(0.1, 10)
        self.upsampling.setValue(1)
        self.param_layout.addWidget(QLabel("Up-sampling Factor:"))
        self.param_layout.addWidget(self.upsampling)

        self.btn_auto_sampling = QPushButton("Auto Suggest Sampling")
        self.btn_auto_sampling.clicked.connect(self.auto_suggest_sampling)
        self.param_layout.addWidget(self.btn_auto_sampling)

        self.lbl_sampling_hint = QLabel("Suggestion: -")
        self.lbl_sampling_hint.setWordWrap(True)
        self.param_layout.addWidget(self.lbl_sampling_hint)

        self.chk_advanced_sampling = QCheckBox("Advanced Sampling Guard")
        self.chk_advanced_sampling.toggled.connect(self.toggle_advanced_sampling)
        self.param_layout.addWidget(self.chk_advanced_sampling)

        self.advanced_sampling_group = QGroupBox("Advanced Guard Settings")
        self.advanced_sampling_layout = QGridLayout(self.advanced_sampling_group)
        self.advanced_sampling_layout.addWidget(QLabel("Suggest Budget Ratio:"), 0, 0)
        self.suggest_budget_ratio = QDoubleSpinBox()
        self.suggest_budget_ratio.setRange(0.05, 0.95)
        self.suggest_budget_ratio.setSingleStep(0.01)
        self.suggest_budget_ratio.setDecimals(2)
        self.suggest_budget_ratio.setValue(0.30)
        self.suggest_budget_ratio.setToolTip(
            "Fraction of available RAM used by auto-suggest when proposing parameters.\n"
            "Lower = safer suggestions, higher = more aggressive."
        )
        self.advanced_sampling_layout.addWidget(self.suggest_budget_ratio, 0, 1)

        self.advanced_sampling_layout.addWidget(QLabel("Run Budget Ratio:"), 1, 0)
        self.run_budget_ratio = QDoubleSpinBox()
        self.run_budget_ratio.setRange(0.05, 0.95)
        self.run_budget_ratio.setSingleStep(0.01)
        self.run_budget_ratio.setDecimals(2)
        self.run_budget_ratio.setValue(0.35)
        self.run_budget_ratio.setToolTip(
            "Fraction of available RAM allowed before UI-side guard auto-clamps settings.\n"
            "Lower = clamps earlier, higher = allows larger runs."
        )
        self.advanced_sampling_layout.addWidget(self.run_budget_ratio, 1, 1)

        self.advanced_sampling_layout.addWidget(QLabel("Backend Stack Ratio:"), 2, 0)
        self.backend_stack_ratio = QDoubleSpinBox()
        self.backend_stack_ratio.setRange(0.05, 0.95)
        self.backend_stack_ratio.setSingleStep(0.01)
        self.backend_stack_ratio.setDecimals(2)
        self.backend_stack_ratio.setValue(0.60)
        self.backend_stack_ratio.setToolTip(
            "Fraction of available RAM reserved for propagated intensity stack in backend.\n"
            "Lower = safer against OOM, higher = allows larger stack."
        )
        self.advanced_sampling_layout.addWidget(self.backend_stack_ratio, 2, 1)

        self.advanced_sampling_layout.addWidget(QLabel("Worker Mem Ratio:"), 3, 0)
        self.worker_mem_ratio = QDoubleSpinBox()
        self.worker_mem_ratio.setRange(0.05, 0.95)
        self.worker_mem_ratio.setSingleStep(0.01)
        self.worker_mem_ratio.setDecimals(2)
        self.worker_mem_ratio.setValue(0.25)
        self.worker_mem_ratio.setToolTip(
            "Fraction of available RAM used to limit parallel worker count.\n"
            "Lower = fewer workers (safer), higher = more workers (faster if memory permits)."
        )
        self.advanced_sampling_layout.addWidget(self.worker_mem_ratio, 3, 1)

        self.param_layout.addWidget(self.advanced_sampling_group)
        self.advanced_sampling_group.hide()
        
        self.direction = QComboBox()
        self.direction.addItems(["forward", "backward"])
        self.param_layout.addWidget(QLabel("Direction:"))
        self.param_layout.addWidget(self.direction)

        self.method = QComboBox()
        self.method.addItems(["TF", "IR", "RS", "QPF", "Wofry", "default"])
        self.param_layout.addWidget(QLabel("Method:"))
        self.param_layout.addWidget(self.method)
        
        # Magnification parameters (for Wofry)
        self.lbl_mag_x = QLabel("Mag X:")
        self.mag_x = QDoubleSpinBox()
        self.mag_x.setRange(0.001, 1000)
        self.mag_x.setValue(1.0)
        self.param_layout.addWidget(self.lbl_mag_x)
        self.param_layout.addWidget(self.mag_x)

        self.lbl_mag_y = QLabel("Mag Y:")
        self.mag_y = QDoubleSpinBox()
        self.mag_y.setRange(0.001, 1000)
        self.mag_y.setValue(1.0)
        self.param_layout.addWidget(self.lbl_mag_y)
        self.param_layout.addWidget(self.mag_y)
        
        self.method.currentTextChanged.connect(self.toggle_mag_params)
        self.toggle_mag_params(self.method.currentText())
        
        self.padding_scale = QDoubleSpinBox()
        self.padding_scale.setRange(1.0, 10.0)
        self.padding_scale.setValue(1.0)
        self.padding_scale.setSingleStep(0.1)
        self.param_layout.addWidget(QLabel("Padding Scale:"))
        self.param_layout.addWidget(self.padding_scale)
        
        self.calc_sigma = QCheckBox("Calculate Beam Size")
        self.calc_sigma.setChecked(False)
        self.param_layout.addWidget(self.calc_sigma)
        
        self.btn_calc = QPushButton("Calculate Focus")
        self.btn_calc.clicked.connect(self.run_analysis)
        self.param_layout.addWidget(self.btn_calc)
        
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop_analysis)
        self.param_layout.addWidget(self.btn_stop)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.param_layout.addWidget(self.progress_bar)
        
        self.param_layout.addStretch()
        
        # Statistics
        self.stats_label = QLabel("Results:\nSigma: -\nFocal Length: -\nStrehl: -")
        self.param_layout.addWidget(self.stats_label)

        # --- Right: Figures ---
        self.tabs = QTabWidget()
        self.main_layout.addWidget(self.tabs, stretch=4)
        
        # Tab 1: 2D Focus & Cuts
        self.tab_2d = QWidget()
        self.layout_2d = QVBoxLayout(self.tab_2d)
        
        # Use GraphicsLayoutWidget for custom layout (Profiles + Image)
        self.glw = pg.GraphicsLayoutWidget()
        self.layout_2d.addWidget(self.glw)
        
        # Layout:
        # Row 0: [Empty, X-Plot, Empty]
        # Row 1: [Y-Plot, ViewBox, Histogram]
        
        # X Profile (Top)
        self.plot_x = self.glw.addPlot(row=0, col=1)
        self.plot_x.setLabel('left', "Int")
        self.plot_x.showGrid(x=True, y=True)
        self.plot_x.setMouseEnabled(x=True, y=False)
        self.plot_x.setMaximumHeight(150)
        
        # Y Profile (Left)
        self.plot_y = self.glw.addPlot(row=1, col=0)
        self.plot_y.setLabel('bottom', "Int")
        self.plot_y.showGrid(x=True, y=True)
        self.plot_y.setMouseEnabled(x=False, y=True)
        self.plot_y.setMaximumWidth(150)
        # self.plot_y.invertY(True) # Match image coordinates if needed
        
        # Image View (Center)
        # CHANGE: Use addPlot instead of addViewBox to support labels
        self.plot_center = self.glw.addPlot(row=1, col=1)
        self.plot_center.setAspectLocked(True)
        self.plot_center.setLabel('bottom', "X (um)")
        self.plot_center.setLabel('left', "Y (um)")
        
        # Retrieve the ViewBox from the PlotItem for image handling and linking
        self.view_main = self.plot_center.getViewBox()
        
        self.img_item = pg.ImageItem()
        self.view_main.addItem(self.img_item)
        
        # Link axes
        self.plot_x.setXLink(self.view_main)
        self.plot_y.setYLink(self.view_main)
        
        # Crosshairs
        self.v_line = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('y', style=Qt.PenStyle.DashLine, width=1))
        self.h_line = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen('y', style=Qt.PenStyle.DashLine, width=1))
        self.v_line.setOpacity(0.5)
        self.h_line.setOpacity(0.5)
        self.view_main.addItem(self.v_line)
        self.view_main.addItem(self.h_line)
        
        # Histogram (Right)
        self.hist = pg.HistogramLUTItem()
        self.hist.setImageItem(self.img_item)
        self.glw.addItem(self.hist, row=1, col=2)
        
        # Connect histogram levels change to update profile axes
        self.hist.sigLevelsChanged.connect(self.sync_profile_levels)
        
        # Layout sizing
        self.glw.ci.layout.setRowStretchFactor(0, 1)
        self.glw.ci.layout.setRowStretchFactor(1, 4)
        self.glw.ci.layout.setColumnStretchFactor(0, 1)
        self.glw.ci.layout.setColumnStretchFactor(1, 4)
        
        # Mouse Click Event
        self.view_main.scene().sigMouseClicked.connect(self.on_image_clicked)
        
        # State
        self.selected_pos = None # (x_um, y_um)
        
        # Slider for Z propagation
        self.slider_layout = QHBoxLayout()
        self.slider_layout.addWidget(QLabel("Z Position:"))
        self.slider_z = QSlider(Qt.Orientation.Horizontal)
        self.slider_z.valueChanged.connect(self.update_slice)
        self.slider_layout.addWidget(self.slider_z)
        self.lbl_z_pos = QLabel("0 mm")
        self.slider_layout.addWidget(self.lbl_z_pos)
        self.layout_2d.addLayout(self.slider_layout)
        
        self.tabs.addTab(self.tab_2d, "2D Focus")
        
        # Tab 2: 3D Focus
        self.tab_3d = QWidget()
        self.layout_3d = QVBoxLayout(self.tab_3d)
        
        # Controls for 3D view - Compact Layout
        controls_group = QGroupBox("3D Settings")
        controls_layout = QGridLayout(controls_group)
        controls_layout.setContentsMargins(5, 5, 5, 5)
        controls_layout.setSpacing(5)
        
        # Row 0: Threshold & Opacity
        controls_layout.addWidget(QLabel("Thresh:"), 0, 0)
        self.slider_threshold = QSlider(Qt.Orientation.Horizontal)
        self.slider_threshold.setRange(0, 99)
        self.slider_threshold.setValue(10)
        self.slider_threshold.valueChanged.connect(self.update_3d_view)
        controls_layout.addWidget(self.slider_threshold, 0, 1)
        self.lbl_threshold = QLabel("10%")
        controls_layout.addWidget(self.lbl_threshold, 0, 2)
        
        controls_layout.addWidget(QLabel("Opacity:"), 0, 3)
        self.slider_opacity = QSlider(Qt.Orientation.Horizontal)
        self.slider_opacity.setRange(0, 100)
        self.slider_opacity.setValue(50)
        self.slider_opacity.valueChanged.connect(self.update_3d_view)
        controls_layout.addWidget(self.slider_opacity, 0, 4)
        self.lbl_opacity = QLabel("50%")
        controls_layout.addWidget(self.lbl_opacity, 0, 5)

        # Row 1: Color & Scales
        controls_layout.addWidget(QLabel("Color:"), 1, 0)
        self.combo_color = QComboBox()
        self.combo_color.addItems(["Green", "Red", "Blue", "Fire", "Gray"])
        self.combo_color.currentTextChanged.connect(self.update_3d_view)
        controls_layout.addWidget(self.combo_color, 1, 1)
        
        controls_layout.addWidget(QLabel("Scale XY:"), 1, 2)
        self.slider_scale_xy = QSlider(Qt.Orientation.Horizontal)
        self.slider_scale_xy.setRange(10, 1000)
        self.slider_scale_xy.setValue(100)
        self.slider_scale_xy.valueChanged.connect(self.update_3d_view)
        controls_layout.addWidget(self.slider_scale_xy, 1, 3)
        self.lbl_scale_xy = QLabel("100")
        controls_layout.addWidget(self.lbl_scale_xy, 1, 4)

        controls_layout.addWidget(QLabel("Scale Z:"), 1, 5)
        self.slider_scale_z = QSlider(Qt.Orientation.Horizontal)
        self.slider_scale_z.setRange(10, 2000)
        self.slider_scale_z.setValue(200)
        self.slider_scale_z.valueChanged.connect(self.update_3d_view)
        controls_layout.addWidget(self.slider_scale_z, 1, 6)
        self.lbl_scale_z = QLabel("200")
        controls_layout.addWidget(self.lbl_scale_z, 1, 7)

        # --- INSERT THIS LINE HERE ---
        controls_group.setMaximumHeight(100) 
        # -----------------------------

        self.layout_3d.addWidget(controls_group)
        
        # Use pyqtgraph GLViewWidget for interactive 3D volume rendering.
        # Some platforms (e.g. CentOS 7 without full OpenGL stack) do not support QOpenGLWidget.
        self.gl_supported = False
        self.view_3d = None
        self.lbl_3d_fallback = None
        try:
            import pyqtgraph.opengl as gl
            self._gl = gl
            self.view_3d = gl.GLViewWidget()
            self.view_3d.setBackgroundColor('w')
            self.view_3d.setCameraPosition(distance=200)
            self.layout_3d.addWidget(self.view_3d)
            
            # Add grid
            g = gl.GLGridItem()
            g.scale(10, 10, 1)
            g.setColor((0, 0, 0, 255)) # Black grid for white background
            self.view_3d.addItem(g)
            self.gl_supported = True
        except Exception as e:
            self.lbl_3d_fallback = QLabel(
                "3D Focus is disabled on this platform (OpenGL not available).\n"
                f"Reason: {e}\n"
                "2D/XZ/YZ focus analysis still works."
            )
            self.lbl_3d_fallback.setWordWrap(True)
            self.layout_3d.addWidget(self.lbl_3d_fallback)
            self.slider_threshold.setEnabled(False)
            self.slider_opacity.setEnabled(False)
            self.combo_color.setEnabled(False)
            self.slider_scale_xy.setEnabled(False)
            self.slider_scale_z.setEnabled(False)
        
        self.tabs.addTab(self.tab_3d, "3D Focus")
        
        # Tab 3: XZ Cut
        self.tab_xz = QWidget()
        self.layout_xz = QVBoxLayout(self.tab_xz)
        self.plot_xz = pg.ImageView(view=pg.PlotItem())
        # self.plot_xz.ui.roiBtn.hide()
        # self.plot_xz.ui.menuBtn.hide()
        self.plot_xz.getView().setLabel('left', "Distance (mm)")
        self.plot_xz.getView().setLabel('bottom', "X (um)")
        self.layout_xz.addWidget(self.plot_xz)
        self.tabs.addTab(self.tab_xz, "XZ Cut")
        
        # Tab 4: YZ Cut
        self.tab_yz = QWidget()
        self.layout_yz = QVBoxLayout(self.tab_yz)
        self.plot_yz = pg.ImageView(view=pg.PlotItem())
        # self.plot_yz.ui.roiBtn.hide()
        # self.plot_yz.ui.menuBtn.hide()
        self.plot_yz.getView().setLabel('left', "Distance (mm)")
        self.plot_yz.getView().setLabel('bottom', "Y (um)")
        self.layout_yz.addWidget(self.plot_yz)
        self.tabs.addTab(self.tab_yz, "YZ Cut")


    def update_slice(self, index):
        """
        Updates the 2D focus image based on the selected Z-slice (distance).
        """
        if self.results is None:
            return
            
        intensity_profiles = self.results.get("intensity_profiles")
        L_z_list = self.results.get("prop_Lz")
        prop_dist = self.results.get("prop_distance")
        
        if intensity_profiles is None or index >= len(intensity_profiles):
            return

        # Get the 2D image at the selected index
        focus_image = intensity_profiles[index]
        L_z = L_z_list[index]

        self.lbl_z_pos.setText(f"Z: {prop_dist[index]*1e3:.2f} mm")

        # Update the 2D view
        # We need to set the image and update the scale/rect
        self.img_item.setImage(focus_image.transpose())
        
        # Update the scale (physical units)
        h, w = focus_image.shape
        # L_z is [height_mm, width_mm] usually, convert to um for display if needed
        # Assuming L_z is in meters based on talbot.py, let's convert to microns
        width_um = L_z[1] * 1e6
        height_um = L_z[0] * 1e6
        
        # Set the bounding rect for the image item
        # pyqtgraph uses (x, y, w, h)

        rect = QRectF(-width_um/2, -height_um/2, width_um, height_um)
        self.img_item.setRect(rect)
        # self.view_main.autoRange()
        
        self.update_profiles()
        
        # Update label to show current distance offset
        # Assuming the slider index maps to the prop_dist_range used in calculation
        # We can't easily get the exact 'z' value unless we stored it, 
        # but we can show the index or calculate if we knew the step.
        # Better: Store the z-axis values in results.
        
        # For now, just update the image.
        # Stats
        # Extract sigma values (default to 0 if not present)
        sigma_x = self.results.get('sigma_x_list', 0.0)[index]
        sigma_y = self.results.get('sigma_y_list', 0.0)[index]
        fwhm_x = self.results.get('fwhm_x_list', 0.0)[index]
        fwhm_y = self.results.get('fwhm_y_list', 0.0)[index]
        d_prop = self.results.get('prop_distance', 0.0)[index] * 1e3 # in mm
        stats_text = (
            f"Results:\n"
            f"Sigma X: {sigma_x*1e6:.2f} um\n"
            f"Sigma Y: {sigma_y*1e6:.2f} um\n"
            f"FWHM X: {fwhm_x*1e6:.2f} um\n"
            f"FWHM Y: {fwhm_y*1e6:.2f} um\n"
            f"Prop Length: {d_prop:.2f} mm\n"
            f"Strehl: {self.results.get('strehl', 0):.2f}"
        )
        self.stats_label.setText(stats_text)

    def run_analysis(self):
        self.btn_calc.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.stop_requested = False
        self.progress_bar.setValue(0)
        QApplication.processEvents() # Force UI update
        
        params = {
            "distance_mm": self.distance.value(),
            "range_mm": self.dist_range.value(),
            "step_mm": self.dist_step.value(),
            "direction": self.direction.currentText(),
            "real_intensity": self.real_intensity.isChecked(),
            "upsampling": self.upsampling.value(),
            "method": self.method.currentText(),
            "calc_sigma": self.calc_sigma.isChecked(),
            "magnification_x": self.mag_x.value(),
            "magnification_y": self.mag_y.value(),
            "padding_scale": self.padding_scale.value(),
            "available_memory_bytes": self._get_available_memory_bytes(),
            **self._advanced_sampling_params()
        }
        params, guard_msg = self._guard_sampling_memory(params)
        if guard_msg:
            self.lbl_sampling_hint.setText(f"Suggestion: {guard_msg}")
        
        # Define a callback to update progress
        def progress_callback(percent):
            self.progress_bar.setValue(int(percent))
            QApplication.processEvents()
            
        try:
            self.results = self.processor.propagate_focus(
                self.phase_map, 
                params, 
                self.transmission, 
                progress_callback=progress_callback,
                check_stop=lambda: self.stop_requested
            )
            if self.results:
                self.update_plots(self.results)
        except Exception as e:
            print(f"Analysis stopped or failed: {e}")
            self.lbl_sampling_hint.setText(f"Suggestion: run failed ({e}). Try lower upsampling/padding.")
        finally:
            self.btn_calc.setEnabled(True)
            self.btn_stop.setEnabled(False)
            self.progress_bar.setValue(100)

        
    def update_plots(self, results):
        # Stats
        self.stats_label.setText(f"Results:\nFocal Length: {results['focal_length']:.2f} mm\nStrehl: {results['strehl']:.2f}")
        
        # --- FIX: Update Slider Range to match data length ---
        num_steps = len(results['prop_distance'])
        self.slider_z.blockSignals(True) # Prevent updates while changing range
        self.slider_z.setRange(0, num_steps - 1)
        self.slider_z.setValue(num_steps // 2) # Set to center (focus)
        self.slider_z.blockSignals(False)
        
        # Trigger update for the initial slice
        self.update_slice(self.slider_z.value())

        # # 2D Focus (Interactive)
        # spot = results['focus_2d']
        # self.plot_2d.setImage(spot.T, autoRange=False, autoLevels=False)
        
        self.update_3d_view()
        

        prop_dist_range = results['prop_distance'] - results['prop_center_distance']
        # XZ Cut - Use PyQtGraph
        # Assuming results['cut_xz'] is a 2D numpy array
        self.plot_xz.setImage(results['cut_xz'].transpose(), autoRange=True, autoLevels=True)
        # Enable linear interpolation for the underlying ImageItem
        # self.plot_xz.getImageItem().setOpts(axisOrder='row-major') # Ensure axis order matches
        self.plot_xz.imageItem.setPxMode(False)

        X_um = results['focus_Lz'][1] * 1e6
        Y_um = results['focus_Lz'][0] * 1e6
        
        # Set the bounding rect for the image item
        # pyqtgraph uses (x, y, w, h)
        rect = QRectF(-X_um/2, prop_dist_range[0]*1e3, X_um, (prop_dist_range[-1]-prop_dist_range[0])*1e3)

        self.plot_xz.getImageItem().setRect(rect)
        # Force the view to fit the new physical rect
        self.plot_xz.getView().autoRange() 
        
        # YZ Cut - Use PyQtGraph
        self.plot_yz.setImage(results['cut_yz'].transpose(), autoRange=True, autoLevels=True)
        # Enable linear interpolation for the underlying ImageItem
        # self.plot_yz.getImageItem().setOpts(axisOrder='row-major') # Ensure axis order matches
        self.plot_yz.imageItem.setPxMode(False)

        # Set the bounding rect for the image item
        # pyqtgraph uses (x, y, w, h)
        rect = QRectF(-Y_um/2, prop_dist_range[0]*1e3, Y_um, (prop_dist_range[-1]-prop_dist_range[0])*1e3)
        self.plot_yz.getImageItem().setRect(rect)
        # Force the view to fit the new physical rect
        self.plot_yz.getView().autoRange()

    def toggle_mag_params(self, method_name):
        is_wofry = (method_name == "Wofry")
        self.mag_x.setVisible(is_wofry)
        self.mag_y.setVisible(is_wofry)
        self.lbl_mag_x.setVisible(is_wofry)
        self.lbl_mag_y.setVisible(is_wofry)

    def on_image_clicked(self, event):
        if self.view_main.sceneBoundingRect().contains(event.scenePos()):
            mouse_point = self.view_main.mapSceneToView(event.scenePos())
            self.selected_pos = (mouse_point.x(), mouse_point.y())
            self.update_profiles()

    def update_profiles(self):
        if self.img_item.image is None:
            return
            
        # Get current image data (transposed in display: x, y)
        # img_data is (w, h) because we transposed it in update_slice
        img_data = self.img_item.image 
        w, h = img_data.shape
        
        # Get physical bounds
        rect = self.img_item.boundingRect()
        x_min, y_min = rect.left(), rect.top()
        width_um, height_um = rect.width(), rect.height()
        
        # Determine selection
        if self.selected_pos is None:
            # Default to center
            cx_um = x_min + width_um / 2
            cy_um = y_min + height_um / 2
            self.selected_pos = (cx_um, cy_um)
            
        x_um, y_um = self.selected_pos
        
        # Update crosshairs
        self.v_line.setPos(x_um)
        self.h_line.setPos(y_um)
        
        # --- FIX 1: Use mapFromParent for robust coordinate conversion ---
        # Maps View Coordinates (microns) -> Image Coordinates (pixels)
        # This works regardless of how setRect/transforms are applied
        pt = self.img_item.mapFromParent(pg.Point(x_um, y_um))
        x_idx = int(pt.x())
        y_idx = int(pt.y())
        
        # Clamp indices to valid range
        x_idx = max(0, min(w - 1, x_idx))
        y_idx = max(0, min(h - 1, y_idx))

        # Extract profiles
        # img_data is [x, y]
        prof_x = img_data[:, y_idx]
        prof_y = img_data[x_idx, :]
        
        # --- FIX 2: Generate Axes using Physical Dimensions from Results ---
        if self.results:
            index = self.slider_z.value()
            # Ensure index is safe
            if index < len(self.results['prop_Lz']):
                L_z = self.results['prop_Lz'][index]
                width_um = L_z[1] * 1e6
                height_um = L_z[0] * 1e6
                
                # Calculate start positions (centered)
                x_min = -width_um / 2
                y_min = -height_um / 2
                
                x_axis = np.linspace(x_min, x_min + width_um, w)
                y_axis = np.linspace(y_min, y_min + height_um, h)
                
                # Plot
                self.plot_x.plot(x_axis, prof_x, clear=True, pen='y')
                self.plot_y.plot(prof_y, y_axis, clear=True, pen='y')
                
                # Sync levels
                self.sync_profile_levels()

    def stop_analysis(self):
        self.stop_requested = True
        self.btn_stop.setEnabled(False)

    def auto_suggest_sampling(self):
        params = {
            "distance_mm": self.distance.value(),
            "range_mm": self.dist_range.value(),
            "step_mm": self.dist_step.value(),
            "direction": self.direction.currentText(),
            "upsampling": self.upsampling.value(),
            "method": self.method.currentText(),
            "magnification_x": self.mag_x.value(),
            "magnification_y": self.mag_y.value(),
            "padding_scale": self.padding_scale.value(),
            "available_memory_bytes": self._get_available_memory_bytes(),
            **self._advanced_sampling_params()
        }
        suggested, msg = self._suggest_sampling_params(params)
        self.upsampling.setValue(float(suggested["upsampling"]))
        self.padding_scale.setValue(float(suggested["padding_scale"]))
        if self.method.currentText() == "Wofry":
            self.mag_x.setValue(float(suggested["magnification_x"]))
            self.mag_y.setValue(float(suggested["magnification_y"]))
        self.lbl_sampling_hint.setText(f"Suggestion: {msg}")

    def _get_available_memory_bytes(self):
        # Best effort without extra dependencies.
        try:
            import psutil
            return int(psutil.virtual_memory().available)
        except Exception:
            pass
        try:
            pages = os.sysconf("SC_AVPHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            return int(pages * page_size)
        except Exception:
            # Fallback: assume 4 GB available
            return int(4 * 1024**3)

    def _count_prop_steps(self, dist_range_m, dist_step_m):
        if dist_step_m <= 0:
            return 1
        return int(np.floor(dist_range_m / dist_step_m + 1e-9)) + 1

    def _estimate_memory_bytes(self, params):
        h, w = self.phase_map.shape
        up = max(float(params.get("upsampling", 1.0)), 0.1)
        pad = max(float(params.get("padding_scale", 1.0)), 1.0)
        steps = self._count_prop_steps(float(params.get("range_mm", 10.0)) * 1e-3,
                                       float(params.get("step_mm", 1.0)) * 1e-3)
        hh = max(1, int(h * up * pad))
        ww = max(1, int(w * up * pad))

        # Approx memory: output stack (float64) + one complex field + temp overhead.
        output_bytes = steps * hh * ww * 8
        field_bytes = hh * ww * 16
        overhead = int(2.5 * field_bytes)
        return output_bytes + field_bytes + overhead

    def _suggest_sampling_params(self, params):
        method = str(params.get("method", "default"))
        up = float(params.get("upsampling", 1.0))
        pad = float(params.get("padding_scale", 1.0))
        mag_x = float(params.get("magnification_x", 1.0))
        mag_y = float(params.get("magnification_y", 1.0))
        avail = int(params.get("available_memory_bytes", self._get_available_memory_bytes()))
        budget = int(avail * float(params.get("sampling_suggest_budget_ratio", 0.30)))

        # Base anti-alias suggestion.
        up_suggest = max(up, 2.0)
        pad_suggest = max(pad, 2.0)
        if method == "Wofry":
            mag_x_suggest = min(mag_x, 0.5)
            mag_y_suggest = min(mag_y, 0.5)
        else:
            mag_x_suggest = mag_x
            mag_y_suggest = mag_y

        trial = dict(params)
        trial["upsampling"] = up_suggest
        trial["padding_scale"] = pad_suggest
        trial["magnification_x"] = mag_x_suggest
        trial["magnification_y"] = mag_y_suggest
        mem_trial = self._estimate_memory_bytes(trial)

        # Memory-aware clamp.
        if mem_trial > budget:
            h, w = self.phase_map.shape
            steps = self._count_prop_steps(float(params.get("range_mm", 10.0)) * 1e-3,
                                           float(params.get("step_mm", 1.0)) * 1e-3)
            coeff = max(1.0, 8.0 * steps * h * w * (pad_suggest ** 2))
            up_max = (budget / coeff) ** 0.5
            up_suggest = float(np.clip(up_max, 0.5, 3.0))
            if up_suggest < 1.0:
                pad_suggest = 1.5

        suggested = {
            "upsampling": float(np.clip(up_suggest, 0.01, 50.0)),
            "padding_scale": float(np.clip(pad_suggest, 1.0, 50.0)),
            "magnification_x": float(np.clip(mag_x_suggest, 0.001, 1000.0)),
            "magnification_y": float(np.clip(mag_y_suggest, 0.001, 1000.0)),
        }
        mem_gb = self._estimate_memory_bytes({**params, **suggested}) / 1024**3
        msg = (
            f"Use upsampling={suggested['upsampling']:.2f}, padding={suggested['padding_scale']:.2f}"
            f"{', mag=0.50' if method == 'Wofry' else ''} (est. RAM ~{mem_gb:.2f} GB)"
        )
        return suggested, msg

    def _guard_sampling_memory(self, params):
        est = self._estimate_memory_bytes(params)
        avail = int(params.get("available_memory_bytes", self._get_available_memory_bytes()))
        budget = int(avail * float(params.get("sampling_run_budget_ratio", 0.35)))
        if est <= budget:
            return params, ""

        suggested, msg = self._suggest_sampling_params(params)
        guarded = dict(params)
        guarded.update(suggested)

        # Reflect applied clamp in UI so user sees actual run params.
        self.upsampling.setValue(float(guarded["upsampling"]))
        self.padding_scale.setValue(float(guarded["padding_scale"]))
        if self.method.currentText() == "Wofry":
            self.mag_x.setValue(float(guarded["magnification_x"]))
            self.mag_y.setValue(float(guarded["magnification_y"]))
        return guarded, f"settings auto-clamped for RAM safety. {msg}"

    def toggle_advanced_sampling(self, checked):
        self.advanced_sampling_group.setVisible(bool(checked))

    def _advanced_sampling_params(self):
        if not self.chk_advanced_sampling.isChecked():
            return {}
        return {
            "sampling_suggest_budget_ratio": self.suggest_budget_ratio.value(),
            "sampling_run_budget_ratio": self.run_budget_ratio.value(),
            "backend_stack_ratio": self.backend_stack_ratio.value(),
            "worker_memory_ratio": self.worker_mem_ratio.value(),
        }

    def sync_profile_levels(self):
        """Synchronize profile plot axes with image contrast levels."""
        levels = self.img_item.getLevels()
        if levels is None:
            return
        min_l, max_l = levels
        # plot_x shows Intensity on Y-axis
        self.plot_x.setYRange(min_l, max_l, padding=0)
        # plot_y shows Intensity on X-axis
        self.plot_y.setXRange(min_l, max_l, padding=0)

    def update_3d_view(self):
        if self.results is None or not self.gl_supported or self.view_3d is None:
            return
            
        gl = self._gl
        import numpy as np
        
        self.view_3d.items = [] # Clear previous items
        
        # Add grid back
        g = gl.GLGridItem()
        g.scale(10, 10, 1)
        g.setColor((0, 0, 0, 255)) # Black grid for white background
        self.view_3d.addItem(g)
        
        # Prepare 3D data: [x, y, z]
        # intensity_profiles is [z, y, x]
        data_3d = self.results['intensity_profiles'] # [z, y, x]
        
        # Transpose to [x, y, z] for GLVolumeItem
        data_3d = np.transpose(data_3d, (2, 1, 0)) 
        
        # --- OPTIMIZATION: Downsample if too large ---
        # Limit max dimension to ~100 voxels for smooth interaction
        max_dim = 100
        step_x = max(1, data_3d.shape[0] // max_dim)
        step_y = max(1, data_3d.shape[1] // max_dim)
        step_z = max(1, data_3d.shape[2] // max_dim)
        
        if step_x > 1 or step_y > 1 or step_z > 1:
            data_3d = data_3d[::step_x, ::step_y, ::step_z]
        
        # Normalize data for display (0 to 1)
        d_min, d_max = data_3d.min(), data_3d.max()
        if d_max == d_min: d_max = d_min + 1
        data_norm = (data_3d - d_min) / (d_max - d_min)
        
        # Get controls
        threshold = self.slider_threshold.value() / 100.0
        opacity = self.slider_opacity.value() / 100.0
        color_mode = self.combo_color.currentText()
        target_xy = self.slider_scale_xy.value()
        target_z = self.slider_scale_z.value()
        
        # Update labels
        self.lbl_threshold.setText(f"{int(threshold*100)}%")
        self.lbl_opacity.setText(f"{int(opacity*100)}%")
        self.lbl_scale_xy.setText(f"{target_xy}")
        self.lbl_scale_z.setText(f"{target_z}")
        
        # --- FIX: Convert to uint8 (0-255) for robust rendering ---
        # This ensures compatibility with OpenGL textures on all drivers
        data_u8 = (data_norm * 255).astype(np.uint8)
        
        # Create RGBA array [x, y, z, 4]
        colors = np.zeros(data_u8.shape + (4,), dtype=np.uint8)
        
        # Color Map
        if color_mode == "Green":
            colors[..., 0] = data_u8          # R
            colors[..., 1] = data_u8          # G
            colors[..., 2] = 0                # B
        elif color_mode == "Red":
            colors[..., 0] = data_u8
            colors[..., 1] = 0
            colors[..., 2] = 0
        elif color_mode == "Blue":
            colors[..., 0] = 0
            colors[..., 1] = 0
            colors[..., 2] = data_u8
        elif color_mode == "Gray":
            colors[..., 0] = data_u8
            colors[..., 1] = data_u8
            colors[..., 2] = data_u8
        elif color_mode == "Fire":
            # Simple Fire map: Red -> Yellow
            colors[..., 0] = data_u8 # Full Red
            colors[..., 1] = (data_u8 * 0.8).astype(np.uint8) # Some Green
            colors[..., 2] = 0
        
        # Alpha Calculation
        # Apply opacity scaling
        alpha = (data_u8 * opacity).astype(np.uint8)
        
        # Apply Thresholding
        # Set alpha to 0 for values below threshold to hide noise
        thresh_val = int(threshold * 255)
        alpha[data_u8 < thresh_val] = 0
        
        colors[..., 3] = alpha
        
        # --- FIX: Ensure contiguous memory layout ---
        colors = np.ascontiguousarray(colors)
        
        # --- Visualization Scaling ---
        nx, ny, nz = data_3d.shape
        
        sx = target_xy / nx
        sy = target_xy / ny
        sz = target_z / nz
        
        vol = gl.GLVolumeItem(colors)
        vol.scale(sx, sy, sz)
        # Center the volume
        vol.translate(-target_xy/2, -target_xy/2, -target_z/2)
        self.view_3d.addItem(vol)
        
        # Add axes
        ax = gl.GLAxisItem()
        ax.setSize(target_xy/2, target_xy/2, target_z/2)
        self.view_3d.addItem(ax)
        
        # Set Camera
        # self.view_3d.setCameraPosition(distance=target_z * 1.5, elevation=30, azimuth=45)
