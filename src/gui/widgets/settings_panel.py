from PyQt6.QtWidgets import (QWidget, QFormLayout, QDoubleSpinBox, QComboBox, 
                             QLabel, QCheckBox, QLineEdit, QPushButton, QHBoxLayout, QFileDialog, QGroupBox)

class SettingsPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QFormLayout(self)

        # All numeric inputs should support high precision.
        precision_decimals = 4
        precision_step = 0.0001
        
        # --- Mode Selection ---
        self.mode_selector = QComboBox()
        self.mode_selector.addItems(["Talbot Interferometry", "Hartmann Sensor"])
        self.mode_selector.currentTextChanged.connect(self.update_visibility)
        self.layout.addRow("Measurement Mode:", self.mode_selector)

        # --- Common Parameters ---
        self.energy = QDoubleSpinBox()
        self.energy.setRange(0.1, 100.0)
        self.energy.setDecimals(precision_decimals)
        self.energy.setSingleStep(precision_step)
        self.energy.setValue(10.0)
        self.energy.setSuffix(" keV")
        self.layout.addRow("X-ray Energy:", self.energy)

        self.pixel_size = QDoubleSpinBox()
        self.pixel_size.setRange(0.1, 100.0)
        self.pixel_size.setDecimals(precision_decimals)
        self.pixel_size.setSingleStep(precision_step)
        self.pixel_size.setValue(0.65)
        self.pixel_size.setSuffix(" um")
        self.layout.addRow("Pixel Size:", self.pixel_size)

        self.distance = QDoubleSpinBox()
        self.distance.setRange(1.0, 10000.0)
        self.distance.setDecimals(precision_decimals)
        self.distance.setSingleStep(precision_step)
        self.distance.setValue(100.0)
        self.distance.setSuffix(" mm")
        self.layout.addRow("Distance (G/P to Det):", self.distance)

        self.period = QDoubleSpinBox()
        self.period.setRange(0.1, 1000.0)
        self.period.setDecimals(precision_decimals)
        self.period.setSingleStep(precision_step)
        self.period.setValue(4.8)
        self.period.setSuffix(" um")
        self.layout.addRow("Ideal GT Period:", self.period)

        self.analysis_mode = QComboBox()
        self.analysis_mode.addItems(["Absolute", "Relative"])
        self.layout.addRow("Analysis Mode:", self.analysis_mode)

        self.run_mode = QComboBox()
        self.run_mode.addItems(["Single", "Monitor"])
        self.layout.addRow("Run Mode:", self.run_mode)

        self.correct_angle = QCheckBox("Correct Angle")
        self.layout.addRow(self.correct_angle)

        self.use_mask = QCheckBox("Use Mask")
        self.layout.addRow(self.use_mask)

        # --- Crop Settings ---
        crop_group = QGroupBox("Crop (L, T, R, B)")
        crop_layout = QHBoxLayout()
        
        self.btn_apply_crop = QPushButton("Apply")
        self.btn_apply_crop.setFixedWidth(50)
        crop_layout.addWidget(self.btn_apply_crop)
        
        self.crop_l = QDoubleSpinBox()
        self.crop_t = QDoubleSpinBox()
        self.crop_r = QDoubleSpinBox()
        self.crop_b = QDoubleSpinBox()
        for w in [self.crop_l, self.crop_t, self.crop_r, self.crop_b]:
            w.setRange(0, 10000)
            w.setDecimals(precision_decimals)
            w.setSingleStep(precision_step)
            w.setValue(0)
            crop_layout.addWidget(w)
        crop_group.setLayout(crop_layout)
        self.layout.addRow(crop_group)

        # --- Talbot Specific ---
        self.wavefront_type = QComboBox()
        self.wavefront_type.addItems(["Effective WF", "Real WF"])
        self.lbl_wf_type = QLabel("Wavefront:")
        self.layout.addRow(self.lbl_wf_type, self.wavefront_type)

        self.source_dist_v = QDoubleSpinBox()
        self.source_dist_v.setRange(-1e9, 1e9)
        self.source_dist_v.setDecimals(precision_decimals)
        self.source_dist_v.setSingleStep(precision_step)
        self.source_dist_v.setValue(0.0)
        self.source_dist_v.setSuffix(" m")
        self.lbl_source_v = QLabel("Source Distance V:")
        self.layout.addRow(self.lbl_source_v, self.source_dist_v)

        self.source_dist_h = QDoubleSpinBox()
        self.source_dist_h.setRange(-1e9, 1e9)
        self.source_dist_h.setDecimals(precision_decimals)
        self.source_dist_h.setSingleStep(precision_step)
        self.source_dist_h.setValue(0.0)
        self.source_dist_h.setSuffix(" m")
        self.lbl_source_h = QLabel("Source Distance H:")
        self.layout.addRow(self.lbl_source_h, self.source_dist_h)

        # --- Hartmann Specific ---
        self.plate_angle = QDoubleSpinBox()
        self.plate_angle.setRange(-180, 180)
        self.plate_angle.setDecimals(precision_decimals)
        self.plate_angle.setSingleStep(precision_step)
        self.plate_angle.setSuffix(" deg")
        self.lbl_p_angle = QLabel("Plate Angle:")
        self.layout.addRow(self.lbl_p_angle, self.plate_angle)
        
        self.mask_threshold = QDoubleSpinBox()
        self.mask_threshold.setRange(0.0, 1.0)
        self.mask_threshold.setDecimals(precision_decimals)
        self.mask_threshold.setSingleStep(precision_step)
        self.mask_threshold.setValue(0.2)
        self.lbl_mask_thresh = QLabel("Mask Threshold:")
        self.layout.addRow(self.lbl_mask_thresh, self.mask_threshold)

        # --- Save Path ---
        path_layout = QHBoxLayout()
        self.save_path = QLineEdit()
        self.btn_browse = QPushButton("...")
        self.btn_browse.clicked.connect(self.browse_folder)
        path_layout.addWidget(self.save_path)
        path_layout.addWidget(self.btn_browse)
        self.layout.addRow("Folder Path:", path_layout)

        # --- Reference Image (Relative Mode) ---
        self.ref_group = QGroupBox("Reference Image")
        ref_layout = QHBoxLayout()
        self.ref_path = QLineEdit()
        self.btn_ref_browse = QPushButton("...")
        self.btn_ref_browse.clicked.connect(self.browse_ref)
        ref_layout.addWidget(self.ref_path)
        ref_layout.addWidget(self.btn_ref_browse)
        self.ref_group.setLayout(ref_layout)
        self.layout.addRow(self.ref_group)

        self.analysis_mode.currentTextChanged.connect(self.update_visibility)
        self.update_visibility(self.mode_selector.currentText())

    def browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Save Folder")
        if folder:
            self.save_path.setText(folder)

    def browse_ref(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Reference Image", "", "Images (*.png *.jpg *.bmp *.tif *.tiff)")
        if file_name:
            self.ref_path.setText(file_name)

    def update_visibility(self, mode_or_text):
        # Handle both mode change and analysis mode change
        mode = self.mode_selector.currentText()
        is_talbot = (mode == "Talbot Interferometry")
        
        self.wavefront_type.setVisible(is_talbot)
        self.lbl_wf_type.setVisible(is_talbot)
        self.source_dist_v.setVisible(is_talbot)
        self.lbl_source_v.setVisible(is_talbot)
        self.source_dist_h.setVisible(is_talbot)
        self.lbl_source_h.setVisible(is_talbot)
        
        self.plate_angle.setVisible(not is_talbot)
        self.lbl_p_angle.setVisible(not is_talbot)
        self.mask_threshold.setVisible(not is_talbot)
        self.lbl_mask_thresh.setVisible(not is_talbot)
        
        # Reference visibility
        is_relative = (self.analysis_mode.currentText() == "Relative")
        self.ref_group.setVisible(is_relative)

    def get_settings(self):
        return {
            "mode": self.mode_selector.currentText(),
            "energy_kev": self.energy.value(),
            "pixel_size_um": self.pixel_size.value(),
            "distance_mm": self.distance.value(),
            "period_um": self.period.value(),
            "analysis_mode": self.analysis_mode.currentText(),
            "run_mode": self.run_mode.currentText(),
            "correct_angle": self.correct_angle.isChecked(),
            "use_mask": self.use_mask.isChecked(),
            "crop": (self.crop_l.value(), self.crop_t.value(), self.crop_r.value(), self.crop_b.value()),
            "real_wf": self.wavefront_type.currentText() == "Real WF",
            "source_distance_v_m": self.source_dist_v.value(),
            "source_distance_h_m": self.source_dist_h.value(),
            "plate_angle": self.plate_angle.value(),
            "mask_threshold": self.mask_threshold.value(),
            "save_path": self.save_path.text(),
            "ref_image_path": self.ref_path.text()
        }
