from PyQt6.QtWidgets import QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QFileDialog, QMenuBar, QInputDialog, QLabel
from PyQt6.QtGui import QAction
from PyQt6.QtCore import QTimer
import cv2
import numpy as np
import os
import json
from PIL import Image
import warnings
from pathlib import Path
from datetime import datetime

try:
    import tifffile
except ImportError:
    tifffile = None

from .widgets.camera_view import CameraView
from .widgets.settings_panel import SettingsPanel
from .widgets.analysis_window import AnalysisResultWindow
from .widgets.period_dialog import PeriodCalcDialog, TalbotPeriodDialog
from .widgets.mask_preview import MaskPreviewDialog

from ..hardware.dummy_camera import DummyCamera
from ..core.talbot import TalbotProcessor
from ..core.hartmann import HartmannProcessor
from ..core.calculations import period_calc, calculate_envelope

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("WaveTool - Wavefront Sensing")
        self.resize(1400, 900)

        # Hardware & Logic
        self.camera = DummyCamera()
        self.processors = {
            "Talbot Interferometry": TalbotProcessor(),
            "Hartmann Sensor": HartmannProcessor()
        }
        self.current_processor = self.processors["Talbot Interferometry"]
        self.live_mode = False
        self.active_image_target = "sample"
        self.monitor_mode_active = False
        self.monitor_folder = ""
        self.monitor_seen_files = set()
        
        # State for tools
        self.saved_image_before_period = None
        self.envelope_period = 20.0 # Default period

        # UI Setup
        self.setup_menus()
        
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)

        # Left: Sample View
        self.sample_panel = QWidget()
        self.sample_layout = QVBoxLayout(self.sample_panel)
        self.sample_layout.setContentsMargins(0, 0, 0, 0)
        self.sample_label = QLabel("Sample")
        self.sample_layout.addWidget(self.sample_label)
        self.camera_view = CameraView()
        self.sample_layout.addWidget(self.camera_view)
        self.main_layout.addWidget(self.sample_panel, stretch=2)
        
        # Reference View (Hidden by default)
        self.ref_panel = QWidget()
        self.ref_layout = QVBoxLayout(self.ref_panel)
        self.ref_layout.setContentsMargins(0, 0, 0, 0)
        self.ref_label = QLabel("Ref")
        self.ref_layout.addWidget(self.ref_label)
        self.ref_view = CameraView()
        self.ref_layout.addWidget(self.ref_view)
        self.ref_panel.hide()
        self.main_layout.addWidget(self.ref_panel, stretch=2)

        # Right: Controls & Results
        self.right_panel = QWidget()
        self.right_layout = QVBoxLayout(self.right_panel)
        self.main_layout.addWidget(self.right_panel, stretch=1)

        self.settings_panel = SettingsPanel()
        self.right_layout.addWidget(self.settings_panel)

        self.btn_snap = QPushButton("Snap & Analyze")
        self.btn_snap.clicked.connect(self.snap_and_process)
        self.right_layout.addWidget(self.btn_snap)
        
        self.btn_live = QPushButton("Toggle Live View")
        self.btn_live.clicked.connect(self.toggle_live)
        self.right_layout.addWidget(self.btn_live)

        # self.results_display = ResultsDisplay()
        # self.right_layout.addWidget(self.results_display)
        self.right_layout.addStretch()

        # Connect signals
        self.settings_panel.mode_selector.currentTextChanged.connect(self.change_mode)
        self.settings_panel.analysis_mode.currentTextChanged.connect(self.toggle_ref_view)
        self.settings_panel.analysis_mode.currentTextChanged.connect(self.on_analysis_mode_changed)
        self.settings_panel.run_mode.currentTextChanged.connect(self.on_run_mode_changed)
        self.settings_panel.ref_path.textChanged.connect(self.load_reference_preview)
        self.settings_panel.btn_apply_crop.clicked.connect(self.apply_crop_manually)
        self.camera_view.clicked.connect(self.on_sample_view_clicked)
        self.ref_view.clicked.connect(self.on_ref_view_clicked)
        self.on_run_mode_changed(self.settings_panel.run_mode.currentText())

        # Initialize Camera
        self.camera.connect()
        
        # Live View Timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_live_view)
        self.timer.start(100) # 10 FPS

        # Folder monitor timer
        self.monitor_timer = QTimer()
        self.monitor_timer.timeout.connect(self.check_monitor_folder)
        self.monitor_timer.setInterval(2000)

    def setup_menus(self):
        menu_bar = self.menuBar()
        
        # File Menu
        file_menu = menu_bar.addMenu("File")
        
        open_action = QAction("Open Image...", self)
        open_action.triggered.connect(self.load_image)
        file_menu.addAction(open_action)
        
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Tools Menu
        tools_menu = menu_bar.addMenu("Tools")
        
        self.period_action = QAction("Image Period Calc", self)
        self.period_action.triggered.connect(self.open_period_calc)
        tools_menu.addAction(self.period_action)

        self.talbot_action = QAction("Period Calc", self)
        self.talbot_action.triggered.connect(self.open_talbot_calc)
        tools_menu.addAction(self.talbot_action)
        
        self.envelope_action = QAction("Beam Envelope", self)
        self.envelope_action.setCheckable(True)
        self.envelope_action.triggered.connect(self.toggle_envelope)
        tools_menu.addAction(self.envelope_action)

    def load_image(self):
        self._open_image_for_target(self.active_image_target)

    def _open_image_for_target(self, target="sample"):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.bmp *.tif *.tiff)")
        if file_name:
            # Stop live view to show loaded image
            self.live_mode = False
            self.timer.stop()
            
            img = self._read_grayscale_image(file_name)
            
            if img is not None:
                if target == "ref":
                    self.ref_view.update_image(img, force_autorange=True)
                    self.settings_panel.ref_path.setText(file_name)
                else:
                    self.camera_view.update_image(img, force_autorange=True)
                self._set_default_layout_for_image(img)
                print(f"Loaded {target} image: {file_name}, Shape: {img.shape}, Dtype: {img.dtype}")
            else:
                print("Failed to load image")

    def toggle_live(self):
        self.live_mode = not self.live_mode
        if self.live_mode:
            self.timer.start(100)
        else:
            self.timer.stop()

    def change_mode(self, mode_name):
        if mode_name in self.processors:
            self.current_processor = self.processors[mode_name]
            print(f"Switched to {mode_name}")

    def update_live_view(self):
        if not self.live_mode:
            return
        try:
            img = self.camera.snap()
            crop = self.get_crop_rect()
            self.camera_view.update_image(img, crop_rect=crop)
            
            # If envelope is enabled, update it in real-time too
            if self.envelope_action.isChecked():
                # Note: Envelope calculation should probably happen on the cropped image if displayed
                # For now, we use the full image or displayed image logic in camera_view
                # But camera_view.get_image() returns the raw full image currently.
                # Let's stick to full image for envelope for now or update logic later.
                envelope = calculate_envelope(img, self.envelope_period)
                self.camera_view.show_envelope(envelope)
                
        except Exception as e:
            print(f"Camera Error: {e}")

    def snap_and_process(self):
        try:
            settings = self.settings_panel.get_settings()
            run_mode = settings.get("run_mode", "Single")
            if run_mode == "Monitor":
                self.toggle_monitor_mode(settings)
                return

            # If live, grab fresh frame. If not (loaded image), use current display
            if self.live_mode:
                img = self.camera.snap()
                crop = self.get_crop_rect()
                self.camera_view.update_image(img, crop_rect=crop)
                self.live_mode = False # Pause to show results
                self.timer.stop()
            else:
                img = self.camera_view.get_image()
            
            if img is None:
                print("No image to process")
                return

            # Hartmann Mask Preview Step
            if isinstance(self.current_processor, HartmannProcessor):
                pitch_um = settings.get("period_um", 150.0)
                pixel_size_um = settings.get("pixel_size_um", 5.0)
                pitch_px = pitch_um / pixel_size_um
                initial_thresh = settings.get("mask_threshold", 0.2)
                
                preview = MaskPreviewDialog(img, pitch_px, initial_thresh, self)
                if preview.exec():
                    # Update threshold with user selected value
                    settings["mask_threshold"] = preview.threshold
                else:
                    # User cancelled
                    return
            
            # Pass settings to processor
            results = self.current_processor.process(img, params=settings)
            
            # Open Analysis Window
            self.analysis_window = AnalysisResultWindow(self.current_processor, results)
            self.analysis_window.show()

            # Save results if a folder path is provided
            if settings.get("save_path", "").strip():
                self.save_analysis_results(
                    results=results,
                    settings=settings,
                    output_folder=settings["save_path"],
                    image_stem="manual",
                    is_monitor=False,
                    source_image_path=None,
                )
            
        except Exception as e:
            print(f"Processing Error: {e}")
            import traceback
            traceback.print_exc()

    def closeEvent(self, event):
        if self.monitor_mode_active:
            self.stop_monitor_mode()
        self.camera.disconnect()
        event.accept()

    def open_period_calc(self):
        # Stop live view if running
        if self.live_mode:
            self.toggle_live()
            
        # Get current image
        img = self.camera_view.get_image()
        if img is None:
            print("No image for period calculation")
            return
            
        self.saved_image_before_period = img.copy()
        
        # Open Dialog
        self.period_dialog = PeriodCalcDialog(self)
        self.period_dialog.calculate_requested.connect(self.run_period_calc)
        self.period_dialog.closed.connect(self.restore_image_after_period)
        self.period_dialog.show()

    def run_period_calc(self, mask_radius):
        if self.saved_image_before_period is None:
            return
            
        spectrum, period = period_calc(self.saved_image_before_period, mask_radius)
        
        # Display spectrum
        self.camera_view.update_image(spectrum)
        
        # Update dialog result
        self.period_dialog.update_result(period)

    def restore_image_after_period(self):
        if self.saved_image_before_period is not None:
            self.camera_view.update_image(self.saved_image_before_period)
            self.saved_image_before_period = None

    def toggle_envelope(self, checked):
        if checked:
            # Ask for period
            period, ok = QInputDialog.getDouble(self, "Beam Envelope", "Enter Grating Period (px):", 
                                              value=self.envelope_period, decimals=2)
            if ok:
                self.envelope_period = period
                img = self.camera_view.get_image()
                if img is not None:
                    envelope = calculate_envelope(img, self.envelope_period)
                    self.camera_view.show_envelope(envelope)
            else:
                # If user cancels, uncheck the menu item
                self.envelope_action.setChecked(False)
        else:
            self.camera_view.hide_envelope()

    def toggle_ref_view(self, mode):
        if mode == "Relative":
            self.ref_panel.show()
            ref_img = self.ref_view.get_image()
            if ref_img is not None:
                self.ref_view.update_image(ref_img, force_autorange=True)
        else:
            self.ref_panel.hide()
            self.active_image_target = "sample"

    def on_analysis_mode_changed(self, mode):
        if mode != "Relative":
            self.active_image_target = "sample"

    def on_run_mode_changed(self, mode):
        if mode == "Monitor":
            self.btn_snap.setText("Start Monitor")
            if self.live_mode:
                self.toggle_live()
        else:
            if self.monitor_mode_active:
                self.stop_monitor_mode()
            self.btn_snap.setText("Snap & Analyze")

    def load_reference_preview(self, path):
        if not path:
            return
        try:
            img = self._read_grayscale_image(path)
            if img is not None:
                self.ref_view.update_image(img, force_autorange=True)
                self._set_default_layout_for_image(img)
            else:
                print("Failed to load reference image")
        except Exception as e:
            print(f"Error loading ref: {e}")

    def get_crop_rect(self):
        s = self.settings_panel
        return (s.crop_l.value(), s.crop_t.value(), s.crop_r.value(), s.crop_b.value())

    def apply_crop_manually(self):
        """Force update the display with current crop settings, even if not live."""
        img = self.camera_view.get_image()
        if img is not None:
            crop = self.get_crop_rect()
            self.camera_view.update_image(img, crop_rect=crop)

    def open_talbot_calc(self):
        self.talbot_dialog = TalbotPeriodDialog(self)
        self.talbot_dialog.show()

    def _read_grayscale_image(self, path):
        """
        Robust image reader for GUI load/preview.
        TIFFs are loaded without OpenCV first, because some TIFF variants
        trigger OpenCV decoder failures.
        """
        img = None
        suffix = Path(path).suffix.lower()

        if suffix in (".tif", ".tiff"):
            # 1) PIL first (handles most 8/16-bit TIFFs)
            try:
                with warnings.catch_warnings():
                    # Corrupt EXIF is not fatal for image pixel data.
                    warnings.filterwarnings("ignore", category=UserWarning, module="PIL.TiffImagePlugin")
                    with Image.open(path) as pil_img:
                        img = np.array(pil_img)
            except Exception:
                img = None

            # 2) tifffile fallback for special/compressed TIFFs
            if img is None and tifffile is not None:
                try:
                    img = tifffile.imread(path)
                except Exception:
                    img = None
            if img is None and tifffile is not None:
                try:
                    with tifffile.TiffFile(path) as tf:
                        img = tf.asarray(key=0)
                except Exception:
                    img = None
        else:
            # For non-TIFF formats, decode from bytes to avoid path encoding issues.
            try:
                data = np.fromfile(path, dtype=np.uint8)
                if data.size > 0:
                    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
            except Exception:
                img = None

        if img is None:
            return None

        # Handle stacks by selecting the first frame.
        if img.ndim > 3:
            img = img[0]

        if img.ndim == 3:
            if img.shape[2] >= 3:
                # Works for RGB/BGR/RGBA by taking first 3 channels.
                img = np.dot(img[..., :3], [0.114, 0.587, 0.299])
                img = np.asarray(img, dtype=np.float32)
            else:
                img = img[..., 0]

        return img

    def _set_default_layout_for_image(self, img):
        """
        Resize the app and image panes to sensible defaults based on loaded image size.
        """
        if img is None or img.ndim < 2:
            return
        h, w = img.shape[:2]
        scale = min(1.0, 720.0 / max(h, w))
        img_w = max(320, min(int(w * scale), 760))
        img_h = max(240, min(int(h * scale), 620))

        self.camera_view.setMinimumSize(img_w, img_h)
        self.ref_view.setMinimumSize(img_w, img_h)

        n_views = 2 if self.ref_panel.isVisible() else 1
        right_panel_w = 420
        total_w = max(980, min(n_views * img_w + right_panel_w + 100, 1920))
        total_h = max(680, min(img_h + 200, 1080))
        self.resize(total_w, total_h)

    def on_sample_view_clicked(self):
        self.active_image_target = "sample"
        if self.settings_panel.analysis_mode.currentText() == "Relative":
            self._open_image_for_target("sample")

    def on_ref_view_clicked(self):
        self.active_image_target = "ref"
        if self.settings_panel.analysis_mode.currentText() == "Relative":
            self._open_image_for_target("ref")

    def toggle_monitor_mode(self, settings):
        if settings.get("run_mode", "Single") != "Monitor":
            return

        if self.monitor_mode_active:
            self.stop_monitor_mode()
            return

        folder = settings.get("save_path", "").strip()
        if not folder:
            print("Monitor mode needs Folder Path.")
            return
        if not os.path.isdir(folder):
            print(f"Monitor folder does not exist: {folder}")
            return

        self.monitor_folder = folder
        self.monitor_seen_files = set(self._list_tiff_files(folder))
        self.monitor_mode_active = True
        self.monitor_timer.start()
        self.btn_snap.setText("Stop Monitor")
        print(f"Monitor started: {folder}")

    def stop_monitor_mode(self):
        self.monitor_mode_active = False
        self.monitor_timer.stop()
        self.btn_snap.setText("Start Monitor")
        print("Monitor stopped.")

    def _list_tiff_files(self, folder):
        p = Path(folder)
        files = list(p.glob("*.tif")) + list(p.glob("*.tiff")) + list(p.glob("*.TIF")) + list(p.glob("*.TIFF"))
        return sorted([str(f) for f in files], key=lambda x: os.path.getmtime(x))

    def check_monitor_folder(self):
        if not self.monitor_mode_active:
            return
        if not self.monitor_folder or not os.path.isdir(self.monitor_folder):
            print("Monitor folder unavailable; stopping monitor.")
            self.stop_monitor_mode()
            return

        current_files = self._list_tiff_files(self.monitor_folder)
        new_files = [f for f in current_files if f not in self.monitor_seen_files]
        for file_path in new_files:
            self.monitor_seen_files.add(file_path)
            self.process_monitor_file(file_path)

    def process_monitor_file(self, file_path):
        try:
            img = self._read_grayscale_image(file_path)
            if img is None:
                print(f"Monitor skip (cannot load): {file_path}")
                return

            self.camera_view.update_image(img, force_autorange=True)
            self._set_default_layout_for_image(img)

            settings = self.settings_panel.get_settings()
            results = self.current_processor.process(img, params=settings)

            self.save_analysis_results(
                results=results,
                settings=settings,
                output_folder=self.monitor_folder,
                image_stem=Path(file_path).stem,
                is_monitor=True,
                source_image_path=file_path,
            )
            print(f"Monitor processed: {file_path}")
        except Exception as e:
            print(f"Monitor processing error ({file_path}): {e}")

    def save_analysis_results(self, results, settings, output_folder, image_stem, is_monitor=False, source_image_path=None):
        if not output_folder:
            return
        os.makedirs(output_folder, exist_ok=True)

        tag = "monitor" if is_monitor else "manual"
        prefix = f"{image_stem}_{tag}"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = f"{prefix}_{timestamp}"

        saved_files = {}
        for key, suffix in [
            ("phase_map", "phase"),
            ("phase_residual_2nd", "phase_residual_2nd"),
            ("transmission", "transmission"),
            ("displacement_x", "disp_x"),
            ("displacement_y", "disp_y"),
        ]:
            arr = results.get(key)
            out = self._save_array(output_folder, f"{base}_{suffix}", arr)
            if out:
                saved_files[key] = out

        run_data = {
            "run_name": base,
            "image_stem": image_stem,
            "is_monitor": bool(is_monitor),
            "source_image_path": source_image_path,
            "timestamp": timestamp,
            "processor": getattr(self.current_processor, "name", str(type(self.current_processor))),
            "settings": self._jsonify(settings),
            "results_scalar": self._extract_scalar_results(results),
            "saved_files": saved_files,
        }
        json_path = os.path.join(output_folder, f"{base}_params.json")
        with open(json_path, "w", encoding="utf-8") as fp:
            json.dump(run_data, fp, indent=2, ensure_ascii=False)
        print(f"Saved result json: {json_path}")

    def _save_array(self, folder, base_name, arr):
        if arr is None:
            return None
        arr = np.asarray(arr)
        if arr.size == 0:
            return None

        if arr.ndim == 2:
            path = os.path.join(folder, f"{base_name}.tiff")
            if tifffile is not None:
                tifffile.imwrite(path, arr.astype(np.float32))
            else:
                Image.fromarray(arr.astype(np.float32)).save(path)
            return path

        path = os.path.join(folder, f"{base_name}.npy")
        np.save(path, arr)
        return path

    def _extract_scalar_results(self, results):
        scalar = {}
        for k, v in results.items():
            if isinstance(v, (int, float, str, bool)):
                scalar[k] = v
            elif isinstance(v, (list, tuple)) and all(isinstance(x, (int, float, str, bool)) for x in v):
                scalar[k] = list(v)
            elif isinstance(v, np.ndarray) and v.size == 1:
                scalar[k] = float(v.ravel()[0])
        return scalar

    def _jsonify(self, obj):
        if isinstance(obj, dict):
            return {str(k): self._jsonify(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._jsonify(v) for v in obj]
        if isinstance(obj, np.ndarray):
            if obj.size <= 64:
                return obj.tolist()
            return {"type": "ndarray", "shape": list(obj.shape), "dtype": str(obj.dtype)}
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return obj
