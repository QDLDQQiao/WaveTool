from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSpinBox,
                             QPushButton, QGridLayout, QScrollArea, QTabWidget, QFileDialog,
                             QMessageBox)
try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT
except ImportError:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT
from matplotlib.figure import Figure
import numpy as np
import os
import json
from datetime import datetime

from ...core.calculations import fit_zernike
from .focus_window import FocusAnalysisWindow

class AnalysisResultWindow(QWidget):
    def __init__(self, processor, results, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Analysis Results")
        self.resize(1680, 1260)
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
        
        if 'period_real' in results:
            p_real = results.get('period_real', [0.0, 0.0])
            stats_text += f" | P_real: {p_real[0]:.3f}x{p_real[1]:.3f} um"

        if 'period_effective' in results:
            p_eff = results.get('period_effective', [0.0, 0.0])
            stats_text += f" | P_eff: {p_eff[0]:.3f}x{p_eff[1]:.3f} um"

        if 'period_method' in results:
            pm = results.get('period_method', ["", ""])
            stats_text += f" | EffMethod(V/H): {pm[0]}/{pm[1]}"

        if 'source_distance_m' in results:
            sd = results.get('source_distance_m', [0.0, 0.0])
            stats_text += f" | SrcD(V/H): {sd[0]:.4f}/{sd[1]:.4f} m"
        elif 'source_distance_mm' in results:
            sd = results.get('source_distance_mm', [0.0, 0.0])
            stats_text += f" | SrcD(V/H): {sd[0]:.3f}/{sd[1]:.3f} mm"

        if 'source_wavefront_added' in results:
            stats_text += f" | RealWF: {'ON' if results.get('source_wavefront_added') else 'OFF'}"

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

        phase_for_shape = results.get("phase_residual_2nd", results.get("phase_map"))
        nx = int(phase_for_shape.shape[1]) if isinstance(phase_for_shape, np.ndarray) and phase_for_shape.ndim == 2 else 1
        ny = int(phase_for_shape.shape[0]) if isinstance(phase_for_shape, np.ndarray) and phase_for_shape.ndim == 2 else 1

        top_layout.addWidget(QLabel("LineCut X(px):"))
        self.spin_line_x = QSpinBox()
        self.spin_line_x.setRange(0, max(0, nx - 1))
        self.spin_line_x.setValue(nx // 2)
        top_layout.addWidget(self.spin_line_x)

        top_layout.addWidget(QLabel("Y(px):"))
        self.spin_line_y = QSpinBox()
        self.spin_line_y.setRange(0, max(0, ny - 1))
        self.spin_line_y.setValue(ny // 2)
        top_layout.addWidget(self.spin_line_y)

        self.btn_update_linecut = QPushButton("Update Line Cut")
        self.btn_update_linecut.clicked.connect(self.update_line_cut_plot)
        top_layout.addWidget(self.btn_update_linecut)
        
        self.layout.addLayout(top_layout)

        # --- Center: Tabbed Figures ---
        self.tabs = QTabWidget()
        self.layout.addWidget(self.tabs)

        # Tab 1: Main Maps
        self.tab_main = QWidget()
        self.tab_main_layout = QVBoxLayout(self.tab_main)
        self.tab_main_scroll = QScrollArea()
        self.tab_main_layout.addWidget(self.tab_main_scroll)
        self.main_content = QWidget()
        self.tab_main_scroll.setWidget(self.main_content)
        self.tab_main_scroll.setWidgetResizable(True)
        self.grid_main = QGridLayout(self.main_content)
        self.grid_main.setHorizontalSpacing(4)
        self.grid_main.setVerticalSpacing(4)

        self.add_plot(self.grid_main, 0, 0, "Displacement X", results.get('displacement_x'))
        self.add_plot(self.grid_main, 0, 1, "Displacement Y", results.get('displacement_y'))
        self.add_plot(self.grid_main, 0, 2, "Transmission", results.get('transmission'), cmap='gray')
        self.add_plot(self.grid_main, 1, 0, "Integrated Phase", results.get('phase_map'), cmap='jet')
        self.add_plot(self.grid_main, 1, 1, "Residual (2nd Order Removed)", results.get('phase_residual_2nd'), cmap='jet')
        self.add_plot(self.grid_main, 1, 2, "Analysis Mask", results.get('mask'), cmap='gray')
        self.add_line_cut_plot(self.grid_main, 2, 0, "Residual Phase Line Cut (X/Y)", results.get('phase_residual_2nd', results.get('phase_map')))
        self.tabs.addTab(self.tab_main, "Main Results")

        # Tab 2: Zernike
        self.tab_zernike = QWidget()
        self.tab_zernike_layout = QVBoxLayout(self.tab_zernike)
        self.tab_zernike_scroll = QScrollArea()
        self.tab_zernike_layout.addWidget(self.tab_zernike_scroll)
        self.zernike_content = QWidget()
        self.tab_zernike_scroll.setWidget(self.zernike_content)
        self.tab_zernike_scroll.setWidgetResizable(True)
        self.grid_zernike = QGridLayout(self.zernike_content)
        self.grid_zernike.setHorizontalSpacing(4)
        self.grid_zernike.setVerticalSpacing(4)

        self.add_bar_plot(self.grid_zernike, 0, 0, "Zernike Coefficients", results.get('zernike_coeffs'))
        self.add_plot(self.grid_zernike, 0, 1, "Zernike Fitted Phase", results.get('zernike_fitted'), cmap='jet')
        self.add_plot(self.grid_zernike, 0, 2, "Zernike Residual", results.get('zernike_residual'), cmap='jet')
        self.tabs.addTab(self.tab_zernike, "Zernike")
        
        # --- Bottom: Action Buttons ---
        btn_row = QHBoxLayout()

        self.btn_focus = QPushButton("Focus Analysis")
        self.btn_focus.setStyleSheet("font-size: 16px; padding: 10px;")
        self.btn_focus.clicked.connect(self.open_focus_window)
        btn_row.addWidget(self.btn_focus)

        self.btn_save_hdf5 = QPushButton("Save HDF5")
        self.btn_save_hdf5.setStyleSheet("font-size: 16px; padding: 10px;")
        self.btn_save_hdf5.clicked.connect(self.save_hdf5_dialog)
        btn_row.addWidget(self.btn_save_hdf5)

        self.layout.addLayout(btn_row)

    def add_plot(self, grid, row, col, title, data, cmap='viridis'):
        if data is None: return
        
        fig = Figure(figsize=(3.0, 2.0))
        canvas = FigureCanvas(fig)
        canvas.setMinimumHeight(170)
        ax = fig.add_subplot(111)
        im = ax.imshow(data, cmap=cmap, aspect='auto')
        fig.colorbar(im, ax=ax)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout(pad=0.2)
        
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
        layout.setSpacing(0)
        layout.addWidget(canvas)
        toolbar = NavigationToolbar2QT(canvas, container)
        layout.addWidget(toolbar)
        
        grid.addWidget(container, row, col)

    def add_bar_plot(self, grid, row, col, title, data):
        if data is None: return
        
        fig = Figure(figsize=(3.0, 2.0))
        canvas = FigureCanvas(fig)
        canvas.setMinimumHeight(170)
        ax = fig.add_subplot(111)
        ax.bar(range(len(data)), data)
        ax.set_title(title)
        fig.tight_layout(pad=0.2)
        
        # Add toolbar for interaction
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(canvas)
        toolbar = NavigationToolbar2QT(canvas, container)
        layout.addWidget(toolbar)
        
        grid.addWidget(container, row, col)

    def add_line_cut_plot(self, grid, row, col, title, phase_data):
        if phase_data is None:
            return
        self.line_fig = Figure(figsize=(9, 2.2))
        self.line_canvas = FigureCanvas(self.line_fig)
        self.line_canvas.setMinimumHeight(180)
        self.line_ax = self.line_fig.add_subplot(111)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.line_canvas)
        toolbar = NavigationToolbar2QT(self.line_canvas, container)
        layout.addWidget(toolbar)
        grid.addWidget(container, row, col, 1, 3)

        self.update_line_cut_plot()

    def update_line_cut_plot(self):
        phase = self.results.get("phase_residual_2nd", self.results.get("phase_map"))
        if phase is None or not hasattr(self, "line_fig"):
            return
        y = int(self.spin_line_y.value())
        x = int(self.spin_line_x.value())
        y = max(0, min(phase.shape[0] - 1, y))
        x = max(0, min(phase.shape[1] - 1, x))

        line_x = phase[y, :]
        line_y = phase[:, x]
        xs = np.arange(phase.shape[1])
        ys = np.arange(phase.shape[0])

        self.line_ax.clear()
        self.line_ax.plot(xs, line_x, lw=1.5, label=f"X-cut @ Y={y}")
        self.line_ax.plot(ys, line_y, lw=1.5, label=f"Y-cut @ X={x}")
        self.line_ax.set_title("Residual Phase Line Cut")
        self.line_ax.set_xlabel("Pixel Index")
        self.line_ax.set_ylabel("Phase (rad)")
        self.line_ax.legend(loc="best", fontsize=9)
        self.line_canvas.draw_idle()

    def open_focus_window(self):
        if 'phase_map' in self.results:
            self.focus_win = FocusAnalysisWindow(self.processor, 
                                               self.results['phase_map'],
                                               self.results.get('transmission'))
            self.focus_win.show()

    def clear_plot(self, grid, row, col):
        item = grid.itemAtPosition(row, col)
        if item:
            widget = item.widget()
            if widget:
                grid.removeWidget(widget)
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
        self.clear_plot(self.grid_zernike, 0, 0)
        self.add_bar_plot(self.grid_zernike, 0, 0, "Zernike Coefficients", coeffs)
        
        self.clear_plot(self.grid_zernike, 0, 1)
        self.add_plot(self.grid_zernike, 0, 1, "Zernike Fitted Phase", fitted, cmap='jet')
        
        self.clear_plot(self.grid_zernike, 0, 2)
        self.add_plot(self.grid_zernike, 0, 2, "Zernike Residual", residual, cmap='jet')

        # Keep line-cut tied to latest phase map
        self.update_line_cut_plot()

    def save_hdf5_dialog(self):
        """Prompt the user for a save path and write a comprehensive HDF5 file."""
        default_name = f"wavetool_{datetime.now().strftime('%Y%m%d_%H%M%S')}.hdf5"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save HDF5", default_name, "HDF5 Files (*.hdf5 *.h5)"
        )
        if not path:
            return
        try:
            self._write_hdf5(path, run_name=os.path.splitext(os.path.basename(path))[0], settings={})
            QMessageBox.information(self, "Saved", f"HDF5 saved to:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", str(e))

    def _write_hdf5(self, h5_path, run_name, settings):
        """Write a structured HDF5 file with all result arrays and metadata."""
        import h5py

        # 2D / nD array keys to store under /arrays/
        array_keys = [
            "displacement_x", "displacement_y",
            "phase_map", "phase_residual_2nd",
            "transmission", "mask",
            "zernike_fitted", "zernike_residual", "zernike_coeffs",
        ]

        with h5py.File(h5_path, "w") as f:
            # Root attributes
            f.attrs["run_name"] = run_name
            f.attrs["created"] = datetime.now().isoformat()
            f.attrs["line_cut_x_px"] = int(self.spin_line_x.value())
            f.attrs["line_cut_y_px"] = int(self.spin_line_y.value())

            # /arrays group — compressed raw data
            arr_grp = f.create_group("arrays")
            for key in array_keys:
                arr = self.results.get(key)
                if arr is None:
                    continue
                data = np.asarray(arr)
                # 1-D arrays (e.g. zernike_coeffs) don't need 2-D compression hint
                arr_grp.create_dataset(
                    key, data=data.astype(np.float32 if np.issubdtype(data.dtype, np.floating) else data.dtype),
                    compression="gzip", compression_opts=6
                )
                arr_grp[key].attrs["description"] = key.replace("_", " ").title()

            # /scalars group — scalar and small-vector results
            scalar_grp = f.create_group("scalars")
            for k, v in self.results.items():
                if k in array_keys:
                    continue
                if isinstance(v, (int, float, bool, np.integer, np.floating)):
                    scalar_grp.attrs[k] = float(v)
                elif isinstance(v, (list, tuple)) and len(v) <= 16 and all(
                    isinstance(x, (int, float, np.integer, np.floating)) for x in v
                ):
                    scalar_grp.create_dataset(k, data=np.array(v, dtype=np.float64))
                elif isinstance(v, str):
                    scalar_grp.attrs[k] = v
                elif isinstance(v, (list, tuple)) and all(isinstance(x, str) for x in v):
                    scalar_grp.attrs[k] = ", ".join(v)

            # /settings group — all GUI settings
            if settings:
                settings_grp = f.create_group("settings")
                for k, v in settings.items():
                    try:
                        if isinstance(v, (int, float, bool, np.integer, np.floating)):
                            settings_grp.attrs[str(k)] = float(v) if isinstance(v, (float, np.floating)) else int(v)
                        elif isinstance(v, str):
                            settings_grp.attrs[str(k)] = v
                        elif isinstance(v, (list, tuple)):
                            settings_grp.attrs[str(k)] = json.dumps(v)
                        else:
                            settings_grp.attrs[str(k)] = str(v)
                    except Exception:
                        pass

        return h5_path

    def save_result_bundle(self, output_folder, run_name, settings):
        """
        Save result bundle:
        - PNG figures: disp X/Y, phase, residual, zernike coeff/fitted/residual, line cut.
        - JSON: settings + scalar metrics + line-cut position.
        - HDF5: all arrays + metadata.
        """
        os.makedirs(output_folder, exist_ok=True)

        def save_img_png(data, title, file_name, cmap='viridis'):
            if data is None:
                return None
            fig = Figure(figsize=(5, 4))
            ax = fig.add_subplot(111)
            im = ax.imshow(data, cmap=cmap, aspect='auto')
            fig.colorbar(im, ax=ax)
            ax.set_title(title)
            fig.tight_layout()
            path = os.path.join(output_folder, file_name)
            fig.savefig(path, dpi=150)
            return path

        def save_bar_png(data, title, file_name):
            if data is None:
                return None
            fig = Figure(figsize=(5, 4))
            ax = fig.add_subplot(111)
            ax.bar(range(len(data)), data)
            ax.set_title(title)
            fig.tight_layout()
            path = os.path.join(output_folder, file_name)
            fig.savefig(path, dpi=150)
            return path

        files = {}
        files["disp_x_png"] = save_img_png(self.results.get("displacement_x"), "Displacement X", f"{run_name}_disp_x.png")
        files["disp_y_png"] = save_img_png(self.results.get("displacement_y"), "Displacement Y", f"{run_name}_disp_y.png")
        files["phase_png"] = save_img_png(self.results.get("phase_map"), "Integrated Phase", f"{run_name}_phase.png", cmap='jet')
        files["phase_residual_png"] = save_img_png(self.results.get("phase_residual_2nd"), "Residual (2nd Order Removed)", f"{run_name}_phase_residual_2nd.png", cmap='jet')
        files["zernike_coeff_png"] = save_bar_png(self.results.get("zernike_coeffs"), "Zernike Coefficients", f"{run_name}_zernike_coeffs.png")
        files["zernike_fitted_png"] = save_img_png(self.results.get("zernike_fitted"), "Zernike Fitted Phase", f"{run_name}_zernike_fitted.png", cmap='jet')
        files["zernike_residual_png"] = save_img_png(self.results.get("zernike_residual"), "Zernike Residual", f"{run_name}_zernike_residual.png", cmap='jet')

        if hasattr(self, "line_fig"):
            line_path = os.path.join(output_folder, f"{run_name}_line_cut.png")
            self.line_fig.savefig(line_path, dpi=150)
            files["line_cut_png"] = line_path

        # HDF5 — comprehensive structured file
        h5_path = os.path.join(output_folder, f"{run_name}.hdf5")
        self._write_hdf5(h5_path, run_name=run_name, settings=settings)
        files["hdf5"] = h5_path

        # JSON for parameters and scalar summary
        summary = {}
        for k, v in self.results.items():
            if isinstance(v, (int, float, str, bool)):
                summary[k] = v
            elif isinstance(v, (list, tuple)) and all(isinstance(x, (int, float, str, bool)) for x in v):
                summary[k] = list(v)

        json_data = {
            "run_name": run_name,
            "settings": settings,
            "line_cut_position_px": {
                "x": int(self.spin_line_x.value()),
                "y": int(self.spin_line_y.value())
            },
            "result_summary": summary,
            "saved_files": files
        }
        json_path = os.path.join(output_folder, f"{run_name}_params.json")
        with open(json_path, "w", encoding="utf-8") as fp:
            json.dump(json_data, fp, indent=2, ensure_ascii=False)
        files["params_json"] = json_path
        return files
