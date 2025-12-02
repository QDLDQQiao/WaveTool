from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, QDoubleSpinBox, 
                             QPushButton, QRadioButton, QButtonGroup, QComboBox, QGroupBox, QTextEdit)
from PyQt6.QtCore import pyqtSignal
import numpy as np

class PeriodCalcDialog(QDialog):
    calculate_requested = pyqtSignal(float)
    closed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Period Calculator")
        self.resize(300, 150)
        
        layout = QVBoxLayout(self)
        
        # Input
        input_layout = QHBoxLayout()
        input_layout.addWidget(QLabel("Mask Radius (px):"))
        self.spin_radius = QDoubleSpinBox()
        self.spin_radius.setRange(0, 1000)
        self.spin_radius.setValue(20)
        input_layout.addWidget(self.spin_radius)
        layout.addLayout(input_layout)
        
        # Calculate Button
        self.btn_calc = QPushButton("Calculate")
        self.btn_calc.clicked.connect(self.on_calculate)
        layout.addWidget(self.btn_calc)
        
        # Result
        self.lbl_result = QLabel("Period: -")
        self.lbl_result.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(self.lbl_result)
        
    def on_calculate(self):
        self.calculate_requested.emit(self.spin_radius.value())
        
    def update_result(self, period):
        self.lbl_result.setText(f"Period: {period:.2f} px")

    def closeEvent(self, event):
        self.closed.emit()
        super().closeEvent(event)

class TalbotPeriodDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Period Calc")
        self.resize(400, 500)
        
        layout = QVBoxLayout(self)
        
        # 1. Mode: Talbot or Fractional Talbot
        self.mode_group = QButtonGroup(self)
        mode_layout = QHBoxLayout()
        self.rb_talbot = QRadioButton("Talbot")
        self.rb_fractional = QRadioButton("Fractional Talbot")
        self.rb_talbot.setChecked(True)
        self.mode_group.addButton(self.rb_talbot)
        self.mode_group.addButton(self.rb_fractional)
        mode_layout.addWidget(self.rb_talbot)
        mode_layout.addWidget(self.rb_fractional)
        layout.addLayout(mode_layout)
        
        self.rb_fractional.toggled.connect(self.toggle_fractional_inputs)
        
        # 2. Grating Period (um)
        layout.addWidget(QLabel("Grating Period (um):"))
        self.spin_period = QDoubleSpinBox()
        self.spin_period.setRange(0.001, 10000)
        self.spin_period.setValue(4.8)
        self.spin_period.setDecimals(3)
        layout.addWidget(self.spin_period)
        
        # 3. X-ray Energy (eV)
        layout.addWidget(QLabel("X-ray Energy (eV):"))
        self.spin_energy = QDoubleSpinBox()
        self.spin_energy.setRange(1, 100000)
        self.spin_energy.setValue(10000)
        self.spin_energy.setDecimals(1)
        layout.addWidget(self.spin_energy)

        # 4. Source Distance (m)
        layout.addWidget(QLabel("Source Distance (m):"))
        self.spin_source = QDoubleSpinBox()
        self.spin_source.setRange(0.1, 1000000)
        self.spin_source.setValue(50.0)
        layout.addWidget(self.spin_source)
        
        # 5. Orientation: Diag or 0 deg
        layout.addWidget(QLabel("Orientation:"))
        self.combo_orient = QComboBox()
        self.combo_orient.addItems(["0 deg", "Diag (45 deg)"])
        layout.addWidget(self.combo_orient)
        
        # 5. Pattern: Checkboard or Grid
        layout.addWidget(QLabel("Pattern:"))
        self.combo_pattern = QComboBox()
        self.combo_pattern.addItems(["Checkerboard", "Grid"])
        layout.addWidget(self.combo_pattern)

        # 6. Phase Level: Pi or Pi/2
        layout.addWidget(QLabel("Phase Level:"))
        self.combo_phase = QComboBox()
        self.combo_phase.addItems(["Pi", "Pi/2"])
        layout.addWidget(self.combo_phase)
        
        # 7. Distance Range (mm) - Only for Fractional
        self.group_range = QGroupBox("Fractional Settings")
        range_layout = QHBoxLayout(self.group_range)
        
        range_layout.addWidget(QLabel("Dist Min (mm):"))
        self.spin_min = QDoubleSpinBox()
        self.spin_min.setRange(0, 10000)
        self.spin_min.setValue(100)
        range_layout.addWidget(self.spin_min)
        
        range_layout.addWidget(QLabel("Max (mm):"))
        self.spin_max = QDoubleSpinBox()
        self.spin_max.setRange(0, 10000)
        self.spin_max.setValue(500)
        range_layout.addWidget(self.spin_max)

        range_layout.addWidget(QLabel("Min Period (um):"))
        self.spin_min_period = QDoubleSpinBox()
        self.spin_min_period.setRange(0, 1000)
        self.spin_min_period.setValue(2.0)
        range_layout.addWidget(self.spin_min_period)
        
        layout.addWidget(self.group_range)
        self.group_range.hide() # Hidden by default
        
        # 7. Calculate Button
        self.btn_calc = QPushButton("Calculate")
        self.btn_calc.clicked.connect(self.calculate)
        layout.addWidget(self.btn_calc)
        
        # 8. Results Textbox
        layout.addWidget(QLabel("Results:"))
        self.txt_results = QTextEdit()
        self.txt_results.setReadOnly(True)
        layout.addWidget(self.txt_results)
        
    def toggle_fractional_inputs(self, checked):
        self.group_range.setVisible(checked)
        
    def calculate(self):
        try:
            p_gt = self.spin_period.value() * 1e-6
            E_eV = self.spin_energy.value()
            R_m = self.spin_source.value()
            orient = self.combo_orient.currentText()
            pattern = self.combo_pattern.currentText()
            phase_level = self.combo_phase.currentText()
            is_fractional = self.rb_fractional.isChecked()
            
            wl = 1.23984198 / E_eV * 1e-6

            if not is_fractional:
                if pattern == 'Checkerboard':
                    if phase_level == 'Pi/2':
                        if orient == '0 deg':
                            zt = p_gt**2/wl
                            zt_effect = zt / (1 - zt / R_m)
                            nTalbot = (np.arange(1, 10) * 2 -1) / 4
                            distance = np.round(zt_effect* nTalbot, 5)  # nth talbot distance
                            p_theta = p_gt
                            p_theta_effect = p_theta * (1+ distance / R_m)
                        elif orient == 'Diag (45 deg)':
                            zt = p_gt**2/2/wl
                            zt_effect = zt / (1 - zt / R_m)
                            nTalbot = (np.arange(1, 10) * 2 -1) / 4
                            distance = np.round(zt_effect* nTalbot, 5)  # nth talbot distance
                            p_theta = p_gt  * np.sqrt(2)
                            p_theta_effect = p_theta * (1+ distance / R_m)
                    elif phase_level == 'Pi':
                        if orient == '0 deg':
                            zt = p_gt**2/wl
                            zt_effect = zt / (1 - zt / R_m)
                            nTalbot = (np.arange(1, 10) * 2 -1) / 8
                            distance = np.round(zt_effect* nTalbot, 5)  # nth talbot distance
                            p_theta = p_gt / 2
                            p_theta_effect = p_theta * (1+ distance / R_m)
                        elif orient == 'Diag (45 deg)':
                            zt = p_gt**2/wl
                            zt_effect = zt / (1 - zt / R_m)
                            nTalbot = (np.arange(1, 10) * 2 -1) / 8
                            distance = np.round(zt_effect* nTalbot, 5)  # nth talbot distance
                            p_theta = p_gt /2 * np.sqrt(2)
                            p_theta_effect = p_theta * (1+ distance / R_m)

                result_text = f"Wavelength: {wl*1e9:.4f} nm\n"
                result_text += f"Plane Wave Z_T: {zt*1e3:.4f} mm\n"
                result_text += f"Source Dist: {R_m:.2f} m\n"
                result_text += f"Physical Z_T : {zt_effect*1e3:.4f} mm\n"
                
                for d, p in zip(distance, p_theta_effect):
                    result_text += f"Dist: {d*1e3:.4f} mm  |  Period: {p*1e6:.4f} um\n"
            
            elif is_fractional:
                d_min = self.spin_min.value() * 1e-3
                d_max = self.spin_max.value() * 1e-3
                min_period_um = self.spin_min_period.value() * 1e-6
                
                def gcd(a, b):
                    while b!=0:
                        a, b = b, a%b
                    return a

                def all_coprime(a):
                    k = []
                    for i in range(a):
                        if gcd(i, a) == 1:
                            k.append(i)
                            i+=1
                        else:
                            i+=1
                    return np.array(k)

                def fractional_Talbot(gt_wl, gt_period, R, N, order=0):
                    Z0 = 2 * gt_period**2 / gt_wl 

                    M_list = all_coprime(N)
                    if order != 0:
                        for k in range(1, order+1):
                            M_list = np.append(M_list, M_list+k*N)
                            
                    if (N % 2) == 0:
                        Compress_ratio = 2/N
                    else:
                        Compress_ratio = 1/N
                    
                    Z_t0 = Z0 * M_list / N
                    Z_t = R/(R-Z_t0) * Z_t0
                    P = (R+Z_t)/R * Compress_ratio * gt_period

                    return M_list, P, Z_t, Compress_ratio
    
                N_list = np.arange(1, 15)

                m_list = []
                P_list = []
                Z_t_list = []
                Compress_ratio_list = []

                for n in N_list:
                    m, P, Z_t, Compress_ratio = fractional_Talbot(wl, p_gt, R_m, n, order=1)
                    m_list.append(m)
                    P_list.append(P)
                    Z_t_list.append(Z_t)
                    Compress_ratio_list.append(Compress_ratio)
                
                # for m, n, P, Z_t, Compress_ratio in zip(m_list, N_list, P_list, Z_t_list, Compress_ratio_list):
                #     print('------------------------------------------------------------------------')
                #     print('N: ', n, ' | M: ', m, ' | Compression ratio: {:.3f}'.format(Compress_ratio))
                #     print('distance: ', Z_t, 'm')
                #     print('grating period: ', P*1e6, 'um')

                # print('min period', min_period_um)
                condition = []
                for m, n, P, Z_t, Compress_ratio in zip(m_list, N_list, P_list, Z_t_list, Compress_ratio_list):

                    for kk, z in enumerate(Z_t):
                        if z >= d_min and z <= d_max:
                            if P[kk] >= min_period_um:
                                condition.append([m[kk], n, P[kk], z, Compress_ratio])

                condition.sort(key=lambda elem:elem[3])
                # for cond in condition:
                    
                #     result_text += 'N: ', cond[1], ' | M: ', cond[0], ' | Compression ratio: {:.3f}'.format(cond[4]), '| distance: {:.4f}'.format(cond[3]), '| grating period: {:.3f} um'.format(cond[2]*1e6)

                result_text = f"\nFractional Orders in [{d_min}, {d_max}] m (Period > {min_period_um * 1e6} um):\n"

                for cond in condition:
                    m_val = cond[0]
                    n_val = cond[1]
                    period_val = cond[2] * 1e6 # convert to um
                    dist_val = cond[3] * 1e3   # convert to mm
                    
                    # Display String:
                    result_text += f"N: {n_val} | M: {m_val} | Comp Ratio: {cond[4]:.3f} | Dist: {dist_val:.4f} mm | Period: {period_val:.3f} um\n"

                if not condition:
                    result_text += "No orders found in range."

            self.txt_results.setText(result_text)
            
        except Exception as e:
            self.txt_results.setText(f"Error: {str(e)}")
