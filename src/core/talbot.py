import numpy as np
from .func import fft2, ifft2, frankotchellappa, load_image, prColor, diffraction_prop, calculate_sigma_width
from skimage.restoration import unwrap_phase
import scipy.ndimage as snd
from .calculations import calculate_envelope, fit_remove_2nd_order, fit_zernike, fit_2nd_order_coeffs
import scipy.constants as sc
from .processor import WavefrontProcessor
from concurrent.futures import ThreadPoolExecutor
import os

class TalbotProcessor(WavefrontProcessor):
    def __init__(self):
        super().__init__(name="Talbot Interferometry")
        self.params = {
            "grating_period": 10.0,
            "talbot_distance": 50.0
        }

    def process(self, image: np.ndarray, params: dict = None) -> dict:
        # Extract parameters
        if params is None: params = {}
        
        image = np.array(image, dtype=np.float32)

        self.energy = params.get("energy_kev", 10.0) * 1e3
        self.dist = params.get("distance_mm", 100.0) * 1e-3
        self.gt_period = params.get("period_um", 4.8) * 1e-6
        self.p_x = params.get("pixel_size_um", 6.5) * 1e-6
        self.mode = str(params.get("analysis_mode", "Absolute")).strip().lower()
        self.correct_angle = params.get("correct_angle", False)
        self.use_mask = params.get("use_mask", False)
        self.real_wf = params.get("real_wf", False)
        source_distance_default_m = float(params.get("source_distance_m", 0.0))
        if "source_distance_m" not in params and "source_distance_mm" in params:
            source_distance_default_m = float(params.get("source_distance_mm", 0.0)) * 1e-3

        if "source_distance_v_m" in params:
            source_distance_v_m = float(params.get("source_distance_v_m", 0.0))
        elif "source_distance_v_mm" in params:
            source_distance_v_m = float(params.get("source_distance_v_mm", 0.0)) * 1e-3
        else:
            source_distance_v_m = source_distance_default_m

        if "source_distance_h_m" in params:
            source_distance_h_m = float(params.get("source_distance_h_m", 0.0))
        elif "source_distance_h_mm" in params:
            source_distance_h_m = float(params.get("source_distance_h_mm", 0.0)) * 1e-3
        else:
            source_distance_h_m = source_distance_default_m

        self.source_d = [source_distance_v_m, source_distance_h_m]  # [V, H], meters
        self.crop_rect = params.get("crop", [0, 0, 0, 0])
        self.save_path = params.get("save_path", None)

        print(self.crop_rect, self.gt_period, self.p_x)

        M_shape = image.shape
        if self.crop_rect[0] + self.crop_rect[2]> 0:
            x_crop = [self.crop_rect[0], self.crop_rect[2]]
        else:
            x_crop = [0, M_shape[1]]
        if self.crop_rect[1] + self.crop_rect[3]> 0:
            y_crop = [self.crop_rect[1], self.crop_rect[3]]
        else:
            y_crop = [0, M_shape[0]]

        img_crop = lambda img: img[int(y_crop[0]):int(y_crop[1]),
                int(x_crop[0]):int(x_crop[1])] 
        
        image = img_crop(image)
        
        print('image crop size: ', image.shape)

        # Wavelength (m) = 1.2398e-9 / E(keV)
        self.wavelength = sc.value('inverse meter-electron volt relationship') / self.energy
        
        if not self.use_mask:
            mask = np.ones(image.shape)
            mask_threshold = 0
        else:
            # calculate the intensity using the fft filtering
            # get the mask for the main beam in the image
            img_filter = calculate_envelope(image, self.gt_period/self.p_x)

            mask_threshold = 0.05
            mask = np.ones(img_filter.shape) * ((img_filter-img_filter.min()) > mask_threshold*(img_filter.max()-img_filter.min()))

        angle_error, period_real = self.find_rotation_angle(self.p_x, self.gt_period, image)

        M_factor = [period_real[0] / self.gt_period, period_real[1] / self.gt_period]
        print('\nCalculated real period from the image is: {}V {}H'.format(period_real[0]/self.p_x, period_real[1]/self.p_x), ' pixels')
        print('\nCalculated M factor from the image is: {}V {}H'.format(M_factor[0], M_factor[1]))

        # Effective period selection:
        # 1) If source distance is provided (>0), use geometric magnification.
        # 2) If source distance is 0, keep the current method (measured period from image).
        self.gt_period_effect = [0.0, 0.0]
        self.period_source_model = [0.0, 0.0]
        self.period_method = ["image", "image"]  # [V, H]

        for i in range(2):
            if abs(self.source_d[i]) > 1e-12:
                period_model = self.gt_period * (1.0 + self.dist / self.source_d[i])
                if period_model > 0:
                    self.period_source_model[i] = period_model
                    self.gt_period_effect[i] = period_model
                    self.period_method[i] = "source_distance"
                else:
                    # Non-physical effective period from source model: fallback.
                    self.period_source_model[i] = period_model
                    self.gt_period_effect[i] = period_real[i]
                    self.period_method[i] = "image_fallback"
            else:
                self.period_source_model[i] = self.gt_period
                self.gt_period_effect[i] = period_real[i]

        print(
            "Effective period method [V,H]: {} / {}\n"
            "Source model period [V,H] (um): {:.6f}, {:.6f}\n"
            "Used effective period [V,H] (um): {:.6f}, {:.6f}".format(
                self.period_method[0],
                self.period_method[1],
                self.period_source_model[0] * 1e6,
                self.period_source_model[1] * 1e6,
                self.gt_period_effect[0] * 1e6,
                self.gt_period_effect[1] * 1e6,
            )
        )

        if self.mode == "absolute":
            
            dxy, dxy_raw, dpc, _, dark_field, int00, virtual_pixelsize = self.solve(image, None)
        
        else:  # Relative
            ref_path = params.get("ref_image_path", params.get("ref_path", None))
            if not ref_path:
                raise ValueError("Reference image path must be provided for Relative mode.")
            
            ref_image = load_image(ref_path)
            ref_image = img_crop(ref_image)

            ref_image = np.array(ref_image, dtype=np.float32)
            if np.sum(self.crop_rect or 0) > 0:
                ref_image = img_crop(ref_image)
            dxy, dxy_raw, dpc, _, dark_field, int00, virtual_pixelsize = self.solve(image, ref_image)

        if True:
            dy = snd.zoom(dxy[1], (image.shape[0]/dpc[1].shape[0], image.shape[1]/dpc[1].shape[1]), order=1)
            dx = snd.zoom(dxy[0], (image.shape[0]/dpc[1].shape[0], image.shape[1]/dpc[1].shape[1]), order=1)
            int00 = snd.zoom(int00, (image.shape[0]/dpc[1].shape[0], image.shape[1]/dpc[1].shape[1]), order=1)

            dx -= np.mean(dx[mask==1])
            dy -= np.mean(dy[mask==1])
            dx = dx * mask * virtual_pixelsize[1] /self.p_x
            dy = dy * mask * virtual_pixelsize[0] /self.p_x

        print('virtual pixel size: ', virtual_pixelsize)

        phase = frankotchellappa(dx * self.p_x**2, dy * self.p_x**2) / self.wavelength /self.dist *2*np.pi

        # --- 2nd Order Removal ---
        _, residual_2nd, coeffs_2nd = fit_remove_2nd_order(phase)
        
        # Calculate Radius of Curvature (ROC)
        # Phase phi(x) approx - pi * x^2 / (lambda * R)
        # coeffs[3] is x^2 term (c3), coeffs[5] is y^2 term (c5)
        # c3 = - pi * p^2 / (lambda * Rx)  (if x is in pixels)
        # Rx = - pi * p^2 / (lambda * c3)
        
        c3 = coeffs_2nd[3]
        c5 = coeffs_2nd[5]
        print(c3, c5)
        if abs(c3) < 1e-12: roc_x = float('inf')
        else: roc_x = - np.pi * self.p_x**2 / (self.wavelength * c3)
            
        if abs(c5) < 1e-12: roc_y = float('inf')
        else: roc_y = - np.pi * self.p_x**2 / (self.wavelength * c5)

        # --- Zernike Analysis ---
        zernike_order = params.get("zernike_order", 15)
        zernike_coeffs, zernike_fitted, zernike_residual = fit_zernike(phase, n_terms=zernike_order)

        return {
            "type": "Talbot",
            "transmission": int00, # Use calculated intensity
            "displacement_x": dx,
            "displacement_y": dy,
            "phase_map": phase,
            "phase_residual_2nd": residual_2nd,
            "zernike_coeffs": zernike_coeffs[1:],
            "zernike_fitted": zernike_fitted,
            "zernike_residual": zernike_residual,
            "pv_value": np.ptp(residual_2nd),
            "rms_value": np.std(residual_2nd),
            "roc_x": roc_x,
            "roc_y": roc_y,
            "period_real": [period_real[0] * 1e6, period_real[1] * 1e6],
            "period_effective": [self.gt_period_effect[0] * 1e6, self.gt_period_effect[1] * 1e6],
            "source_distance_m": [self.source_d[0], self.source_d[1]],
            "period_method": self.period_method,
            "source_wavefront_added": bool(self.real_wf)
        }

    def solve(self, img, ref=None):
        """
            use ref and img to get phase distortion
        """
        dxy, dxy_raw, dpc, phase, dark_field, int00, virtual_pixelsize = self.grating_2D(ref, img)
    
        return dxy, dxy_raw, dpc, phase, dark_field, int00, virtual_pixelsize
    
    def grating_2D(self, ref, img):
        """
            phase retrieval for 2D grating
        """
        M_shape = img.shape
        
        # # calculate the theoretical position of the hamonics
        # period_harm_Vert = np.int16(self.p_x/self.gt_period*M_shape[0] /
        #                         (self.source_d[0] + self.distance)*self.source_d[0])
        # period_harm_Hor = np.int16(self.p_x/self.gt_period*M_shape[1] /
        #                         (self.source_d[1] + self.distance)*self.source_d[1])
        # print(period_harm_Hor, period_harm_Vert)
        # use effective period for highly focused beam

        print('effect gt period ', self.gt_period_effect)
        period_harm_Vert = max(2, int(round(self.p_x / self.gt_period_effect[0] * M_shape[0])))
        period_harm_Hor = max(2, int(round(self.p_x / self.gt_period_effect[1] * M_shape[1])))
        # print(period_harm_Hor, period_harm_Vert)
        
        searchRegion = [max(1, int(period_harm_Vert / 2)), max(1, int(period_harm_Hor / 2))]
        
        img_fft = fft2(img)

        def extract_subimage(im_fft, Harm=(0, 0), calculate_angle=False):
            """
            docstring
            """
            idx_peak = [M_shape[0] // 2 + Harm[0] * period_harm_Vert, M_shape[1] // 2 + Harm[1] * period_harm_Hor]
            # find center for vertical
            maskSearchRegion = np.zeros(M_shape)

            maskSearchRegion[idx_peak[0] - searchRegion[0]:
                            idx_peak[0] + searchRegion[0],
                            idx_peak[1] - searchRegion[1]:
                            idx_peak[1] + searchRegion[1]] = 1.0

            idxPeak_ij_exp = np.where(np.abs(im_fft) * maskSearchRegion ==
                                    np.max(np.abs(im_fft) * maskSearchRegion)) 

            prColor('searched peak position: {}(vertical) {}(horizontal)\nerror of harmonic position: {}(vertical) {}(horizontal)'.format(idxPeak_ij_exp[0][0], idxPeak_ij_exp[1][0], (idxPeak_ij_exp[0][0]-idx_peak[0]), (idxPeak_ij_exp[1][0]-idx_peak[1])), 'green')
            # if True:
            #     idx_peak = [idxPeak_ij_exp[0][0], idxPeak_ij_exp[1][0]]
            sub_img_fft = im_fft[idx_peak[0] - period_harm_Vert//2:idx_peak[0] + period_harm_Vert//2,
                                idx_peak[1] - period_harm_Hor//2:idx_peak[1] + period_harm_Hor//2]

            # use the peak position to get the relative angle to the X-Y axis
            
            if calculate_angle:
                angle_error = np.arctan((Harm[1] * (idxPeak_ij_exp[0][0] - M_shape[0] // 2) + Harm[0] * (idxPeak_ij_exp[1][0] - M_shape[1] // 2)) / (Harm[0] * (idxPeak_ij_exp[0][0] - M_shape[0] // 2) + Harm[1] * (idxPeak_ij_exp[1][0] - M_shape[1] // 2)))
                print('angle error for harm: {} is {}'.format(Harm, angle_error/np.pi*180))
                
                return ifft2(sub_img_fft), angle_error
            else:
                angle_error = 0
                return ifft2(sub_img_fft)
        
        # calculate the rotation angle first
        _, angle_error01 = extract_subimage(img_fft, Harm=(0, 1), calculate_angle=True)
        _, angle_error10 = extract_subimage(img_fft, Harm=(1, 0), calculate_angle=True)
        angle_error = (angle_error01 - angle_error10)/2
        prColor('angle error of the grating image: \n 01:{}; 10:{}\naverage angle error: {}'.format(angle_error01/np.pi*180, angle_error10/np.pi*180, angle_error/np.pi*180), 'cyan')
        
        if self.correct_angle:
            img_rot = snd.rotate(img, angle_error/np.pi*180, reshape=False, order=3)
            _, angle_error01 = extract_subimage(fft2(img_rot), Harm=(0, 1), calculate_angle=True)
            _, angle_error10 = extract_subimage(fft2(img_rot), Harm=(1, 0), calculate_angle=True)
            angle_error_corrected = (angle_error01 - angle_error10)/2
            prColor('angle error of the grating image after correction: \n 01:{}; 10:{}\naverage angle error: {}'.format(angle_error01/np.pi*180, angle_error10/np.pi*180, angle_error_corrected/np.pi*180), 'cyan')
            
            img_fft = fft2(img_rot)
            # plt.figure()
            # plt.imshow(np.log(np.abs(img_fft)))
            # plt.title('corrected FFT image')
            # plt.show()

        if self.mode == 'relative':
            if self.correct_angle:
                ref_rot = snd.rotate(ref, angle_error/np.pi*180, reshape=False, order=3)
                ref_fft = fft2(ref_rot)
            else:
                ref_fft = fft2(ref)
            ref_11 = extract_subimage(ref_fft, Harm=(1, 1))
            ref_01 = extract_subimage(ref_fft, Harm=(0, 1))
            ref_10 = extract_subimage(ref_fft, Harm=(1, 0))
            ref_00 = extract_subimage(ref_fft, Harm=(0, 0))

        else:
            ref_01 = 1
            ref_10 = 1
            ref_00 = 1

        img_11 = extract_subimage(img_fft, Harm=(1, 1))
        img_01 = extract_subimage(img_fft, Harm=(0, 1))
        img_10 = extract_subimage(img_fft, Harm=(1, 0))
        img_00 = extract_subimage(img_fft, Harm=(0, 0))

        # plt.figure()
        # plt.imshow(np.log(np.abs(self.fft2(img_01))))
        # plt.title('croped 01 sub FFT')
        # plt.show()

        int00 = np.abs(img_00)/np.abs(ref_00)
        int01 = np.abs(img_01)/np.abs(ref_01)
        int10 = np.abs(img_10)/np.abs(ref_10)
        if self.mode == 'relative':
            arg01 = unwrap_phase(np.angle(img_01)) - unwrap_phase(np.angle(ref_01))
            arg10 = unwrap_phase(np.angle(img_10)) - unwrap_phase(np.angle(ref_10))
        else:
            arg11 = unwrap_phase(np.angle(img_11))
            arg01 = unwrap_phase(np.angle(img_01))
            arg10 = unwrap_phase(np.angle(img_10))
        arg01 = arg01 - np.mean(arg01)
        arg10 = arg10 - np.mean(arg10)

        darkField01 = int01/int00
        darkField10 = int10/int00

        virtual_pixelsize = [0, 0]
        virtual_pixelsize[1] = self.p_x*M_shape[0]/int00.shape[0]
        virtual_pixelsize[0] = self.p_x*M_shape[1]/int00.shape[1]
        
        dx = -arg01 /2 /np.pi
        dy = -arg10 /2 /np.pi

        # add back the source distance induced displacement
        gt_period_effect_h = self.gt_period_effect[1]
        gt_period_effect_v = self.gt_period_effect[0]
        dx_diff = (np.arange(dx.shape[1]) - dx.shape[1] / 2.0) * (gt_period_effect_h - self.gt_period) / virtual_pixelsize[0]
        dy_diff = (np.arange(dx.shape[0]) - dx.shape[0] / 2.0) * (gt_period_effect_v - self.gt_period) / virtual_pixelsize[1]

        XX_diff, YY_diff = np.meshgrid(dx_diff, dy_diff)
        dx_all = dx + XX_diff
        dy_all = dy + YY_diff

        if self.real_wf:
            dx_used = dx_all
            dy_used = dy_all
        else:
            dx_used = dx
            dy_used = dy

        diffPhase01 = dx_used * 2*np.pi *virtual_pixelsize[1]/self.dist
        diffPhase10 = dy_used * 2*np.pi *virtual_pixelsize[0]/self.dist

        if self.correct_angle:
            # crop the boundary due to the image rotation
            N_crop = [int(arg01.shape[0] /2 * np.arctan(angle_error))+1, int(arg01.shape[1] /2 * np.arctan(angle_error))+1]
            crop = lambda x: x[N_crop[0]:-N_crop[0], N_crop[1]:-N_crop[1]]
            dx = crop(dx)
            dy = crop(dy)
            dx_all = crop(dx_all)
            dy_all = crop(dy_all)
            dx_used = crop(dx_used)
            dy_used = crop(dy_used)
            diffPhase01 = crop(diffPhase01)
            diffPhase10 = crop(diffPhase10)
            darkField01 = crop(darkField01)
            darkField10 = crop(darkField10)
            int00 = crop(int00)

        phase = frankotchellappa(diffPhase01 * virtual_pixelsize[1], diffPhase10 * virtual_pixelsize[0]) / self.wavelength

        return [dx_used, dy_used], [dx, dy], [diffPhase01, diffPhase10], phase - np.amin(phase), [darkField01, darkField10], int00, virtual_pixelsize
    
    
    def find_rotation_angle(self, p_x, period, img):
        # find the rotation angle
        M_shape = img.shape
        period_harm = [int(p_x/period*M_shape[0]), int(p_x/period*M_shape[1])]
        searchRegion = [int(period_harm[0]/2), int(period_harm[1]/2)]

        img_fft = fft2(img)

        def extract_subimage(im_fft, Harm=(0, 0)):
            """
            extract the sub image
            """
            idx_peak = [M_shape[0] // 2 + Harm[0] * period_harm[0], M_shape[1] // 2 + Harm[1] * period_harm[1]]
            # find center for vertical
            maskSearchRegion = np.zeros(M_shape)

            maskSearchRegion[idx_peak[0] - searchRegion[0]:
                            idx_peak[0] + searchRegion[0],
                            idx_peak[1] - searchRegion[1]:
                            idx_peak[1] + searchRegion[1]] = 1.0

            idxPeak_ij_exp = np.where(np.abs(im_fft) * maskSearchRegion ==
                                    np.max(np.abs(im_fft) * maskSearchRegion)) 
            # print(idxPeak_ij_exp)
            print('searched peak position: {}(vertical) {}(horizontal)\nerror of harmonic position: {}(vertical) {}(horizontal)'.format(idxPeak_ij_exp[0][0], idxPeak_ij_exp[1][0], (idxPeak_ij_exp[0][0]-idx_peak[0]), (idxPeak_ij_exp[1][0]-idx_peak[1])))

            # if True:
            #     idx_peak = [idxPeak_ij_exp[0][0], idxPeak_ij_exp[1][0]]
            sub_img_fft = im_fft[idx_peak[0] - period_harm[0]//2:idx_peak[0] + period_harm[0]//2,
                                idx_peak[1] - period_harm[1]//2:idx_peak[1] + period_harm[1]//2]

            # use the peak position to get the relative angle to the X-Y axis
            angle_error = np.arctan((Harm[1] * (idxPeak_ij_exp[0][0] - M_shape[0] // 2) + Harm[0] * (idxPeak_ij_exp[1][0] - M_shape[1] // 2)) / (Harm[0] * (idxPeak_ij_exp[0][0] - M_shape[0] // 2) + Harm[1] * (idxPeak_ij_exp[1][0] - M_shape[1] // 2)))
            print('angle error for harm: {} is {}'.format(Harm, angle_error/np.pi*180))
            
            return ifft2(sub_img_fft), angle_error, [idxPeak_ij_exp[0][0] - idx_peak[0], idxPeak_ij_exp[1][0]-idx_peak[1]]

        _, angle_error01, peak_error01 = extract_subimage(img_fft, Harm=(0, 1))
        _, angle_error10, peak_error10 = extract_subimage(img_fft, Harm=(1, 0))
        angle_error = (angle_error01 - angle_error10)/2
        print('angle error of the grating image: \n 01:{}; 10:{}\naverage angle error: {}'.format(angle_error01/np.pi*180, angle_error10/np.pi*180, angle_error/np.pi*180))    
        
        # period_harm_exp = (np.sqrt((period_harm + peak_error01[1])**2 + (peak_error01[0])**2) + np.sqrt((peak_error10[1])**2 + (period_harm + peak_error10[0])**2)) / 2
        period_harm_exp_Horz = np.sqrt((period_harm[1] + peak_error01[1])**2 + (peak_error01[0])**2)
        period_harm_exp_Vert = np.sqrt((peak_error10[1])**2 + (period_harm[0] + peak_error10[0])**2)

        # print('calculated harm period: {}\ntheoretical harm period: {}'.format(period_harm_exp, period_harm))
        # period_real = period * period_harm / period_harm_exp
        print('calculated harm period: {}V {}H\ntheoretical harm period: {}'.format(period_harm_exp_Vert, period_harm_exp_Horz, period_harm))
        period_real = [period * period_harm[0] / period_harm_exp_Vert, period * period_harm[1] / period_harm_exp_Horz]

        return angle_error, period_real


    def data_nanMask(self, data, mask):
        '''
            replace the value outside the mask with nan
        '''
        data_mask = np.copy(data)
        data_mask[np.logical_not(mask)] = np.nan

        return data_mask

    def data_removeNaN(self, data, val=0):
        # replate NaN with val
        data_new = np.copy(data)
        data_new[np.isnan(data_new)] = val
        return data_new
    
    def propagate_focus(self, phase_map: np.ndarray, params: dict, transmission: np.ndarray = None, progress_callback=None, check_stop=None) -> dict:
        # Extract params
        center_dist = params.get("distance_mm", 100.0)*1e-3
        dist_range = params.get("range_mm", 10.0)*1e-3
        dist_step = params.get("step_mm", 1)*1e-3
        direction = params.get("direction", 'backward')
        method = params.get("method", 'default')
        upsampling_factor = params.get("upsampling", 'default')
        
        # If transmission is not provided or not used, use uniform amplitude
        if transmission is None or not params.get("real_intensity", False):
            amplitude = np.ones_like(phase_map)
        else:
            amplitude = np.sqrt(transmission) # Amplitude is sqrt of intensity
            
        # Initial Field
        field = amplitude * np.exp(1j * phase_map)
        
        # Apply Padding and Smoothing if requested
        padding_scale = params.get("padding_scale", 1.0)
        if padding_scale > 1.0:
            h, w = field.shape
            
            # 1. Apply Super-Gaussian Window to smooth edges
            # W(x,y) = exp( - ( (x/wx)^n + (y/wy)^n ) )
            # We want the window to be 1 inside and drop to 0 at the edges
            y, x = np.mgrid[:h, :w]
            cy, cx = h/2, w/2
            # Normalize coordinates to [-1, 1] at the edges
            ny = (y - cy) / (h/2)
            nx = (x - cx) / (w/2)
            
            # Super-Gaussian order (higher = steeper edges)
            sg_order = 20 
            # Scale factor to ensure it drops to ~0 at the very edge (e.g. at 0.95)
            # Let's make it drop at 95% of the field
            window = np.exp( - ( (np.abs(nx)/0.95)**sg_order + (np.abs(ny)/0.95)**sg_order ) )
            
            field = field * window
            
            # 2. Pad with zeros
            new_h = int(h * padding_scale)
            new_w = int(w * padding_scale)
            
            pad_h = (new_h - h) // 2
            pad_w = (new_w - w) // 2
            
            # Use pad_width format: ((top, bottom), (left, right))
            # Handle odd sizes correctly
            pad_top = pad_h
            pad_bottom = new_h - h - pad_top
            pad_left = pad_w
            pad_right = new_w - w - pad_left
            
            field = np.pad(field, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=0)
            
            print(f"Field padded from {h}x{w} to {new_h}x{new_w} (Scale: {padding_scale})")
        
        # Estimate Focal Length from Phase
        # c3 is coeff of x^2. Phase ~ c3 * x^2
        # Parabolic phase: - pi * x^2 * p^2 / (lambda * f)
        # So c3 = - pi * p^2 / (lambda * f)
        # f = - pi * p^2 / (lambda * c3)
        
        coeffs = fit_2nd_order_coeffs(phase_map)
        c3 = coeffs[3] # x^2 term
        c5 = coeffs[5] # y^2 term
        
        # Average x and y curvature for focal length estimate
        c_avg = (c3 + c5) / 2.0
        
        if abs(c_avg) < 1e-9:
            f_est = 0.0
        else:
            f_est = - np.pi * self.p_x**2 / (self.wavelength * c_avg)
            
        print(f"Estimated Focal Length: {f_est*1e3:.2f} mm")
        
        # Propagate to the specified distance (Back propagation)
        # We assume the wavefront is diverging from a source/focus, so we propagate back by -distance

        if direction == 'forward':
            prop_dist = center_dist
        else:
            prop_dist = -center_dist
        
        if upsampling_factor != 1:
            p_x_new = self.p_x / upsampling_factor
            field = snd.zoom(field, upsampling_factor, order=3)
            self.p_x_bp = p_x_new
        else:
            self.p_x_bp = self.p_x

        prop_dist_range = np.arange(-dist_range/2, dist_range/2 + dist_step, dist_step)
        n_steps = len(prop_dist_range)

        # Memory guard for large upsampling/padding combinations.
        # intensity_profiles will be float64 [steps, H, W].
        output_est_bytes = n_steps * field.shape[0] * field.shape[1] * 8
        available_bytes = int(params.get("available_memory_bytes", 0) or 0)
        if available_bytes > 0:
            backend_stack_ratio = float(params.get("backend_stack_ratio", 0.60))
            safe_budget = int(available_bytes * backend_stack_ratio)
            if output_est_bytes > safe_budget:
                raise MemoryError(
                    "Projected result stack is too large for available RAM. "
                    f"Need ~{output_est_bytes/1024**3:.2f} GB; budget ~{safe_budget/1024**3:.2f} GB. "
                    "Reduce upsampling/padding or increase step size."
                )

        intensity_profiles = []
        L_z_list = []
        sigma_x_list = []
        sigma_y_list = []
        fwhm_x_list = []
        fwhm_y_list = []
        new_p_x_list = []
        
        calc_sigma = params.get("calc_sigma", True)
        mag_x = params.get("magnification_x", 1.0)
        mag_y = params.get("magnification_y", 1.0)
        args_list = [(dz, field, self.p_x_bp, prop_dist, self.wavelength, method, calc_sigma, mag_x, mag_y) for dz in prop_dist_range]
        
        # Use ThreadPoolExecutor to avoid multiprocessing issues on Windows (pickling, spawn overhead)
        # Numpy FFTs often release GIL, so threading provides speedup without the stability risks.
        max_workers = min(os.cpu_count() or 4, len(args_list), 16)
        if available_bytes > 0:
            # Rough working-set estimate per worker for propagation intermediates.
            per_worker_bytes = max(1, int(field.shape[0] * field.shape[1] * 64))
            worker_memory_ratio = float(params.get("worker_memory_ratio", 0.25))
            mem_limited_workers = max(1, int((available_bytes * worker_memory_ratio) // per_worker_bytes))
            max_workers = min(max_workers, mem_limited_workers)
        prColor('use ThreadPoolExecutor with {} workers for propagation'.format(max_workers), 'cyan')
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            futures = [executor.submit(process_single_slice, arg) for arg in args_list]
            
            # Wait for results and update progress
            total_tasks = len(futures)
            for i, future in enumerate(futures):
                if check_stop and check_stop():
                    print("Propagation stopped by user.")
                    executor.shutdown(wait=False, cancel_futures=True)
                    return None
                    
                res = future.result() # This blocks until the specific task is done
                intensity, L_z, sx, sy, fx, fy, npx = res
                intensity_profiles.append(intensity)
                L_z_list.append(L_z)
                sigma_x_list.append(sx)
                sigma_y_list.append(sy)
                fwhm_x_list.append(fx)
                fwhm_y_list.append(fy)
                new_p_x_list.append(npx)
                
                if progress_callback:
                    percent = (i + 1) / total_tasks * 100
                    progress_callback(percent)
        
        sigma_x_list = np.array(sigma_x_list)
        sigma_y_list = np.array(sigma_y_list)
        fwhm_x_list = np.array(fwhm_x_list)
        fwhm_y_list = np.array(fwhm_y_list)
        new_p_x_list = np.array(new_p_x_list)

        # Propagate to the estimater focus position
        field_prop_focus, L_z_focus = diffraction_prop(field, self.p_x_bp, prop_dist, self.wavelength, method, mag_x, mag_y)
        intensity_profile_focus= np.abs(field_prop_focus)**2

        intensity_profiles = np.array(intensity_profiles)
        L_z_list = np.array(L_z_list)
        print(intensity_profiles.shape)
        # Generate cuts
        n, h, w = intensity_profiles.shape
        cy, cx = h // 2, w // 2
        
        # Find peak
        py, px = np.unravel_index(np.argmax(intensity_profile_focus), intensity_profile_focus.shape)
        
        # Cuts through the peak
        cut_xz = intensity_profiles[:, py, :]
        cut_yz = intensity_profiles[:, :, px]
        
        return {
            "prop_distance": prop_dist_range+prop_dist, # in m
            "prop_center_distance": prop_dist, # in m
            "focus_2d": intensity_profile_focus,
            "focus_Lz": L_z_focus,
            "intensity_profiles": intensity_profiles,
            "prop_Lz": L_z_list,
            "cut_xz": cut_xz, # Expand for image view
            "cut_yz": cut_yz,
            "focal_length": f_est * 1e3, # mm
            "strehl": 0.0,
            # ... existing keys ...
            "sigma_list": (sigma_x_list + sigma_y_list) / 2, # Average sigma in um
            "sigma_x_list": sigma_x_list,
            "sigma_y_list": sigma_y_list,
            "fwhm_x_list": fwhm_x_list,
            "fwhm_y_list": fwhm_y_list,# ... inside propagate_focus ...
            "new_pixel_size_list": new_p_x_list

        }
    
def process_single_slice(args):
    dz, field, p_x_bp, prop_dist, wavelength, method, calc_sigma, mag_x, mag_y = args
    field_prop, L_z = diffraction_prop(field, p_x_bp, prop_dist + dz, wavelength, method, mag_x, mag_y)
    intensity = np.abs(field_prop)**2
    
    new_p_x = [L_z[0] / intensity.shape[0], L_z[1] / intensity.shape[1]]
    
    if calc_sigma:
        sigma_x, sigma_y, fwhm_x, fwhm_y = calculate_sigma_width(intensity, new_p_x)
    else:
        sigma_x, sigma_y, fwhm_x, fwhm_y = 0.0, 0.0, 0.0, 0.0
    
    return intensity, L_z, sigma_x, sigma_y, fwhm_x, fwhm_y, new_p_x
