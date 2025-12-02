import numpy as np
import scipy.ndimage as snd
from scipy.interpolate import griddata
import concurrent.futures
import multiprocessing
from .processor import WavefrontProcessor
from .calculations import fit_zernike, frankot_chellappa, fit_remove_2nd_order
from .func import fft2, ifft2

def process_patches_chunk(indices, points_r, points_c, img_flat, half_width, template, XX_template, YY_template, mean_threshold):
    dx_list = []
    dy_list = []
    pos_list = []
    intensity_list = []
    
    for i in indices:
        pos = [points_r[i], points_c[i]]
        
        patch = extract_subimage(img_flat, pos, half_width)
        if patch is None: continue
        
        # Check if patch has signal
        if np.mean(patch) < mean_threshold: continue 
        
        # Correlation
        patch_norm = patch - np.mean(patch)
        temp_norm = template - np.mean(template)
        
        # FFT Correlation
        corr = np.abs(ifft2(fft2(patch_norm) * fft2(temp_norm)))
        
        dy_local, dx_local = find_disp(corr, XX_template, YY_template)
        
        dx_list.append(dx_local)
        dy_list.append(dy_local)
        pos_list.append(pos)
        intensity_list.append(np.sum(patch))
        
    return dx_list, dy_list, pos_list, intensity_list

def find_rotation_angle(p_x, period, img):
    # find the rotation angle
    M_shape = img.shape
    period_harm = [int(p_x/period*M_shape[0]), int(p_x/period*M_shape[1])]
    searchRegion = [int(period_harm[0]/2), int(period_harm[1]/2)]

    img_fft = fft2(img)

    def extract_subimage_fft(im_fft, Harm=(0, 0)):
        idx_peak = [M_shape[0] // 2 + Harm[0] * period_harm[0], M_shape[1] // 2 + Harm[1] * period_harm[1]]
        maskSearchRegion = np.zeros(M_shape)

        # Boundary check
        r_min = max(0, idx_peak[0] - searchRegion[0])
        r_max = min(M_shape[0], idx_peak[0] + searchRegion[0])
        c_min = max(0, idx_peak[1] - searchRegion[1])
        c_max = min(M_shape[1], idx_peak[1] + searchRegion[1])

        maskSearchRegion[r_min:r_max, c_min:c_max] = 1.0

        masked_fft = np.abs(im_fft) * maskSearchRegion
        if np.max(masked_fft) == 0:
            return None, 0, [0, 0]

        idxPeak_ij_exp = np.unravel_index(np.argmax(masked_fft), M_shape)
        
        # Angle error
        denom = (Harm[0] * (idxPeak_ij_exp[0] - M_shape[0] // 2) + Harm[1] * (idxPeak_ij_exp[1] - M_shape[1] // 2))
        if denom == 0:
            angle_error = 0
        else:
            angle_error = np.arctan((Harm[1] * (idxPeak_ij_exp[0] - M_shape[0] // 2) + Harm[0] * (idxPeak_ij_exp[1] - M_shape[1] // 2)) / denom)
        
        return None, angle_error, [idxPeak_ij_exp[0] - idx_peak[0], idxPeak_ij_exp[1]-idx_peak[1]]

    _, angle_error01, peak_error01 = extract_subimage_fft(img_fft, Harm=(0, 1))
    _, angle_error10, peak_error10 = extract_subimage_fft(img_fft, Harm=(1, 0))
    angle_error = (angle_error01 - angle_error10)/2
    
    period_harm_exp_Horz = np.sqrt((period_harm[1] + peak_error01[1])**2 + (peak_error01[0])**2)
    period_harm_exp_Vert = np.sqrt((peak_error10[1])**2 + (period_harm[0] + peak_error10[0])**2)

    period_real = [period * period_harm[0] / period_harm_exp_Vert if period_harm_exp_Vert != 0 else period, 
                   period * period_harm[1] / period_harm_exp_Horz if period_harm_exp_Horz != 0 else period]

    return angle_error, period_real

def find_disp(Corr_img, XX_axis, YY_axis, sub_resolution=True):
    pos = np.unravel_index(np.argmax(Corr_img, axis=None), Corr_img.shape)
    Corr_max = np.amax(Corr_img)
    
    # Sub-pixel interpolation
    Corr_img_pad = np.pad(Corr_img, ((1,1), (1,1)), 'edge')
    max_pos_y = pos[0] + 1
    max_pos_x = pos[1] + 1

    dy = (Corr_img_pad[max_pos_y + 1, max_pos_x] - Corr_img_pad[max_pos_y - 1, max_pos_x]) / 2.0
    dyy = (Corr_img_pad[max_pos_y + 1, max_pos_x] + Corr_img_pad[max_pos_y - 1, max_pos_x] - 2.0 * Corr_img_pad[max_pos_y, max_pos_x])

    dx = (Corr_img_pad[max_pos_y, max_pos_x + 1] - Corr_img_pad[max_pos_y, max_pos_x - 1]) / 2.0
    dxx = (Corr_img_pad[max_pos_y, max_pos_x + 1] + Corr_img_pad[max_pos_y, max_pos_x - 1] - 2.0 * Corr_img_pad[max_pos_y, max_pos_x])

    dxy = (Corr_img_pad[max_pos_y + 1, max_pos_x + 1] - Corr_img_pad[max_pos_y + 1, max_pos_x - 1] 
           - Corr_img_pad[max_pos_y - 1, max_pos_x + 1] + Corr_img_pad[max_pos_y - 1, max_pos_x - 1]) / 4.0
    
    det = dxx * dyy - dxy * dxy
    if det != 0.0:
        det = 1.0 / det
    
    pixel_res_x = XX_axis[0, 1] - XX_axis[0,0]
    pixel_res_y = YY_axis[1, 0] - YY_axis[0,0]
    Minor_disp_x = (- (dyy * dx - dxy * dy) * det) * pixel_res_x
    Minor_disp_y = (- (dxx * dy - dxy * dx) * det) * pixel_res_y

    if sub_resolution:
        disp_x = Minor_disp_x + XX_axis[pos[0], pos[1]]  
        disp_y = Minor_disp_y + YY_axis[pos[0], pos[1]]
    else:
        disp_x = XX_axis[pos[0], pos[1]]  
        disp_y = YY_axis[pos[0], pos[1]]
        
    return disp_y, disp_x

def extract_subimage(img, pos, half_width):
    r, c = int(pos[0]), int(pos[1])
    hw_r, hw_c = int(half_width[0]), int(half_width[1])
    
    r_min, r_max = r - hw_r, r + hw_r + 1
    c_min, c_max = c - hw_c, c + hw_c + 1
    
    if r_min < 0 or r_max > img.shape[0] or c_min < 0 or c_max > img.shape[1]:
        return None
    
    return img[r_min:r_max, c_min:c_max]

def grid_generate(mask, img_flat, p_x, hole_period, period_effect, angle_error):
    center_mask = snd.center_of_mass(img_flat * mask)
    
    # Refine center
    pe_r_px = period_effect[0] / p_x
    pe_c_px = period_effect[1] / p_x
    
    r_start = int(center_mask[0] - pe_r_px/2)
    r_end = int(center_mask[0] + pe_r_px/2)
    c_start = int(center_mask[1] - pe_c_px/2)
    c_end = int(center_mask[1] + pe_c_px/2)
    
    # Boundary check
    r_start = max(0, r_start); r_end = min(img_flat.shape[0], r_end)
    c_start = max(0, c_start); c_end = min(img_flat.shape[1], c_end)
    
    sub_img = img_flat[r_start:r_end, c_start:c_end]
    if sub_img.size == 0:
        return None, None, None, center_mask

    center_pos = np.array(center_mask) # Initial guess
    # Find peak in the center patch to align grid
    peak_idx = np.unravel_index(np.argmax(sub_img), sub_img.shape)
    
    # Adjust center_pos to align with the peak
    # The sub_img top-left is (r_start, c_start)
    peak_r_global = r_start + peak_idx[0]
    peak_c_global = c_start + peak_idx[1]
    
    center_pos = np.array([peak_r_global, peak_c_global])

    # Generate grid
    # Extend grid to cover image
    # We generate grid points relative to center_pos
    
    # Number of points needed
    n_r = int(img_flat.shape[0] / pe_r_px) * 2 + 2
    n_c = int(img_flat.shape[1] / pe_c_px) * 2 + 2
    
    y_idx = np.arange(-n_r, n_r)
    x_idx = np.arange(-n_c, n_c)
    
    XX_idx, YY_idx = np.meshgrid(x_idx, y_idx)
    
    # Rotate grid
    theta = -angle_error
    
    # Grid coordinates relative to center (in pixels)
    # X corresponds to column index, Y to row index
    X_grid_rel = XX_idx * pe_c_px
    Y_grid_rel = YY_idx * pe_r_px
    
    X_rot = X_grid_rel * np.cos(theta) + Y_grid_rel * np.sin(theta)
    Y_rot = -X_grid_rel * np.sin(theta) + Y_grid_rel * np.cos(theta)
    
    X_final = center_pos[1] + X_rot
    Y_final = center_pos[0] + Y_rot
    
    # Filter points inside image and mask
    valid_points = (X_final >= 0) & (X_final < img_flat.shape[1]) & \
                   (Y_final >= 0) & (Y_final < img_flat.shape[0])
                   
    # Also check if inside mask (roughly)
    # We can check integer coordinates
    valid_indices = np.where(valid_points)
    
    final_points_r = Y_final[valid_indices]
    final_points_c = X_final[valid_indices]
    
    # Check mask
    mask_vals = mask[final_points_r.astype(int), final_points_c.astype(int)]
    valid_mask = mask_vals > 0
    
    final_points_r = final_points_r[valid_mask]
    final_points_c = final_points_c[valid_mask]
    
    # Calculate theoretical difference (if effective period != design period)
    # This is for DPC or magnification analysis, maybe less critical for basic wavefront?
    # Hartman_solver uses it.
    
    return [final_points_r, final_points_c], center_pos

class HartmannProcessor(WavefrontProcessor):
    def __init__(self):
        super().__init__(name="Hartmann Sensor")
        self.params = {
            "period_um": 150.0,      # Lenslet pitch
            "pixel_size_um": 5.0,    # Camera pixel size
            "distance_mm": 20.0,     # Distance lenslet -> sensor
            "energy_kev": 10.0,      # For wavelength (optional)
            "hole_size_um": 30.0,    # Hole size for template
            "mask_threshold": 0.2    # Threshold for beam mask
        }
 
    def process(self, image: np.ndarray, params: dict = None) -> dict:
        if params is None: params = {}
        
        # 1. Parameters
        pitch_um = params.get("period_um", 150.0)
        pixel_size_um = params.get("pixel_size_um", 5.0)
        dist_mm = params.get("distance_mm", 20.0)
        hole_size_um = params.get("hole_size_um", 30.0)
        mask_thresh = params.get("mask_threshold", 0.2)
        
        p_x = pixel_size_um * 1e-6
        pitch = pitch_um * 1e-6
        hole_size = hole_size_um * 1e-6
        
        # 2. Preprocessing
        # Gaussian filter to remove noise
        sigma = int(hole_size / p_x / 4)
        if sigma < 1: sigma = 1
        img_filter = snd.gaussian_filter(image, sigma=sigma)
        
        # Flat field correction (remove low freq background)
        sigma_flat = int(pitch / p_x)
        flat = snd.gaussian_filter(img_filter, sigma_flat)
        img_flat = img_filter / (flat + 1e-6) # Avoid div by zero
        
        # 3. Mask Generation
        # Maximum filter to find peaks, then smooth to get envelope
        img_max = snd.maximum_filter(img_flat, size=int(pitch/p_x))
        img_env = snd.gaussian_filter(img_max, sigma=int(pitch/p_x))
        
        mask_val = (img_env - img_env.min()) > mask_thresh * (img_env.max() - img_env.min())
        mask = mask_val.astype(float)
        
        # 4. Grid Detection
        angle_error, period_real = find_rotation_angle(p_x, pitch, img_flat)
        
        # Use calculated period
        period_effect = period_real
        
        grid_points, center_pos = grid_generate(mask, img_flat, p_x, pitch, period_effect, angle_error)
        
        if grid_points is None or len(grid_points[0]) == 0:
             return {"error": "Grid generation failed. Check parameters."}
             
        # 5. Patch Extraction & Shift Calculation
        # Generate Template
        half_width = [int(period_effect[0]/p_x/2), int(period_effect[1]/p_x/2)]
        
        x_axis_template = np.arange(half_width[1]*2+1) - half_width[1]
        y_axis_template = np.arange(half_width[0]*2+1) - half_width[0]
        XX_template, YY_template = np.meshgrid(x_axis_template, y_axis_template)
        template = np.exp(-(XX_template**2 + YY_template**2)/((hole_size/p_x/2)**2))
        
        dx_list = []
        dy_list = []
        pos_list = []
        intensity_list = []
        
        points_r, points_c = grid_points
        
        # Multiprocessing setup
        num_points = len(points_r)
        # Use fewer workers to avoid overhead for small tasks, but max out for large ones
        num_workers = min(multiprocessing.cpu_count(), 16) 
        chunk_size = max(1, num_points // num_workers)
        
        indices = np.arange(num_points)
        chunks = [indices[i:i + chunk_size] for i in range(0, num_points, chunk_size)]
        
        mean_threshold = np.mean(img_flat) * 0.5
        
        dx_list = []
        dy_list = []
        pos_list = []
        intensity_list = []
        
        # Use ProcessPoolExecutor for parallel processing
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for chunk in chunks:
                futures.append(executor.submit(
                    process_patches_chunk, 
                    chunk, 
                    points_r, 
                    points_c, 
                    img_flat, 
                    half_width, 
                    template, 
                    XX_template, 
                    YY_template, 
                    mean_threshold
                ))
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    dx_chunk, dy_chunk, pos_chunk, int_chunk = future.result()
                    dx_list.extend(dx_chunk)
                    dy_list.extend(dy_chunk)
                    pos_list.extend(pos_chunk)
                    intensity_list.extend(int_chunk)
                except Exception as e:
                    print(f"Error in worker: {e}")

        dx_array = np.array(dx_list)
        dy_array = np.array(dy_list)
        pos_array = np.array(pos_list)
        intensity_array = np.array(intensity_list)
        
        if len(dx_array) == 0:
            return {"error": "No valid spots found."}
            
        # Remove mean tilt (optional, but usually done)
        dx_array -= np.mean(dx_array)
        dy_array -= np.mean(dy_array)
        
        # 6. Interpolation to Dense Map
        # Create grid for interpolation
        h, w = image.shape
        y_grid, x_grid = np.mgrid[0:h, 0:w]
        
        # Interpolate
        # griddata expects points as (N, 2)
        # pos_array is (N, 2) -> (r, c) -> (y, x)
        
        # Use linear interpolation
        dx_map = griddata(pos_array, dx_array, (y_grid, x_grid), method='linear', fill_value=0)
        dy_map = griddata(pos_array, dy_array, (y_grid, x_grid), method='linear', fill_value=0)
        int_map = griddata(pos_array, intensity_array, (y_grid, x_grid), method='linear', fill_value=0)
        
        # Apply mask
        dx_map *= mask
        dy_map *= mask
        int_map *= mask
        
        # Convert to physical units
        # dx in pixels -> um
        disp_x_map = dx_map * pixel_size_um
        disp_y_map = dy_map * pixel_size_um
        
        # 7. Slope & Integration
        slope_x = disp_x_map / (dist_mm * 1000)
        slope_y = disp_y_map / (dist_mm * 1000)
        
        # Integration
        # Note: frankot_chellappa assumes grid step of 1.
        # We need to scale by pixel size?
        # No, frankot_chellappa integrates gradients.
        # If slope is dW/dx, and we integrate over pixels, we get W in units of slope * pixel_size.
        # slope is dimensionless (um/um).
        # So result is in um if we multiply by pixel_size_um.
        
        phase_recon_um = frankot_chellappa(slope_x, slope_y) * pixel_size_um
        phase_recon_um *= mask
        
        # Wavelength
        wavelength_um = 1.2398e-3 / params.get("energy_kev", 10.0)
        phase_recon_rad = phase_recon_um * (2 * np.pi / wavelength_um)
        
        # 8. Zernike & Fitting
        coeffs, fitted_phase, residual = fit_zernike(phase_recon_rad, n_terms=15, mask=mask>0)
        
        _, residual_2nd, poly_coeffs = fit_remove_2nd_order(phase_recon_rad, mask=mask>0)
        residual_2nd[mask==0] = 0
        
        # Calculate Radius of Curvature
        # poly_coeffs: [c0, c1, c2, c3, c4, c5] for [1, x, y, x^2, xy, y^2]
        # x, y are in pixels. Phase is in radians.
        # R = pi * p_x^2 / (lambda * c)
        
        c3 = poly_coeffs[3] # x^2
        c5 = poly_coeffs[5] # y^2
        
        wavelength_m = wavelength_um * 1e-6
        p_x_m = pixel_size_um * 1e-6
        
        if abs(c3) > 1e-10:
            roc_x = (np.pi * p_x_m**2) / (wavelength_m * c3)
        else:
            roc_x = float('inf')
            
        if abs(c5) > 1e-10:
            roc_y = (np.pi * p_x_m**2) / (wavelength_m * c5)
        else:
            roc_y = float('inf')
        
        # Statistics on Residual Phase (within mask)
        if np.any(mask):
            valid_residual = residual_2nd[mask > 0]
            pv_val = np.ptp(valid_residual)
            rms_val = np.std(valid_residual)
        else:
            pv_val = 0.0
            rms_val = 0.0

        return {
            "displacement_x": disp_x_map,
            "displacement_y": disp_y_map,
            "phase_map": phase_recon_rad,
            "phase_residual_2nd": residual_2nd,
            "zernike_coeffs": coeffs,
            "zernike_fitted": fitted_phase,
            "zernike_residual": residual,
            "transmission": int_map,
            "mask": mask,
            "pv_value": pv_val,
            "rms_value": rms_val,
            "roc_x": roc_x,
            "roc_y": roc_y,
            "rotation_angle": np.degrees(angle_error),
            "period_real": [p * 1e6 for p in period_real]
        }

    def propagate_focus(self, phase_map: np.ndarray, params: dict) -> dict:
        # Dummy propagation logic
        dist = params.get("distance_mm", 100.0)
        
        h, w = phase_map.shape
        y, x = np.mgrid[0:h, 0:w]
        cy, cx = h // 2, w // 2
        
        sigma = 8.0
        spot_2d = np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * sigma**2))
        
        cut_xz = np.tile(np.exp(-(np.arange(w) - cx)**2 / (2 * sigma**2)), (100, 1))
        cut_yz = np.tile(np.exp(-(np.arange(h) - cy)**2 / (2 * sigma**2)), (100, 1)).T
        
        return {
            "focus_2d": spot_2d,
            "cut_xz": cut_xz,
            "cut_yz": cut_yz,
            "sigma": sigma,
            "focal_length": dist,
            "strehl": 0.88
        }
