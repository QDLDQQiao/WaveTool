import os
import numpy as np
import argparse
from PIL import Image
import glob
import time
import multiprocessing as ms
import concurrent.futures
from multiprocessing import Pool
from matplotlib import pyplot as plt
import scipy.signal as ssignal
import h5py
import json
import scipy.ndimage as snd
from tqdm import tqdm
import scipy.constants as sc
from matplotlib import cm
from func import load_image, frankotchellappa, write_h5, write_json, prColor
from scipy.interpolate import griddata
from diffraction_process import diffraction_prop

def fft2(img):
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(img)))
    
def ifft2(img):
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(img)))

def fft(img, dire=0):
    return np.fft.fftshift(np.fft.fft(np.fft.ifftshift(img, axes=dire), axis=dire), axes=dire)

def ifft(img, dire=0):
    return np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(img, axes=dire), axis=dire), axes=dire)

def remove_bg(img, width=50):
    return np.fmax(img - np.mean(img[0:width, 0:width]), 0)

def find_rotation_angle(p_x, period, img):
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

def find_disp(Corr_img, XX_axis, YY_axis, sub_resolution=True):
    '''
        find the peak value in the Corr_img
        the XX_axis, YY_axis is the corresponding position for the Corr_img
    '''

    # find the maximal value and postion
    
    pos = np.unravel_index(np.argmax(Corr_img, axis=None), 
                            Corr_img.shape)
    Corr_max = np.amax(Corr_img)
    # Calculate the average to which to compare the signal
    avg = (np.sum(np.abs(Corr_img)) - np.abs(Corr_max)) \
           / (Corr_img.shape[0] * Corr_img.shape[1] -1) if (Corr_img.shape[0] * Corr_img.shape[1] -1) != 0 else 0
    
    # Assign the signal-to-noise
    SN_ratio = Corr_max / avg if avg !=0 else 0
    if(avg):
        SN_ratio = Corr_max / float(avg)
    else:
        SN_ratio = 0.0
    # SN_ratio, Corr_max = 0, 0
    # Compute displacement on both axes
    Corr_img_pad = np.pad(Corr_img, ((1,1), (1,1)), 'edge')
    max_pos_y = pos[0] + 1
    max_pos_x = pos[1] + 1

    dy = (Corr_img_pad[max_pos_y + 1, max_pos_x] - Corr_img_pad[max_pos_y - 1, max_pos_x]) / 2.0
    dyy = (Corr_img_pad[max_pos_y + 1, max_pos_x] + Corr_img_pad[max_pos_y - 1, max_pos_x] 
           - 2.0 * Corr_img_pad[max_pos_y, max_pos_x])

    dx = (Corr_img_pad[max_pos_y, max_pos_x + 1] - Corr_img_pad[max_pos_y, max_pos_x - 1]) / 2.0
    dxx = (Corr_img_pad[max_pos_y, max_pos_x + 1] + Corr_img_pad[max_pos_y, max_pos_x - 1] 
           - 2.0 * Corr_img_pad[max_pos_y, max_pos_x])

    dxy = (Corr_img_pad[max_pos_y + 1, max_pos_x + 1] - Corr_img_pad[max_pos_y + 1, max_pos_x - 1] 
           - Corr_img_pad[max_pos_y - 1, max_pos_x + 1] + Corr_img_pad[max_pos_y - 1, max_pos_x - 1]) / 4.0
    
    if ((dxx * dyy - dxy * dxy) != 0.0):
        det = 1.0 / (dxx * dyy - dxy * dxy)
    else:
        det = 0.0
    # the XX, YY axis resolution
    pixel_res_x = XX_axis[0, 1] - XX_axis[0,0]
    pixel_res_y = YY_axis[1, 0] - YY_axis[0,0]
    Minor_disp_x = (- (dyy * dx - dxy * dy) * det) * pixel_res_x
    Minor_disp_y = (- (dxx * dy - dxy * dx) * det) * pixel_res_y

    # if np.abs(Minor_disp_x) > 1:
    #     # print('x:{}'.format(Minor_disp_x))
    #     Minor_disp_x = 0
    # if np.abs(Minor_disp_y) > 1:
    #     # print('y:{}'.format(Minor_disp_y))
    #     Minor_disp_y = 0
    if sub_resolution:
        disp_x = Minor_disp_x + XX_axis[pos[0], pos[1]]  
        disp_y = Minor_disp_y + YY_axis[pos[0], pos[1]]
    else:
        disp_x = XX_axis[pos[0], pos[1]]  
        disp_y = YY_axis[pos[0], pos[1]]
    
    max_x = XX_axis[0, -1]
    min_x = XX_axis[0, 0]
    max_y = YY_axis[-1, 0]
    min_y = YY_axis[0, 0]

    if disp_x > max_x:
        disp_x = max_y
    elif disp_x < min_x:
        disp_x = min_x

    if disp_y > max_y:
        disp_y = max_y
    elif disp_y < min_y:
        disp_y = min_y

    return disp_y, disp_x, SN_ratio, Corr_max

def extract_subimage(img, pos, half_width):
    corner = np.array([pos[0] - half_width[0], pos[0] + half_width[0]+1, pos[1] - half_width[1], pos[1] + half_width[1]+1])
    if np.sum(np.sum(corner < 0) or corner[1]>img.shape[0] or corner[3]>img.shape[1]):
        # print('skip this point at the boundary')
        return None
    else:
        sub_img = img[corner[0]:corner[1], corner[2]:corner[3]]
        return sub_img

def data_nanMask(data, mask):
    '''
        replace the value outside the mask with nan
    '''
    data_mask = np.copy(data)
    data_mask[np.logical_not(mask)] = np.nan

    return data_mask

def data_removeNaN(data, val=0):
    # replate NaN with val
    data_new = np.copy(data)
    data_new[np.isnan(data_new)] = val
    return data_new

def get_interp_map(mask, points, data, order=1):
    M_shape = mask.shape
    XX_grid_interp, YY_grid_interp = np.meshgrid(np.arange(M_shape[1]), np.arange(M_shape[0]))
    if order == 0:
        method = 'nearest'
    elif order == 1:
        method = 'linear'
    elif order == 2:
        method = 'cubic'
    else:
        print('only 0~2 orders are supported')
    return griddata(points, data, (YY_grid_interp, XX_grid_interp), method=method)

def statistic_lens(data):
    '''
        remove the affect of nan in the array
    '''
    argNotNAN = np.isfinite(data)

    ptp = np.ptp(data[argNotNAN].flatten())

    sigmaError = np.std(data[argNotNAN].flatten())

    return ptp, sigmaError
    
def _lsq_fit_parabola(zz, mask, pixelsize):

    # get the center and the lens size
    xx, yy = np.meshgrid((np.arange(zz.shape[1]) - zz.shape[1]/2) * pixelsize[1],
                         (np.arange(zz.shape[0]) - zz.shape[0]/2) * pixelsize[0])
    select_index = np.where(mask != 0)
    f = zz[select_index]
    x = xx[select_index]
    y = yy[select_index]
    X_matrix = np.vstack([x**2, y**2, x, y, x*y, x*0.0 + 1]).T

    beta_matrix = np.linalg.lstsq(X_matrix, f, rcond=None)[0]

    fit = (beta_matrix[0]*(xx**2) +
           beta_matrix[1]*(yy**2) +
           beta_matrix[2]*xx +
           beta_matrix[3]*yy +
           beta_matrix[4]*xx*yy +
           beta_matrix[5])

    R_x = 1/2/beta_matrix[0]
    R_y = 1/2/beta_matrix[1]
    x_o = -beta_matrix[2]/beta_matrix[0]/2
    y_o = -beta_matrix[3]/beta_matrix[1]/2
    offset = beta_matrix[5]

    popt = [R_x, R_y, x_o, y_o, offset]

    return fit, popt

def wavefront_processing_phase(phase, mask, p_x, wavelength):
    # calculate the phase error
    pv_error_phase, rms_error_phase = statistic_lens(phase / 2 / np.pi)
    print(
        'phase error before fitting: {:.4f} \u03BB (PV); {:4f} \u03BB (RMS)'.format(
            pv_error_phase, rms_error_phase))
    
    phase_2nd_order_lsq, popt = _lsq_fit_parabola(phase, mask, p_x)

    _, popt = _lsq_fit_parabola(1/2/np.pi*phase*wavelength, mask, p_x)

    print(popt)

    print('Curvature Radius of WF x: {:.3g} m'.format(popt[0]))
    print('Curvature Radius of WF y: {:.3g} m'.format(popt[1]))

    phase_err = (phase - phase_2nd_order_lsq)  # [rad]
    phase_err -= np.mean(phase_err * mask)
    
    pv_error_phase, rms_error_phase = statistic_lens(phase_err * mask / 2 / np.pi)
    print(
        'phase error after 2D fitting: {:.4f} \u03BB (PV); {:4f} \u03BB (RMS)'.format(
            pv_error_phase, rms_error_phase))

    return phase_err, phase_2nd_order_lsq, popt

def _lsq_fit_1D(zz, mask, pixelsize, mode='x'):

    # get the center and the lens size
    xx, yy = np.meshgrid((np.arange(zz.shape[1]) - zz.shape[1]/2) * pixelsize[1],
                         (np.arange(zz.shape[0]) - zz.shape[0]/2) * pixelsize[0])
    select_index = np.where(mask != 0)
    f = zz[select_index]
    x = xx[select_index]
    y = yy[select_index]
    if mode == 'x':
        X_matrix = np.vstack([x, x*0.0 + 1]).T
        beta_matrix = np.linalg.lstsq(X_matrix, f, rcond=None)[0]

        fit = (beta_matrix[0]*(xx) + beta_matrix[1])
        
    elif mode == 'y':
        X_matrix = np.vstack([y, y*0.0 + 1]).T
        
        beta_matrix = np.linalg.lstsq(X_matrix, f, rcond=None)[0]

        fit = (beta_matrix[0]*(yy) + beta_matrix[1])

    R = 1/beta_matrix[0]
    offset = beta_matrix[1]

    popt = [R, offset]

    return fit, popt

def wavefront_processing_dpc(dxy, mask, p_x, wavelength, distance):
    
    dx_1st_order_lsq, popt = _lsq_fit_1D(dxy[0], mask, p_x, mode='x')
    dy_1st_order_lsq, popt = _lsq_fit_1D(dxy[1], mask, p_x, mode='y')

    _, popt_x = _lsq_fit_1D(dxy[0] * p_x[1] / distance, mask, p_x, mode='x')
    _, popt_y = _lsq_fit_1D(dxy[1] * p_x[0] / distance, mask, p_x, mode='y')

    print(popt_x, popt_y)

    print('Curvature Radius of WF x: {:.3g} m'.format(popt_x[0]))
    print('Curvature Radius of WF y: {:.3g} m'.format(popt_y[0]))

    dx_err = (dxy[0] - dx_1st_order_lsq) * mask# [rad]
    dx_err -= np.mean(dx_err)
    dy_err = (dxy[1] - dy_1st_order_lsq) * mask# [rad]
    dy_err -= np.mean(dy_err)

    return [dx_err, dy_err], [dx_1st_order_lsq, dy_1st_order_lsq], [popt_x, popt_y]

def FFT_filter(img, period_px):
    """
        phase retrieval for 2D grating
    """
    M_shape = img.shape
    
    # calculate the theoretical position of the hamonics
    period_harm_Vert = np.int16(1/period_px*M_shape[0])
    period_harm_Hor = np.int16(1/period_px*M_shape[1])
    correct_peak = True

    searchRegion = int(period_harm_Hor/2 + period_harm_Vert/2)

    img_fft = fft2(img)

    def extract_subimage(im_fft, Harm=(0, 0)):
        """
        docstring
        """
        idx_peak = [M_shape[0] // 2 + Harm[0] * period_harm_Vert, M_shape[1] // 2 + Harm[1] * period_harm_Hor]
        # find center for vertical
        maskSearchRegion = np.zeros(M_shape)

        maskSearchRegion[idx_peak[0] - searchRegion:
                        idx_peak[0] + searchRegion+1,
                        idx_peak[1] - searchRegion:
                        idx_peak[1] + searchRegion+1] = 1.0

        idxPeak_ij_exp = np.where(np.abs(im_fft) * maskSearchRegion ==
                                np.max(np.abs(im_fft) * maskSearchRegion))                    
        print('error of harmonic position: {}(vertical) {}(horizontal)'.format((idxPeak_ij_exp[0][0]-idx_peak[0]), (idxPeak_ij_exp[1][0]-idx_peak[1])))
        if correct_peak:
            idx_peak = [idxPeak_ij_exp[0][0], idxPeak_ij_exp[1][0]]
        sub_img_fft = im_fft[idx_peak[0] - period_harm_Vert//2:idx_peak[0] + period_harm_Vert//2+1,
                            idx_peak[1] - period_harm_Hor//2:idx_peak[1] + period_harm_Hor//2+1]
        return ifft2(sub_img_fft)

    img_00 = extract_subimage(img_fft, Harm=(0, 0))
    int00 = np.abs(img_00)

    return snd.zoom(int00, zoom=(M_shape[0]/int00.shape[0], M_shape[1]/int00.shape[1]))

def grid_generate(mask, img_flat, p_x, hole_period, period_effect, angle_error):
    
    x_axis = np.arange(mask.shape[1])
    y_axis = np.arange(mask.shape[0])

    center_mask = snd.center_of_mass(img_flat * mask)
    print(center_mask)

    sub_img = img_flat[int(center_mask[0]-period_effect[0]/2/p_x):int(center_mask[0]+period_effect[0]/2/p_x),
                    int(center_mask[1]-period_effect[1]/2/p_x):int(center_mask[1]+period_effect[1]/2/p_x)]
    center_pos = np.array(center_mask) + np.array(snd.center_of_mass(sub_img)) - np.array(period_effect)/2/p_x

    center_sub_img = img_flat[int(center_pos[0]-period_effect[0]/2/p_x):int(center_pos[0]+period_effect[0]/2/p_x),
                    int(center_pos[1]-period_effect[1]/2/p_x):int(center_pos[1]+period_effect[1]/2/p_x)]

    peak_pos = np.unravel_index(np.argmax(center_sub_img, axis=None), center_sub_img.shape)
    print('peak_pos:', peak_pos)

    center_pos = np.array([center_pos[0] + peak_pos[0]-int(period_effect[0]/2/p_x), 
                        center_pos[1] + peak_pos[1]-int(period_effect[1]/2/p_x)])

    a_x = np.flip(np.arange(center_pos[1], -mask.shape[1]/2, -period_effect[1]/p_x))
    b_x = np.arange(center_pos[1], img_flat.shape[1]*1.5, period_effect[1]/p_x)
    x_grid = np.concatenate([a_x, b_x]) - center_pos[1]

    a_y = np.flip(np.arange(center_pos[0], -mask.shape[0]/2, -period_effect[0]/p_x))
    b_y = np.arange(center_pos[0], img_flat.shape[0]*1.5, period_effect[0]/p_x)
    y_grid = np.concatenate([a_y, b_y]) - center_pos[0]

    theta_rot = -angle_error
    XX_grid, YY_grid = np.meshgrid(x_grid, y_grid)
    # XX_grid_rot = center_pos[1] + (XX_grid * np.sin(theta_rot) + YY_grid * np.cos(theta_rot) )
    # YY_grid_rot = center_pos[0] + (XX_grid * np.cos(theta_rot) - YY_grid * np.sin(theta_rot) )
    XX_grid_rot = center_pos[1] + (XX_grid * np.cos(theta_rot) + YY_grid * np.sin(theta_rot) )
    YY_grid_rot = center_pos[0] + (-XX_grid * np.sin(theta_rot) + YY_grid * np.cos(theta_rot) )
    
    grid_nrow, grid_ncol = XX_grid.shape

    select_index = np.where((XX_grid_rot>=0) * (XX_grid_rot<=mask.shape[1]) * (YY_grid_rot>=0) * (YY_grid_rot<=mask.shape[0]) == 1)

    XX_grid_rot = (XX_grid_rot[select_index])
    YY_grid_rot = (YY_grid_rot[select_index])

    # the XX_grid_rot and YY_grid_rot above are calculated from the calculated effective period, so here need to add the difference of the effective period to the real design plate period
    x_grid_diff = (np.arange(grid_ncol) - grid_ncol/2) * (period_effect[1] - hole_period)/p_x
    y_grid_diff = (np.arange(grid_nrow) - grid_nrow/2) * (period_effect[0] - hole_period)/p_x

    XX_grid_diff, YY_grid_diff = np.meshgrid(x_grid_diff, y_grid_diff)

    # XX_grid_rot_diff = (XX_grid_diff * np.sin(theta_rot) + YY_grid_diff * np.cos(theta_rot) )
    # YY_grid_rot_diff = (XX_grid_diff * np.cos(theta_rot) - YY_grid_diff * np.sin(theta_rot) )

    XX_grid_rot_diff = (XX_grid_diff * np.cos(theta_rot) + YY_grid_diff * np.sin(theta_rot) )
    YY_grid_rot_diff = (-XX_grid_diff * np.sin(theta_rot) + YY_grid_diff * np.cos(theta_rot) )

    XX_grid_rot_diff_list = (XX_grid_rot_diff[select_index])
    YY_grid_rot_diff_list = (YY_grid_rot_diff[select_index])

    return [XX_grid_rot, YY_grid_rot], [XX_grid_rot_diff_list, YY_grid_rot_diff_list], [XX_grid_diff, YY_grid_diff], center_pos

class Extract_patches:
    def __init__(self, half_width, n_cores, hole_size, hole_period, period_effect) -> None:
        self.half_width = half_width
        self.n_cores = n_cores
        self.hole_size = hole_size
        self.hole_period = hole_period
        self.period_effect = period_effect
        prColor('use {} cpu cores'.format(self.n_cores), 'cyan')
        self.template, self.XX_template, self.YY_template = self.generate_template()

    def generate_template(self):
        # calculate the relative movement of each hole
        # use the average image as template
        # average_sub_img = np.mean(sub_img_list, axis=0)

        # generate template distribution
        x_axis_template = np.arange(self.half_width[1]*2+1) - self.half_width[1]
        y_axis_template = np.arange(self.half_width[0]*2+1) - self.half_width[0]
        XX_template, YY_template = np.meshgrid(x_axis_template, y_axis_template)
        template = np.exp(-(XX_template**2 + YY_template**2)/(self.hole_size/2)**2)
        return template, XX_template, YY_template

    def find_shift(self, sub_img, template_img, XX, YY, method='corr'):
        if method == 'corr':
            # use correlation to find he displacement 
            # Corr_img = ssignal.correlate2d(sub_img, template_img) / (np.std(sub_img) * np.std(template_img))
            Corr_img = np.abs(ifft2(fft2(sub_img-np.mean(sub_img)) * fft2(template_img-np.mean(template_img)))) / (np.std(sub_img) * np.std(template_img))/(sub_img.shape[0]*sub_img.shape[1])
            dy, dx, SN_ratio, max_corr = find_disp(
                        Corr_img, XX, YY, sub_resolution=True)
            
            return dy, dx, SN_ratio, max_corr, Corr_img
        elif method == 'central_mass':
            center_pos = snd.center_of_mass(sub_img)
            half_width = sub_img.shape[0]//2
            return center_pos[0] - half_width, center_pos[1] - half_width, 1, 1, None 
    
    def extract_patches(self, img, img_flat, mask, point_pos_list, point_pos_cen_subpixel_list, disp_diff_list, id_list, id_sort):
            
        # patch_mask_list = []
        # patch_list = []
        # patch_raw_list = []
        point_pos_list_select = []
        point_pos_cen_subpixel_list_select = []
        disp_diff_list_select = []
        intensity_list = []
        dy_list = []
        dx_list = []
        SN_ratio_list = []
        max_corr_list = []

        for kk, point_pos in tqdm(enumerate(point_pos_list)):
            patch = extract_subimage(img_flat, point_pos, self.half_width)
          
            if patch is not None:
                patch_mask = extract_subimage(img_flat*mask, point_pos, self.half_width)
                if np.sum(patch_mask) != 0:
                    patch_raw = extract_subimage(img, point_pos, self.half_width)

                    dy, dx, SN_ratio, max_corr, Corr_image = self.find_shift(patch, self.template, self.XX_template, self.YY_template, method='corr')

                    # dx_list.append(dx + point_pos_cen_subpixel_list[kk][1] + disp_diff_list[kk][1])
                    # dy_list.append(dy + point_pos_cen_subpixel_list[kk][0] + disp_diff_list[kk][0])
                    dx_list.append(dx + point_pos_cen_subpixel_list[kk][1])
                    dy_list.append(dy + point_pos_cen_subpixel_list[kk][0])
                    SN_ratio_list.append(SN_ratio)
                    max_corr_list.append(max_corr)

                    point_pos_list_select.append(point_pos)
                    # record subpixel center movement 
                    point_pos_cen_subpixel_list_select.append(point_pos_cen_subpixel_list[kk])
                    # record the difference between effective period and design period
                    disp_diff_list_select.append(disp_diff_list[kk])
                    # get the relative intensity change
                    intensity_list.append(np.sum(patch_raw))

        return dx_list, dy_list, intensity_list, point_pos_list_select, disp_diff_list_select, id_list, id_sort

    def extract_patches_multiprocess(self, flat, mask, img, grid_rot, grid_diff_list):
        """
        use multi-process to accelerate mini-patches extraction from image
        img is in a shape of [H, W]

        Args:
            img (ndarray): image data
            n_cores (int): number of cpu cores used
            wavelet_method (str, optional): wavelets used. Defaults to 'db6'.
            w_level (int, optional): wavelet level. Defaults to 1.
            return_level (int, optional): return wavelet level. Defaults to 1.

        Returns:
            wavelet coefficients, level name
        """

        cores = ms.cpu_count()
        prColor('Computer available cores: {}'.format(cores), 'green')

        if cores > self.n_cores:
            cores = self.n_cores
        else:
            cores = ms.cpu_count()
        prColor('Use {} cores'.format(cores), 'light_purple')
        n_tasks = cores

        # split the points into small groups
        point_num = grid_rot[0].shape[0]
        y_axis = np.arange(point_num)

        chunks_idx_y = np.array_split(y_axis, n_tasks)

        dim = img.shape

        # use CPU parallel to calculate
        result_list = []

        with concurrent.futures.ProcessPoolExecutor(
                    max_workers=cores) as executor:

                futures = []
                for id_sort, y_list in enumerate(chunks_idx_y):
                    point_pos_patch = [[int(grid_rot[1][kk]), int(grid_rot[0][kk])] for kk in y_list]
                    point_pos_cen_subpixel_patch = [[grid_rot[0][kk] - int(grid_rot[0][kk]), grid_rot[1][kk] - int(grid_rot[1][kk])] for kk in y_list]
                    disp_diff = [[grid_diff_list[1][kk], grid_diff_list[0][kk]] for kk in y_list]

                    futures.append(
                        executor.submit(self.extract_patches, img, flat, mask, point_pos_patch, point_pos_cen_subpixel_patch, disp_diff, y_list, id_sort))

                for future in concurrent.futures.as_completed(futures):
                    result_list.append(future.result())
                    # display the status of the program
                    Total_iter = cores
                    Current_iter = len(result_list)
                    percent_iter = Current_iter / Total_iter * 100
                    str_bar = '>' * (int(np.ceil(percent_iter / 2))) + ' ' * (int(
                        (100 - percent_iter) // 2))
                    prColor(
                        '\r' + str_bar + 'processing: [%3.1f%%] ' % (percent_iter),
                        'purple')
        
        dx_list = [item[0] for item in result_list]
        dy_list = [item[1] for item in result_list]
        intensity_list = [item[2] for item in result_list]
        point_pos_list_select = [item[3] for item in result_list]
        disp_diff_list_select = [item[4] for item in result_list] 
        id_list = [item[5] for item in result_list]
        id_sort = [item[6] for item in result_list]
        # print('before sorting: ', [id[0] for id in id_list])

        sort_order = lambda x: [i for _,i in sorted(zip(id_sort,x))]
        flatten_list = lambda x: [j for sub in x for j in sub]

        dx_list = flatten_list(sort_order(dx_list))
        dy_list = flatten_list(sort_order(dy_list))
        intensity_list = flatten_list(sort_order(intensity_list))
        point_pos_list_select = flatten_list(sort_order(point_pos_list_select))
        disp_diff_list_select = flatten_list(sort_order(disp_diff_list_select))
        id_list = sort_order(id_list)

        # print(len(dx_list))
        # print('after sorting: ', [id[0] for id in id_list])
        # print(id_sort)
        # for y, dx, dy in zip(y_list, disp_x_list, disp_y_list):
        #     disp_x[y, :] = dx
        #     disp_y[y, :] = dy

        return dx_list, dy_list, intensity_list, point_pos_list_select, disp_diff_list_select
    
    def reconstruct(self, img_flat, mask, img, grid_rot, grid_diff_list, interp_order=1):

        dx_list, dy_list, intensity_list, point_pos_list, disp_diff_list = self.extract_patches_multiprocess(img_flat, mask, img, grid_rot, grid_diff_list)

        # plot the dx and dy with different color
        dx_array = np.array(dx_list)
        dy_array = np.array(dy_list)
        intensity_list = np.array(intensity_list)
        point_pos_list = np.array(point_pos_list)
        
        intensity_list -= np.amin(intensity_list)
        dx_array -= np.mean(dx_array)
        dy_array -= np.mean(dy_array)

        # interpolation to the original size
        dx_interp = get_interp_map(mask, point_pos_list, dx_array, order=interp_order) * mask
        dy_interp = get_interp_map(mask, point_pos_list, dy_array, order=interp_order) * mask
        dx_interp = data_removeNaN(dx_interp, val=0)
        dy_interp = data_removeNaN(dy_interp, val=0)

        intensity_interp = get_interp_map(mask, point_pos_list, intensity_list, order=interp_order) * mask
        intensity_interp = data_removeNaN(intensity_interp, val=0)

        # magnification factor
        M_factor = [self.period_effect[0] / self.hole_period, self.period_effect[1] / self.hole_period]
        prColor('Magnification factor estimated by hole period: {} V {} H'.format(M_factor[0], M_factor[1]), 'green')

        dx_mag = (np.arange(mask.shape[1]) - mask.shape[1]/2) * (M_factor[1] -1)
        dy_mag = (np.arange(mask.shape[0]) - mask.shape[0]/2) * (M_factor[0] -1)

        XX_dx_mag, YY_dy_mag = np.meshgrid(dx_mag, dy_mag)

        # dx = data_removeNaN(mask * (dx_interp + XX_dx_mag))
        # dy = data_removeNaN(mask * (dy_interp + YY_dy_mag))
        dx = dx_interp + XX_dx_mag
        dy = dy_interp + YY_dy_mag
        dx -= np.mean(dx)
        dy -= np.mean(dy)
        
        return [dx, dy], [dx_interp, dy_interp], [XX_dx_mag, YY_dy_mag],  intensity_interp, [dx_array, dy_array], point_pos_list, intensity_list

if __name__ == "__main__":
    # paremater settings
    parser = argparse.ArgumentParser(
        description='experimental data analysis for Hartman wavefront sensing',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # shared args
    # ============================================================
    parser.add_argument('--img',
                        type=str,
                        default='./testdata/Image40.tif',
                        help='path to image')

    parser.add_argument('--result_folder',
                        type=str,
                        default='./testdata/result_folder',
                        help='saving folder')

    parser.add_argument('--p_x',
                        default=3.25e-6,
                        type=float,
                        help='pixel size')
    parser.add_argument('--energy',
                        default=420,
                        type=float,
                        help='X-ray energy')
    parser.add_argument('--distance',
                        default=445e-3,
                        type=float,
                        help='detector to Hartman plate distance')
    
    parser.add_argument('--fitting_2D',
                        action='store_true',
                        default=False,
                        help='do 2D fitting or not')
    
    parser.add_argument('--fitting_2D_method',
                        default='phase',
                        type=str,
                        help='use phase or dpc for 2D fitting')

    parser.add_argument('--down_sampling', type=float, default=1, help='down-sample images to reduce memory cost and accelerate speed.')

    parser.add_argument('--hole_period', type=float, default=120e-6, help='Hartman plate hole period')

    parser.add_argument('--hole_size', type=float, default=30e-6, help='Hartman plate hole size')

    parser.add_argument('--mask_threshold', type=float, default=0.2, help='mask threshold to select the effective hole area')

    parser.add_argument('--nCores',
                        default=4,
                        type=int,
                        help='number of CPU cores used for calculation.')

    parser.add_argument('--show_figure',
                        action='store_true',
                        default=False,
                        help='show figure or not')
    
    parser.add_argument('--post_processing',
                        action='store_true',
                        default=False,
                        help='do post processing for wavefront curvature or not')

    
    args = parser.parse_args()

    for key, value in args.__dict__.items():
        prColor('{}: {}'.format(key, value), 'cyan')
    
    if not os.path.exists(args.result_folder):
        os.makedirs(args.result_folder)
    
    # save setting files into 
    write_json(args.result_folder, 'setting', args.__dict__)

    wl = sc.value('inverse meter-electron volt relationship') / args.energy

    # load data and do pre-processing
    img = load_image(args.img)
    # img = remove_bg(img)

    img_filter = snd.gaussian_filter(img, sigma=int(args.hole_size/args.p_x/4))

    flat = snd.gaussian_filter(img_filter, int(args.hole_period/args.p_x))

    img_flat = img_filter / flat

    img_filter2 = snd.maximum_filter(img_flat, size=(int(args.hole_period/args.p_x), int(args.hole_period/args.p_x)))
    img_filter2 = snd.gaussian_filter(img_filter2, int(args.hole_period/args.p_x))

    mask = np.ones(img_filter2.shape) * ((img_filter2-img_filter2.min()) > args.mask_threshold*(img_filter2.max()-img_filter2.min()))
    
    plt.figure(figsize=(10,8))
    plt.subplot(221)
    plt.imshow(img_filter)
    plt.colorbar()
    plt.title('filtered image')
    plt.subplot(222)
    plt.imshow(img_flat)
    plt.colorbar()
    plt.title('flat-corr image')
    plt.subplot(223)
    plt.imshow(mask * img_flat)
    plt.colorbar()
    plt.title('masked flat-corr image')
    plt.subplot(224)
    plt.imshow(mask)
    plt.colorbar()
    plt.title('mask image')
    plt.savefig(os.path.join(args.result_folder, 'image_preprocess.png'))

    if args.show_figure:
        plt.show()

    '''
        the calculated period_real is the real hole distance, so use the calculated hole distance to calculate the hole displacement to implement the ability for highly focused beam
    '''
    angle_error, period_real = find_rotation_angle(args.p_x, args.hole_period, img_flat)

    print('\nCalculated real period from the image is: {}V {}H'.format(period_real[0]/args.p_x, period_real[1]/args.p_x), ' pixels')

    # use the calculated period to find the peak displacement
    # period_effect = period_real
    period_effect = [period_real[0], period_real[1]]
    # period_effect = peak_period
    print('use calculated hole period: {}V px {}H px {}V um {}H um'.format(period_effect[0]/args.p_x, period_effect[1]/args.p_x, period_effect[0]*1e6, period_effect[1]*1e6))

    grid_rot, grid_diff_list, grid_diff, center_pos = grid_generate(mask, img_flat, args.p_x, args.hole_period, period_effect, angle_error)

    plt.figure()
    plt.plot(grid_rot[0], img_flat.shape[0] - grid_rot[1], 'b*', ms=0.5)
    plt.figure(figsize=(10,9))
    plt.imshow(img_flat * mask)
    plt.plot(center_pos[1], center_pos[0], 'r*')
    plt.plot(grid_rot[0], grid_rot[1], 'w.', ms=1)
    plt.plot(grid_rot[0][100], grid_rot[1][100], 'r*')
    plt.savefig(os.path.join(args.result_folder, 'grid_peak.png'))
    if args.show_figure:
        plt.show()
    
    # get the sub-image around the searched grid points
    point_num = grid_rot[0].shape[0]
    print('total point number: ', point_num)

    half_width = [int(period_effect[0]/args.p_x/2), int(period_effect[1]/args.p_x/2)]

    img_patch_extractor = Extract_patches(half_width=half_width, n_cores=args.nCores, hole_size=args.hole_size, hole_period=args.hole_period, period_effect=period_effect)

    [dx, dy], [dx_interp, dy_interp], [dx_mag, dy_mag],  intensity_interp, [dx_raw, dy_raw], point_pos_list, intensity_list = img_patch_extractor.reconstruct(img_flat, mask, img, grid_rot, grid_diff_list, interp_order=2)

    print('None zero points:', len(point_pos_list))

    DPC_y = (dy) * args.p_x / args.distance
    DPC_x = (dx) * args.p_x / args.distance

    phase = frankotchellappa(DPC_x, DPC_y) * args.p_x * 2 * np.pi / wl

    phase -= np.mean(phase)

    color_range = [np.amin([dx, dy]), np.amax([dx, dy])]
    
    plt.figure(figsize=(15, 8))

    plt.subplot(231)
    plt.imshow(dx * mask, cmap='plasma')
    plt.colorbar()
    plt.clim(color_range)
    plt.title('fitted final dx')
    plt.subplot(232)
    plt.imshow(dy * mask, cmap='plasma')
    plt.colorbar()
    plt.clim(color_range)
    plt.title('fitted final dy')
    plt.subplot(233)
    plt.imshow(intensity_interp * mask, cmap='plasma')
    plt.colorbar()
    plt.title('fitted intensity')

    plt.subplot(234)
    plt.xlim([0, img_flat.shape[1]])
    plt.ylim([0, img_flat.shape[0]])
    plt.scatter(point_pos_list[:, 1], img_flat.shape[0] - point_pos_list[:, 0], c=np.clip(dx_raw, a_min=color_range[0], a_max=color_range[1]), s=8, cmap='plasma')
    plt.colorbar()
    plt.title('raw dx')
    plt.subplot(235)
    plt.xlim([0, img_flat.shape[1]])
    plt.ylim([0, img_flat.shape[0]])
    plt.scatter(point_pos_list[:, 1], img_flat.shape[0] - point_pos_list[:, 0], c=np.clip(dy_raw, a_min=color_range[0], a_max=color_range[1]), s=8, cmap='plasma')
    plt.colorbar()
    plt.title('raw dy')

    plt.subplot(236)
    plt.xlim([0, img_flat.shape[1]])
    plt.ylim([0, img_flat.shape[0]])
    plt.scatter(point_pos_list[:, 1], img_flat.shape[0] - point_pos_list[:, 0], c=intensity_list, s=8, cmap='plasma')
    plt.colorbar()
    plt.title('raw int')
    plt.savefig(os.path.join(args.result_folder, 'displacement_processed.png'))

    if args.show_figure:
        plt.show()

    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.imshow(data_nanMask(phase, mask), cmap='plasma')
    plt.colorbar()
    plt.title('Integrated phase')
    plt.subplot(122)
    plt.imshow(data_nanMask(intensity_interp, mask), cmap='plasma')
    plt.colorbar()
    plt.title('Interp intensity')
    plt.savefig(os.path.join(args.result_folder, 'phase_preprocess.png'))
    
    if args.show_figure:
        plt.show()   
    #  post-process the phase and get the residual
    if args.fitting_2D:

        if args.fitting_2D_method == 'phase':
            prColor('Use phase for 2D fitting', 'cyan')
            phase_err, phase_fit_2nd, popt = wavefront_processing_phase(phase, mask, [args.p_x, args.p_x], wl)

            pv_error_phase, rms_error_phase = statistic_lens(phase_err * mask /2/np.pi)
            Curve_x = popt[0]
            Curve_y = popt[1]
            print('Curvature X: {}\nCurvature Y: {}'.format(Curve_x, Curve_y))
        else:
            prColor('Use dpc for 2D fitting', 'cyan')
            [dx_err, dy_err], [dx_1st_order_lsq, dy_1st_order_lsq], [popt_x, popt_y] = wavefront_processing_dpc([dx, dy], mask, [args.p_x, args.p_x], wl, args.distance)

            Curve_x = popt_x[0]
            Curve_y = popt_y[0]
            print('Curvature X: {}\nCurvature Y: {}'.format(Curve_x, Curve_y))

            phase_err = frankotchellappa(dx_err, dy_err) * args.p_x **2 * 2 * np.pi / wl / args.distance
            phase_err -= np.mean(phase_err)
            pv_error_phase, rms_error_phase = statistic_lens(phase_err * mask /2 /np.pi)
            print(
                'phase error after 2D fitting: {:.4f} \u03BB (PV); {:4f} \u03BB (RMS)'.format(
                    pv_error_phase, rms_error_phase))
            phase_fit_2nd = frankotchellappa(dx_1st_order_lsq, dy_1st_order_lsq) * args.p_x **2 * 2 * np.pi / wl / args.distance
            phase_fit_2nd -= np.mean(phase_fit_2nd)
           
        # get the center and the lens size
        XX, YY = np.meshgrid(np.arange(phase.shape[1]),
                                np.arange(phase.shape[0]))

        '''
            plot the figure and save
        '''
        extent_data = np.array([
            -phase.shape[1] / 2 * args.p_x, phase.shape[1] / 2 * args.p_x,
            -phase.shape[0] / 2 * args.p_x, phase.shape[0] / 2 * args.p_x
        ])
        fig=plt.figure(figsize=(15,10))
        plt.subplot(221)
        plt.imshow(data_nanMask(phase / 2 / np.pi, mask),
                    cmap=cm.get_cmap('plasma'),
                    interpolation='bilinear',
                    extent=extent_data * 1e6,
                    aspect='equal')
        plt.xlabel('x [$\mu m$]', fontsize=14)
        plt.ylabel('y [$\mu m$]', fontsize=14)
        cbar = plt.colorbar()
        cbar.set_label('\u03BB', rotation=90, fontsize=14)
        plt.title('2D_phase')

        plt.subplot(222)
        plt.imshow(data_nanMask(phase_fit_2nd / 2 / np.pi, mask),
                    cmap=cm.get_cmap('plasma'),
                    interpolation='bilinear',
                    extent=extent_data * 1e6,
                    aspect='equal')
        plt.xlabel('x [$\mu m$]', fontsize=14)
        plt.ylabel('y [$\mu m$]', fontsize=14)
        cbar = plt.colorbar()
        cbar.set_label('\u03BB', rotation=90, fontsize=14)
        plt.title('Fitted 2D_phase')

        plt.subplot(223)
        plt.imshow(data_nanMask(phase_err / 2 / np.pi , mask),
                    cmap=cm.get_cmap('plasma'),
                    interpolation='bilinear',
                    extent=extent_data * 1e6,
                    aspect='equal')
        plt.xlabel('x [$\mu m$]', fontsize=14)
        plt.ylabel('y [$\mu m$]', fontsize=14)
        cbar = plt.colorbar()
        cbar.set_label('\u03BB', rotation=90, fontsize=14)
        plt.title('2D_phase error')

        # fig = plt.figure(figsize=(14,8))
        ax1 = fig.add_subplot(224, projection='3d')
        ax1.plot_surface(XX, YY, data_nanMask(phase_err, mask), cmap=cm.get_cmap('plasma'))
        ax1.set_xlabel('x [$\mu m$]')
        ax1.set_ylabel('y [$\mu m$]')
        ax1.set_zlabel('phase error [$rad$]')
        plt.tight_layout()
        plt.savefig(os.path.join(args.result_folder, 'phase_error.png'))
        
        if args.show_figure:
            plt.show()

    data_save = {'dx': dx, 'dy': dy, 'dx_raw': dx_raw, 'dy_raw': dy_raw, 'phase': phase, 'mask':mask, 'pos_list_raw': point_pos_list, 'intensity_list':intensity_list, 'intensity': intensity_interp}
    para_json = {
        'energy': args.energy,
        'p_x': args.p_x,
        'distance': args.distance,
        'period_real': period_real,
        'angle_err': angle_error,
        'mask_threshold': args.mask_threshold
    }
    if args.fitting_2D:
        data_save.update({'phase_err': phase_err, 'intensity_int': intensity_interp})
        para_json.update({'curve_x': Curve_x, 'curve_y': Curve_y, 'phase_err_rms': rms_error_phase, 'phase_err_PV':pv_error_phase})
            
    write_h5(args.result_folder, 'result_Hartman', data_dict=data_save)
    write_json(args.result_folder, 'result_Hartman_para', para_json)
