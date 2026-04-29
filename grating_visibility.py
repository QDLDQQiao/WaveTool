import os
import sys
import json
import numpy as np
import argparse
import glob
from tqdm import tqdm 
import scipy.ndimage as snd
from matplotlib import pyplot as plt
from func import load_image, prColor, write_json
from gui_func import crop_gui, gui_load_data

def fft2(img):
        return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(img)))
    
def ifft2(img):
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(img)))

def get_visibility(img, corner, method='FFT'):
    """
    get_visibility get the grating visibility

    Args:
        img (_type_): _raw image
        corner (_type_): FFT peak position
        method (str, optional): Defaults to 'FFT'.
                                FFT: use fourier peak ratio
                                Filter: uniform filter to get contrast

    Returns:
        _type_: _description_
    """
    if method == 'FFT':
        img_fft = np.abs(fft2(img))
        Int_max = np.amax(img_fft)

        img_fft_sub = img_fft[int(corner[0][0]):int(corner[1][0]),
                    int(corner[0][1]):int(corner[1][1])]
        Int_peak = np.amax(img_fft_sub)

        # find the peak position
        idxPeak_ij_exp = np.where(np.abs(img_fft_sub) == np.max(np.abs(img_fft_sub)))
        peak_pos = [int(corner[0][0] + idxPeak_ij_exp[0] - img_fft.shape[0]//2), int(corner[0][1] + idxPeak_ij_exp[1] - img_fft.shape[1]//2)]
        # print(peak_pos)
        # print(Int_peak, Int_max)
        return Int_peak/Int_max, peak_pos
    else:
        img_fft = np.abs(fft2(img))
        img_fft_sub = img_fft[int(corner[0][0]):int(corner[1][0]),
                    int(corner[0][1]):int(corner[1][1])]
        idxPeak_ij_exp = np.where(np.abs(img_fft_sub) == np.max(np.abs(img_fft_sub)))
        peak_pos = [int(corner[0][0] + idxPeak_ij_exp[0] - img_fft.shape[0]//2), int(corner[0][1] + idxPeak_ij_exp[1] - img_fft.shape[1]//2)]
        # print(corner, idxPeak_ij_exp, peak_pos)
        gt_period_est = np.amax(peak_pos)
        filter_sigma = [int(img_fft.shape[0]/gt_period_est)*2, int(img_fft.shape[1]/gt_period_est)*2]
        img_filtered = snd.uniform_filter(img, size=filter_sigma)
        img_gt = img_first - img_filtered

        # print(filter_sigma)
        # plt.figure()
        # plt.subplot(131)
        # plt.imshow(img)
        # plt.colorbar()
        # plt.subplot(132)
        # plt.imshow(img_filtered)
        # plt.colorbar()
        # plt.subplot(133)
        # plt.imshow(img_gt)
        # plt.colorbar()
        # plt.show()

        return np.std(img_gt) / np.mean(img_filtered), peak_pos

if __name__ == "__main__":
    """
    get_visibility:
        calculate the grating visibility along the Z scanning direction
        N : repeated acquired images at each scanning position
        z0: starting position of the Z scan
        z_step: step size of the Z scan

    """
    # file_list = sorted(glob.glob(os.path.join('E:/SXFEL_Grating_20230520/20230520/rough_scan/', '*mm.tif*')), key=lambda f: int(os.path.basename(f).split('.')[0].split('mm')[0].split('_')[-1]))

    file_list = glob.glob(os.path.join('/Volumes/dataFat32/scan_00053', '*.tif*'))
    file_list.sort(key=lambda x: int(os.path.basename(x).split('.')[0].split('_')[-1]))

    result_folder = os.path.dirname(file_list[0])
    print(file_list)
    # bg_file = 'D:/data_qiaozhi/2023/wfs202311/20231118/scan_Z/beijing29_1.tif'
    # bg_img = load_image(bg_file)
    bg_img = 0
    print('bg mean: ', np.mean(bg_img))
    method = 'FFT' #FFT: fourier method; Filter: contrast method
    print('file number: ', len(file_list))
    N = 10
    z0 = -9
    z_step = 0.5
    p_x = 6.5e-6
    # z_pos = z0 + np.floor(np.arange(len(file_list)) / N) * z_step
    n_test = 11

    img_first = load_image(file_list[n_test]) - bg_img
    _, corner_crop = crop_gui(img_first)
    plt.show()
    img_first = img_first[int(corner_crop[0][0]):int(corner_crop[1][0]),
                int(corner_crop[0][1]):int(corner_crop[1][1])]
    
    img_fft_first = fft2(img_first)
    _, corner_gt = crop_gui(np.log(np.abs(img_fft_first)))
    plt.show()
    
    visibility = []
    gt_peak = []
    for f in tqdm(file_list):
        img = load_image(f) - bg_img
        
        # img -= 99
        img = img[int(corner_crop[0][0]):int(corner_crop[1][0]),
                int(corner_crop[0][1]):int(corner_crop[1][1])]
        temp, peak_pos = get_visibility(img, corner_gt, method)
        # print(temp)
        gt_peak.append([p_x * img.shape[1] / np.sqrt(peak_pos[0]**2 + peak_pos[1]**2)])
        visibility.append(temp)
    
    N_num = int(len(visibility)/N)
    vis_array = np.array(np.array_split(np.array(visibility[0:N_num*N]), N_num))

    vis_array = np.mean(vis_array, axis=1)

    gt_peak_avg = np.array(np.array_split(np.array(gt_peak[0:N_num*N]), N_num))

    gt_peak_avg = np.mean(gt_peak_avg, axis=1)

    z_pos = z0 + z_step * np.arange(N_num)
    
    result = {
        #  'bg': bg_file,
         'p_x': p_x,
         'N_repeat': N,
         'z_pos': z_pos.tolist(),
         'visibility': vis_array.tolist(),
         'gt_period': gt_peak,

    }
    
    write_json(result_folder, 'processed_result', result)

    plt.figure(figsize=(12,8))
    plt.plot(z_pos, vis_array, '--*')
    plt.grid()
    plt.xlabel('Z scan [mm]')
    plt.ylabel('visibility')
    plt.savefig(os.path.join(result_folder, 'visibility_scan.png'), dpi=150)

    plt.figure(figsize=(12,8))
    plt.plot(z_pos, gt_peak_avg, '-d')
    plt.grid()
    plt.xlabel('Z scan [mm]')
    plt.ylabel('gt_period')
    plt.savefig(os.path.join(result_folder, 'period_scan.png'), dpi=150)
    plt.show()

