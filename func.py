import numpy as np
import os
import sys
from PIL import Image
import h5py
import json
from matplotlib import pyplot as plt
import matplotlib
font = {'family' : 'sans-serif',
        # 'weight' : 'bold',
        'size'   : 14}

matplotlib.rc('font', **font)


def prColor(word, color_type):
    ''' function to print color text in terminal
        input:
            word:           word to print
            color_type:     which color
                            'red', 'green', 'yellow'
                            'light_purple', 'purple'
                            'cyan', 'light_gray'
                            'black'
    '''
    end_c = '\033[00m'
    if color_type == 'red':
        start_c = '\033[91m'
    elif color_type == 'green':
        start_c = '\033[92m'
    elif color_type == 'yellow':
        start_c = '\033[93m'
    elif color_type == 'light_purple':
        start_c = '\033[94m'
    elif color_type == 'purple':
        start_c = '\033[95m'
    elif color_type == 'cyan':
        start_c = '\033[96m'
    elif color_type == 'light_gray':
        start_c = '\033[97m'
    elif color_type == 'black':
        start_c = '\033[98m'
    else:
        print('color not right')
        sys.exit()

    print(start_c + str(word) + end_c)

def load_image(file_path, stack=False):
    if os.path.exists(file_path):
        if not stack:
            img = np.array(Image.open(file_path))
        else:
            dataset = Image.open(file_path)
            h,w = np.shape(dataset)
            img = np.zeros((dataset.n_frames, h,w))
            for i in range(dataset.n_frames):
                dataset.seek(i)
                img[i, :,:] = np.array(dataset)
            
    else:
        prColor('Error: wrong data path. No data is loaded.', 'red')
        sys.exit()
    return np.array(img).astype(np.float32)

def img_save(folder_path, filename, img):
    if not os.path.exists(os.path.join(folder_path)):
        os.makedirs(folder_path)
    # np.savetxt(os.path.join(folder_path, filename+'.npy.gz'), img)
    # np.save(os.path.join(folder_path, filename+'.npy'), img)
    # print(img.shape)
    
    if len(img.shape) == 2:
        im = Image.fromarray(img)
        im.save(os.path.join(folder_path, filename+'.tiff'), save_all=True)
        
        prColor('image saved: {}'.format(filename), 'green')
    elif len(img.shape) == 3:
        im = Image.fromarray(img[0,:,:])
        img_append = []
        for e in img[1:,:,:]:
            img_append.append(Image.fromarray(e))
        im.save(os.path.join(folder_path, filename+'.tiff'), save_all=True, append_images=img_append)
        
        prColor('image saved: {}'.format(filename), 'green')

def image_roi(img, M):
    '''
        take out the interested area of the all data.
        input:
            img:            image data, 2D or 3D array
            M:              the interested array size
                            if M = 0, use the whole size of the data
        output:
            img_data:       the area of the data
    '''
    img_size = img.shape
    if M == 0:
        return img
    elif len(img_size) == 2:
        if M > min(img_size):
            return img
        else:
            pos_0 = np.arange(M) - np.round(M/2) + np.round(img_size[0]/2)
            pos_0 = pos_0.astype('int')
            pos_1 = np.arange(M) - np.round(M/2) + np.round(img_size[1]/2)
            pos_1 = pos_1.astype('int')
            img_data = img[pos_0[0]:pos_0[-1]+1, pos_1[0]:pos_1[-1]+1]
    elif len(img_size) == 3:
        if M > min(img_size[1:]):
            return img
        else:
            pos_0 = np.arange(M) - np.round(M/2) + np.round(img_size[1]/2)
            pos_0 = pos_0.astype('int')
            pos_1 = np.arange(M) - np.round(M/2) + np.round(img_size[2]/2)
            pos_1 = pos_1.astype('int')
            img_data = np.zeros((img_size[0], M, M))
            for kk, pp in enumerate(img):
                img_data[kk] = pp[pos_0[0]:pos_0[-1]+1, pos_1[0]:pos_1[-1]+1]

    return img_data


def get_delta_beta(energy, material='Be'):
    '''
        use xraylib to get the delta of the material
    '''
    import xraylib

    try: 
        elementZnumber = xraylib.SymbolToAtomicNumber(material)
        density = xraylib.ElementDensity(elementZnumber)
        delta = 1 - xraylib.Refractive_Index_Re(material, energy/1e3, density)
        beta = xraylib.Refractive_Index_Im(material, energy/1e3, density)
    except:
        err_msg = 'error in getting delta for material'
        prColor('xraylib'+ err_msg, 'red')
        sys.exit()

    return beta, delta

def frankotchellappa(dpc_x, dpc_y):
    '''
        Frankt-Chellappa Algrotihm
        input:
            dpc_x:              the differential phase along x
            dpc_y:              the differential phase along y       
        output:
            phi:                phase calculated from the dpc
    '''
    fft2 = lambda x: np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x)))
    ifft2 = lambda x: np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(x)))
    fftshift = lambda x: np.fft.fftshift(x)
    # ifftshift = lambda x: np.fft.ifftshift(x)

    NN, MM = dpc_x.shape

    wx, wy = np.meshgrid(np.fft.fftfreq(MM) * 2 * np.pi,
                         np.fft.fftfreq(NN) * 2 * np.pi,
                         indexing='xy')
    wx = fftshift(wx)
    wy = fftshift(wy)
    numerator = -1j * wx * fft2(dpc_x) - 1j * wy * fft2(dpc_y)
    # here use the np.fmax method to eliminate the zero point of the division
    denominator = np.fmax((wx)**2 + (wy)**2, np.finfo(float).eps)

    div = numerator / denominator

    phi = np.real(ifft2(div))

    phi -= np.mean(np.real(phi))

    return phi


def frankotchellappa_1D(dpc_x, axis=0):
    '''
        Frankt-Chellappa Algrotihm for 1D problem
        input:
            dpc_x:              the differential phase along x
        output:
            phi:                phase calculated from the dpc
    '''
    fft = lambda x: np.fft.fftshift(np.fft.fft(np.fft.ifftshift(x, axes=axis), axis=axis), axes=axis)
    ifft = lambda x: np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(x, axes=axis), axis=axis), axes=axis)
    fftshift = lambda x: np.fft.fftshift(x, axes=axis)
    # ifftshift = lambda x: np.fft.ifftshift(x)

    NN, MM = dpc_x.shape

    wx, wy = np.meshgrid(np.fft.fftfreq(MM) * 2 * np.pi,
                         np.fft.fftfreq(NN) * 2 * np.pi,
                         indexing='xy')
    wx = fftshift(wx)
    wy = fftshift(wy)

    if axis == 0:
        numerator = -1j * wy * fft(dpc_x)
        # here use the np.fmax method to eliminate the zero point of the division
        denominator = np.fmax((wy)**2, np.finfo(float).eps)

        div = numerator / denominator
    else:
        numerator = -1j * wx * fft(dpc_x)
        # here use the np.fmax method to eliminate the zero point of the division
        denominator = np.fmax((wx)**2, np.finfo(float).eps)

        div = numerator / denominator
        
    phi = np.real(ifft(div))

    phi -= np.mean(np.real(phi))

    return phi


def write_h5(result_path, file_name, data_dict):
    ''' this function is used to save the variables in *args to hdf5 file
        args are in format: {'name': data}
    '''
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    with h5py.File(os.path.join(result_path, file_name+'.hdf5'), 'w') as f:
        for key_name in data_dict:
            f.create_dataset(key_name, data=data_dict[key_name], compression="gzip", compression_opts=9)
    prColor('result hdf5 file : {} saved'.format(file_name+'.hdf5'), 'green')

def read_h5(file_path, key_name, print_key=False):
    '''
        read the data with the key_name in the h5 file
    '''
    if not os.path.exists(file_path):
        prColor('Wrong file path', 'red')
        sys.exit()

    with h5py.File(file_path, 'r') as f:
        # List all groups
        if print_key:
            prColor("Keys: {}".format(list(f.keys())), 'green')

        # Get the data
        data = f[key_name][:]
    return data


def write_json(result_path, file_name, data_dict):
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    file_name_para = os.path.join(result_path, file_name+'.json')
    with open(file_name_para, 'w') as fp:
        json.dump(data_dict, fp, indent=0)
    
    prColor('result json file : {} saved'.format(file_name+'.json'), 'green')


def read_json(filepath, print_para=False):
    if not os.path.exists(filepath):
        prColor('Wrong file path', 'red')
        sys.exit()
    # file_name_para = os.path.join(result_path, file_name+'.json')
    with open(filepath, 'r') as fp:
        data = json.load(fp)
        if print_para:
            prColor('parameters: {}'.format(data), 'green')
    
    return data


def save_img(img_data, px, data_title, cbar_name, file_path, fig_size=(5, 5), fig_show=False):

    cbar_shrink = 0.1
    # impad = 0.01
    font_size = 14
    extent = np.array([
            -img_data.shape[1]/2*px[1]*1e3, img_data.shape[1]/2*px[1]*1e3,
            -img_data.shape[0]/2*px[1]*1e3, img_data.shape[0]/2*px[1]*1e3,
            ])
    fig, axs = plt.subplots(figsize=fig_size)
    im = axs.imshow(img_data, cmap='Spectral', extent=extent)
    cbar = plt.colorbar(im, ax=axs, fraction=cbar_shrink)
    plt.xlabel('mm')
    plt.ylabel('mm')
    axs.set_title(data_title)
    cbar.ax.set_ylabel(cbar_name)
    plt.tight_layout()
    plt.savefig(file_path)
    if not fig_show:
        plt.close()
    else:
        plt.show()

def save_plot(line_data, px, data_title, cbar, file_path, fig_size=(5, 5), fig_show=False):

    cbar_shrink = 0.1
    # impad = 0.01
    font_size = 14
    x_axis = px * (np.arange(len(line_data)) - len(line_data) / 2)

    fig, axs = plt.subplots(figsize=fig_size)
    im = axs.plot(x_axis*1e3, line_data)
    plt.xlabel('mm')
    plt.ylabel(cbar)
    axs.set_title(data_title)
    plt.tight_layout()
    plt.savefig(file_path)
    if not fig_show:
        plt.close()
    else:
        plt.show()

def fft2(img):
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(img)))

def ifft2(img):
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(img)))