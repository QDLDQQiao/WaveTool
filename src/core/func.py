import numpy as np
import os
import sys
from PIL import Image
import h5py
import json
import warnings
from pathlib import Path
from matplotlib import pyplot as plt
import matplotlib
from scipy.optimize import curve_fit
from wofry.propagator.wavefront2D.generic_wavefront import GenericWavefront2D
from wofryimpl.propagator.propagators2D.fresnel_zoom_xy import FresnelZoomXY2D

try:
    import tifffile
except ImportError:
    tifffile = None

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
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Image file not found: {file_path}")

    suffix = Path(file_path).suffix.lower()
    img = None

    # Robust TIFF path first.
    if suffix in ('.tif', '.tiff'):
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning, module="PIL.TiffImagePlugin")
                with Image.open(file_path) as dataset:
                    if stack and getattr(dataset, "n_frames", 1) > 1:
                        frames = []
                        for i in range(dataset.n_frames):
                            dataset.seek(i)
                            frames.append(np.array(dataset))
                        img = np.stack(frames, axis=0)
                    else:
                        img = np.array(dataset)
        except Exception:
            img = None

        if img is None and tifffile is not None:
            try:
                if stack:
                    img = tifffile.imread(file_path)
                else:
                    with tifffile.TiffFile(file_path) as tf:
                        img = tf.asarray(key=0)
            except Exception:
                img = None
    else:
        try:
            with Image.open(file_path) as dataset:
                img = np.array(dataset)
        except Exception:
            img = None

    if img is None:
        raise ValueError(f"Failed to load image: {file_path}")

    # Convert color image to grayscale for analysis consistency.
    if img.ndim == 3 and not stack:
        if img.shape[2] >= 3:
            img = np.dot(img[..., :3], [0.114, 0.587, 0.299])
        else:
            img = img[..., 0]

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
        # arr = np.asarray(img)
        # if arr.ndim != 2:
        #     raise ValueError(f"fft2 expects a 2D array, got shape={arr.shape}")

        # wy = np.hanning(arr.shape[0])
        # wx = np.hanning(arr.shape[1])
        # window = np.outer(wy, wx)

        # # Soft-edge apodization to suppress sharp-boundary FFT artifacts
        # return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(arr * window))) / window.mean()

        return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(img)))
    
def ifft2(img):
    # arr = np.asarray(img)
    # if arr.ndim != 2:
    #     raise ValueError(f"ifft2 expects a 2D array, got shape={arr.shape}")

    # wy = np.hanning(arr.shape[0])
    # wx = np.hanning(arr.shape[1])
    # window = np.outer(wy, wx)

    # # Soft-edge apodization to suppress edge-related artifacts
    # return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(arr * window))) / window.mean()
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(img)))

def fft(img, dire=0):
    return np.fft.fftshift(np.fft.fft(np.fft.ifftshift(img, axes=dire), axis=dire), axes=dire)

def ifft(img, dire=0):
    return np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(img, axes=dire), axis=dire), axes=dire)

'''
This is the python version code for diffraction
                dxy             the pixel pitch of the object
                z               the distance of the propagation
                wavelength      the wave length
                X,Y             meshgrid of coordinate
                data            input object
                diff_method     calculation method:
                                QPF: quadratic phase fresnel diffraction
                                IR:  convolution transfer methord (for far field)
                                TF: spectral transfer method    (for near field)
                                RS: Rayleigh-Sommerfield method
                                default: use IR(far) and TF(near)
'''
def diffraction_prop(data, dxy, z, wavelength, diff_method='default', magnification_x=1, magnification_y=1):
    '''
    This is the python version code for diffraction
        dxy             the pixel pitch of the object
        z               the distance of the propagation
        wavelength      the wave length
        X,Y             meshgrid of coordinate
        data            input object
        diff_method     calculation method:
                        QPF: quadratic phase fresnel diffraction for near field with upsampling
                        IR:  convolution transfer methord (for far field)
                        TF: spectral transfer method    (for near field)
                        RS: Rayleigh-Sommerfield method
                        default: use IR(far) and TF(near)
        if donot specify, the method will be determined by dxy and L(source length)
    '''
    if diff_method == 'default':
        # the method is not defined
        # the array size
        M = data.shape[0]
        # the source plane size
        L = M * dxy
        if dxy > wavelength * z / L:
            # use TF method for near field
            diff, L_out = prop_TF(data, dxy, z, wavelength)
        else:
            # use IR method for far field
            diff, L_out = prop_IR(data, dxy, z, wavelength)

    elif diff_method == 'QPF':
        # the method is defined
        # use QPF method
        diff, L_out = prop_QPF(data, dxy, z, wavelength)
    elif diff_method == 'IR':
        # use IR method
        diff, L_out = prop_IR(data, dxy, z, wavelength)
    elif diff_method == 'TF':
        # use TF method
        diff, L_out = prop_TF(data, dxy, z, wavelength)
    elif diff_method == 'RS':
        # use RS method
        diff, L_out = prop_RS(data, dxy, z, wavelength)
    elif diff_method == 'Wofry':
        # use phase gradient shift method
        diff, L_out = prop_worfy(data, dxy, z, wavelength,magnification_x, magnification_y)
    else:
        sys.exit('Error: no such diffraction method; must be TF, IR, RS, QPF, Worfy or default')

    return diff, L_out

def prop_worfy(data, dxy, z , wavelength, magnification_x=1, magnification_y=1):
    """
    Propagate a wavefront using Fresnel diffraction with zoom capabilities.
    This function uses the Wolfry library to perform Fresnel propagation of a 2D wavefront
    with optional magnification in both x and y directions.
    Parameters
    ----------
    data : numpy.ndarray
        2D array representing the complex amplitude of the input wavefront.
        Will be transposed internally for processing.
    dxy : float
        Pixel size (sampling interval) of the input wavefront in meters.
        Assumed to be the same in both x and y directions.
    z : float
        Propagation distance in meters. Positive values propagate forward.
    wavelength : float
        Wavelength of the light in meters.
    magnification_x : float, optional
        Magnification factor in the x direction (default is 1, no magnification).
    magnification_y : float, optional
        Magnification factor in the y direction (default is 1, no magnification).
    Returns
    -------
    WF_wolfry : numpy.ndarray
        2D complex array representing the propagated wavefront amplitude.
    L_out : list of float
        Physical dimensions of the output wavefront [height, width] in meters,
        calculated as [new_pixel_size_y * height, new_pixel_size_x * width].
    Notes
    -----
    - The function uses FresnelZoomXY2D propagator from the Wolfry library.
    - The shift_half_pixel parameter is set to True internally.
    - The input data is transposed at the beginning and the output is transposed back.
    - The coordinate system is centered around zero.
    """
    
    wavefront = data.T
    x_array = np.linspace(-dxy*wavefront.shape[0]/2, dxy*wavefront.shape[0]/2, wavefront.shape[0])
    y_array = np.linspace(-dxy*wavefront.shape[1]/2, dxy*wavefront.shape[1]/2, wavefront.shape[1])

    # Initialize wavefront object from the complex amplitude array
    initial_wavefront = GenericWavefront2D.initialize_wavefront_from_arrays(x_array=x_array,y_array=y_array,z_array=wavefront,wavelength=wavelength)

    propagation_distance = z  # Propagation distance in meters
    shift_half_pixel = True  # Whether to shift half a pixel

    # Instantiate the propagator
    fresnel_propagator = FresnelZoomXY2D()
    # Perform the propagation
    
    propagated_wavefront_wolfry = fresnel_propagator.propagate_wavefront(initial_wavefront,propagation_distance, magnification_x=magnification_x, magnification_y=magnification_y, shift_half_pixel=shift_half_pixel)

    WF_wolfry = propagated_wavefront_wolfry.get_complex_amplitude().T
    # New pixel sizes after magnification
    new_pixel_size_x = dxy * magnification_x
    new_pixel_size_y = dxy * magnification_y

    L_out = [new_pixel_size_y * WF_wolfry.shape[0], new_pixel_size_x * WF_wolfry.shape[1]]

    return WF_wolfry, L_out

def prop_RS(data, dxy, z, wavelength):
    '''
    Use Rayleigh-Sommerfield method for the diffraction. Assume same x and y lengthss and uniform sampling
        data:        source plane field
        dxy:         source and observation plane pixel size
        wavelength:  wavelength
        z:           propagation distance
        RS method
        u2(x,y)=ifft(fft(u1)*H); H=exp(jkz(1-(lambdafx)^2-(lambdafy)^2)^0.5)

    '''
    # the array size
    M = data.shape[0]
    N = data.shape[1]
    # source plane size
    L_y = dxy * M
    L_x = dxy * N
    # wavenumber
    k = 2 * np.pi / wavelength
    # frequency resolution
    dfx = 1/L_x
    dfy = 1/L_y
    fy = np.arange(-M/2, M/2) * dfy
    fx = np.arange(-N/2, N/2) * dfx
    FX, FY = np.meshgrid(fx, fy)
    L_out = [L_y, L_x]
    if z > 0:
        # transform function
        H = np.exp(1j*k*z*np.sqrt(1-(wavelength*FX)**2-(wavelength*FY)**2))
        # u2(x,y)=ifft(fft(u1)*H)
        diff = np.fft.fftshift(np.fft.ifft2(np.fft.fft2(np.fft.ifftshift(data)) * np.fft.ifftshift(H)))
    else:
        # transform function
        H = np.exp(1j*k*z*np.sqrt(1-(wavelength*FX)**2-(wavelength*FY)**2))
        # u2(x,y)=ifft(fft(u1)*H)
        diff = np.fft.fftshift(np.fft.ifft2(np.fft.fft2(np.fft.ifftshift(data)) * np.fft.ifftshift(H)))

    return diff, L_out


def prop_QPF(data, dxy, z, wavelength):
    '''
    this method use the quadratic phase method to calculate fresnel diffraction
        data:        source plane field
        dxy:         source and observation plane pixel size
        wavelength:  wavelength
        z:           propagation distance
        QPF:
            U2(x,y) = exp(jkz)/(j*lambda*z)*exp(jk/2z*(x^2+y^2))*int(U1(xx,yy)exp(jk/2z*(xx^2+yy^2))*exp(-jk/z(x*xx+y*yy))dxxdyy)

    '''
    # the array size
    M = data.shape[0]
    N = data.shape[1]
    
    # wavenumber
    k = 2 * np.pi / wavelength
    
    # Input spatial coordinates
    y = np.arange(-M/2, M/2) * dxy
    x = np.arange(-N/2, N/2) * dxy
    XX, YY = np.meshgrid(x, y)
    
    # Output pixel size (Scaling factor for single-step Fresnel)
    # dx_out = lambda * |z| / L_in
    # L_in = N * dxy
    dxy_out_x = wavelength * abs(z) / (N * dxy)
    dxy_out_y = wavelength * abs(z) / (M * dxy)
    
    # Output spatial coordinates
    fy = np.arange(-M/2, M/2) * dxy_out_y
    fx = np.arange(-N/2, N/2) * dxy_out_x
    FX, FY = np.meshgrid(fx, fy) # These are x2, y2 in the formula

    # Total output size
    L_out_x = N * dxy_out_x
    L_out_y = M * dxy_out_y

    # 1. Input Quadratic Phase Factor
    # Q1 = exp(j * k / 2z * (x1^2 + y1^2))
    Q1 = np.exp(1j * k / (2*z) * (XX**2 + YY**2))
    
    # 2. Output Quadratic Phase Factor
    # Q2 = exp(j * k / 2z * (x2^2 + y2^2))
    Q2 = np.exp(1j * k / (2*z) * (FX**2 + FY**2))
    
    # 3. Constant Pre-factor
    # C = exp(jkz) / (j * lambda * z)
    C = np.exp(1j * k * z) / (1j * wavelength * z)

    # 4. Perform the Integral via FFT
    # The integral is Fourier Transform of (U1 * Q1)
    # We must multiply by pixel area (dxy^2) to convert sum to integral
    field_in = data * Q1
    
    # Note: The Fourier kernel in Fresnel is exp(-j 2pi ...)
    # Standard FFT is exp(-j 2pi ...). 
    # The scaling relation x2 = f * lambda * z handles the coordinate transform.
    
    if z > 0:
        fft_field = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field_in))) * (dxy * dxy)
    else:
        # For negative z, the kernel sign in the integral effectively flips.
        # However, mathematically, the Fresnel formula holds for negative z directly 
        # if we respect the sign in the exponents.
        # The standard FFT definition works if we trust the sign of z in Q1 and Q2.
        fft_field = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field_in))) * (dxy * dxy)

    diff = C * Q2 * fft_field

    return diff, [L_out_y, L_out_x]


def prop_TF(data, dxy, wavelength, z):
    '''
    this method use the transfer function approach to calculate fresnel diffraction
        data:        source plane field
        dxy:         source and observation plane pixel size
        wavelength:  wavelength
        z:           propagation distance
        TF:
            U2(x,y)=ifft(fft(u1)*H); H=exp(jkz)*exp(-j*pi*lambda*z*(fx^2+fy^2))

    '''
    # the array size
    M = data.shape[0]
    N = data.shape[1]
    # source plane size
    L_x = dxy * N
    L_y = dxy * M
    # wavenumber
    k = 2 * np.pi / wavelength
    # frequency resolution
    dfx = 1/L_x
    dfy = 1/L_y
    fy = np.arange(-M/2, M/2) * dfy
    fx = np.arange(-N/2, N/2) * dfx

    FX, FY = np.meshgrid(fx, fy)

    L_out = [L_y, L_x]

    if z > 0:
        # transfer function
        H = np.exp(1j*k*z) * np.exp(-1j*wavelength*z*np.pi*(FX**2 + FY**2))
        diff = np.fft.fftshift(np.fft.ifft2(np.fft.fft2(np.fft.ifftshift(data)) * np.fft.ifftshift(H)))
    else:
        # transfer function
        H = np.exp(1j*k*z) * np.exp(-1j*wavelength*z*np.pi*(FX**2 + FY**2))
        diff = np.fft.fftshift(np.fft.ifft2(np.fft.fft2(np.fft.ifftshift(data)) * np.fft.ifftshift(H)))

    return diff, L_out


def prop_IR(data, dxy, wavelength, z):
    '''
    this method use the impulse response approach to calculate fresnel diffraction
        data:        source plane field
        dxy:         source and observation plane pixel size
        wavelength:  wavelength
        z:           propagation distance
        IR:
            U2(x,y)=ifft(fft(u1)*H); H=fft(exp(jkz)/(jlambda*z)*exp(j*k/2z*(x^2+y^2)))

    '''
    # the array size
    M = data.shape[0]
    N = data.shape[1]
    # source plane size
    L_y = dxy * M
    L_x = dxy * N
    # wavenumber
    k = 2 * np.pi / wavelength
    # spatial resolution
    y = np.arange(-M/2, M/2) * dxy
    x = np.arange(-N/2, N/2) * dxy
    XX, YY = np.meshgrid(x, y)

    L_out = [L_y, L_x]

    if z > 0:
        # impule response
        h = 1/(1j*wavelength*z) * np.exp(1j*k/(2*z)*(XX**2 + YY**2))
        # transfer function
        H = np.fft.fft2(np.fft.ifftshift(h)) * dxy**2
        U1 = np.fft.fft2(np.fft.ifftshift(data))
        diff = np.fft.fftshift(np.fft.ifft2(U1 * H))
    else:
        # impule response
        h = 1/(-1j*wavelength*z) * np.exp(1j*k/(-2*z)*(XX**2 + YY**2))
        # transfer function
        H = np.fft.fft2(np.fft.ifftshift(h)) * dxy**2
        U1 = np.fft.fft2(np.fft.ifftshift(data))
        diff = np.fft.fftshift(np.fft.ifft2(U1 / H))

    return diff, L_out


def gaussian_2d(xy, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    x, y = xy
    xo = float(xo)
    yo = float(yo)
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) + c*((y-yo)**2)))
    return g.ravel()

def calculate_sigma_width(image: np.ndarray, pixel_size: list):
    """
    Calculates the Sigma width (standard deviation) of a 2D focal spot 
    using 2D Gaussian fitting. Falls back to moments if fitting fails.
    
    Args:
        image: 2D numpy array of intensity.
        pixel_size: Physical size of one pixel [size_y, size_x] (e.g., in microns).
        
    Returns:
        sigma_x: Standard deviation in X direction (physical units).
        sigma_y: Standard deviation in Y direction (physical units).
        fwhm_x: Full Width at Half Maximum in X.
        fwhm_y: Full Width at Half Maximum in Y.
    """
    # 1. Pre-processing
    img = image - np.min(image)
    total_energy = np.sum(img)
    if total_energy == 0:
        return 0.0, 0.0, 0.0, 0.0

    h, w = img.shape
    y, x = np.mgrid[:h, :w]
    
    # 2. Initial Guess using Moments
    cx = np.sum(x * img) / total_energy
    cy = np.sum(y * img) / total_energy
    
    var_x = np.sum((x - cx)**2 * img) / total_energy
    var_y = np.sum((y - cy)**2 * img) / total_energy
    
    sigma_x_guess = np.sqrt(var_x)
    sigma_y_guess = np.sqrt(var_y)
    
    # If moments give NaN or 0 (single pixel), handle it
    if np.isnan(sigma_x_guess) or sigma_x_guess == 0: sigma_x_guess = 1.0
    if np.isnan(sigma_y_guess) or sigma_y_guess == 0: sigma_y_guess = 1.0
    
    # Optimization: Crop ROI around the centroid to speed up fitting
    # Use a window of +/- 4 sigma (covers >99.9% of Gaussian energy)
    roi_r_x = int(max(4 * sigma_x_guess, 10))
    roi_r_y = int(max(4 * sigma_y_guess, 10))
    
    x_min = max(0, int(cx) - roi_r_x)
    x_max = min(w, int(cx) + roi_r_x)
    y_min = max(0, int(cy) - roi_r_y)
    y_max = min(h, int(cy) + roi_r_y)
    
    img_roi = img[y_min:y_max, x_min:x_max]
    x_roi = x[y_min:y_max, x_min:x_max]
    y_roi = y[y_min:y_max, x_min:x_max]
    
    initial_guess = (np.max(img_roi), cx, cy, sigma_x_guess, sigma_y_guess, 0, 0)
    
    # 3. Fit Gaussian
    try:
        # Flatten x, y grids
        xy = (x_roi.ravel(), y_roi.ravel())
        
        popt, pcov = curve_fit(gaussian_2d, xy, img_roi.ravel(), p0=initial_guess, 
                               bounds=([0, 0, 0, 0, 0, -np.pi, 0], [np.inf, w, h, w, h, np.pi, np.max(img)]),
                               maxfev=1000)
        
        # Extract fitted parameters
        amplitude, xo, yo, sigma_x_fit, sigma_y_fit, theta, offset = popt
        
        sigma_x_px = abs(sigma_x_fit)
        sigma_y_px = abs(sigma_y_fit)
        
    except Exception as e:
        # Fallback to moments
        # print(f"Gaussian fit failed: {e}. Using moments.")
        sigma_x_px = sigma_x_guess
        sigma_y_px = sigma_y_guess

    # 4. Convert to physical units
    sigma_x = sigma_x_px * pixel_size[1]
    sigma_y = sigma_y_px * pixel_size[0]
    
    # 5. Calculate FWHM
    fwhm_x = 2.355 * sigma_x
    fwhm_y = 2.355 * sigma_y
    
    return sigma_x, sigma_y, fwhm_x, fwhm_y
