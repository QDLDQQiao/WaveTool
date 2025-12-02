import numpy as np
import math
from scipy.ndimage import gaussian_filter

def period_calc(image: np.ndarray, mask_radius: float):
    """
    Calculates the spectrum and period of the grating from a Talbot image.
    
    Args:
        image: 2D numpy array (grayscale image)
        mask_radius: Radius in pixels to mask the DC component (center)
        
    Returns:
        magnitude_spectrum: Log-scaled magnitude spectrum for display
        period: Calculated period in pixels
    """
    # FFT
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
    
    h, w = image.shape
    cy, cx = h // 2, w // 2
    
    # Mask the DC term (center) to find the first order peak
    y, x = np.ogrid[:h, :w]
    mask = (x - cx)**2 + (y - cy)**2 <= mask_radius**2
    
    # Search for peak outside the mask
    search_spectrum = np.abs(fshift).copy()
    search_spectrum[mask] = 0
    
    # Find peak location
    if np.max(search_spectrum) == 0:
        return magnitude_spectrum, 0.0
    # return the masked spectrum for visualization
    magnitude_spectrum[mask] = 0

    py, px = np.unravel_index(np.argmax(search_spectrum), search_spectrum.shape)
    
    # Calculate distance from center
    dist_px = np.sqrt((px - cx)**2 + (py - cy)**2)
    
    if dist_px == 0:
        period = 0.0
    else:
        # Assuming square pixels and period is in pixels
        # Spatial frequency k = dist_px / N
        # Period = 1/k = N / dist_px
        # We use the dimension corresponding to the distance direction, 
        # but assuming square image/pixels, we can approximate.
        # More accurately: period = 1 / sqrt((kx/w)^2 + (ky/h)^2)
        # where kx = px-cx, ky = py-cy
        
        kx = abs(px - cx)
        ky = abs(py - cy)
        
        if kx == 0 and ky == 0:
            period = 0.0
        else:
            # Frequency in cycles per pixel
            freq_sq = (kx / w)**2 + (ky / h)**2
            period = 1.0 / np.sqrt(freq_sq)
        
    return magnitude_spectrum, period

def calculate_envelope(image: np.ndarray, period: float):
    """
    Calculates the beam envelope by masking out high spatial frequencies.
    The cutoff frequency is set to 1 / (2 * period).
    
    Args:
        image: 2D numpy array
        period: Grating period in pixels
    """
    if period <= 0:
        # Fallback to simple smoothing if period is invalid
        return gaussian_filter(image, sigma=20)
    
    # FFT
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    
    h, w = image.shape
    cy, cx = h // 2, w // 2
    
    # Create Low Pass Filter Mask
    # We want to keep frequencies f < f_cutoff
    # f_cutoff = 1 / (2 * period)
    
    y, x = np.ogrid[:h, :w]
    
    # Normalized frequency coordinates (cycles/pixel)
    # (x - cx) / w ranges from -0.5 to 0.5
    freq_y = (y - cy) / h
    freq_x = (x - cx) / w
    
    freq_sq = freq_x**2 + freq_y**2
    cutoff_sq = (1.0 / (2.0 * period))**2
    
    mask = freq_sq <= cutoff_sq
    
    # Apply mask
    fshift_filtered = fshift * mask
    
    # Inverse FFT
    f_ishift = np.fft.ifftshift(fshift_filtered)
    img_back = np.fft.ifft2(f_ishift)
    
    return np.abs(img_back)

def fit_remove_2nd_order(phase_map: np.ndarray, mask: np.ndarray = None):
    """
    Fits a 2nd order polynomial (paraboloid) to the phase map and removes it.
    Model: Z = c0 + c1*x + c2*y + c3*x^2 + c4*x*y + c5*y^2
    """
    h, w = phase_map.shape
    y, x = np.mgrid[:h, :w]
    
    # Flatten
    x_flat = x.flatten()
    y_flat = y.flatten()
    z_flat = phase_map.flatten()
    
    # Design matrix for 2nd order polynomial
    # [1, x, y, x^2, xy, y^2]
    A = np.column_stack((np.ones_like(x_flat), x_flat, y_flat, x_flat**2, x_flat*y_flat, y_flat**2))
    
    if mask is not None:
        mask_flat = mask.flatten()
        A_fit = A[mask_flat]
        z_fit = z_flat[mask_flat]
        coeffs, _, _, _ = np.linalg.lstsq(A_fit, z_fit, rcond=None)
    else:
        # Least squares fit
        # coeffs: [c0, c1, c2, c3, c4, c5]
        coeffs, _, _, _ = np.linalg.lstsq(A, z_flat, rcond=None)
    
    # Reconstruct fitted surface
    fitted_flat = A @ coeffs
    fitted_surface = fitted_flat.reshape(h, w)
    
    residual = phase_map - fitted_surface
    
    return fitted_surface, residual, coeffs

def zernike_radial(n, m, rho):
    """Radial component of Zernike polynomial."""
    R = np.zeros_like(rho)
    if (n - m) % 2 == 0:
        for k in range((n - m) // 2 + 1):
            c = ((-1)**k * math.factorial(n - k)) / (
                math.factorial(k) * math.factorial((n + m) // 2 - k) * math.factorial((n - m) // 2 - k)
            )
            R += c * rho**(n - 2 * k)
    return R

def zernike_polynomial(n, m, rho, theta):
    """Generate Zernike polynomial (n, m)."""
    R = zernike_radial(n, abs(m), rho)
    if m >= 0:
        return R * np.cos(m * theta)
    else:
        return R * np.sin(abs(m) * theta)

def fit_zernike(phase_map: np.ndarray, n_terms: int = 15, mask: np.ndarray = None):
    """
    Fits Zernike polynomials up to n_terms to the phase map.
    Returns coefficients, fitted phase, and residual.
    """
    h, w = phase_map.shape
    y, x = np.mgrid[:h, :w]
    
    # Normalize coordinates to unit disk [-1, 1]
    # Center
    cy, cx = h / 2, w / 2
    # Radius (use min dimension to fit inside, or max to cover corners)
    # Usually we fit over a circular aperture. Here we fit over the whole rectangular image 
    # mapped to a unit circle or just use normalized coordinates.
    # Let's normalize so the image fits within the unit circle (r=1 at corners might be > 1)
    # or r=1 at edges.
    # Standard practice: normalize by half-width/height.
    max_rad = min(h, w) / 2
    
    rho = np.sqrt((x - cx)**2 + (y - cy)**2) / max_rad
    theta = np.arctan2(y - cy, x - cx)
    
    # Mask outside unit disk? 
    # Zernike are orthogonal on unit disk. If we fit on a rectangle, they are not orthogonal, 
    # but we can still use least squares to find coefficients.
    # Let's use all pixels.
    
    # Generate Zernike modes
    # We need a sequence of (n, m). Using Noll index or similar.
    # Simple sequence:
    # j=0: (0,0) Piston
    # j=1: (1,1) Tilt X
    # j=2: (1,-1) Tilt Y
    # j=3: (2,0) Defocus
    # ...
    # Let's generate a list of (n, m) pairs up to n_terms.
    
    modes = []
    count = 0
    n = 0
    while count < n_terms:
        for m in range(-n, n + 1, 2):
            modes.append((n, m))
            count += 1
            if count >= n_terms:
                break
        n += 1
        
    # Flatten for fitting
    z_flat = phase_map.flatten()
    
    # Build design matrix
    A_cols = []
    for (n, m) in modes:
        Z = zernike_polynomial(n, m, rho, theta)
        A_cols.append(Z.flatten())
        
    A = np.column_stack(A_cols)
    
    # Apply mask if provided
    if mask is not None:
        mask_flat = mask.flatten()
        # Only use valid points for fitting
        A_fit = A[mask_flat]
        z_fit = z_flat[mask_flat]
        coeffs, _, _, _ = np.linalg.lstsq(A_fit, z_fit, rcond=None)
    else:
        # Least squares fit on all points
        coeffs, _, _, _ = np.linalg.lstsq(A, z_flat, rcond=None)
    
    # Reconstruct (on full grid)
    fitted_flat = A @ coeffs
    fitted_phase = fitted_flat.reshape(h, w)
    residual = phase_map - fitted_phase
    
    # If mask provided, mask the residual too (optional, but good for display)
    if mask is not None:
        # We don't mask the output arrays here, we let the caller handle it.
        # But residual outside mask is meaningless if we didn't fit there.
        pass
    
    return coeffs, fitted_phase, residual

def fit_2nd_order_coeffs(phase_map: np.ndarray):
    """
    Fits a 2nd order polynomial to the phase map and returns coefficients.
    Model: Z = c0 + c1*x + c2*y + c3*x^2 + c4*x*y + c5*y^2
    Returns: coeffs [c0, c1, c2, c3, c4, c5]
    """
    h, w = phase_map.shape
    y, x = np.mgrid[:h, :w]
    
    x_flat = x.flatten()
    y_flat = y.flatten()
    z_flat = phase_map.flatten()
    
    A = np.column_stack((np.ones_like(x_flat), x_flat, y_flat, x_flat**2, x_flat*y_flat, y_flat**2))
    coeffs, _, _, _ = np.linalg.lstsq(A, z_flat, rcond=None)
    
    return coeffs

def frankot_chellappa(gx, gy):
    """
    Integrate gradients gx and gy to reconstruct the surface using the Frankot-Chellappa algorithm.
    
    Args:
        gx (np.ndarray): Gradient in x-direction (2D array).
        gy (np.ndarray): Gradient in y-direction (2D array).
        
    Returns:
        np.ndarray: Reconstructed phase/surface.
    """
    rows, cols = gx.shape
    
    # Frequency coordinates
    wx = np.fft.fftfreq(cols) * 2 * np.pi
    wy = np.fft.fftfreq(rows) * 2 * np.pi
    wx, wy = np.meshgrid(wx, wy)
    
    # FFT of gradients
    Gx = np.fft.fft2(gx)
    Gy = np.fft.fft2(gy)
    
    # Numerator and Denominator
    # Equation: phi_hat = -j * (wx*Gx + wy*Gy) / (wx^2 + wy^2)
    numerator = -1j * (wx * Gx + wy * Gy)
    denominator = wx**2 + wy**2
    
    # Avoid division by zero at (0,0)
    denominator[0, 0] = 1.0 # Arbitrary non-zero value, result will be 0 anyway
    
    phi_hat = numerator / denominator
    phi_hat[0, 0] = 0 # Set DC component to 0
    
    # Inverse FFT
    phi = np.real(np.fft.ifft2(phi_hat))
    
    return phi
