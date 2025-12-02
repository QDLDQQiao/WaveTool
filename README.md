# WaveTool - Wavefront Sensing Software

## Overview
WaveTool is a modular GUI software designed for wavefront sensing measurements. It supports extensible modules for different sensing techniques, such as Talbot grating interferometry and Hartmann wavefront sensors. The application is built using Python and PyQt6, ensuring cross-platform compatibility and ease of maintenance.

## Features
-   **Modular Design**: Easily extensible for new algorithms and hardware.
-   **Live Camera Feed**: Real-time display with zoom, contrast, and colormap adjustments.
-   **Image Loading**: Support for loading static images (PNG, JPG, BMP, TIF) for analysis.
-   **Advanced Analysis**:
    -   2D Phase Map reconstruction.
    -   Zernike Polynomial decomposition (Coefficients & Residuals).
    -   Focus Analysis: 3D intensity profile, 2D spot view with X/Y line cuts, FWHM, and Sigma metrics.
-   **Tools**:
    -   **Period Calc**: Calculate grating period from Talbot images using FFT spectrum analysis.
    -   **Beam Envelope**: Real-time Gaussian-smoothed beam envelope overlay.
-   **Hardware Support**: Abstract interface for cameras (Dummy camera included).

## Project Structure

The project follows a modular architecture to separate the User Interface (GUI), Core Logic (Algorithms), and Hardware Abstraction.

```
WaveTool/
├── main.py                 # Entry point of the application
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
├── src/
│   ├── __init__.py
│   ├── core/               # Core analysis algorithms
│   │   ├── __init__.py
│   │   ├── processor.py    # Base class for wavefront processors
│   │   ├── talbot.py       # Talbot grating specific implementation
│   │   ├── hartmann.py     # Hartmann sensor specific implementation
│   ├── gui/                # Graphical User Interface
│   │   ├── __init__.py
│   │   ├── main_window.py  # Main application window
│   │   └── widgets/        # Reusable UI components
│   │       ├── __init__.py
│   │       ├── camera_view.py      # Pyqtgraph-based image viewer
│   │       ├── settings_panel.py   # Widget for algorithm parameters
│   │       └── results_display.py  # Matplotlib-based results visualization
│   └── hardware/           # Hardware abstraction layer
│       ├── __init__.py
│       ├── camera_interface.py # Abstract base class for cameras
│       └── dummy_camera.py     # Simulated camera for testing
└── tests/                  # Unit tests
```

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/QDLDQQiao/WaveTool.git
    cd WaveTool
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # Linux/macOS
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To run the application:
```bash
python main.py
```

### Controls
-   **File Menu**: Open static images for analysis.
-   **Camera View**: Right-click the image to access contrast (Histogram) and colormap options. Scroll to zoom.
-   **Snap & Analyze**: Captures the current frame (or uses the loaded image) and runs the selected analysis algorithm.
-   **Results**: Switch tabs to view Phase Maps, Zernike Coefficients, or detailed Focus Analysis.

### Adding a New Analysis Module
1.  Create a new file in `src/core/` (e.g., `my_algo.py`).
2.  Inherit from the `WavefrontProcessor` class defined in `src/core/processor.py`.
3.  Implement the `process(image)` method.
4.  Register your new processor in the GUI or configuration.

### Adding a New Camera
1.  Create a new file in `src/hardware/`.
2.  Inherit from `CameraInterface` in `src/hardware/camera_interface.py`.
3.  Implement methods like `connect()`, `snap()`, `disconnect()`.

## Dependencies
-   Python 3.8+
-   PyQt6 (GUI Framework)
-   NumPy (Numerical computations)
-   Matplotlib / PyQtGraph (Plotting)
-   OpenCV (Image processing, optional)
