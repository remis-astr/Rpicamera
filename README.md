# RPiCamera — Advanced interface for Raspberry Pi cameras (IMX585)

RPiCamera is a control and recording application for Raspberry Pi cameras, optimized for the IMX585 sensor and designed to work with a custom libcamera build (libcamera-imx585). The software provides a Pygame graphical interface, advanced camera parameter control via Picamera2/libcamera, and features useful for astrophotography (focusing assistance, histograms, live stacking support, etc.).

<img width="1024" height="600" alt="20260322_16h12m36s_grim" src="https://github.com/user-attachments/assets/f69fa734-58c1-48ed-99dc-77e8b3b8b362" />


## Requirements

Recommended OS: Raspberry Pi OS (Bullseye or Bookworm), fully updated. Package names and availability may differ across distributions.


### System packages (APT)

Run:
```bash
sudo apt update
sudo apt install -y \
  python3 python3-pip python3-venv git ffmpeg \
  libatlas-base-dev libopenjp2-7 libtiff5-dev libjpeg-dev libpng-dev \
  libv4l-dev v4l-utils libcamera-dev libcamera-apps \
  gstreamer1.0-tools gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
  gstreamer1.0-plugins-bad
```

Notes:
- `libcamera-dev` and `libcamera-apps` are required for Picamera2 and libcamera-based workflows.
- `ffmpeg` is useful for video encoding and post-processing.
- GStreamer plugins may be required for RTSP and advanced streaming setups.
- On Raspberry Pi it is often faster and more reliable to install heavy Python packages via apt (for example `python3-astropy`, `python3-scipy`, `python3-opencv`) to avoid long pip compilations.

### Python dependencies (pip)


Minimal Python dependencies:
- pygame
- numpy
- matplotlib
- picamera2
- gpiozero
- pillow

Additional dependencies:
- astropy (used: astropy.stats.sigma_clipped_stats)
- scipy (used: scipy.ndimage)
- scikit-image (used: skimage.feature.peak_local_max)
- opencv-python (cv2), used by several scripts for image processing

Recommended pip install command:
```bash
pip install pygame numpy matplotlib picamera2 gpiozero pillow \
            astropy scipy scikit-image opencv-python
```

APT alternatives on Raspberry Pi OS:
```bash
sudo apt install -y python3-astropy python3-scipy python3-opencv python3-skimage
```

---

## Quick installation

1. Clone the repository:
```bash
git clone https://github.com/remis-astr/Rpicamera.git
cd Rpicamera
```

2. Prepare the Python environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
# install dependencies (see Requirements section)
pip install -r requirements.txt  # if a requirements file is present
```

3. Install or prepare the IMX585 driver (if applicable):
- Follow the instructions in `will12753/libcamera-imx585` to build and install the driver and configuration files.

4. Run the application (replace with actual entry script if different):
```bash
python RPiCamera2.py
```


