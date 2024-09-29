# OpenCV Color Calibration

A Python script for color correction and calibration using OpenCV and a ColorChecker chart. This tool detects a ColorChecker in an image, calibrates the colors, and applies the calibration to images or live video streams.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
  - [Calibrate an Image](#calibrate-an-image)
  - [Apply Calibration to an Image](#apply-calibration-to-an-image)
  - [Apply Calibration to Live Camera Feed](#apply-calibration-to-live-camera-feed)
- [Code Structure](#code-structure)
- [Contributing](#contributing)
- [License](#license)

## Prerequisites

- Python 3.x
- A ColorChecker chart (Macbeth ColorChecker)
- OpenCV with extra modules (`opencv-contrib-python`)

## Installation

First, uninstall any existing `opencv-python` package:

```bash
python -m pip uninstall opencv-python
```

Then, install the `opencv-contrib-python` package:

```bash
python -m pip install opencv-contrib-python==4.10.0.84
```

This version includes the `cv2.ccm` module required for color calibration.

## Usage

### Calibrate an Image

1. Place a ColorChecker chart in your scene and capture an image.
2. Run the script to detect the ColorChecker and calibrate the image:

   ```bash
   python your_script.py
   ```

   In the script, ensure the `main` function is called with the path to your image:

   ```python
   if __name__ == "__main__":
       main(image_path='your_image.jpg')
   ```

3. The script will:

   - Detect the ColorChecker in the image.
   - Save the calibration parameters to `calibration_params.pickle`.
   - Calibrate the image and save it as `calibrated_output.jpg`.
   - Display the original and calibrated images.

### Apply Calibration to an Image

To apply the saved calibration to another image:

1. Ensure you have the `calibration_params.pickle` file generated from the previous step.
2. Run the script with the calibration file and the image to calibrate:

   ```bash
   python your_script.py
   ```

   Modify the `main` function:

   ```python
   if __name__ == "__main__":
       main(calibration_file='calibration_params.pickle', calibrate_image='image_to_calibrate.jpg')
   ```

3. The script will:

   - Load the calibration parameters.
   - Apply the calibration to the specified image.
   - Save the calibrated image with a prefixed name (e.g., `calibrated_image_to_calibrate.jpg`).
   - Display the original and calibrated images.

### Apply Calibration to Live Camera Feed

To apply the calibration to a live video stream from your camera:

1. Ensure you have the `calibration_params.pickle` file.
2. Run the script with the calibration file:

   ```bash
   python your_script.py
   ```

   Modify the `main` function:

   ```python
   if __name__ == "__main__":
       main(calibration_file='calibration_params.pickle')
   ```

3. The script will:

   - Load the calibration parameters.
   - Open a video stream from your default camera.
   - Apply the calibration to each frame in real-time.
   - Display the original and calibrated video streams.
   - Press `q` to exit the video stream.

## Code Structure

- **detect_color_checker(image)**: Detects the ColorChecker chart in the image and extracts color patches.
- **calibrate_image(image, color_patches)**: Calibrates the image using the extracted color patches.
- **apply_color_correction(image, model)**: Applies the color correction model to an image.
- **save_calibration_params(color_patches, filename)**: Saves the calibration parameters to a file.
- **load_calibration_params(filename)**: Loads calibration parameters from a file.
- **reconstruct_model_from_params(params)**: Reconstructs the color correction model from saved parameters.
- **calibrate_camera(calibration_file)**: Applies the calibration to a live video stream.
- **main()**: Orchestrates the calibration and application processes based on provided arguments.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---