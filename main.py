# You must first uninstall
# python -m pip uninstall opencv-python
# And then install
# python -m pip install opencv-contrib-python==4.10.0.84

import cv2
import numpy as np
import pickle

def detect_color_checker(image):
    # Create a ColorChecker detector
    detector = cv2.mcc.CCheckerDetector_create()
    
    # Process the image to detect the ColorChecker
    detected = detector.process(image, cv2.mcc.MCC24, 1)
    
    if not detected:
        print("No ColorChecker pattern detected in the image.")
        return None

    # Get the list of detected ColorCheckers
    checkers = detector.getListColorChecker()
    
    for checker in checkers:
        # Create a CCheckerDraw object to visualize the ColorChecker
        cdraw = cv2.mcc.CCheckerDraw_create(checker)
        img_draw = image.copy()
        cdraw.draw(img_draw)
        
        # Display the image with the ColorChecker visualization
        cv2.namedWindow('Detected ColorChecker', cv2.WINDOW_NORMAL)
        cv2.imshow('Detected ColorChecker', img_draw)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Get the detected color patches and rearrange them
        chartsRGB = checker.getChartsRGB()
        width, height = chartsRGB.shape[:2]
        src = chartsRGB[:, 1].copy().reshape(int(width / 3), 1, 3) / 255.0

        # Check the content of src
        print(f"Content of src:\n{src}")
        
        return src
    
    return None

def calibrate_image(image, color_patches):
    try:
        # Create the color correction model
        model = cv2.ccm_ColorCorrectionModel(color_patches, cv2.ccm.COLORCHECKER_Macbeth)
        
        # Configure the model
        model.setColorSpace(cv2.ccm.COLOR_SPACE_sRGB)
        model.setCCM_TYPE(cv2.ccm.CCM_3x3)
        model.setDistance(cv2.ccm.DISTANCE_CIE2000)
        model.setLinear(cv2.ccm.LINEARIZATION_GAMMA)
        model.setLinearGamma(2.2)
        model.setLinearDegree(3)
        model.setSaturatedThreshold(0, 0.98)
        
        # Run the model
        model.run()
        
        # Get the color correction matrix and loss
        ccm = model.getCCM()
        print(f'ccm:\n{ccm}\n')
        loss = model.getLoss()
        print(f'loss:\n{loss}\n')
        
        return model

    except cv2.error as e:
        print(f"Error running the color correction model: {e}")
        return None
    except Exception as e:
        print(f"Unknown exception: {e}")
        return None

def apply_color_correction(image, model):
    # Apply color correction to the image
    img_ = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_ = img_.astype(np.float64) / 255.0

    # Perform inference with the model
    calibrated_image = model.infer(img_)
    out_ = calibrated_image * 255
    out_[out_ < 0] = 0
    out_[out_ > 255] = 255
    out_ = out_.astype(np.uint8)

    # Convert back to BGR
    out_img = cv2.cvtColor(out_, cv2.COLOR_RGB2BGR)
    return out_img

def save_calibration_params(color_patches, filename):
    # Save the color patches and configuration to a pickle file
    params = {
        'color_patches': color_patches
    }
    with open(filename, 'wb') as f:
        pickle.dump(params, f)

def load_calibration_params(filename):
    # Load the color patches and configuration from a pickle file
    with open(filename, 'rb') as f:
        params = pickle.load(f)
    return params

def reconstruct_model_from_params(params):
    # Reconstruct the color correction model from parameters
    color_patches = params['color_patches']
    model = cv2.ccm_ColorCorrectionModel(color_patches, cv2.ccm.COLORCHECKER_Macbeth)
    
    # Configure the model
    model.setColorSpace(cv2.ccm.COLOR_SPACE_sRGB)
    model.setCCM_TYPE(cv2.ccm.CCM_3x3)
    model.setDistance(cv2.ccm.DISTANCE_CIE2000)
    model.setLinear(cv2.ccm.LINEARIZATION_GAMMA)
    model.setLinearGamma(2.2)
    model.setLinearDegree(3)
    model.setSaturatedThreshold(0, 0.98)
    
    # Run the model
    model.run()
    return model

def calibrate_camera(calibration_file):
    # Load parameters and reconstruct the model
    params = load_calibration_params(calibration_file)
    model = reconstruct_model_from_params(params)
    cap = cv2.VideoCapture(0)

    # Create resizable windows
    cv2.namedWindow('Original Video', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Corrected Video', cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Apply color correction
        corrected_frame = apply_color_correction(frame, model)

        # Display the original and corrected video
        cv2.imshow('Original Video', frame)
        cv2.imshow('Corrected Video', corrected_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main(image_path=None, calibration_file=None, image_to_calibrate=None):
    if image_path and not calibration_file and not image_to_calibrate:
        # Step 1: Load image and detect ColorChecker
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not read the image {image_path}")
            return
        color_patches = detect_color_checker(image)

        if color_patches is not None:
            # Save calibration parameters
            save_calibration_params(color_patches, 'calibration_params.pickle')
            print("Calibration parameters saved in 'calibration_params.pickle'.")
            
            # Step 2: Calibrate the image
            model = calibrate_image(image, color_patches)
            if model is not None:
                # Apply color correction
                calibrated_image = apply_color_correction(image, model)
                cv2.imwrite('calibrated_output.jpg', calibrated_image)
                print("Calibrated image saved as 'calibrated_output.jpg'.")
                
                # Display the original and calibrated images
                cv2.namedWindow('Original Image', cv2.WINDOW_NORMAL)
                cv2.imshow('Original Image', image)
                cv2.namedWindow('Calibrated Image', cv2.WINDOW_NORMAL)
                cv2.imshow('Calibrated Image', calibrated_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                print("Could not calibrate the image.")
        else:
            print("Could not detect the ColorChecker.")

    elif calibration_file and image_to_calibrate and not image_path:
        # Apply calibration directly to an image using saved parameters
        # Load calibration parameters
        params = load_calibration_params(calibration_file)
        model = reconstruct_model_from_params(params)
        # Load the image
        image = cv2.imread(image_to_calibrate)
        if image is None:
            print(f"Could not read the image {image_to_calibrate}")
            return
        # Apply color correction
        calibrated_image = apply_color_correction(image, model)
        # Save the calibrated image
        output_filename = 'calibrated_' + image_to_calibrate
        cv2.imwrite(output_filename, calibrated_image)
        print(f"Calibrated image saved as '{output_filename}'")
        # Display the original and calibrated images
        cv2.namedWindow('Original Image', cv2.WINDOW_NORMAL)
        cv2.imshow('Original Image', image)
        cv2.namedWindow('Calibrated Image', cv2.WINDOW_NORMAL)
        cv2.imshow('Calibrated Image', calibrated_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    elif calibration_file and not image_to_calibrate and not image_path:
        # Calibrate the camera directly using the calibration file
        calibrate_camera(calibration_file)

    else:
        print("Please provide an image path for calibration, or a calibration file and an image to calibrate, or a calibration file to process video.")


if __name__ == "__main__":
    # To calibrate an image and save the calibration
    # main(image_path='DSC09033.JPG')

    # To apply the calibration directly to an image
    # main(calibration_file='calibration_params.pickle', calibrate_image="DSC09035.JPG")

    # To apply the calibration directly to the camera
    main(calibration_file='calibration_params.pickle')
