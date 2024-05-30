import cv2
import numpy as np

# Attach camera indexed as 0
camera = cv2.VideoCapture(0)

# Setting frame width and frame height as 640x480
camera.set(3, 640)
camera.set(4, 480)

# Loading the mountain image
mountain = cv2.imread('mount_everest.jpg')

# Resizing the mountain image to 640x480
mountain_resized = cv2.resize(mountain, (640, 480))

while True:
    # Read a frame from the attached camera
    status, frame = camera.read()

    # If we got the frame successfully
    if status:
        # Flip the frame horizontally
        frame = cv2.flip(frame, 1)

        # Convert the image to RGB for easy processing
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Creating thresholds
        lower_bound = np.array([100, 100, 100])
        upper_bound = np.array([255, 255, 255])

        # Thresholding image
        mask = cv2.inRange(frame_rgb, lower_bound, upper_bound)

        # Inverting the mask
        mask_inv = cv2.bitwise_not(mask)

        # Create a 3-channel version of the mask
        mask_inv_colored = cv2.cvtColor(mask_inv, cv2.COLOR_GRAY2BGR)

        # Use np.where to create the final image
        final_image = np.where(mask_inv_colored == 0, mountain_resized, frame)

        # Show the final image
        cv2.imshow('frame', final_image)

        # Wait for 1ms before displaying another frame
        code = cv2.waitKey(1)
        if code == 32:  # Space key to exit
            break

# Release the camera and close all opened windows
camera.release()
cv2.destroyAllWindows()
