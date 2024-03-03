import os
import numpy as np
import cv2

# Load the input image
input_image_path = "backcode\\finished\\try_edited.jpg"
output_folder = "backcode\\fin"

# Check if the output folder exists, if not create it
os.makedirs(output_folder, exist_ok=True)

image = cv2.imread(input_image_path)

# Check if the image was loaded successfully
if image is None:
    print("Error: Unable to load the input image.")
    exit()

# Create a mask initialized with zeros
mask = np.zeros(image.shape[:2], np.uint8)

# Define the region of interest (ROI) covering a slightly expanded area
height, width = image.shape[:2]
startY = height // 4 - 20  # Expand the top boundary
endY = 3 * height // 4 + 20  # Expand the bottom boundary
startX = width // 4 - 20  # Expand the left boundary
endX = 3 * width // 4 + 20  # Expand the right boundary
mask[startY:endY, startX:endX] = cv2.GC_PR_FGD

# Run GrabCut algorithm with the refined mask
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)
cv2.grabCut(image, mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)

# Modify the mask to obtain the final segmentation result
mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

# Apply the mask to the input image to extract the shirt
result = cv2.bitwise_and(image, image, mask=mask2)
result = cv2.GaussianBlur(result, (15, 15), 0)
# Construct the output image path within the output folder
output_image_name = os.path.splitext(os.path.basename(input_image_path))[0] + "fin.jpg"
output_image_path = os.path.join(output_folder, output_image_name)

# Save the extracted shirt image in the output folder
cv2.imwrite(output_image_path, result)

# Display the result
cv2.imshow("Extracted Shirt", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
