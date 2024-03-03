import cv2
import numpy as np
import os

def apply_paper_texture(image, texture_path):
    # Load the paper texture image
    texture = cv2.imread(texture_path)
    
    # Resize the texture to match the size of the input image
    texture = cv2.resize(texture, (image.shape[1], image.shape[0]))
    
    # Convert the image to the HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Increase the saturation of the image to make colors more intense
    hsv_image[:, :, 1] += 50  # Adjust the value as needed
    
    # Convert the image back to the BGR color space
    saturated_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    
    # Blend the saturated image with the paper texture using alpha blending
    alpha = 0.5  # Adjust alpha for desired blending effect
    blended = cv2.addWeighted(saturated_image, alpha, texture, 1 - alpha, 0)
    
    return blended

# Path to the original image and paper texture image
original_image_path = 'backcode\\try.jpg'
paper_texture_path = 'backcode\\paper_texture.jpg'

# Load the original image
original_image = cv2.imread(original_image_path)

# Apply paper texture to the foreground image
result = apply_paper_texture(original_image, paper_texture_path)

# Display the result
cv2.imshow('Scrapbooky Image', result)
cv2.waitKey(0)
cv2.destroyAllWindows()

output_folder = "backcode\\finished"

# Check if the output folder exists, if not create it
os.makedirs(output_folder, exist_ok=True)
# Construct the output image path within the output folder
output_image_name = os.path.splitext(os.path.basename(original_image_path))[0] + "_edited.jpg"
output_image_path = os.path.join(output_folder, output_image_name)
cv2.imwrite(output_image_path, result)