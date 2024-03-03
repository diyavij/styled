import os
import cv2
import numpy as np
import random

def random_select_images(folder, identifier, num_images):
    images = os.listdir(folder)
    # Filter images based on the identifier
    selected_images = [img for img in images if img.startswith(identifier)]
    # Randomly select specified number of images
    selected_images = random.sample(selected_images, num_images)
    return selected_images


def smooth_edges(image):
    mask = np.zeros(image.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect = (10, 10, image.shape[1]-10, image.shape[0]-10)
    cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    result = cv2.bitwise_and(image, image, mask=mask2)
    return result

def apply_paper_texture(image, texture_path, alpha=0.5):
    texture = cv2.imread(texture_path, cv2.IMREAD_UNCHANGED)
    if texture is None:
        raise ValueError("Failed to load texture image")

    # Resize the texture to match the input image dimensions
    texture_resized = cv2.resize(texture[..., :3], (image.shape[1], image.shape[0]))

    # Check if the input image has an alpha channel
    if image.shape[2] == 4:
        # If the input image has an alpha channel, use it as the mask
        mask = image[..., 3]
    else:
        # If the input image doesn't have an alpha channel, create a mask based on the intensity
        mask = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

    # Apply the texture only to non-transparent regions of the input image
    result = np.zeros_like(image)
    for c in range(3):
        result[..., c] = np.where(mask > 0, texture_resized[..., c], image[..., c])
    if image.shape[2] == 4:
        result[..., 3] = mask  # Preserve the alpha channel

    # Blend the result with the input image based on alpha
    blended = cv2.addWeighted(image, 1 - alpha, result, alpha, 0)

    return blended

def enhance_colors(image, contrast=1.5, saturation=1.5):
    # Convert image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Apply contrast adjustment
    hsv[..., 1] = np.clip(contrast * hsv[..., 1], 0, 255)
    # Apply saturation adjustment
    hsv[..., 2] = np.clip(saturation * hsv[..., 2], 0, 255)
    # Convert back to BGR color space
    enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return enhanced

def main():
    print("start main")
    try:
        image_folder = 'C:/Users/vijdi/OneDrive/Desktop/CSProjects/athenahacks24/backcode/images2/'
        save_folder = 'C:/Users/vijdi/OneDrive/Desktop/CSProjects/athenahacks24/backcode/sample1/'

        # Create save folder if it doesn't exist
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        # Randomly select one image for each type
        selected_tops = random_select_images(image_folder, 'h1', 1)
        selected_bottoms = random_select_images(image_folder, 'h2', 1)
        selected_shoes = random_select_images(image_folder, 'h3', 1)
        selected_accessories = random_select_images(image_folder, 'h4', 1)

        selected_images = selected_tops + selected_bottoms + selected_shoes + selected_accessories

        # Smooth edges and apply paper texture to the selected images
        for img_name in selected_images:
            img_path = os.path.join(image_folder, img_name)
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError(f"Failed to load image: {img_path}")
            img_smoothed = smooth_edges(img)
            img_paperized = apply_paper_texture(img_smoothed, 'backcode/paper_texture.jpg', alpha=0.5)  # Adjust alpha
            img_enhanced = enhance_colors(img_paperized, contrast=1.5, saturation=1.5)  # Adjust contrast and saturation
            save_path = os.path.join(save_folder, img_name)
            if not cv2.imwrite(save_path, img_enhanced):
                raise ValueError(f"Failed to write image: {save_path}")

    except Exception as e:
        print("An error occurred:", e)

if __name__ == '__main__':
    main()
