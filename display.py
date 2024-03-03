import os
import cv2
import numpy as np
import random

def random_select_images(folder, identifier, num_images):
    images = os.listdir(folder)
    # Filter images based on the identifier
    selected_images = [img for img in images if identifier in img]
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

def apply_paper_texture(image, texture_path):
    texture = cv2.imread(texture_path)
    texture = cv2.resize(texture, (image.shape[1], image.shape[0]))
    alpha = 0.5
    blended = cv2.addWeighted(image, alpha, texture, 1 - alpha, 0)
    return blended

def main():
    image_folder = 'C:/Users/vijdi/OneDrive/Desktop/CSProjects/athenahacks24/backcode/images2/'
    save_folder = 'C:/Users/vijdi/OneDrive/Desktop/CSProjects/athenahacks24/backcode/sample1/'

    # Create save folder if it doesn't exist
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Randomly select one image for each type
    selected_tops = random_select_images(image_folder + 'h1','h1', 6)
    selected_bottoms = random_select_images(image_folder + 'h2','h2', 6)
    selected_shoes = random_select_images(image_folder + 'h3','h3', 6)
    selected_accessories = random_select_images(image_folder + 'h4','h4', 6)

    selected_images = selected_tops + selected_bottoms + selected_shoes + selected_accessories

    # Smooth edges and apply paper texture to the selected images
    for img_name in selected_images:
        img_path = os.path.join(image_folder, img_name)
        img = cv2.imread(img_path)
        img_smoothed = smooth_edges(img)
        img_paperized = apply_paper_texture(img_smoothed, 'paper_texture.jpg')
        save_path = os.path.join(save_folder, img_name)
        cv2.imwrite(save_path, img_paperized)

if __name__ == '__main__':
    main()
