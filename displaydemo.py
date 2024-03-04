import os
import cv2
import numpy as np
import random
import shutil
from flask import Flask, request, jsonify


entry = "h"

# Function to delete the previous sample folder
def delete_previous_sample_folder():
    folder_path = "backcode\sample2"
    try:
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
            print(f"Deleted previous {folder_path} folder")
        else:
            print(f"No previous {folder_path} folder found")
    except Exception as e:
        print(f"Error deleting {folder_path} folder: {e}")



def random_select_images(folder, identifier, num_images):
    images = os.listdir(folder)
    selected_images = [img for img in images if img.startswith(identifier)]
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

def apply_paper_texture(image, texture_path, alpha=0.2):  # Adjusted alpha value
    texture = cv2.imread(texture_path, cv2.IMREAD_UNCHANGED)
    if texture is None:
        raise ValueError("Failed to load texture image")

    texture_resized = cv2.resize(texture[..., :3], (image.shape[1], image.shape[0]))

    soft_light_blend = cv2.addWeighted(image, 1 - alpha, texture_resized, alpha, 0)

    return soft_light_blend

def enhance_colors(image, contrast=1.7, saturation=1.7):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[..., 1] = np.clip(contrast * hsv[..., 1], 0, 255)
    hsv[..., 2] = np.clip(saturation * hsv[..., 2], 0, 255)
    enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return enhanced

def main(option):  
    print("start main")
    try:
        image_folder = 'C:/Users/vijdi/OneDrive/Desktop/CSProjects/athenahacks24/backcode/images2/'
        save_folder = 'backcode\sample2'

        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        selected_tops = random_select_images(image_folder, f'{entry}1', 1)
        selected_bottoms = random_select_images(image_folder, f'{entry}2', 1)
        selected_shoes = random_select_images(image_folder, f'{entry}3', 1)
        selected_accessories = random_select_images(image_folder, f'{entry}4', 1)

        selected_images = selected_tops + selected_bottoms + selected_shoes + selected_accessories

        for img_name in selected_images:
            img_path = os.path.join(image_folder, img_name)
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError(f"Failed to load image: {img_path}")
            img_smoothed = smooth_edges(img)
            img_paperized = apply_paper_texture(img_smoothed, 'backcode/paper_texture.jpg', alpha=0.2)  
            img_enhanced = enhance_colors(img_paperized, contrast=1.5, saturation=1.5)  
            save_path = os.path.join(save_folder, img_name)
            if not cv2.imwrite(save_path, img_enhanced):
                raise ValueError(f"Failed to write image: {save_path}")
    except Exception as e:
        print("An error occurred:", e)


