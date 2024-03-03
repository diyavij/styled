import os
import cv2
import numpy as np
import random
import shutil
from flask import Flask, render_template, request, jsonify
import display

app = Flask(__name__)

def delete_previous_sample_folder():
    folder_path = "sample1"
    try:
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
            print(f"Deleted previous {folder_path} folder")
    except Exception as e:
        print(f"Error deleting {folder_path} folder: {e}")

delete_previous_sample_folder()

def main(option):  
    print("start main")
    try:
        image_folder = 'C:/Users/vijdi/OneDrive/Desktop/CSProjects/athenahacks24/backcode/images2/'
        save_folder = 'C:/Users/vijdi/OneDrive/Desktop/CSProjects/athenahacks24/backcode/sample1/'

        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        selected_tops = display.random_select_images(image_folder, f'{option}1', 1)
        selected_bottoms = display.random_select_images(image_folder, f'{option}2', 1)
        selected_shoes = display.random_select_images(image_folder, f'{option}3', 1)
        selected_accessories = display.random_select_images(image_folder, f'{option}4', 1)

        selected_images = selected_tops + selected_bottoms + selected_shoes + selected_accessories

        for img_name in selected_images:
            img_path = os.path.join(image_folder, img_name)
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError(f"Failed to load image: {img_path}")
            img_smoothed = display.smooth_edges(img)
            img_paperized = display.apply_paper_texture(img_smoothed, 'backcode/paper_texture.jpg', alpha=0.2)  
            img_enhanced = display.enhance_colors(img_paperized, contrast=1.5, saturation=1.5)  
            save_path = os.path.join(save_folder, img_name)
            if not cv2.imwrite(save_path, img_enhanced):
                raise ValueError(f"Failed to write image: {save_path}")

    except Exception as e:
        print("An error occurred:", e)

@app.route('/process_option', methods=['POST'])
def process_option():
    option = request.json['option']

    try:
        delete_previous_sample_folder()
        
        # Call the main function with the selected option
        main(option)

        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/output')
def output():
    # Load and display the generated images
    image_folder = 'C:/Users/vijdi/OneDrive/Desktop/CSProjects/athenahacks24/backcode/sample1/'
    images = os.listdir(image_folder)
    return render_template('output.html', images=images)

if __name__ == '__main__':
    app.run(debug=True)
