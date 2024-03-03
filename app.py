import os
from flask import Flask, session, render_template, request, jsonify, redirect
import display
import shutil
import cv2

app = Flask(__name__)

# Render the index.html template for the root route
@app.route("/")
def index():
    return render_template('index.html')

# Function to delete the previous sample folder
def delete_previous_sample_folder():
    folder_path = "backcode\static\sample1"
    try:
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
            print(f"Deleted previous {folder_path} folder")
        else:
            print(f"No previous {folder_path} folder found")
    except Exception as e:
        print(f"Error deleting {folder_path} folder: {e}")

# Main function to generate the images
def main(option):
    print("start main")
    
    try:
        delete_previous_sample_folder()
        image_folder = 'backcode/images2/'
        save_folder = 'backcode\static\sample1'

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

# Handle the POST request sent to /process_option
@app.route('/process_option', methods=['POST'])
def process_option():
    try:
        option = request.json['option']
        session['option'] = option
        return redirect('/output')
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# Render the output.html template for the /output route
@app.route('/output')
def output():
    try:
        option = session.get('option')
        if option:
            main(option)
        image_folder = 'backcode\static\sample1'
        images = os.listdir(image_folder)
        return render_template('output.html', images=images)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/input')
def input_page():
    return render_template('input.html')

if __name__ == '__main__':
    app.run(host="127.0.0.1", port=8080, debug=True)
