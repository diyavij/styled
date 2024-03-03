import os
import requests
from bs4 import BeautifulSoup
import cv2

# Function to download image from URL
def download_image(url, save_path):
    response = requests.get(url)
    with open(save_path, 'wb') as f:
        f.write(response.content)

# Function to save image with specified naming convention
def save_image_with_naming_convention(url, aesthetic, clothing_type, clothing_id, save_folder):
    # Define filename based on naming convention
    filename = f"{aesthetic}_{clothing_type}_{clothing_id}.jpg"
    save_path = os.path.join(save_folder, filename)

    # Download image from URL and save with specified filename
    download_image(url, save_path)

# Function to scrape clothing items from website
def scrape_clothing_items(url, save_folder):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Find all image elements on the webpage
    image_elements = soup.find_all('img')
    
    # Define aesthetic and clothing type as "hippie" and "pants" respectively
    aesthetic = "h"
    clothing_type = "2"  # Assuming pants are type 2
    
    # Extract relevant information from image URLs
    for idx, img in enumerate(image_elements):
        image_url = img['src']
        clothing_id = idx + 1  # ID value based on the index in the loop
        save_image_with_naming_convention(image_url, aesthetic, clothing_type, clothing_id, save_folder)

# Main function
def main():
    # Define URL to scrape
    url = 'https://www.thelittlebazaar.com/category/Clothing.html'

    # Define save folder
    save_folder = r'C:\Users\vijdi\OneDrive\Desktop\CSProjects\athenahacks24\backcode\images2'

    # Create save folder if it doesn't exist
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Scrape clothing items and save them to the specified folder
    scrape_clothing_items(url, save_folder)

if __name__ == '__main__':
    main()
