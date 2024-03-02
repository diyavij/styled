import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, datasets, optimizers
from PIL import Image

# Load and preprocess dataset
def load_dataset(data_dir):
    image_paths = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.jpg'):
                image_paths.append(os.path.join(root, file))

    images = []
    for image_path in image_paths:
        with Image.open(image_path) as img:
            img = img.resize((28, 28))  # Resize to match input size of the GAN
            img = np.array(img.convert('L'))  # Convert to grayscale
            img = (img.astype(np.float32) - 127.5) / 127.5  # Normalize to [-1, 1]
            images.append(img)
    return np.array(images)

# Define generator network
def build_generator(latent_dim):
    model = models.Sequential([
        layers.Dense(7 * 7 * 256, input_dim=latent_dim),
        layers.LeakyReLU(alpha=0.2),
        layers.Reshape((7, 7, 256)),
        layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same'),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same'),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', activation='tanh')
    ])
    return model

# Define discriminator network
def build_discriminator():
    model = models.Sequential([
        layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=(28, 28, 1)),
        layers.LeakyReLU(alpha=0.2),
        layers.Dropout(0.3),
        layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'),
        layers.LeakyReLU(alpha=0.2),
        layers.Dropout(0.3),
        layers.Flatten(),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

# Define the GAN
def build_gan(generator, discriminator):
    discriminator.trainable = False
    model = models.Sequential([
        generator,
        discriminator
    ])
    return model

# Define hyperparameters
latent_dim = 100
epochs = 50
batch_size = 128

# Load dataset
data_dir = "C:/Users/vijdi/OneDrive/Desktop/CSProjects/athenahacks24/fashion_data"
x_train = load_dataset(data_dir)

# Reshape and add channel dimension
x_train = np.expand_dims(x_train, axis=-1)

# Build and compile models
generator = build_generator(latent_dim)
discriminator = build_discriminator()
gan = build_gan(generator, discriminator)
gan.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(lr=0.0002, beta_1=0.5))

# Training loop
for epoch in range(epochs):
    np.random.shuffle(x_train)
    for batch in range(x_train.shape[0] // batch_size):
        # Train
