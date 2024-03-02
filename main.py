import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, datasets, optimizers
from PIL import Image

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
data_dir = "C:/Users/vijdi/OneDrive/Desktop/CSProjects/athenahacks24/fashion_data/images"
x_train = load_dataset(data_dir)

# Reshape and add channel dimension
x_train = np.expand_dims(x_train, axis=-1)

# Build and compile models
generator = build_generator(latent_dim)
discriminator = build_discriminator()
gan = build_gan(generator, discriminator)
gan.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(lr=0.0002, beta_1=0.5))

# Define hyperparameters
latent_dim = 100
epochs = 50
batch_size = 128
save_interval = 10  # Save generated images every 10 epochs

# Function to save generated images
def save_generated_images(generator, epoch, examples=100, dim=(10, 10), figsize=(10, 10)):
    noise = np.random.normal(0, 1, (examples, latent_dim))
    generated_images = generator.predict(noise)

    # Rescale images 0 - 1
    generated_images = 0.5 * generated_images + 0.5

    fig, axs = plt.subplots(dim[0], dim[1], figsize=figsize, sharex=True, sharey=True)
    cnt = 0
    for i in range(dim[0]):
        for j in range(dim[1]):
            axs[i, j].imshow(generated_images[cnt, :, :, 0], cmap='gray')
            axs[i, j].axis('off')
            cnt += 1
    fig.savefig(f"generated_images_epoch_{epoch}.png")
    plt.close()

# Training loop
for epoch in range(epochs):
    np.random.shuffle(x_train)
    for batch in range(x_train.shape[0] // batch_size):
        # Train Discriminator
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        fake_images = generator.predict(noise)

        real_images = x_train[batch * batch_size: (batch + 1) * batch_size]

        # Labels for real and fake images
        real_labels = np.ones((batch_size, 1))
        fake_labels = np.zeros((batch_size, 1))

        # Train the discriminator
        d_loss_real = discriminator.train_on_batch(real_images, real_labels)
        d_loss_fake = discriminator.train_on_batch(fake_images, fake_labels)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Train Generator via GAN
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        misleading_targets = np.ones((batch_size, 1))

        # Update the generator (via the GAN model)
        g_loss = gan.train_on_batch(noise, misleading_targets)

    # Print progress
    print(f"Epoch {epoch+1}/{epochs}, Discriminator Loss: {d_loss}, Generator Loss: {g_loss}")

    # Save generated images at specified intervals
    if epoch % save_interval == 0:
        save_generated_images(generator, epoch)
