# Generative Adversarial Neural Network

A TensorFlow/Keras notebook project that trains a Generative Adversarial Network (GAN) on Fashion MNIST and generates new 28x28 grayscale clothing-like images.

## What is a GAN?

A Generative Adversarial Network is made of two neural networks trained against each other:

- The **generator** creates synthetic samples from random noise.
- The **discriminator** tries to distinguish real training samples from generated samples.

During training, the generator improves by learning to produce images that increasingly fool the discriminator, while the discriminator improves by learning to detect generated images.

## How this project works

The main notebook, [`Generative-Adversarial-Neural-Network.ipynb`](Generative-Adversarial-Neural-Network.ipynb), builds and trains a GAN using the Fashion MNIST dataset from `tensorflow_datasets`.

The workflow is:

1. Install and import the required Python libraries.
2. Load the Fashion MNIST training split.
3. Normalize images from pixel values in `[0, 255]` to `[0, 1]`.
4. Build a TensorFlow dataset pipeline with caching, shuffling, batching, and prefetching.
5. Define the generator and discriminator models.
6. Train both models with a custom Keras training loop.
7. Save generated images at the end of each epoch.
8. Plot discriminator and generator loss values.

## Model architecture

### Generator

The generator receives a 128-dimensional random latent vector and produces a `28x28x1` grayscale image.

It uses:

- `Dense` projection into a `7x7x128` tensor
- `Reshape`
- Two `UpSampling2D` blocks
- Several `Conv2D` layers
- `LeakyReLU` activations
- Final `Conv2D` layer with `sigmoid` activation

### Discriminator

The discriminator receives a `28x28x1` image and predicts whether it is real or generated.

It uses:

- Four `Conv2D` blocks with increasing filters: 32, 64, 128, and 256
- `LeakyReLU` activations
- `Dropout` for regularization
- `Flatten`
- Final `Dense(1, activation="sigmoid")` output layer

### Training loop

The notebook defines a subclassed Keras model named `FashionGAN` with a custom `train_step` method. Each step:

1. Generates fake images from random noise.
2. Trains the discriminator on real Fashion MNIST images and generated images.
3. Adds label noise to discriminator labels.
4. Trains the generator to fool the discriminator.

The optimizers and losses used in the notebook are:

- Generator optimizer: `Adam(learning_rate=0.0001)`
- Discriminator optimizer: `Adam(learning_rate=0.00001)`
- Loss function: `BinaryCrossentropy`

## Technologies and dependencies

This project uses Python in a Jupyter Notebook environment with:

- TensorFlow / Keras
- TensorFlow Datasets
- Matplotlib
- NumPy
- ipywidgets

The notebook installs the required packages with:

```bash
pip3 install tensorflow matplotlib tensorflow-datasets ipywidgets
```

## Installation

Clone the repository and install the dependencies used by the notebook:

```bash
git clone https://github.com/Jazbarrionuev0/Generative-Adversarial-Neural-Network.git
cd Generative-Adversarial-Neural-Network
pip3 install tensorflow matplotlib tensorflow-datasets ipywidgets
```

## Usage

Open [`Generative-Adversarial-Neural-Network.ipynb`](Generative-Adversarial-Neural-Network.ipynb) in a Jupyter-compatible environment and run the cells from top to bottom.

The notebook includes sections for:

- Loading and previewing Fashion MNIST samples
- Building the generator
- Building the discriminator
- Training the GAN
- Plotting training losses
- Loading archived generator weights from `archive/generatormodel.h5`
- Saving generated model files as `generator.h5` and `discriminator.h5`

## Training process

The main notebook trains the GAN with batches of 128 Fashion MNIST images. A `ModelMonitor` callback generates three images at the end of each epoch and saves them to the `images/` directory using this naming pattern:

```text
images/generated_img_<epoch>_<index>.png
```

The main training cell is configured as:

```python
hist = fashgan.fit(ds, epochs=100, callbacks=[ModelMonitor()])
```

Training time depends on the available hardware. The notebook also configures TensorFlow GPU memory growth when a GPU is available.

## Expected output

Running the notebook can produce:

- Sample visualizations of Fashion MNIST training images
- Generated grayscale fashion images
- PNG files saved in `images/`
- A loss plot for `d_loss` and `g_loss`
- Optional saved model files: `generator.h5` and `discriminator.h5`

The repository already includes generated sample images in the [`images/`](images/) directory.

## Repository structure

```text
.
├── Generative-Adversarial-Neural-Network.ipynb
├── archive/
│   ├── FashionGAN-Tutorial.ipynb
│   └── generatormodel.h5
└── images/
    └── generated_img_*.png
```

## Potential improvements

Future documentation or project improvements could include:

- Add a `requirements.txt` or environment file for reproducible setup.
- Move reusable model and training code from the notebook into Python modules.
- Add a standalone training script.
- Document tested Python and TensorFlow versions.
- Add clearer instructions for using or regenerating saved model files.
- Add a license file if the project is intended for open-source reuse.
