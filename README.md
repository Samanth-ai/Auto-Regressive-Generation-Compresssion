# Auto-Regressive Image Generation and Compression in PyTorch

This project is a complete, from-scratch implementation of an auto-regressive model for generating and compressing images. The model learns the statistical patterns of images from the SuperTuxKart dataset and uses this knowledge for two primary tasks:

1.  **Image Generation**: Producing novel images by sampling from the learned probability distribution.
2.  **Image Compression**: Acting as a highly efficient image compressor using entropy coding principles.

The core of this project is a **decoder-only Transformer** that operates on image tokens created by a custom **Patch-level Auto-Encoder** combined with **Binary Spherical Quantization (BSQ)**.

---

## Key Features

* **Patch-Level Auto-Encoder**: An auto-encoder (`ae.py`) that learns compact feature representations for image patches.
* **Binary Spherical Quantization (BSQ)**: A quantizer (`bsq.py`) that converts the auto-encoder's features into a discrete codebook of tokens. This is based on a simplified implementation of the method from the paper "[Binary Spherical Quantization](https://arxiv.org/abs/2406.07548)".
* **Auto-Regressive Transformer**: A decoder-only Transformer model (`autoregressive.py`) trained to predict the next image token in a sequence.
* **Full Training & Generation Pipeline**: Includes scripts for training (`train.py`), tokenizing a dataset (`tokenize.py`), and generating new images (`generation.py`).
* **High-Efficiency Compression**: The trained model can be used for image compression (`compress.py`), achieving rates significantly better than standard JPG for the given dataset.

---

## Setup

### 1. Environment

We recommend using [Miniconda](https://docs.conda.io/en/latest/miniconda.html) to manage dependencies.

```bash
# Create and activate a new conda environment
conda create --name gen_img python=3.12 -y
conda activate gen_img

# Install dependencies
pip install -r requirements.txt
```

### 2. Dataset

Download and unzip the SuperTuxKart image dataset.

```bash
# Download the dataset
wget [https://utexas.box.com/shared/static/qubjm5isldqvyimfj9rsmbnvnbezwcv4.zip](https://utexas.box.com/shared/static/qubjm5isldqvyimfj9rsmbnvnbezwcv4.zip) -O supertux_data.zip

# Unzip into the 'data' directory
unzip supertux_data.zip -d data/
```

---

## Usage Workflow

The project is broken down into three main stages: training a tokenizer, training a generative model, and using the models for generation or compression.

### Stage 1: Train the Tokenizer (`BSQPatchAutoEncoder`)

The first step is to train the `BSQPatchAutoEncoder`, which will learn to convert images into sequences of tokens.

```bash
python -m image_generator.train BSQPatchAutoEncoder --epochs 20
```

* Training progress can be monitored using TensorBoard: `tensorboard --logdir logs`
* A trained model will be saved to `image_generator/BSQPatchAutoEncoder.pth`.

### Stage 2: Tokenize the Dataset

Use the trained tokenizer to convert the training and validation image sets into token files.

```bash
# Create the directory for tokenized data if it doesn't exist
mkdir -p data/tokens

# Tokenize the training set
python -m image_generator.tokenize image_generator/BSQPatchAutoEncoder.pth data/tokens/tokenized_train.pth data/train/

# Tokenize the validation set
python -m image_generator.tokenize image_generator/BSQPatchAutoEncoder.pth data/tokens/tokenized_valid.pth data/valid/
```

### Stage 3: Train the Auto-Regressive Model

Now, train the `AutoregressiveModel` on the tokenized dataset. The model's cross-entropy loss directly corresponds to the compression rate it can achieve. A lower loss means better compression.

```bash
python -m image_generator.train AutoregressiveModel --epochs 50
```

* A trained model will be saved to `image_generator/AutoregressiveModel.pth`.

### Stage 4: Generate New Images

With both models trained, you can now generate novel images.

```bash
# Create an output directory for generated images
mkdir -p generated_images

python -m image_generator.generation \
    image_generator/BSQPatchAutoEncoder.pth \
    image_generator/AutoregressiveModel.pth \
    16 \
    generated_images/
```

This command will generate 16 new images and save them in the `generated_images/` directory.

---
## Model Checkpoints

During training, model checkpoints are automatically saved in the `checkpoints/` directory. The final trained model from a session is also saved directly inside the `image_generator/` directory (e.g., `image_generator/AutoregressiveModel.pth`) and is used by other scripts by default.

If you wish to use a specific checkpoint from the `checkpoints/` directory, you can manually copy it into the `image_generator/` directory, overwriting the existing model file.

For example:
```bash
# To use a specific checkpoint for generation
cp checkpoints/YYYY-MM-DD_HH-MM-SS_AutoregressiveModel.pth image_generator/AutoregressiveModel.pth
```
---

## Note for Apple Silicon (MPS) Users

There is a known PyTorch bug affecting bitwise operations on Apple Silicon. This implementation avoids the issue by using multiplication with powers of 2 instead of bit-shifting, ensuring compatibility.
