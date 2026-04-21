# Handwritten Digit Recognition (CV Mini Project)

This project recognizes a single handwritten digit from an image.

It includes:
- Model training on MNIST (`train.py`)
- Command-line prediction (`predict.py`)
- Tkinter GUI prediction (`gui.py`)

## 1) Prerequisites

- Miniconda is installed on your machine
- You are in the project root directory

## 2) Create and activate conda environment

```bash
conda create -n cv-digit python=3.11 -y
conda activate cv-digit
```

## 3) Install dependencies

Dependencies are managed by `requirements.txt`:

- `torch`
- `torchvision`
- `opencv-python`
- `Pillow`
- `numpy`

Install them with:

```bash
pip install -r requirements.txt
```

## 4) Train the model (generate weights)

Run:

```bash
python train.py --epochs 3 --batch-size 64
```

After training, weights are saved to:

```text
checkpoints/mnist_cnn.pt
```

## 5) Predict from command line

```bash
python predict.py --image samples/mnist_test_0.png
```

Optional: specify custom weights

```bash
python predict.py --image path/to/your_digit.png --weights checkpoints/mnist_cnn.pt
```

## 6) Run GUI

```bash
python gui.py
```

In the GUI:
1. Click `Select Image`
2. Choose an image file (`.png`, `.jpg`, `.jpeg`, `.bmp`)
3. See the uploaded image and prediction result

## 7) Quick environment check

After activation, verify key packages:

```bash
python -c "import torch, torchvision, cv2, PIL, numpy; print('Environment OK')"
```

## 8) Common issues

- **`Model weights not found`**
  - Run training first: `python train.py --epochs 3 --batch-size 64`
- **Image path error**
  - Check that `--image` points to an existing file
- **GUI fails due to Tk runtime**
  - Recreate conda env with Python 3.11 and retry
