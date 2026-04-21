from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from model import DigitCNN
from preprocess import preprocess_digit_image


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict a handwritten digit from an image.")
    parser.add_argument("--image", required=True, help="Path to input image.")
    parser.add_argument("--weights", default="checkpoints/mnist_cnn.pt", help="Path to trained model weights.")
    args = parser.parse_args()

    image_path = Path(args.image)
    weights_path = Path(args.weights)
    try:
        if not weights_path.exists():
            raise FileNotFoundError(f"Model weights not found: {weights_path}. Run train.py first.")

        input_tensor = torch.from_numpy(preprocess_digit_image(image_path))
        model = DigitCNN()
        model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        model.eval()

        with torch.no_grad():
            logits = model(input_tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0)
            pred = int(torch.argmax(probs).item())
            conf = float(probs[pred].item())
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print(f"Predicted digit: {pred}")
    print(f"Confidence: {conf:.4f}")


if __name__ == "__main__":
    main()
