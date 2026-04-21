from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk

import torch
from PIL import Image, ImageTk

from model import DigitCNN
from preprocess import preprocess_digit_image


class DigitRecognizerGUI:
    def __init__(self, root: tk.Tk, weights_path: str = "checkpoints/mnist_cnn.pt") -> None:
        self.root = root
        self.root.title("Digit Recognizer")
        self.root.geometry("520x620")

        self.weights_path = Path(weights_path)
        self.model = self._load_model(self.weights_path)
        self.preview_photo: ImageTk.PhotoImage | None = None

        self._build_layout()

    def _build_layout(self) -> None:
        container = ttk.Frame(self.root, padding=12)
        container.pack(fill=tk.BOTH, expand=True)

        title = ttk.Label(container, text="Handwritten Digit Recognition", font=("Arial", 14, "bold"))
        title.pack(pady=(0, 10))

        upload_button = ttk.Button(container, text="Select Image", command=self.on_select_image)
        upload_button.pack(pady=(0, 10))

        self.path_label = ttk.Label(container, text="No image selected", wraplength=480)
        self.path_label.pack(pady=(0, 10))

        self.preview_label = ttk.Label(container, text="Image preview will appear here", anchor="center")
        self.preview_label.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        self.result_label = ttk.Label(container, text="Result: N/A", font=("Arial", 12, "bold"))
        self.result_label.pack(pady=(0, 8))

        self.message_label = ttk.Label(container, text="", wraplength=480, foreground="#444444")
        self.message_label.pack(pady=(0, 8))

    def _load_model(self, weights_path: Path) -> DigitCNN:
        if not weights_path.exists():
            raise FileNotFoundError(f"Model weights not found: {weights_path}. Run train.py first.")
        model = DigitCNN()
        model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        model.eval()
        return model

    def _predict(self, image_path: Path) -> tuple[int, float]:
        input_tensor = torch.from_numpy(preprocess_digit_image(image_path))
        with torch.no_grad():
            logits = self.model(input_tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0)
            pred = int(torch.argmax(probs).item())
            conf = float(probs[pred].item())
        return pred, conf

    def _show_preview(self, image_path: Path) -> None:
        preview = Image.open(image_path).convert("RGB")
        preview.thumbnail((420, 360))
        self.preview_photo = ImageTk.PhotoImage(preview)
        self.preview_label.configure(image=self.preview_photo, text="")

    def on_select_image(self) -> None:
        selected = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp")],
        )
        if not selected:
            return

        image_path = Path(selected)
        self.path_label.configure(text=f"Selected: {image_path}")

        try:
            self._show_preview(image_path)
            pred, conf = self._predict(image_path)
            self.result_label.configure(text=f"Result: Predicted digit = {pred}, Confidence = {conf:.4f}")
            self.message_label.configure(text="Prediction successful.")
        except (FileNotFoundError, ValueError, OSError) as exc:
            self.result_label.configure(text="Result: N/A")
            self.message_label.configure(text=f"Error: {exc}")


def main() -> None:
    root = tk.Tk()
    try:
        app = DigitRecognizerGUI(root)
        _ = app
    except FileNotFoundError as exc:
        # Show startup error inside the window if weights are missing.
        label = ttk.Label(root, text=f"Error: {exc}", wraplength=480, foreground="#aa0000")
        label.pack(padx=16, pady=20)
    root.mainloop()


if __name__ == "__main__":
    main()
