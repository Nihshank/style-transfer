# Neural Style Transfer

Applies the artistic style of a painting to a content image using a pretrained VGG19 CNN. Built from scratch in Python with PyTorch.

## How It Works

VGG19 is used purely as a feature extractor — weights are frozen. The pixels of a generated image are optimized over thousands of iterations to minimize two losses:

- **Content loss** — preserves the structure of the content image via feature map comparison at a deep layer
- **Style loss** — matches texture and color patterns of the style image via Gram matrix comparison across multiple layers

## Setup

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Add a content image to `images/content/` and a style image to `images/style/`, then run:

```
python style_transfer.py
```

Results are saved to `result/` every 500 steps.

## Configuration

```python
StyleTransfer(
    content_weight=1,     # lower = less content structure
    style_weight=1e9,     # higher = stronger style
    image_size=244,       # higher = better quality, slower run time
)
```

Paintings with bold consistent brushstrokes produce the best results. Higher resolution yields better quality but increases run time significantly.

## Hardware

Automatically uses MPS (Apple Silicon), CUDA (NVIDIA), or falls back to CPU.
