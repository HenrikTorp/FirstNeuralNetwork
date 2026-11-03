from pathlib import Path
import math
from PIL import Image, ImageOps, ImageFilter
import numpy as np


MNIST_PATH = Path(__file__).resolve().parents[1] / "data" / "mnist.npz"


def load_mnist(path=MNIST_PATH):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"MNIST archive not found: {path}. "
            "Place mnist.npz there or edit MNIST_PATH in src/data.py."
        )
    with np.load(path, allow_pickle=False) as archive:
        return (
            (archive["x_train"], archive["y_train"]),
            (archive["x_test"], archive["y_test"]),
        )


def preprocess_digit(image):
    # Crop bright ink on a black background.
    ink = ImageOps.invert(image.convert("L"))
    bounds = ink.getbbox()
    if bounds is None:
        return None

    # Fit inside 20x20 without stretching the digit.
    digit = ink.crop(bounds)
    scale = 20 / max(digit.size)
    size = tuple(max(1, round(dimension * scale)) for dimension in digit.size)
    digit = digit.resize(size, Image.Resampling.LANCZOS)

    # Compute the center of mass using grayscale intensity as weight.
    pixels = np.asarray(digit, dtype=np.float64)
    total_ink = pixels.sum()
    if total_ink == 0:
        return None
    rows, columns = np.indices(pixels.shape)
    center_x = (columns * pixels).sum() / total_ink
    center_y = (rows * pixels).sum() / total_ink

    # Move that center toward (14, 14) using whole-pixel translation.
    processed_image = Image.new("L", (28, 28), 0)
    position = (round(14 - center_x), round(14 - center_y))
    # Ink shifted beyond the fixed 28x28 boundary is clipped.
    processed_image.paste(digit, position)
    return np.asarray(processed_image, dtype=np.float32) / 255.0


def augment_batch(
    images, rng, max_shift=3, noise_std=0.05, max_rotation=10,
    max_shear=5, max_scale=0.1, max_stretch=0.1,
    stroke_probability=0.5, stroke_strength=0.25,
    blur_probability=0.25, blur_strength=0.1,
):
    # images: (n, 784) float32 in [0, 1]. Returns a new, randomly altered copy.
    n = images.shape[0]
    pixels = images.reshape(n, 28, 28)
    angles = rng.uniform(-max_rotation, max_rotation, size=n)
    shears = rng.uniform(-max_shear, max_shear, size=n) if max_shear else np.zeros(n)
    scales = rng.uniform(1 - max_scale, 1 + max_scale, size=n) if max_scale else np.ones(n)
    stretches = rng.uniform(1 - max_stretch, 1 + max_stretch, size=n) if max_stretch else np.ones(n)

    # Blend thinner/thicker ink before transforming its geometry.
    stroke_modes = np.zeros(n, dtype=np.int8)
    if stroke_probability and stroke_strength:
        stroke_modes = rng.choice(
            (-1, 0, 1), size=n,
            p=(stroke_probability / 2, 1 - stroke_probability, stroke_probability / 2),
        )
    transformed = np.empty_like(pixels)
    for i in range(n):
        image = Image.fromarray(pixels[i])
        if stroke_modes[i]:
            filter_type = ImageFilter.MinFilter if stroke_modes[i] < 0 else ImageFilter.MaxFilter
            filtered = np.asarray(image.filter(filter_type(3)))
            blended = (1 - stroke_strength) * pixels[i] + stroke_strength * filtered
            image = Image.fromarray(blended)

        # Forward: stretch -> scale -> shear -> rotation, centered at (14, 14).
        radians = math.radians(float(angles[i]) % 360.0)
        cosine = round(math.cos(radians), 15)
        sine = round(math.sin(radians), 15)
        shear = math.tan(math.radians(float(shears[i])))
        a = (cosine - shear * sine) / (scales[i] * stretches[i])
        b = (-sine - shear * cosine) / (scales[i] * stretches[i])
        d = sine / scales[i]
        e = cosine / scales[i]
        matrix = (
            a, b, a * -14 + b * -14 + 14,
            d, e, d * -14 + e * -14 + 14,
        )
        transformed[i] = np.asarray(image.transform(
            (28, 28), Image.Transform.AFFINE, matrix,
            resample=Image.Resampling.BILINEAR, fillcolor=0,
        ))

    # Keep the existing integer shifts and intermediate frame clipping.
    padded = np.pad(transformed, ((0, 0), (max_shift, max_shift), (max_shift, max_shift)))
    shift_y = rng.integers(-max_shift, max_shift + 1, size=n)
    shift_x = rng.integers(-max_shift, max_shift + 1, size=n)
    grid = np.arange(28)
    rows = (grid[None, :] + max_shift - shift_y[:, None])[:, :, None]
    columns = (grid[None, :] + max_shift - shift_x[:, None])[:, None, :]
    shifted = padded[np.arange(n)[:, None, None], rows, columns]

    # Occasionally soften the transformed ink with a float32 three-tap blur.
    if blur_probability and blur_strength:
        blur_mask = rng.random(n) < blur_probability
        if np.any(blur_mask):
            selected = shifted[blur_mask]
            padded = np.pad(selected, ((0, 0), (0, 0), (1, 1)))
            softened = (
                blur_strength * padded[:, :, :-2]
                + (1 - 2 * blur_strength) * padded[:, :, 1:-1]
                + blur_strength * padded[:, :, 2:]
            )
            padded = np.pad(softened, ((0, 0), (1, 1), (0, 0)))
            shifted[blur_mask] = (
                blur_strength * padded[:, :-2, :]
                + (1 - 2 * blur_strength) * padded[:, 1:-1, :]
                + blur_strength * padded[:, 2:, :]
            )

    # Noise comes last; clip and return a new float32 array.
    shifted = shifted + rng.normal(0.0, noise_std, size=shifted.shape)
    return np.clip(shifted, 0.0, 1.0).astype(np.float32).reshape(n, 784)
