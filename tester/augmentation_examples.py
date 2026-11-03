from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from src.data import augment_batch, load_mnist


class ExampleRNG:
    """Choose fixed strong examples; only the Gaussian noise remains random."""

    def __init__(self, stroke=1):
        self.stroke = stroke
        self.noise = np.random.default_rng(42)

    def uniform(self, low, high, size):
        return np.full(size, high)

    def integers(self, low, high, size):
        return np.full(size, high - 1)

    def choice(self, values, size, p):
        return np.full(size, self.stroke)

    def random(self, size):
        return np.zeros(size)

    def normal(self, loc, scale, size):
        return self.noise.normal(loc, scale, size)


def save_examples(path):
    (images, labels), _ = load_mnist()
    digits = (1, 5, 8)
    indices = [np.flatnonzero(labels == digit)[0] for digit in digits]
    samples = images[indices].reshape(len(digits), 784).astype(np.float32) / 255
    disabled = dict(
        max_shift=0, noise_std=0, max_rotation=0, max_shear=0,
        max_scale=0, max_stretch=0, stroke_probability=0, blur_probability=0,
    )
    cases = [
        ("Original", {}, 1),
        ("Shift\n(+3, +3)", dict(max_shift=3), 1),
        ("Rotation\n+10 degrees", dict(max_rotation=10), 1),
        ("Shear\n+5 degrees", dict(max_shear=5), 1),
        ("Scale\n1.1x", dict(max_scale=0.1), 1),
        ("Stretch\n1.1x width", dict(max_stretch=0.1), 1),
        ("Thinner\n25% blend", dict(stroke_probability=1), -1),
        ("Thicker\n25% blend", dict(stroke_probability=1), 1),
        ("Blur\nstrength 0.1", dict(blur_probability=1), 1),
        ("Noise\nstd. 0.05", dict(noise_std=0.05), 1),
        ("Combined\nstrong case", None, 1),
    ]
    figure = Figure(figsize=(17.6, 5.8), facecolor="white")
    FigureCanvasAgg(figure)
    axes = figure.subplots(len(digits), len(cases))
    for column, (title, settings, stroke) in enumerate(cases):
        options = {} if settings is None else disabled | settings
        variants = augment_batch(samples, ExampleRNG(stroke), **options)
        for row, digit in enumerate(digits):
            axis = axes[row, column]
            axis.imshow(variants[row].reshape(28, 28), cmap="gray", vmin=0, vmax=1,
                        interpolation="nearest")
            axis.set_xticks([])
            axis.set_yticks([])
            if row == 0:
                axis.set_title(title, fontsize=10, pad=9)
            if column == 0:
                axis.set_ylabel(f"Digit {digit}", fontsize=11, labelpad=10)
    figure.suptitle("Training augmentation examples", fontsize=17, y=0.99)
    figure.text(0.5, 0.04,
                "Individual effects are isolated. Combined uses the configured strong values with thicker ink and blur forced on.",
                ha="center", fontsize=10)
    figure.text(0.5, 0.01,
                "Noise is seeded Gaussian noise, not a maximum. The combined column is an extreme case, not a typical random sample.",
                ha="center", fontsize=10)
    figure.subplots_adjust(left=0.045, right=0.995, top=0.83, bottom=0.11,
                           wspace=0.13, hspace=0.13)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    print(f"Augmentation examples saved to {path}")


if __name__ == "__main__":
    save_examples(Path(__file__).resolve().parents[1] / "assets" / "augmentation_examples.png")
