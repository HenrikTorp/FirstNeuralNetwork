import json
from pathlib import Path
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


def plot_training(history, best_epoch, path):
    """Save epoch-end training and validation loss without opening a window."""
    figure = Figure(figsize=(7, 4))
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    epochs = range(1, len(history["train_loss"]) + 1)
    axis.plot(epochs, history["train_loss"], label="Training")
    axis.plot(epochs, history["validation_loss"], label="Validation")
    axis.axvline(
        best_epoch, color="gray", linestyle="--",
        label=f"Selected epoch ({best_epoch})",
    )
    axis.set(xlabel="Epoch", ylabel="Mean loss", title="Cross-entropy loss")
    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, format="png", dpi=150)


if __name__ == "__main__":
    history_path = Path(__file__).resolve().parents[2] / "models" / "digit_recog_history.json"
    with history_path.open(encoding="utf-8") as file:
        run = json.load(file)
    plot_path = history_path.with_name(history_path.stem.removesuffix("_history") + "_training.png")
    plot_training(run["history"], run["best_epoch"], plot_path)
    print(f"Training plot saved to {plot_path}")
