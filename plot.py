import matplotlib.pyplot as plt
import numpy as np

def plot_errors(models):
    plt.figure(figsize=(12, 8))

    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))

    for i, (model, color) in enumerate(zip(models, colors)):
        plt.plot(
            model.errors,
            label=f"Cyfra {i}",
            color=color,
            linewidth=2
        )

    plt.title("Funkcja błędu (MSE) dla jednostek ADALINE", fontsize=14)
    plt.xlabel("Epoka", fontsize=12)
    plt.ylabel("MSE", fontsize=12)

    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()