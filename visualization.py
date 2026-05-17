import numpy as np
import matplotlib.pyplot as plt


def plot_source_data(X, y, title="Исходные данные"):
    plt.figure(figsize=(7, 5))
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor="k", alpha=0.8)
    plt.xlabel("Признак 1")
    plt.ylabel("Признак 2")
    plt.title(title)
    plt.grid(True)
    plt.show()


def plot_losses(model, title="Изменение функции потерь", ylabel="Loss"):
    plt.figure(figsize=(8, 5))
    plt.plot(model.train_losses, label="Train loss")
    plt.plot(model.test_losses, label="Test loss")
    plt.xlabel("Эпоха")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_decision_boundary(model, X, y, title, wrong_mask=None):
    plt.figure(figsize=(7, 5))

    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor="k", alpha=0.8, label="Объекты")

    if wrong_mask is not None and np.any(wrong_mask):
        plt.scatter(
            X[wrong_mask, 0],
            X[wrong_mask, 1],
            facecolors="none",
            edgecolors="red",
            s=130,
            linewidths=2,
            label="Ошибки",
        )

    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    x_values = np.linspace(x_min, x_max, 200)

    if abs(model.w[1]) > 1e-12:
        y_values = -(model.w[0] * x_values + model.b) / model.w[1]
        plt.plot(x_values, y_values, color="black", linewidth=2, label="Граница")
    else:
        x_line = -model.b / model.w[0]
        plt.axvline(x_line, color="black", linewidth=2, label="Граница")

    plt.xlabel("Признак 1")
    plt.ylabel("Признак 2")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_series(series_by_label, title, ylabel="Test loss"):
    plt.figure(figsize=(9, 6))

    for label, values in series_by_label:
        plt.plot(values, label=label)

    plt.xlabel("Эпоха")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()
