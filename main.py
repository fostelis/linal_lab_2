import numpy as np
from sklearn.datasets import make_classification

from data import (
    generate_circle,
    generate_linear_gaussians,
    generate_xor,
    standardize_fit_transform,
    stratified_train_test_split,
)
from experiments import (
    cross_validate_perceptron,
    run_custom_dataset_experiment,
    run_experiment,
)
from metrics import classification_metrics_manual, print_metrics, print_table
from perceptron import SingleLayerPerceptron
from visualization import plot_decision_boundary, plot_losses, plot_series, plot_source_data


np.set_printoptions(precision=4, suppress=True)


def prepare_base_data():
    X, y = make_classification(
        n_samples=500,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        random_state=42,
        n_clusters_per_class=1,
    )

    print("Размер X:", X.shape)
    print("Размер y:", y.shape)
    print("Класс 0:", np.sum(y == 0))
    print("Класс 1:", np.sum(y == 1))

    plot_source_data(X, y)

    X_train, X_test, y_train, y_test = stratified_train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
    )
    X_train_scaled, X_test_scaled, mean, std = standardize_fit_transform(
        X_train,
        X_test,
    )

    print("Train:", X_train_scaled.shape, y_train.shape)
    print("Test:", X_test_scaled.shape, y_test.shape)
    print("\nКлассы в train:")
    print("0:", np.sum(y_train == 0), "1:", np.sum(y_train == 1))
    print("\nКлассы в test:")
    print("0:", np.sum(y_test == 0), "1:", np.sum(y_test == 1))
    print("\nСреднее train после стандартизации:", np.mean(X_train_scaled, axis=0))
    print("Std train после стандартизации:", np.std(X_train_scaled, axis=0))

    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled


def train_base_model(X_train_scaled, y_train, X_test_scaled, y_test):
    model = SingleLayerPerceptron(
        n_features=X_train_scaled.shape[1],
        init_type="small_random",
        loss_type="bce",
        random_state=42,
    )

    model.fit(
        X_train_scaled,
        y_train,
        X_test_scaled,
        y_test,
        epochs=100,
        lr=0.1,
        batch_size=32,
    )

    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)

    train_metrics = classification_metrics_manual(y_train, y_train_pred)
    test_metrics = classification_metrics_manual(y_test, y_test_pred)

    print_metrics("Метрики на обучающей выборке", train_metrics)
    print_metrics("Метрики на тестовой выборке", test_metrics)

    plot_losses(model, ylabel="Binary cross-entropy loss")

    wrong_mask = y_test_pred != y_test
    plot_decision_boundary(
        model,
        X_test_scaled,
        y_test,
        "Разделяющая граница на test",
        wrong_mask=wrong_mask,
    )

    return model


def learning_rate_experiments(X_train_scaled, y_train, X_test_scaled, y_test):
    learning_rates = [0.001, 0.01, 0.5, 1.0]
    results = []
    series = []

    for lr in learning_rates:
        model, train_acc, test_acc = run_experiment(
            X_train_scaled,
            y_train,
            X_test_scaled,
            y_test,
            lr=lr,
            batch_size=32,
            init_type="small_random",
            epochs=100,
        )

        results.append(
            [
                lr,
                f"{train_acc:.4f}",
                f"{test_acc:.4f}",
                f"{model.train_losses[-1]:.4f}",
                f"{model.test_losses[-1]:.4f}",
            ]
        )
        series.append((f"lr={lr}", model.test_losses))

    plot_series(series, "Влияние learning rate на loss")
    print_table(
        ["lr", "train_acc", "test_acc", "final_train_loss", "final_test_loss"],
        results,
    )


def batch_size_experiments(X_train_scaled, y_train, X_test_scaled, y_test):
    batch_sizes = [1, 16, 64, 256]
    results = []
    series = []

    for batch_size in batch_sizes:
        model, train_acc, test_acc = run_experiment(
            X_train_scaled,
            y_train,
            X_test_scaled,
            y_test,
            lr=0.1,
            batch_size=batch_size,
            init_type="small_random",
            epochs=100,
        )

        results.append(
            [
                batch_size,
                f"{train_acc:.4f}",
                f"{test_acc:.4f}",
                f"{model.train_losses[-1]:.4f}",
                f"{model.test_losses[-1]:.4f}",
            ]
        )
        series.append((f"batch={batch_size}", model.test_losses))

    plot_series(series, "Влияние размера батча на loss")
    print_table(
        ["batch_size", "train_acc", "test_acc", "final_train_loss", "final_test_loss"],
        results,
    )


def initialization_experiments(X_train_scaled, y_train, X_test_scaled, y_test):
    init_types = ["zeros", "small_random", "large_random"]
    results = []
    series = []

    for init_type in init_types:
        model, train_acc, test_acc = run_experiment(
            X_train_scaled,
            y_train,
            X_test_scaled,
            y_test,
            lr=0.1,
            batch_size=32,
            init_type=init_type,
            epochs=100,
        )

        results.append(
            [
                init_type,
                f"{train_acc:.4f}",
                f"{test_acc:.4f}",
                f"{model.train_losses[-1]:.4f}",
                f"{model.test_losses[-1]:.4f}",
            ]
        )
        series.append((f"init={init_type}", model.test_losses))

    plot_series(series, "Влияние инициализации весов на loss")
    print_table(
        ["init_type", "train_acc", "test_acc", "final_train_loss", "final_test_loss"],
        results,
    )


def custom_dataset_experiments():
    datasets = [
        (
            "Линейно разделимые гауссовы облака",
            generate_linear_gaussians(noise=0.05, random_state=42),
        ),
        ("XOR", generate_xor(noise=0.03, random_state=42)),
        ("Окружность", generate_circle(noise=0.03, random_state=42)),
    ]

    results = []

    for title, (X_custom, y_custom) in datasets:
        plot_source_data(X_custom, y_custom, title=title)
        model, metrics, X_custom_test_scaled, y_custom_test = run_custom_dataset_experiment(
            X_custom,
            y_custom,
            stratified_train_test_split,
            standardize_fit_transform,
            random_state=42,
        )

        plot_decision_boundary(
            model,
            X_custom_test_scaled,
            y_custom_test,
            title + ": граница на test",
        )

        results.append(
            [
                title,
                f"{metrics['accuracy']:.4f}",
                f"{metrics['precision']:.4f}",
                f"{metrics['recall']:.4f}",
                f"{metrics['f1']:.4f}",
            ]
        )

    print_table(["dataset", "accuracy", "precision", "recall", "f1"], results)


def loss_function_experiments(X_train_scaled, y_train, X_test_scaled, y_test):
    results = []
    series = []

    for loss_type in ["bce", "hinge"]:
        model, train_acc, test_acc = run_experiment(
            X_train_scaled,
            y_train,
            X_test_scaled,
            y_test,
            lr=0.1,
            batch_size=32,
            init_type="small_random",
            epochs=100,
            loss_type=loss_type,
        )

        results.append(
            [
                loss_type,
                f"{train_acc:.4f}",
                f"{test_acc:.4f}",
                f"{model.train_losses[-1]:.4f}",
                f"{model.test_losses[-1]:.4f}",
            ]
        )
        series.append((loss_type, model.test_losses))

    plot_series(series, "Сравнение BCE и Hinge loss", ylabel="Loss")
    print_table(
        ["loss_type", "train_acc", "test_acc", "final_train_loss", "final_test_loss"],
        results,
    )


def l2_experiments(X_train_scaled, y_train, X_test_scaled, y_test):
    lambdas = [0.0, 0.001, 0.01, 0.1, 1.0]
    results = []
    series = []

    for l2_lambda in lambdas:
        model, train_acc, test_acc = run_experiment(
            X_train_scaled,
            y_train,
            X_test_scaled,
            y_test,
            lr=0.1,
            batch_size=32,
            init_type="small_random",
            epochs=100,
            loss_type="bce",
            l2_lambda=l2_lambda,
        )

        weight_norm = np.linalg.norm(model.w)

        results.append(
            [
                l2_lambda,
                f"{train_acc:.4f}",
                f"{test_acc:.4f}",
                f"{weight_norm:.4f}",
                f"{model.test_losses[-1]:.4f}",
            ]
        )
        series.append((f"lambda={l2_lambda}", model.test_losses))

    plot_series(series, "Влияние L2-регуляризации")
    print_table(["lambda", "train_acc", "test_acc", "||w||", "final_test_loss"], results)


def cross_validation_experiment(X_train, y_train):
    cv_lrs = [0.001, 0.01, 0.1, 0.5]
    cv_batch_sizes = [16, 32, 64, 256]

    cv_results = cross_validate_perceptron(
        X_train,
        y_train,
        lrs=cv_lrs,
        batch_sizes=cv_batch_sizes,
        n_splits=5,
        epochs=100,
    )

    rows = []

    for result in cv_results:
        rows.append(
            [
                result["lr"],
                result["batch_size"],
                f"{result['mean_accuracy']:.4f}",
                f"{result['std_accuracy']:.4f}",
            ]
        )

    print_table(["lr", "batch_size", "mean_accuracy", "std_accuracy"], rows)

    best_result = max(cv_results, key=lambda item: item["mean_accuracy"])

    print("\nЛучшие гиперпараметры:")
    print(best_result)

    return best_result


def train_final_model(X_train_scaled, y_train, X_test_scaled, y_test, best_result):
    best_lr = best_result["lr"]
    best_batch_size = best_result["batch_size"]

    model = SingleLayerPerceptron(
        n_features=X_train_scaled.shape[1],
        init_type="small_random",
        loss_type="bce",
        random_state=42,
    )

    model.fit(
        X_train_scaled,
        y_train,
        X_test_scaled,
        y_test,
        epochs=100,
        lr=best_lr,
        batch_size=best_batch_size,
    )

    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)

    train_metrics = classification_metrics_manual(y_train, y_train_pred)
    test_metrics = classification_metrics_manual(y_test, y_test_pred)

    print_metrics("Финальная модель: train", train_metrics)
    print_metrics("Финальная модель: test", test_metrics)

    plot_losses(model, title="Финальная модель после подбора гиперпараметров")
    plot_decision_boundary(
        model,
        X_test_scaled,
        y_test,
        "Финальная модель: разделяющая граница на test",
    )


def main():
    X_train, _, y_train, y_test, X_train_scaled, X_test_scaled = prepare_base_data()

    train_base_model(X_train_scaled, y_train, X_test_scaled, y_test)

    learning_rate_experiments(X_train_scaled, y_train, X_test_scaled, y_test)
    batch_size_experiments(X_train_scaled, y_train, X_test_scaled, y_test)
    initialization_experiments(X_train_scaled, y_train, X_test_scaled, y_test)

    custom_dataset_experiments()
    loss_function_experiments(X_train_scaled, y_train, X_test_scaled, y_test)
    l2_experiments(X_train_scaled, y_train, X_test_scaled, y_test)

    best_result = cross_validation_experiment(X_train, y_train)
    train_final_model(X_train_scaled, y_train, X_test_scaled, y_test, best_result)


if __name__ == "__main__":
    main()
