import numpy as np

from data import standardize_fit_transform
from metrics import accuracy_score_manual, classification_metrics_manual
from perceptron import SingleLayerPerceptron


def run_experiment(
    X_train_scaled,
    y_train,
    X_test_scaled,
    y_test,
    lr=0.1,
    batch_size=32,
    init_type="small_random",
    epochs=100,
    loss_type="bce",
    l2_lambda=0.0,
    random_state=42,
):
    model = SingleLayerPerceptron(
        n_features=X_train_scaled.shape[1],
        init_type=init_type,
        loss_type=loss_type,
        l2_lambda=l2_lambda,
        random_state=random_state,
    )

    model.fit(
        X_train_scaled,
        y_train,
        X_test_scaled,
        y_test,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
    )

    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)

    train_acc = accuracy_score_manual(y_train, y_train_pred)
    test_acc = accuracy_score_manual(y_test, y_test_pred)

    return model, train_acc, test_acc


def run_custom_dataset_experiment(
    X_custom,
    y_custom,
    split_function,
    standardize_function,
    random_state=42,
):
    X_custom_train, X_custom_test, y_custom_train, y_custom_test = split_function(
        X_custom,
        y_custom,
        test_size=0.3,
        random_state=random_state,
    )
    X_custom_train_scaled, X_custom_test_scaled, _, _ = standardize_function(
        X_custom_train,
        X_custom_test,
    )

    model = SingleLayerPerceptron(
        n_features=2,
        init_type="small_random",
        loss_type="bce",
        random_state=random_state,
    )

    model.fit(
        X_custom_train_scaled,
        y_custom_train,
        X_custom_test_scaled,
        y_custom_test,
        epochs=100,
        lr=0.1,
        batch_size=32,
    )

    y_custom_pred = model.predict(X_custom_test_scaled)
    metrics = classification_metrics_manual(y_custom_test, y_custom_pred)

    return model, metrics, X_custom_test_scaled, y_custom_test


def stratified_kfold_indices(y, n_splits=5, random_state=42):
    rng = np.random.default_rng(random_state)
    folds = [[] for _ in range(n_splits)]

    for cls in np.unique(y):
        cls_indices = np.where(y == cls)[0]
        rng.shuffle(cls_indices)

        split_indices = np.array_split(cls_indices, n_splits)

        for fold_id, part in enumerate(split_indices):
            folds[fold_id].extend(part.tolist())

    folds = [np.array(fold) for fold in folds]
    return folds


def cross_validate_perceptron(
    X_data,
    y_data,
    lrs,
    batch_sizes,
    n_splits=5,
    epochs=100,
):
    folds = stratified_kfold_indices(y_data, n_splits=n_splits, random_state=42)
    results = []

    for lr in lrs:
        for batch_size in batch_sizes:
            fold_scores = []

            for fold_id in range(n_splits):
                validation_indices = folds[fold_id]
                train_indices = np.concatenate(
                    [folds[i] for i in range(n_splits) if i != fold_id]
                )

                X_fold_train = X_data[train_indices]
                y_fold_train = y_data[train_indices]
                X_fold_validation = X_data[validation_indices]
                y_fold_validation = y_data[validation_indices]

                X_fold_train_scaled, X_fold_validation_scaled, _, _ = (
                    standardize_fit_transform(X_fold_train, X_fold_validation)
                )

                model = SingleLayerPerceptron(
                    n_features=X_fold_train_scaled.shape[1],
                    init_type="small_random",
                    loss_type="bce",
                    random_state=42 + fold_id,
                )

                model.fit(
                    X_fold_train_scaled,
                    y_fold_train,
                    X_fold_validation_scaled,
                    y_fold_validation,
                    epochs=epochs,
                    lr=lr,
                    batch_size=batch_size,
                )

                y_fold_pred = model.predict(X_fold_validation_scaled)
                fold_scores.append(
                    accuracy_score_manual(y_fold_validation, y_fold_pred)
                )

            results.append(
                {
                    "lr": lr,
                    "batch_size": batch_size,
                    "mean_accuracy": float(np.mean(fold_scores)),
                    "std_accuracy": float(np.std(fold_scores)),
                }
            )

    return results
