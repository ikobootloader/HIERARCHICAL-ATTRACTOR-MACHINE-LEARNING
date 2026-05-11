"""
MNIST report (lightweight):
- accuracy par epoch (train + validation)
- coherence inter-niveaux
- distribution des forces intra-niveau sur test set
"""

import json
import os
import numpy as np
import torch
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from haml import HAML


def level_predictions(model, final_states):
    preds = []
    for level, state in zip(model.levels, final_states):
        scores = level.predict_class_scores(state)
        preds.append(torch.argmax(scores, dim=1).cpu().numpy())
    return preds


def force_stats(model, final_states):
    stats = {}
    for i, (level, state) in enumerate(zip(model.levels, final_states)):
        with torch.no_grad():
            force = level.intra_level_force(state)
            norms = torch.norm(force, dim=1).cpu().numpy()
        stats[f"level_{i}"] = {
            "mean": float(np.mean(norms)),
            "std": float(np.std(norms)),
            "p50": float(np.percentile(norms, 50)),
            "p90": float(np.percentile(norms, 90)),
        }
    return stats


def main():
    cache_dir = os.path.join(os.getcwd(), ".sklearn_data")
    os.makedirs(cache_dir, exist_ok=True)

    print("Loading MNIST from OpenML...")
    mnist = fetch_openml("mnist_784", version=1, parser="auto", data_home=cache_dir)
    X = mnist.data.to_numpy()
    y = mnist.target.to_numpy().astype(int)

    rng = np.random.RandomState(42)
    idx = rng.choice(len(X), size=1800, replace=False)
    X = X[idx]
    y = y[idx]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=450, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    model = HAML(
        n_levels=3,
        n_attractors_per_class=2,
        lr=0.01,
        n_epochs=5,
        batch_size=128,
        max_steps=15,
        train_on_fit=False,
        device="cpu",
    )
    model.fit(X_train, y_train)
    history = model.train_model(X_train, y_train, X_val, y_val)

    test_acc = model.score(X_test, y_test)

    X_test_t = torch.from_numpy(X_test).float()
    model.eval()
    with torch.no_grad():
        final_states, _, _ = model.forward(X_test_t)
        lvl_preds = level_predictions(model, final_states)
        final_pred = model.predict(X_test)

    lvl_stack = np.stack(lvl_preds, axis=1)
    unanimous = np.all(lvl_stack == lvl_stack[:, [0]], axis=1)
    coherence = {
        "unanimity_rate": float(np.mean(unanimous)),
        "level0_vs_final": float(np.mean(lvl_stack[:, 0] == final_pred)),
        "level1_vs_final": float(np.mean(lvl_stack[:, 1] == final_pred)),
        "level2_vs_final": float(np.mean(lvl_stack[:, 2] == final_pred)),
    }

    report = {
        "epochs": model.n_epochs,
        "train_accuracy_by_epoch": history["accuracy"],
        "val_accuracy_by_epoch": history["val_accuracy"],
        "test_accuracy_final": float(test_acc),
        "coherence": coherence,
        "force_distribution_test": force_stats(model, final_states),
    }

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

