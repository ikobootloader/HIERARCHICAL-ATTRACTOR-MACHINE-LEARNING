"""
Test rapide MNIST avec entraînement (5 epochs, optimisé pour vitesse).
"""

import numpy as np
import time
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from haml import HAML

def main():
    print("="*70)
    print("Quick MNIST Test (5 epochs, optimized)")
    print("="*70)

    # 1. Charger MNIST (subset réduit)
    print("\nLoading MNIST...")
    mnist = fetch_openml('mnist_784', version=1, parser='auto')
    X, y = mnist.data.to_numpy(), mnist.target.to_numpy().astype(int)

    # Subset 2000 samples
    indices = np.random.RandomState(42).choice(len(X), size=2000, replace=False)
    X_subset = X[indices]
    y_subset = y[indices]

    X_train, X_test, y_train, y_test = train_test_split(
        X_subset, y_subset, test_size=500, random_state=42, stratify=y_subset
    )

    # Normalisation
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(f"Dataset: {len(X_train)} train, {len(X_test)} test")

    # 2. Sans entraînement
    print("\n1. Sans entrainement...")
    model_no_train = HAML(
        n_levels=3,
        n_attractors_per_class=2,
        max_steps=30,
        train_on_fit=False
    )
    model_no_train.fit(X_train, y_train)
    acc_no_train = model_no_train.score(X_test, y_test)
    print(f"   Accuracy: {acc_no_train:.3f}")

    # 3. Avec entraînement (rapide)
    print("\n2. Avec entrainement (5 epochs, optimized)...")
    model_train = HAML(
        n_levels=3,
        n_attractors_per_class=2,
        lr=0.01,
        n_epochs=5,
        batch_size=128,  # Plus gros = plus rapide
        max_steps=20,     # Réduit
        train_on_fit=True
    )

    start = time.time()
    model_train.fit(X_train, y_train)
    time_train = time.time() - start

    acc_train = model_train.score(X_test, y_test)

    print(f"\n3. Resultats")
    print("="*70)
    print(f"Sans entrainement:  {acc_no_train:.1%}")
    print(f"Avec entrainement:  {acc_train:.1%}")
    print(f"Amelioration:       +{(acc_train - acc_no_train)*100:.1f} points")
    print(f"Temps entrainement: {time_train:.1f}s")
    print("="*70)

    if acc_train > acc_no_train + 0.05:
        print("\n[SUCCESS] Entrainement ameliore significativement!")
    elif acc_train > acc_no_train:
        print("\n[OK] Entrainement ameliore les performances")
    else:
        print("\n[INFO] Augmenter n_epochs pour plus d'amelioration")


if __name__ == "__main__":
    main()
