"""
Test HAML sur MNIST avec entraînement par gradient.

Compare performances sans/avec entraînement sur dataset 10 classes.
Objectif spec : accuracy >= 85%
"""

import numpy as np
import time
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from haml import HAML
from haml.evaluation import run_benchmark


def load_mnist_subset(n_train=5000, n_test=1000, random_state=42):
    """
    Charge un subset de MNIST pour test rapide.

    Args:
        n_train (int): Nombre d'exemples d'entraînement
        n_test (int): Nombre d'exemples de test
        random_state (int): Seed

    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    print("\nChargement MNIST...")

    # Charger MNIST
    mnist = fetch_openml('mnist_784', version=1, parser='auto')
    X, y = mnist.data.to_numpy(), mnist.target.to_numpy().astype(int)

    # Subset aléatoire
    total_samples = n_train + n_test
    indices = np.random.RandomState(random_state).choice(
        len(X), size=total_samples, replace=False
    )
    X_subset = X[indices]
    y_subset = y[indices]

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_subset, y_subset,
        test_size=n_test,
        random_state=random_state,
        stratify=y_subset
    )

    # Normalisation
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(f"Dataset: {len(X_train)} train, {len(X_test)} test")
    print(f"Features: {X_train.shape[1]}")
    print(f"Classes: {len(np.unique(y_train))}")

    return X_train, X_test, y_train, y_test


def test_mnist_no_training():
    """Test MNIST sans entraînement."""
    print("\n" + "="*70)
    print("TEST 1: MNIST sans entraînement (baseline)")
    print("="*70)

    # Données
    X_train, X_test, y_train, y_test = load_mnist_subset(
        n_train=5000, n_test=1000
    )

    # Modèle sans entraînement
    print("\nHAML sans entraînement...")
    model = HAML(
        n_levels=4,
        n_attractors_per_class=3,
        alpha_bu=1.0,
        alpha_td=1.0,
        max_steps=50,
        tol=1e-3,
        train_on_fit=False,
        device='cpu'
    )

    start = time.time()
    model.fit(X_train, y_train)
    time_fit = time.time() - start

    # Évaluation
    y_pred = model.predict(X_test)
    accuracy = np.mean(y_pred == y_test)

    print(f"\nRésultats:")
    print(f"  Accuracy: {accuracy:.1%}")
    print(f"  Time: {time_fit:.2f}s")

    # Benchmark
    print("\nBenchmark avec baselines...")
    results = run_benchmark(model, X_train, y_train, X_test, y_test)

    print("\nComparaison:")
    print(f"  HAML:  {accuracy:.1%}")
    print(f"  KNN:   {results['KNN']['accuracy']:.1%}")
    print(f"  SVM:   {results['SVM']['accuracy']:.1%}")

    return accuracy


def test_mnist_with_training():
    """Test MNIST avec entraînement par gradient."""
    print("\n" + "="*70)
    print("TEST 2: MNIST avec entraînement par gradient")
    print("="*70)

    # Données
    X_train, X_test, y_train, y_test = load_mnist_subset(
        n_train=5000, n_test=1000
    )

    # Modèle avec entraînement
    print("\nHAML avec entraînement...")
    model = HAML(
        n_levels=4,
        n_attractors_per_class=3,
        alpha_bu=1.0,
        alpha_td=1.0,
        max_steps=30,
        tol=1e-3,
        lr=0.005,  # LR plus petit pour MNIST
        n_epochs=20,
        batch_size=64,
        train_on_fit=True,
        device='cpu'
    )

    start = time.time()
    model.fit(X_train, y_train)
    time_fit = time.time() - start

    # Évaluation
    y_pred = model.predict(X_test)
    accuracy = np.mean(y_pred == y_test)

    print(f"\nRésultats finaux:")
    print(f"  Accuracy: {accuracy:.1%}")
    print(f"  Time: {time_fit:.2f}s")

    # Benchmark
    print("\nBenchmark avec baselines...")
    results = run_benchmark(model, X_train, y_train, X_test, y_test)

    print("\nComparaison:")
    print(f"  HAML:  {accuracy:.1%}")
    print(f"  KNN:   {results['KNN']['accuracy']:.1%}")
    print(f"  SVM:   {results['SVM']['accuracy']:.1%}")

    return accuracy


def main():
    """Point d'entrée principal."""
    print("="*70)
    print("VALIDATION HAML v0.2.1 sur MNIST")
    print("="*70)

    # Test 1: Sans entraînement
    acc_no_train = test_mnist_no_training()

    # Test 2: Avec entraînement
    acc_train = test_mnist_with_training()

    # Résumé
    print("\n" + "="*70)
    print("RÉSUMÉ FINAL")
    print("="*70)
    print(f"Sans entraînement:  {acc_no_train:.1%}")
    print(f"Avec entraînement:  {acc_train:.1%}")
    print(f"Amélioration:       +{(acc_train - acc_no_train)*100:.1f} points")
    print("="*70)

    # Validation objectif
    if acc_train >= 0.85:
        print("\n[SUCCESS] Objectif spec atteint (>= 85%)")
    elif acc_train >= 0.80:
        print("\n[OK] Proche de l'objectif spec (80-85%)")
    else:
        print(f"\n[INFO] Amélioration possible (actuel: {acc_train:.1%})")

    return acc_train


if __name__ == "__main__":
    main()
