"""
Test d'entraînement HAML avec gradient descent.

Compare :
- HAML sans entraînement (v0.2.0)
- HAML avec entraînement (v0.2.1)
- Baselines (KNN, SVM)
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_classification
from sklearn.model_selection import train_test_split
from haml import HAML
from haml.evaluation import run_benchmark
import time


def test_moons_with_training():
    """Test make_moons avec entraînement."""
    print("\n" + "="*70)
    print("TEST: make_moons avec entraînement par gradient")
    print("="*70)

    # Données
    X, y = make_moons(n_samples=300, noise=0.15, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    print(f"\nDataset: {len(X_train)} train, {len(X_test)} test")

    # 1. HAML sans entraînement (baseline v0.2.0)
    print("\n1. HAML sans entraînement (v0.2.0)...")
    model_no_train = HAML(
        n_levels=3,
        n_attractors_per_class=5,
        alpha_bu=1.0,
        alpha_td=1.0,
        max_steps=50,
        tol=1e-3,
        train_on_fit=False,  # Pas d'entraînement
        device='cpu'
    )

    start = time.time()
    model_no_train.fit(X_train, y_train)
    time_no_train = time.time() - start

    y_pred_no_train = model_no_train.predict(X_test)
    acc_no_train = np.mean(y_pred_no_train == y_test)

    print(f"   Accuracy: {acc_no_train:.3f}")
    print(f"   Time: {time_no_train:.2f}s")

    # 2. HAML avec entraînement (v0.2.1)
    print("\n2. HAML avec entraînement par gradient (v0.2.1)...")
    model_train = HAML(
        n_levels=3,
        n_attractors_per_class=5,
        alpha_bu=1.0,
        alpha_td=1.0,
        max_steps=30,  # Réduit (sera optimisé)
        tol=1e-3,
        lr=0.01,
        n_epochs=30,
        batch_size=32,
        train_on_fit=True,  # Entraînement activé
        device='cpu'
    )

    start = time.time()
    model_train.fit(X_train, y_train)
    time_train = time.time() - start

    y_pred_train = model_train.predict(X_test)
    acc_train = np.mean(y_pred_train == y_test)

    print(f"\n   Final accuracy: {acc_train:.3f}")
    print(f"   Total time: {time_train:.2f}s")

    # 3. Benchmark complet
    print("\n3. Benchmark avec baselines...")
    results_no_train = run_benchmark(model_no_train, X_train, y_train, X_test, y_test)
    results_train = run_benchmark(model_train, X_train, y_train, X_test, y_test)

    # 4. Comparaison
    print("\n" + "="*70)
    print("RÉSULTATS COMPARATIFS")
    print("="*70)
    print(f"HAML sans entraînement: {acc_no_train:.1%}")
    print(f"HAML avec entraînement: {acc_train:.1%}")
    print(f"KNN (baseline):         {results_train['KNN']['accuracy']:.1%}")
    print(f"SVM (baseline):         {results_train['SVM']['accuracy']:.1%}")
    print("="*70)

    improvement = (acc_train - acc_no_train) * 100
    print(f"\nAmélioration: +{improvement:.1f} points")

    if acc_train >= 0.95:
        print("[OK] Objectif atteint: accuracy >= 95%")
    else:
        print(f"[INFO] Proche de l'objectif (actuel: {acc_train:.1%})")

    return model_train, acc_train, improvement


def test_classification_with_training():
    """Test make_classification."""
    print("\n" + "="*70)
    print("TEST: make_classification avec entraînement")
    print("="*70)

    # Données
    X, y = make_classification(
        n_samples=400,
        n_features=10,
        n_informative=8,
        n_redundant=2,
        n_classes=3,
        n_clusters_per_class=2,
        random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Sans entraînement
    print("\n1. Sans entraînement...")
    model_no_train = HAML(
        n_levels=3,
        n_attractors_per_class=3,
        n_epochs=0,
        train_on_fit=False
    )
    model_no_train.fit(X_train, y_train)
    acc_no_train = model_no_train.score(X_test, y_test)
    print(f"   Accuracy: {acc_no_train:.3f}")

    # Avec entraînement
    print("\n2. Avec entraînement...")
    model_train = HAML(
        n_levels=3,
        n_attractors_per_class=3,
        lr=0.01,
        n_epochs=40,
        batch_size=32,
        train_on_fit=True
    )
    model_train.fit(X_train, y_train)
    acc_train = model_train.score(X_test, y_test)
    print(f"\n   Final accuracy: {acc_train:.3f}")

    improvement = (acc_train - acc_no_train) * 100
    print(f"\n   Amélioration: +{improvement:.1f} points")

    print("\n" + "="*70)

    return model_train, acc_train, improvement


if __name__ == "__main__":
    print("="*70)
    print("VALIDATION ENTRAÎNEMENT HAML v0.2.1")
    print("="*70)

    # Test 1: make_moons
    model1, acc1, imp1 = test_moons_with_training()

    # Test 2: make_classification
    model2, acc2, imp2 = test_classification_with_training()

    # Résumé final
    print("\n" + "="*70)
    print("RÉSUMÉ FINAL")
    print("="*70)
    print(f"make_moons:          {acc1:.1%} (+{imp1:.1f} pts)")
    print(f"make_classification: {acc2:.1%} (+{imp2:.1f} pts)")
    print("="*70)

    if acc1 >= 0.95 and acc2 >= 0.70:
        print("\n[SUCCESS] Objectifs atteints !")
    elif acc1 >= 0.90:
        print("\n[OK] Amélioration significative validée")
    else:
        print("\n[INFO] Entraînement fonctionnel, optimisation en cours")
