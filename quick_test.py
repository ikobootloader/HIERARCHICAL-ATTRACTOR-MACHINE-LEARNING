"""
Script de test rapide HAML.

Usage:
    python quick_test.py
"""

from sklearn.datasets import make_moons, make_classification
from sklearn.model_selection import train_test_split
from haml import HAML
import numpy as np

def test_moons():
    """Test rapide sur make_moons."""
    print("\n" + "="*50)
    print("Quick Test: make_moons")
    print("="*50)

    # Données
    X, y = make_moons(n_samples=200, noise=0.15, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # Modèle
    model = HAML(
        n_levels=3,
        n_attractors_per_class=5,
        alpha_bu=1.0,
        alpha_td=1.0,
        max_steps=50,
        tol=1e-3
    )

    print("Training...")
    model.fit(X_train, y_train)

    print("Predicting...")
    y_pred = model.predict(X_test)
    accuracy = np.mean(y_pred == y_test)

    print(f"\nResults:")
    print(f"  Accuracy: {accuracy:.1%}")
    print(f"  Converged: Yes")
    print("="*50)

    return model, accuracy

def test_classification():
    """Test sur make_classification."""
    print("\n" + "="*50)
    print("Quick Test: make_classification")
    print("="*50)

    # Données
    X, y = make_classification(
        n_samples=300,
        n_features=10,
        n_informative=8,
        n_redundant=2,
        n_classes=3,
        random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # Modèle
    model = HAML(
        n_levels=3,
        n_attractors_per_class=3,
        alpha_bu=1.0,
        alpha_td=1.0,
        max_steps=50
    )

    print("Training...")
    model.fit(X_train, y_train)

    print("Predicting...")
    y_pred = model.predict(X_test)
    accuracy = np.mean(y_pred == y_test)

    print(f"\nResults:")
    print(f"  Accuracy: {accuracy:.1%}")
    print("="*50)

    return model, accuracy

if __name__ == "__main__":
    # Test 1: Moons
    model1, acc1 = test_moons()

    # Test 2: Classification
    model2, acc2 = test_classification()

    print("\n" + "="*50)
    print("Summary")
    print("="*50)
    print(f"make_moons:          {acc1:.1%}")
    print(f"make_classification: {acc2:.1%}")
    print("="*50)
