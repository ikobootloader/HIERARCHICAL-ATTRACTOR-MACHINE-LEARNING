"""
Test rapide d'entraînement HAML (5 epochs).
"""

import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from haml import HAML

print("="*70)
print("Quick Training Test (5 epochs)")
print("="*70)

# Données
X, y = make_moons(n_samples=200, noise=0.15, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

print(f"\nDataset: {len(X_train)} train, {len(X_test)} test")

# Sans entraînement
print("\n1. Sans entraînement...")
model_no_train = HAML(
    n_levels=3,
    n_attractors_per_class=5,
    train_on_fit=False
)
model_no_train.fit(X_train, y_train)
acc_no_train = model_no_train.score(X_test, y_test)
print(f"   Accuracy: {acc_no_train:.3f}")

# Avec entraînement (rapide)
print("\n2. Avec entraînement (5 epochs)...")
model_train = HAML(
    n_levels=3,
    n_attractors_per_class=5,
    lr=0.01,
    n_epochs=5,
    batch_size=32,
    max_steps=30,
    train_on_fit=True
)
model_train.fit(X_train, y_train)
acc_train = model_train.score(X_test, y_test)

print(f"\n3. Résultats")
print("="*70)
print(f"Sans entraînement:  {acc_no_train:.1%}")
print(f"Avec entraînement:  {acc_train:.1%}")
print(f"Amélioration:       +{(acc_train - acc_no_train)*100:.1f} points")
print("="*70)

if acc_train > acc_no_train:
    print("\n[SUCCESS] Entraînement améliore les performances !")
else:
    print("\n[INFO] Augmenter n_epochs pour plus d'amélioration")
