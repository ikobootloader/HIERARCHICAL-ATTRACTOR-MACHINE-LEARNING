"""
Diagnostic des forces d'attraction pour confirmer saturation gaussienne.
"""

import numpy as np
import torch
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from haml import HAML

print("="*70)
print("Diagnostic: Forces d'attraction en haute dimension")
print("="*70)

# Charger MNIST subset
print("\nLoading MNIST subset...")
mnist = fetch_openml('mnist_784', version=1, parser='auto')
X, y = mnist.data.to_numpy(), mnist.target.to_numpy().astype(int)

indices = np.random.RandomState(42).choice(len(X), size=1000, replace=False)
X_subset = X[indices][:500]  # 500 samples
y_subset = y[indices][:500]

# Normalisation
scaler = StandardScaler()
X_train = scaler.fit_transform(X_subset)

print(f"Data shape: {X_train.shape}")
print(f"Data stats: mean={X_train.mean():.3f}, std={X_train.std():.3f}")

# Créer modèle sans entraînement
print("\nInitializing HAML...")
model = HAML(
    n_levels=3,
    n_attractors_per_class=2,
    train_on_fit=False,
    device='cpu'
)
model.fit(X_train, y_subset)

# Diagnostic des forces
print("\n" + "="*70)
print("DIAGNOSTIC DES FORCES D'ATTRACTION")
print("="*70)

X_sample = torch.from_numpy(X_train[:10]).float()

for l, level in enumerate(model.levels):
    print(f"\nNiveau {l} (dim={level.dim}):")

    # Projeter vers ce niveau
    if l == 0:
        x_level = X_sample
    else:
        x_level = model.spaces.project_down(X_sample, 0, l)

    # Calculer distances aux attracteurs
    for c in range(level.n_classes):
        for m, attr in enumerate(level.attractors[str(c)]):
            # Distance
            dists = torch.norm(x_level - attr.position, dim=-1)
            dist_mean = dists.mean().item()

            # Forces d'attraction
            forces = attr.attraction_force(x_level)
            force_norms = torch.norm(forces, dim=-1)

            # Kernel gaussien
            sigma = attr.sigma.item()
            dist_sq = dists ** 2
            kernels = torch.exp(-dist_sq / (2 * sigma**2))

            print(f"  Attracteur [{c},{m}]: sigma={sigma:.3f}")
            print(f"    Distances: min={dists.min():.2f}, max={dists.max():.2f}, mean={dist_mean:.2f}")
            print(f"    Kernel exp(...): min={kernels.min():.2e}, max={kernels.max():.2e}, mean={kernels.mean():.2e}")
            print(f"    Force norms: min={force_norms.min():.2e}, max={force_norms.max():.2e}, mean={force_norms.mean():.2e}")

            # DIAGNOSTIC SATURATION
            if kernels.max() < 1e-6:
                print(f"    [SATURATION] Kernel < 1e-6 -> forces nulles")
            elif kernels.mean() < 1e-3:
                print(f"    [WARNING] Kernel moyen < 1e-3 -> forces faibles")

print("\n" + "="*70)
print("RECOMMANDATION")
print("="*70)

print("\nSi saturation detectee:")
print("  1. sigma_init devrait etre ~ sqrt(d) * data_std")
print(f"     Pour dim=784: sigma_init ~ sqrt(784) * 1.0 = {np.sqrt(784):.1f}")
print(f"     Pour dim=392: sigma_init ~ sqrt(392) * 1.0 = {np.sqrt(392):.1f}")
print(f"     Pour dim=196: sigma_init ~ sqrt(196) * 1.0 = {np.sqrt(196):.1f}")
print("\n  2. Ou utiliser PCA globale 784->50 en preprocessing")
print("="*70)
