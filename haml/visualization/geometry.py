"""
Module de visualisation géométrique.

Fonctions :
- plot_basins : Bassins d'attraction 2D
- plot_trajectories : Trajectoires de convergence
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import torch


def plot_basins(model, X, y, level=0, resolution=100, ax=None):
    """
    Visualise les bassins d'attraction en 2D.

    Args:
        model (HAML): Modèle entraîné
        X (np.ndarray): Données (n_samples, n_features)
        y (np.ndarray): Labels
        level (int): Niveau hiérarchique à visualiser
        resolution (int): Résolution de la grille
        ax (matplotlib axis, optional): Axes pour le plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    # Projeter les données au niveau l
    final_states = model.get_final_states(X)
    X_level = final_states[level]  # (n_samples, d_l)

    # Réduction à 2D si nécessaire
    if X_level.shape[1] > 2:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X_level)
    else:
        X_2d = X_level

    # Grille 2D
    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution)
    )

    # Prédiction sur la grille (approximatif car nécessite inverse PCA)
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    # Pour simplifier, on utilise les positions des attracteurs
    attractors_2d = []
    attractor_labels = []

    for c in range(model.n_classes_):
        positions = model.levels[level].get_all_attractor_positions(c)
        positions = positions.detach().cpu().numpy()

        if positions.shape[1] > 2:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            # Ajouter du bruit pour PCA
            if len(positions) < 2:
                positions = np.vstack([positions, positions + 0.01])
            positions = pca.fit_transform(positions)[:len(positions)]

        attractors_2d.append(positions)
        attractor_labels.extend([c] * len(positions))

    attractors_2d = np.vstack(attractors_2d)

    # Plot données
    scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap='viridis',
                        alpha=0.6, edgecolors='k', s=50)

    # Plot attracteurs
    ax.scatter(attractors_2d[:, 0], attractors_2d[:, 1],
              c=attractor_labels, cmap='viridis',
              marker='*', s=500, edgecolors='red', linewidths=2)

    ax.set_title(f'Basins of Attraction - Level {level}')
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    plt.colorbar(scatter, ax=ax, label='Class')

    return ax


def plot_trajectories(model, X_sample, max_steps=50, ax=None):
    """
    Visualise les trajectoires de convergence en 2D.

    Args:
        model (HAML): Modèle entraîné
        X_sample (np.ndarray): Échantillon (1, n_features)
        max_steps (int): Nombre de pas à visualiser
        ax (matplotlib axis, optional): Axes pour le plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    # Normaliser
    X_norm = model.scaler.transform(X_sample)
    X_torch = torch.from_numpy(X_norm).float().to(model.device)

    # Intégration avec trajectoire complète
    initial_states = model.spaces(X_torch)

    # Sauvegarder le max_steps original
    original_max_steps = model.integrator.max_steps
    model.integrator.max_steps = max_steps

    final_states, n_steps, converged, trajectory = model.integrator.integrate(
        initial_states, return_trajectory=True
    )

    # Restaurer
    model.integrator.max_steps = original_max_steps

    # Visualiser niveau 0 (2D)
    level_0_traj = [states[0].detach().cpu().numpy()[0] for states in trajectory]
    level_0_traj = np.array(level_0_traj)  # (n_steps, d_0)

    # Réduction à 2D si nécessaire
    if level_0_traj.shape[1] > 2:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        level_0_traj_2d = pca.fit_transform(level_0_traj)
    else:
        level_0_traj_2d = level_0_traj

    # Plot trajectoire
    ax.plot(level_0_traj_2d[:, 0], level_0_traj_2d[:, 1],
           'b-', linewidth=2, alpha=0.7, label='Trajectory')

    # Points initial et final
    ax.scatter(level_0_traj_2d[0, 0], level_0_traj_2d[0, 1],
              c='green', marker='o', s=200, label='Initial', zorder=5)
    ax.scatter(level_0_traj_2d[-1, 0], level_0_traj_2d[-1, 1],
              c='red', marker='X', s=200, label='Final', zorder=5)

    ax.set_title(f'Convergence Trajectory ({n_steps} steps)')
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.legend()
    ax.grid(True, alpha=0.3)

    return ax


def plot_energy_landscape(model, X, y, level=0, resolution=50):
    """
    Visualise le paysage d'énergie E^(l)(x).

    Args:
        model (HAML): Modèle entraîné
        X (np.ndarray): Données de référence
        y (np.ndarray): Labels
        level (int): Niveau hiérarchique
        resolution (int): Résolution de la grille
    """
    fig, ax = plt.subplots(figsize=(12, 10))

    # Projeter au niveau l
    final_states = model.get_final_states(X)
    X_level = final_states[level]

    # Réduction 2D
    if X_level.shape[1] > 2:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X_level)
    else:
        X_2d = X_level

    # Grille
    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution)
    )

    # Note: Calcul d'énergie nécessite de passer par le modèle complet
    # Simplifié ici pour démonstration

    ax.set_title(f'Energy Landscape - Level {level}')
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')

    print("Note: Energy landscape computation requires full model evaluation (not implemented in simplified version)")

    return fig, ax
