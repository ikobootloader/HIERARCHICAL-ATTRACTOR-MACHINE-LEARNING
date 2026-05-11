"""
Métriques de diagnostic pour l'entraînement HAML.

Fonctions utilitaires pour:
- Accuracy par niveau hiérarchique
- Distribution des forces d'attraction
"""

import numpy as np
import torch


def compute_level_accuracy(model, X, y):
    """
    Calcule l'accuracy de prédiction pour chaque niveau indépendamment.

    Args:
        model: Instance HAML entraînée
        X (np.ndarray): Données normalisées (n_samples, n_features)
        y (np.ndarray): Labels vrais (n_samples,)

    Returns:
        list[float]: Accuracy par niveau
    """
    X_torch = torch.from_numpy(X).float().to(model.device)
    y_torch = torch.from_numpy(y).long().to(model.device)

    model.eval()
    with torch.no_grad():
        final_states, _, _ = model.forward(X_torch)
        level_accuracies = []

        for level, state in zip(model.levels, final_states):
            scores = level.predict_class_scores(state)
            preds = torch.argmax(scores, dim=1)
            acc = (preds == y_torch).float().mean().item()
            level_accuracies.append(acc)

    return level_accuracies


def compute_attraction_force_stats(model, X):
    """
    Calcule des statistiques de force d'attraction par niveau.

    Args:
        model: Instance HAML entraînée
        X (np.ndarray): Données (non normalisées, sera normalisé via model.scaler)

    Returns:
        dict: Statistiques par niveau (mean, std, p10, p50, p90)
    """
    X_norm = model.scaler.transform(X)
    X_torch = torch.from_numpy(X_norm).float().to(model.device)

    model.eval()
    with torch.no_grad():
        final_states, _, _ = model.forward(X_torch)
        stats = {}

        for level_idx, (level, state) in enumerate(zip(model.levels, final_states)):
            force = level.intra_level_force(state)
            norm = torch.norm(force, dim=1).detach().cpu().numpy()

            stats[f"level_{level_idx}"] = {
                "mean": float(np.mean(norm)),
                "std": float(np.std(norm)),
                "p10": float(np.percentile(norm, 10)),
                "p50": float(np.percentile(norm, 50)),
                "p90": float(np.percentile(norm, 90)),
            }

    return stats
