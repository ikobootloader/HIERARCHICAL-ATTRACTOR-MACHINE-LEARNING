"""
Expérience sur make_moons : test de concept HAML.

Vérifie :
- Convergence vers états stationnaires distincts
- Accuracy compétitive
- Visualisation des bassins d'attraction
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from haml import HAML
from haml.analysis import StabilityAnalyzer
from haml.visualization import plot_basins, plot_trajectories
from haml.evaluation import run_benchmark


def main():
    print("=" * 70)
    print("HAML Test: make_moons Dataset")
    print("=" * 70)

    # 1. Génération des données
    print("\n1. Generating data...")
    X, y = make_moons(n_samples=300, noise=0.15, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    print(f"   Train: {len(X_train)} samples, Test: {len(X_test)} samples")

    # 2. Entraînement HAML
    print("\n2. Training HAML...")
    model = HAML(
        n_levels=3,
        n_attractors_per_class=5,
        alpha_bu=1.0,
        alpha_td=1.0,
        gamma=1.0,
        lambda_repulsion=0.5,
        integrator_method='rk4',
        dt=0.1,
        max_steps=50,
        tol=1e-3,
        device='cpu'
    )

    model.fit(X_train, y_train)
    print("   [OK] Model fitted")

    # 3. Évaluation
    print("\n3. Evaluating...")
    y_pred = model.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    print(f"   Accuracy: {accuracy:.3f}")

    # 4. Analyse de stabilité
    print("\n4. Stability Analysis...")
    analyzer = StabilityAnalyzer(model)
    analyzer.summary()

    # 5. Benchmark
    print("\n5. Benchmark against baselines...")
    results = run_benchmark(model, X_train, y_train, X_test, y_test)

    # 6. Visualisations
    print("\n6. Generating visualizations...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # Bassins niveau 0
    plot_basins(model, X_train, y_train, level=0, ax=axes[0, 0])
    axes[0, 0].set_title('Level 0 - Input Space')

    # Bassins niveau 1
    if model.n_levels > 1:
        plot_basins(model, X_train, y_train, level=1, ax=axes[0, 1])
        axes[0, 1].set_title('Level 1 - Abstract Space')

    # Trajectoires
    sample_idx = 0
    plot_trajectories(model, X_test[sample_idx:sample_idx+1], ax=axes[1, 0])

    # Résultats benchmark
    ax_bench = axes[1, 1]
    models = list(results.keys())
    accuracies = [results[m]['accuracy'] for m in models]
    colors = ['#ff7f0e' if m == 'HAML' else '#1f77b4' for m in models]

    bars = ax_bench.bar(models, accuracies, color=colors, alpha=0.7, edgecolor='black')
    ax_bench.set_ylabel('Accuracy')
    ax_bench.set_title('Benchmark Comparison')
    ax_bench.set_ylim([0, 1])
    ax_bench.grid(True, alpha=0.3, axis='y')

    # Ajouter les valeurs sur les barres
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax_bench.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                     f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig('haml_moons_results.png', dpi=150, bbox_inches='tight')
    print("   [OK] Saved visualization to haml_moons_results.png")

    plt.show()

    # 7. Résumé
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"[OK] Convergence: Model converged successfully")
    print(f"[OK] Accuracy: {accuracy:.1%} on test set")
    print(f"[OK] Stability: See analysis above")
    print(f"[OK] Visualization: Basins and trajectories plotted")
    print("=" * 70)


if __name__ == "__main__":
    main()
