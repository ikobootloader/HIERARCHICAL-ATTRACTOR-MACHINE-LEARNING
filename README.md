# HAML - Hierarchical Attractor Machine Learning

Système de classification supervisée basé sur une dynamique d'attracteurs hiérarchiques bidirectionnels.

## Description

HAML implémente un système dynamique multi-échelles où la classification émerge de la convergence vers des bassins d'attraction stables. L'architecture hiérarchique couple des niveaux de représentation de dimensions décroissantes via des flux bidirectionnels (bottom-up et top-down), inspirés du Predictive Coding.

### Caractéristiques principales

- **Convergence garantie** : système gradient dissipatif avec analyse de stabilité (théorème de LaSalle)
- **Scalabilité** : temps de convergence indépendant de la dimension (potentiel gaussien)
- **Expressivité hiérarchique** : gain exponentiel en nombre d'attracteurs nécessaires
- **Fondement bayésien** : équivalence MAP exacte sous conditions précises

### Distinction avec SA-nODE

- Hiérarchie de sous-espaces vs espace unique
- Attracteurs continus appris par gradient vs eigenvecteurs binaires fixés
- Couplage bidirectionnel entre niveaux
- Potentiel gaussien paramétrique vs double puits analytique

## Installation

```bash
pip install numpy scikit-learn matplotlib torch torchdiffeq scipy pyyaml
```

## Utilisation rapide

```python
from haml import HAML

# Initialisation
model = HAML(n_levels=3, n_attractors_per_class=5,
             alpha_bu=0.5, alpha_td=0.5)

# Entraînement
model.fit(X_train, y_train)

# Prédiction
y_pred = model.predict(X_test)

# Visualisation des bassins d'attraction
model.visualize_basins(X_test, level=0)
```

## Architecture

```
haml/
├── dynamics/          # Équations de mouvement, attracteurs, couplage
├── model/            # Interface principale HAML
├── training/         # Loss composite, optimiseur avec contraintes
├── projection/       # Tour de sous-espaces PCA
├── analysis/         # Analyse de stabilité spectrale
├── visualization/    # Bassins, trajectoires, énergie
├── evaluation/       # Benchmarks
└── experiments/      # Scripts reproductibles
```

## Fondements théoriques

### Dynamique couplée

Pour chaque niveau $l \in [0, L]$ :

$$\dot{\vec{x}}^{(l)} = \vec{F}^{(l)}_{intra} + \vec{F}^{(l)}_{bu} + \vec{F}^{(l)}_{td} - \gamma\dot{\vec{x}}^{(l)}$$

- $\vec{F}^{(l)}_{intra}$ : gradient du potentiel local (attractions/répulsions)
- $\vec{F}^{(l)}_{bu}$ : erreur de prédiction ascendante
- $\vec{F}^{(l)}_{td}$ : prédiction descendante
- $\gamma$ : amortissement

### Potentiel d'énergie

$$E^{(l)}(\vec{x}) = -\sum_{c,m} w^{(l)}_{c,m} \exp\left(-\frac{\|\vec{x} - \vec{\mu}^{(l)}_{c,m}\|^2}{2(\sigma^{(l)}_{c,m})^2}\right) + \text{termes répulsifs}$$

### Conditions de convergence

**C1** - Dominance locale : attracteur de la bonne classe le plus proche à chaque niveau
**C2** - Couplage suffisant : $\alpha_{bu} + \alpha_{td} > \alpha_{min}$
**C3** - Séparation des bassins : garantie par $\mathcal{L}_{sep}$

### Loss d'entraînement

$$\mathcal{L} = \mathcal{L}_{CE} + \mu_1 \mathcal{L}_{sep} + \mu_2 \mathcal{L}_{dyn}$$

## Résultats

### Version 0.2.2 (fix haute dimension)

**🔧 CORRECTION CRITIQUE** : Saturation gaussienne en haute dimension

En dimension $d$, les distances concentrent autour de $\sqrt{d} \cdot \sigma_{data}$. Avec $\sigma_{kernel} \sim 1.0$ fixe, le kernel gaussien $\exp(-||x-\mu||^2 / 2\sigma^2)$ sature vers 0 dès que $d > 10-20$.

**Impact MNIST (784D)** :
- **Avant fix** : kernel ≈ 0, forces = 0, loss stagne à 2.302 (CE initiale), gradients nuls
- **Après fix** : kernel ∈ [0.1, 0.9], forces actives, apprentissage possible

**Solution** : Rescaling $\sigma_{init} = \sqrt{d_l} \times \sigma_{data}$

```python
# haml/dynamics/level.py ligne 85
sigma = math.sqrt(self.dim) * data_std  # Au lieu de data_std seul
```

**Outil diagnostic** : `diagnose_forces.py` détecte automatiquement la saturation

⚠️ **Sans ce fix, HAML ne fonctionne PAS au-delà de ~20 dimensions.**

### Version 0.2.1 (avec entraînement par gradient)

| Dataset | Sans entraînement | Avec entraînement | Amélioration |
|---------|-------------------|-------------------|--------------|
| make_moons | 62-90% | **92-98%** | +8 à +30 pts |
| make_classification (3 classes) | 47% | **~75%** | +28 pts |

**Entraînement** :
- Stratégie 3 phases (indépendant → progressif → conjoint)
- Backprop à travers trajectoires ODE
- 5-30 epochs (1-5 min selon dataset)
- Optimisation positions μ, portées σ/ρ, poids w

### Version 0.2.0 (architecture sans entraînement)

Résultats avec attracteurs fixes après initialisation k-means :
- make_moons : 83-90%
- make_classification : 47%

## Roadmap

- [x] Proof of concept (v1)
- [x] Architecture modulaire complète (v2)
- [x] Modules découplés (dynamics, model, training)
- [x] Couplage bidirectionnel (F_bu, F_td)
- [x] Intégration ODE (Euler, RK4)
- [x] Analyse de stabilité spectrale
- [x] Visualisations (bassins, trajectoires)
- [x] Benchmarks vs KNN/SVM
- [x] **Entraînement par gradient (backprop ODE)**
- [x] **Couplage progressif (3 phases)**
- [x] **Amélioration +30 pts make_moons**
- [ ] Méthode adjointe (torchdiffeq)
- [ ] Tests MNIST complets
- [ ] Extension VAE génératif

## Références

- **SA-nODE** : Marino et al., 2024 - arXiv:2311.10387
- **Neural ODEs** : Chen et al., NeurIPS 2018
- **Predictive Coding** : Rao & Ballard, 1999 ; Friston, 2005
- **Modern Hopfield** : Ramsauer et al., ICLR 2021

## Licence

MIT

## Contact

Voir [CLAUDE.md](CLAUDE.md) pour les directives de développement.
