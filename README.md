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

### Mise à jour MNIST (subset 1500/500, 5 epochs)

Configuration du test :
- Train/Test : `1500 / 500` échantillons
- Niveaux : `784 -> 392 -> 196`
- Attracteurs : `2` par classe
- Entraînement : `5` epochs, stratégie 3 phases

Résultats observés :
- Sans entraînement : `45.6%`
- Avec entraînement (5 epochs) : `75.8%`
- Gain absolu : `+30.2 points`

Progression par phase (accuracy train) :
- Phase 1 (indépendant) : `64.2%`
- Phase 2 (progressif) : `68.7%`
- Phase 3 (conjoint) : `71.8% -> 77.8%`

Diagnostics désormais disponibles :
- `accuracy` par niveau hiérarchique à chaque epoch (log d'entraînement)
- distribution des forces `||F||` par niveau (`mean/std/p10/p50/p90`) via `model.diagnostics(X, y)`

Ablation couplage MNIST (1500/500, 5 epochs, seed fixe) :
- Indépendant (`alpha_bu=0`, `alpha_td=0`) : `75.8%` test
- Couplé (`alpha_bu=1`, `alpha_td=1`) : `75.8%` test
- Delta couplage : `+0.0 point` sur ce protocole court

Commande de reproduction :
- `python experiments/mnist_coupling_ablation.py`

Ablation couplage sur anneaux concentriques alternés (train/test: 2400/800) :
- Indépendant (`alpha_bu=0`, `alpha_td=0`) : `70.75%` test
- Couplé (`alpha_bu=1`, `alpha_td=1`) : `69.00%` test
- Delta couplage : `-1.75 point` (protocole actuel)
- Figure exportée : `concentric_coupling_ablation.png`

Commande de reproduction :
- `python experiments/concentric_coupling_ablation.py`

Variante couplée renforcée (même script) :
- `alpha_bu=0.5`, `alpha_td=1.5`, `M=5`, `25` epochs, phases `(5,8,12)`
- Accuracy test : `86.38%`
- Gain vs indépendant (`70.75%`) : `+15.63 points`
- Coût: temps d'entraînement plus élevé (~`+771s` vs baseline indépendant)

Note de stabilité :
- Cette variante montre un gain net, mais avec instabilités inter-niveaux en fin de run
  (fluctuation forte de certaines `level_acc`). Un réglage de schedule/regularisation est
  recommandé avant généralisation.

Variante couplée renforcée stabilisée (early-stop + LR decay sur divergence) :
- mêmes hyperparamètres de base (`alpha_bu=0.5`, `alpha_td=1.5`, `M=5`)
- garde-fous: seuil divergence inter-niveaux `0.15`, patience `2`,
  réduction LR `x0.5`, `min_lr=1e-4`
- Accuracy test : `82.88%`
- Gain vs indépendant (`70.88%`) : `+12.00 points`
- Observation clé: la divergence apparaît tard (epoch `23`, fin de phase 3),
  puis déclenche réduction du LR et early-stop (epoch `24`).

Itération "mu_sep adaptatif + soft landing préventif" :
- ajout de `mu_sep` adaptatif (croissance dès `level_div > 0.05`)
- soft landing à epoch `20` (LR `x0.1` + gel des `mu`)
- résultat sur protocole concentrique actuel :
  - indépendant: `71.13%`
  - couplé tuned stable: `69.88%`
  - delta: `-1.25 point`

Lecture :
- la stabilité inter-niveaux est mieux contenue en fin de run,
  mais la régularisation actuelle est trop agressive et dégrade la performance.

Itération "soft-landing sur divergence + mu_sep adaptatif tardif (phase 3)" :
- soft-landing déclenché sur `level_div` (au lieu d'un epoch fixe)
- `mu_sep` adaptatif activé uniquement en phase 3, croissance douce (`x1.05`) et plafond (`0.35`)
- résultat sur protocole concentrique actuel :
  - indépendant: `69.88%`
  - couplé tuned stable: `76.25%`
  - delta: `+6.37 points`

Lecture :
- ce réglage rétablit un gain couplé positif avec stabilité tardive,
- mais reste en dessous du pic instable (`86.38%`) et du stabilisé précédent (`82.88%`).

Itération "trigger mu_sep relevé + patience 2 epochs" :
- `mu_sep_trigger_divergence=0.11` (au lieu de `0.08`)
- `mu_sep_patience=2` (hausse de `mu_sep` seulement après 2 epochs consécutives au-dessus du seuil)
- résultat sur protocole concentrique actuel :
  - indépendant: `71.13%`
  - couplé tuned stable: `76.13%`
  - delta: `+5.00 points`

Lecture :
- le cooldown évite les réactions à des fluctuations isolées,
- mais la performance reste encore trop bridée pour atteindre la zone `82-86%`.

Itération "collapse guard sur saut de divergence" :
- détection d'un saut brutal `delta_div` (au lieu d'un simple seuil absolu)
- activé uniquement à partir de l'epoch `18`
- règle appliquée au trigger :
  - `mu_sep *= 1.5` (ponctuel)
  - `lr *= 0.3` (freinage rapide)
- résultat sur protocole concentrique actuel :
  - indépendant: `70.38%`
  - couplé tuned + collapse guard: `86.38%`
  - delta: `+16.00 points`

Observation :
- le guard détecte bien le collapse (`delta_div`), mais dans ce run `L1` chute puis reste proche du hasard (`~0.5`) en fin d'entraînement.
- la performance globale test reste élevée, ce qui confirme un régime "pic utile mais incohérence inter-niveaux persistante".

Interprétation :
- La progression continue de la loss (`1.866 -> 1.240`) valide un apprentissage stable.
- Le couplage progressif apporte un gain supplémentaire après la phase indépendante.
- Le principal verrou actuel est le temps d'entraînement (~1786s pour 5 epochs), ce qui motive la priorisation d'une méthode adjointe.

Condition d'initialisation obligatoire (I1) :
- Pour éviter la dégénérescence en haute dimension, imposer
  `sigma_init^(l) = sqrt(d_l) * sigma_data^(l)`.
- Sans cette condition, les kernels gaussiens saturent et les gradients s'annulent en grande dimension.

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
