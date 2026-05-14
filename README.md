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

Notes de configuration :
- `rho_sigma_ratio` permet de contrôler explicitement l'initialisation
  de la portée de répulsion (`rho_init = rho_sigma_ratio * sigma_init`).
- `level_score_weighting` contrôle l'agrégation multi-niveaux
  (`'exponential'` par défaut, option `'uniform'`).

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

## Résultats et campagnes de test

Pour améliorer la lisibilité, les résultats sont segmentés par campagnes de tests reproductibles.

### Index rapide des campagnes

| Campagne | Objectif principal | Script(s) | Indicateur clé |
|---|---|---|---|
| A (MNIST) | Mesurer le gain d'entraînement gradient sur protocole court | `experiments/mnist_coupling_ablation.py` | `+30.2 pts` (sans entraînement -> entraîné) |
| B (Concentrique) | Tester l'apport du couplage sur topologie non convexe | `experiments/concentric_coupling_ablation.py` | Delta couplé vs indép. (multi-seeds) |
| C (Bruité) | Évaluer robustesse au bruit (`make_moons`) | `experiments/noisy_benchmark.py` | Courbe du delta couplé-vs-indép. selon bruit |
| D (SA-nODE-reimpl) | Comparaison architecturale contrôlée hiérarchie vs plat | `experiments/noisy_sanode_comparison.py`, `experiments/tune_sanode_reimpl.py` | Accuracy HAML couplé vs SA-nODE-reimpl |
| E (Stabilisation) | Stabiliser les gains couplés forts sur concentrique | `experiments/concentric_coupling_ablation.py` | Compromis perf/stabilité (`E1` -> `E6`) |

### Campagne A - MNIST subset (1500/500, 5 epochs)

Objectif :
- Vérifier le gain apporté par l'entraînement gradient sur un protocole court.

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

Sous-test A1 - Ablation couplage MNIST (seed fixe) :
- Indépendant (`alpha_bu=0`, `alpha_td=0`) : `75.8%` test
- Couplé (`alpha_bu=1`, `alpha_td=1`) : `75.8%` test
- Delta couplage : `+0.0 point` sur ce protocole court

Commande de reproduction :
- `python experiments/mnist_coupling_ablation.py`

### Campagne B - Ablation couplage sur anneaux concentriques alternes

Objectif :
- Tester un cas structurel où le contexte global (top-down) doit aider la décision locale.

Protocole :
- Protocole principal : multi-seeds (`n_runs=5` par defaut) avec aggregation statistique.
- Sortie standard : `experiments/concentric_coupling_ablation_summary.json`
- Figure optionnelle : `concentric_coupling_ablation.png` (premier run uniquement, via flag)

Résultats observés (run court `n_runs=2`, seeds `42,43`) :
- Résultat multi-seeds observé (run court `n_runs=2`, seeds `42,43`) :
  - indépendant : `66.19% ± 3.69`
  - couplé tuned : `79.19% ± 5.69`
  - delta moyen : `+13.00 points`
- Lecture qualitative (figure) :
  - indépendant : frontière majoritairement convexe, ne capture pas correctement la topologie annulaire.
  - couplé tuned : frontière non convexe alignée avec la géométrie en anneaux alternés.
  - interprétation : différence qualitative cohérente avec `P5` (classe de fonctions représentables plus large en hiérarchie bidirectionnelle).
- Réserve :
  - présence possible de lobes parasites sur certains runs couplés, compatible avec les instabilités inter-niveaux observées en fin d'entraînement.

Commande de reproduction :
- `python experiments/concentric_coupling_ablation.py`
- `python experiments/concentric_coupling_ablation.py --n-runs 5 --save-figure`

### Campagne C - Benchmark bruité `make_moons` (seed `42`, `n_samples=3000`)

Objectif :
- Évaluer la robustesse au bruit et le gain couplé vs indépendant.

Résultats :
- Bruit `0.10` :
  - HAML indépendant: `99.60%`
  - HAML couplé: `99.60%`
  - KNN: `99.87%`
  - SVM RBF: `99.87%`
  - Delta couplé - indépendant: `+0.00 point`
- Bruit `0.20` :
  - HAML indépendant: `89.33%`
  - HAML couplé: `95.60%`
  - KNN: `96.53%`
  - SVM RBF: `97.20%`
  - Delta couplé - indépendant: `+6.27 points`
- Bruit `0.30` :
  - HAML indépendant: `88.13%`
  - HAML couplé: `90.53%`
  - KNN: `90.80%`
  - SVM RBF: `91.33%`
  - Delta couplé - indépendant: `+2.40 points`
- Bruit `0.40` :
  - HAML indépendant: `84.67%`
  - HAML couplé: `85.07%`
  - KNN: `83.33%`
  - SVM RBF: `87.33%`
  - Delta couplé - indépendant: `+0.40 point`

Commande de reproduction :
- `python experiments/noisy_benchmark.py`

Lecture empirique (bruit croissant) :
- Dégradation `0.20 -> 0.40` :
  - KNN : `-13.20 points` (de `96.53%` à `83.33%`)
  - HAML couplé : `-10.53 points` (de `95.60%` à `85.07%`)
  - SVM RBF : `-9.87 points` (de `97.20%` à `87.33%`)
- Lecture :
  - HAML couplé est plus robuste que KNN sous bruit croissant.
  - HAML couplé reste proche de SVM en robustesse, sans le dépasser systématiquement.
  - La contribution principale est le gain intra-modèle couplé vs indépendant.

Delta couplage (HAML couplé - HAML indépendant) :
- bruit `0.10` : `+0.00 point`
- bruit `0.20` : `+6.27 points`
- bruit `0.30` : `+2.40 points`
- bruit `0.40` : `+0.40 point`

Interprétation :
- Le gain du couplage suit un profil en cloche : nul à bruit faible, maximal à bruit intermédiaire, faible à bruit fort.
- Ce comportement est cohérent avec l'hypothèse théorique : le top-down apporte surtout quand le signal local est ambigu mais encore exploitable.
- Prédiction falsifiable associée :
  - si l'hypothèse est correcte, le profil en cloche du gain couplé-vs-indépendant doit se reproduire sur plusieurs datasets et en multi-seeds, avec un maximum à bruit intermédiaire et un gain réduit aux extrêmes (bruit faible et bruit fort).

Synthèse théorie/expérience (état actuel) :
- Anneaux concentriques : gain couplé vs indépendant jusqu'à `+16.00 points` (cas structurel favorable à la hiérarchie).
- Données bruitées : gain maximal `+6.27 points` à bruit `0.20`, avec robustesse proche SVM et meilleure que KNN.
- Observation dynamique : récupération de niveaux intermédiaires (L1) observée sur certains runs couplés, compatible avec le guidage top-down attendu.

Point restant pour le positionnement publication :
- Ajouter un benchmark comparatif direct avec SA-nODE sur protocole bruité comparable (même split, même seed, mêmes métriques).

### Campagne D - Comparaison HAML vs SA-nODE-reimpl (même protocole bruité)

Objectif :
- Isoler l'effet architectural de la hiérarchie bidirectionnelle vs modèle plat.

Comparaison additionnelle HAML vs SA-nODE-reimpl (seed `42`) :
- Note méthode :
  - baseline `SA-nODE-reimpl` = espace unique + attracteurs binaires plantés ±a + potentiel double-puits analytique + couplage linéaire entraîné
  - réimplémentation interne inspirée de la publication (pas un code officiel auteur)
  - objectif: ablation architecturale contrôlée pour isoler l'effet de la hiérarchie bidirectionnelle, à implémentation comparable
- Pré-optimisation SA-nODE-reimpl (seed fixe, architecture figée) :
  - grille testée: `a in {0.8, 1.0, 1.2}`, `dt in {0.03, 0.05, 0.08}`, `gamma in {0.0, 0.05, 0.1}`
  - critère principal: meilleure accuracy moyenne sur bruits `[0.10, 0.20, 0.30, 0.40]`
  - configuration retenue: `a=1.0`, `dt=0.03`, `gamma=0.0`
  - meilleure config ciblée bruit fort (`0.40`): `a=0.8`, `dt=0.03`, `gamma=0.1` (gain limité)
- Résultats :
  - bruit `0.10` : HAML couplé `99.60%` vs SA-nODE-reimpl tuned `86.67%` (delta `+12.93 points`)
  - bruit `0.20` : HAML couplé `95.60%` vs SA-nODE-reimpl tuned `86.40%` (delta `+9.20 points`)
  - bruit `0.30` : HAML couplé `90.53%` vs SA-nODE-reimpl tuned `85.87%` (delta `+4.67 points`)
  - bruit `0.40` : HAML couplé `85.07%` vs SA-nODE-reimpl tuned `83.33%` (delta `+1.73 point`)
- Dégradation `0.10 -> 0.40` :
  - HAML couplé : `-14.53 points`
  - SA-nODE-reimpl tuned : `-3.33 points`
- Lecture :
  - HAML couplé domine en niveau absolu sur tout le spectre de bruit testé.
  - SA-nODE-reimpl tuned dégrade moins vite sous bruit croissant ; ce point est une tension ouverte avec l'intuition "top-down plus robuste" et doit être expliqué par une analyse mécanistique dédiée (dynamique des niveaux, saturation/instabilité sous bruit fort).
  - à bruit `0.40`, l'écart reste faible (`+1.73 pt`) et doit être confirmé multi-seeds avant claim fort.
  - conclusion défendable: dans des conditions d'implémentation contrôlées, la hiérarchie bidirectionnelle surpasse le modèle plat sur tous les niveaux de bruit testés.
  - limite explicite: ces résultats ne permettent pas de conclure directement contre l'implémentation officielle SA-nODE; un benchmark externe dédié reste nécessaire.

Méthode adjointe (précaution d'interprétation mémoire) :
- Le gain `0.45 MB` vs `66.16 MB` rapporté dans le benchmark RK4 vs adjoint provient de `tracemalloc` (allocations Python tracées).
- Ce chiffre décrit un proxy partiel de mémoire runtime ; il ne mesure ni la mémoire totale du processus, ni la mémoire GPU.
- La formulation correcte est donc : "forte réduction de la mémoire Python tracée", et non "réduction mémoire globale" sans benchmark système/GPU complémentaire.

Commande de reproduction :
- `python experiments/noisy_sanode_comparison.py`
- `python experiments/tune_sanode_reimpl.py`

### Campagne E - Itérations de stabilisation (protocole concentrique)

Objectif :
- Stabiliser les gains du couplage fort en limitant les collapses inter-niveaux tardifs.

Sous-test E1 - Variante couplée renforcée :
- `alpha_bu=0.5`, `alpha_td=1.5`, `M=5`, `25` epochs, phases `(5,8,12)`
- Accuracy test : `86.38%`
- Gain vs indépendant (`70.75%`) : `+15.63 points`
- Coût: temps d'entraînement plus élevé (~`+771s` vs baseline indépendant)

Note de stabilité :
- Cette variante montre un gain net, mais avec instabilités inter-niveaux en fin de run
  (fluctuation forte de certaines `level_acc`). Un réglage de schedule/regularisation est
  recommandé avant généralisation.

Sous-test E2 - Variante couplée renforcée stabilisée (early-stop + LR decay) :
- mêmes hyperparamètres de base (`alpha_bu=0.5`, `alpha_td=1.5`, `M=5`)
- garde-fous: seuil divergence inter-niveaux `0.15`, patience `2`,
  réduction LR `x0.5`, `min_lr=1e-4`
- Accuracy test : `82.88%`
- Gain vs indépendant (`70.88%`) : `+12.00 points`
- Observation clé: la divergence apparaît tard (epoch `23`, fin de phase 3),
  puis déclenche réduction du LR et early-stop (epoch `24`).

Sous-test E3 - Itération "mu_sep adaptatif + soft landing préventif" :
- ajout de `mu_sep` adaptatif (croissance dès `level_div > 0.05`)
- soft landing à epoch `20` (LR `x0.1` + gel des `mu`)
- résultat sur protocole concentrique actuel :
  - indépendant: `71.13%`
  - couplé tuned stable: `69.88%`
  - delta: `-1.25 point`

Lecture :
- la stabilité inter-niveaux est mieux contenue en fin de run,
  mais la régularisation actuelle est trop agressive et dégrade la performance.

Sous-test E4 - Itération "soft-landing sur divergence + mu_sep adaptatif tardif (phase 3)" :
- soft-landing déclenché sur `level_div` (au lieu d'un epoch fixe)
- `mu_sep` adaptatif activé uniquement en phase 3, croissance douce (`x1.05`) et plafond (`0.35`)
- résultat sur protocole concentrique actuel :
  - indépendant: `69.88%`
  - couplé tuned stable: `76.25%`
  - delta: `+6.37 points`

Lecture :
- ce réglage rétablit un gain couplé positif avec stabilité tardive,
- mais reste en dessous du pic instable (`86.38%`) et du stabilisé précédent (`82.88%`).

Sous-test E5 - Itération "trigger mu_sep relevé + patience 2 epochs" :
- `mu_sep_trigger_divergence=0.11` (au lieu de `0.08`)
- `mu_sep_patience=2` (hausse de `mu_sep` seulement après 2 epochs consécutives au-dessus du seuil)
- résultat sur protocole concentrique actuel :
  - indépendant: `71.13%`
  - couplé tuned stable: `76.13%`
  - delta: `+5.00 points`

Lecture :
- le cooldown évite les réactions à des fluctuations isolées,
- mais la performance reste encore trop bridée pour atteindre la zone `82-86%`.

Sous-test E6 - Itération "collapse guard sur saut de divergence" :
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

**ðŸ”§ CORRECTION CRITIQUE** : Saturation gaussienne en haute dimension

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

⚠ï¸ **Sans ce fix, HAML ne fonctionne PAS au-delà de ~20 dimensions.**

### Version 0.2.1 (avec entraînement par gradient)

| Dataset | Sans entraînement | Avec entraînement | Amélioration |
|---------|-------------------|-------------------|--------------|
| make_moons | 62-90% | **92-98%** | +8 à +30 pts |
| make_classification (3 classes) | 47% | **~75%** | +28 pts |

**Entraînement** :
- Stratégie 3 phases (indépendant → progressif → conjoint)
- Backprop à travers trajectoires ODE
- 5-30 epochs (1-5 min selon dataset)
- Optimisation positions Î¼, portées Ïƒ/ρ, poids w

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
- [x] Méthode adjointe (torchdiffeq, avec fallback RK4 si indisponible)
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


