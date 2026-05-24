# HAML - Hierarchical Attractor Machine Learning

Système de classification supervisée basé sur une dynamique d'attracteurs hiérarchiques bidirectionnels.

DOI (archive) : https://doi.org/10.5281/zenodo.20238600

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
- `use_vectorized_levels=True` active un niveau dynamique vectorisé
  (accélération forward/backward, comportement par défaut inchangé si `False`).
- `repulsion_mode` contrôle la sémantique de répulsion dans le niveau vectorisé
  (`'global'` par défaut, option `'inter_class_only'`).
- `convergence_check_every` contrôle la fréquence de vérification de convergence
  dans l'intégrateur ODE (défaut `5`).
- `tol=None` permet de désactiver complètement l'arrêt anticipé de convergence
  (intégration systématique jusqu'à `max_steps`).
- `mu_dyn=0.0` (profilage) évite de stocker inutilement la trajectoire ODE
  complète pendant l'entraînement.
- Le planning d'entraînement supporte un budget de pas par phase :
  - `phase1_max_steps`, `phase2_max_steps`, `phase3_max_steps`
  (utile pour réduire la profondeur de graphe en début d'entraînement).
- Le planning d'entraînement supporte aussi une méthode d'intégration par phase :
  - `phase1_integrator_method`, `phase2_integrator_method`, `phase3_integrator_method`
  (ex: `euler` en phase 1/2 puis `rk4` en phase 3).
- Ces paramètres sont disponibles directement dans `HAML(...)` et appliqués par
  `train_model()` (pas seulement via le script de profilage).
- Preset pratique :
  - `training_preset='fast_train_cpu'` applique automatiquement :
    - `phase1_max_steps=20`, `phase2_max_steps=40`, `phase3_max_steps=100`
    - `phase1_integrator_method='euler'`, `phase2_integrator_method='euler'`,
      `phase3_integrator_method='rk4'`
    - `learn_projections=True` (si non explicitement fourni)
  - les paramètres explicitement fournis dans `HAML(...)` restent prioritaires.
- Preset ultra-rapide CPU :
  - `training_preset='ultra_fast_train_cpu'` applique automatiquement :
    - `phase1_max_steps=20`, `phase2_max_steps=40`, `phase3_max_steps=100`
    - `phase1_integrator_method='euler'`, `phase2_integrator_method='euler'`,
      `phase3_integrator_method='euler'`
    - `learn_projections=True` (si non explicitement fourni)
  - à utiliser quand l'objectif principal est la réduction maximale du temps de train.
- Presets dédiés Fashion-MNIST CPU :
  - `training_preset='fashion_cpu_accuracy'` applique :
    - `sigma_init_mode='median_pairwise'`
    - `learn_projections=True` (si non explicitement fourni)
    - `phase1/2/3_max_steps=20/40/100`
    - `phase1/2/3_integrator_method='euler'/'euler'/'euler'`
  - `training_preset='fashion_cpu_runtime'` applique :
    - `sigma_init_mode='sqrt_d_std'`
    - `learn_projections=True` (si non explicitement fourni)
    - `phase1/2/3_max_steps=20/40/100`
    - `phase1/2/3_integrator_method='euler'/'euler'/'euler'`
  - utiliser `fashion_cpu_accuracy` pour maximiser l'accuracy,
    et `fashion_cpu_runtime` pour minimiser le temps.
- Les diagnostics de fin d'epoch peuvent être espacés pour réduire le coût :
  - `diagnostics_every_epochs` (défaut `1`),
  - `diagnostics_subset_size` (taille de sous-échantillon pour epochs non full).

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

### Proposition P6 - Récupération niveau-aware en phase 3

Quand un niveau reste proche du hasard alors que le couplage est actif (phase 3), on considère un collapse local de représentation.

Condition de déclenchement (forme opérationnelle) :
- phase courante = 3
- divergence inter-niveaux $\Delta_l = \max(\text{acc}_l)-\min(\text{acc}_l)$ au-dessus d'un seuil
- existence d'un niveau $l^\*$ tel que $|\text{acc}_{l^\*} - 1/K| \le \varepsilon$ pendant `patience` epochs
- accuracy train globale encore modérée (filtre anti faux positifs sur runs déjà sains)

Effet attendu :
- casser le collapse local (de-collapse attracteurs du niveau bloqué),
- réduire la probabilité de run dégradé,
- améliorer la borne basse inter-seeds sans sacrifier le gain moyen du couplage.

### Loss d'entraînement

$$\mathcal{L} = \mathcal{L}_{CE} + \mu_1 \mathcal{L}_{sep} + \mu_2 \mathcal{L}_{dyn}$$

## Résultats et campagnes de test

Pour améliorer la lisibilité, les résultats sont segmentés par campagnes de tests reproductibles.
Note organisation (2026-05-15) :
- Les artefacts historiques volumineux (`.json/.log/.txt`) ont été archivés dans `experiments/_archive_results_2026-05-15`.
- Les scripts de campagne restent inchangés et régénèrent les sorties standard dans `experiments/`.
- Les baselines actives conservées en clair dans `experiments/` :
  - `diag_seed42_46_n3200_diffgate_mutex.json`
  - `diag_noisy_moons_seed42_46_n3200_diffgate_mutex.json`
  - `diag_make_classification_seed42_46_n3200_diffgate_mutex.json`

### Index rapide des campagnes

| Campagne | Objectif principal | Script(s) | Indicateur clé |
|---|---|---|---|
| A (MNIST) | Mesurer le gain d'entraînement gradient sur protocole court | `experiments/mnist_coupling_ablation.py` | `+30.2 pts` (sans entraînement -> entraîné) |
| B (Concentrique) | Tester l'apport du couplage sur topologie non convexe | `experiments/concentric_coupling_ablation.py` | Delta couplé vs indép. (multi-seeds) |
| C (Bruité) | Évaluer robustesse au bruit (`make_moons`) | `experiments/noisy_benchmark.py` | Courbe du delta couplé-vs-indép. selon bruit |
| D (SA-nODE-reimpl) | Comparaison architecturale contrôlée hiérarchie vs plat | `experiments/noisy_sanode_comparison.py`, `experiments/tune_sanode_reimpl.py` | Accuracy HAML couplé vs SA-nODE-reimpl |
| E (Stabilisation) | Stabiliser les gains couplés forts sur concentrique | `experiments/concentric_coupling_ablation.py`, `experiments/seedwise_coupling_diagnostics.py` | E10c (`thr=0.71`): `74.68% ± 5.49`, min `67.25%` |
| F (Fashion-MNIST) | Validation réelle haute dimension (784D, 10 classes) | `experiments/fashion_mnist_short_probe.py` | seed 42: test `80.1%` (10 epochs, n_train=5000) |
| G (Profilage runtime) | Mesurer où passe le temps avant optimisation | `experiments/profile_training_runtime.py` | Artefacts `cProfile` + rapport ligne-à-ligne optionnel |

Commande de profilage recommandée (locale, sans réseau) :
- `python experiments/profile_training_runtime.py --dataset make_moons --n-train 1200 --n-test 400 --n-epochs 2 --max-steps 20 --device cpu --out-dir experiments/profile_runtime_moons`

Commande de profilage pipeline cible (si accès OpenML) :
- `python experiments/profile_training_runtime.py --dataset fashion_mnist --n-train 5000 --n-test 1000 --n-epochs 2 --max-steps 40 --device cpu --out-dir experiments/profile_runtime_fashion`
- variante avec contrôle convergence :
- `python experiments/profile_training_runtime.py --dataset fashion_mnist --n-train 5000 --n-test 1000 --n-epochs 2 --max-steps 40 --use-vectorized-levels --convergence-check-every 5 --device cpu --out-dir experiments/profile_runtime_fashion_vectorized_k5`
- variante scheduler max_steps par phase (conservatrice) :
- `python experiments/profile_training_runtime.py --dataset fashion_mnist --n-train 5000 --n-test 1000 --n-epochs 2 --phase1-epochs 1 --phase2-epochs 1 --phase1-max-steps 50 --phase2-max-steps 80 --phase3-max-steps 100 --use-vectorized-levels --convergence-check-every 5 --mu-dyn 0.0 --skip-line-profiler --device cpu --out-dir experiments/profile_runtime_fashion_vectorized_sched_50_80_100`
- variante avec diagnostics espacés (runs plus longs) :
- `python experiments/profile_training_runtime.py --dataset make_moons --noise 0.3 --n-train 5000 --n-test 1000 --n-epochs 10 --phase1-epochs 3 --phase2-epochs 3 --phase1-max-steps 50 --phase2-max-steps 80 --phase3-max-steps 100 --diagnostics-every-epochs 2 --diagnostics-subset-size 1000 --use-vectorized-levels --convergence-check-every 5 --mu-dyn 0.0 --skip-line-profiler --device cpu --out-dir experiments/profile_runtime_moons_diag_every2`

Note reporting :
- `profile_summary.json` contient désormais `probe.resolved_device` (device réellement utilisé),
  pour éviter toute ambiguïté quand l'argument CLI est `--device auto`.
- `profile_summary.json` contient aussi les paramètres runtime effectivement appliqués :
  - `probe.effective_phase*_max_steps`
  - `probe.effective_phase*_integrator_method`
  - `probe.effective_diagnostics_every_epochs`
  (utile avec `--training-preset` + overrides CLI).

Commande preset (CPU) :
- `python experiments/profile_training_runtime.py --dataset fashion_mnist --n-train 5000 --n-test 1000 --n-epochs 10 --phase1-epochs 3 --phase2-epochs 3 --training-preset fast_train_cpu --use-vectorized-levels --device cpu --skip-line-profiler --out-dir experiments/profile_runtime_fashion_local_long_fastpreset_cpu`
- `python experiments/profile_training_runtime.py --dataset fashion_mnist --n-train 5000 --n-test 1000 --n-epochs 10 --phase1-epochs 3 --phase2-epochs 3 --training-preset ultra_fast_train_cpu --use-vectorized-levels --device cpu --skip-line-profiler --out-dir experiments/profile_runtime_fashion_local_long_ultra_fastpreset_cpu`

Résultats A/B locaux (validation perf scheduler `max_steps`) :
- protocole :
  - dataset `make_moons` (`noise=0.3`), `n_train=5000`, `n_test=1000`
  - `2` epochs, `phase1=1`, `phase2=1`, `use_vectorized_levels=True`
  - `convergence_check_every=5`, `mu_dyn=0.0`, `skip_line_profiler=True`
- run A (`100/100/100`) :
  - `train_time_sec=86.45`
  - `test_accuracy=0.846`
  - `final_train_accuracy=0.845`
  - artefacts : `experiments/profile_runtime_moons_local_ab_nosched_100_100_100/`
- run B (`50/80/100`) :
  - `train_time_sec=51.16`
  - `test_accuracy=0.855`
  - `final_train_accuracy=0.8496`
  - artefacts : `experiments/profile_runtime_moons_local_ab_sched_50_80_100/`
- conclusion :
  - gain temps : `-35.29s` (`~40.8%`, `~1.69x`) pour `50/80/100`
  - accuracy non dégradée (légère hausse observée).

Contrôle diagnostics espacés (même protocole court `2` epochs) :
- baseline `diagnostics_every_epochs=1` (`50/80/100`) :
  - `train_time_sec=51.16`, `test_accuracy=0.855`
- variante `diagnostics_every_epochs=2`, `diagnostics_subset_size=1000` :
  - `train_time_sec=58.63`, `test_accuracy=0.855`
- lecture :
  - sur ce protocole court, l'espacement n'est pas rentable (`+7.47s`, `+14.6%`).
  - ce levier reste à revalider sur des runs plus longs (où les diagnostics
    full-train répétés pèsent davantage).

Itération scheduler `max_steps` (même protocole local `2` epochs) :
- baseline scheduler `50/80/100` :
  - `train_time_sec=51.16`
  - `test_accuracy=0.855`
  - `final_train_accuracy=0.8496`
- variante scheduler `30/60/100` :
  - `train_time_sec=41.19`
  - `test_accuracy=0.855`
  - `final_train_accuracy=0.8506`
- conclusion :
  - gain additionnel `-9.97s` (`~19.5%`, `~1.24x`) vs `50/80/100`
  - accuracy stable sur ce protocole.

Itération scheduler agressive `20/40/100` (même protocole local `2` epochs) :
- `train_time_sec=25.26`
- `test_accuracy=0.855`
- `final_train_accuracy=0.8528`
- comparaison :
  - vs `30/60/100` : `-15.93s` (`~38.7%`, `~1.63x`)
  - vs `50/80/100` : `-25.90s` (`~50.6%`, `~2.03x`)
- lecture :
  - sur ce protocole court, `20/40/100` est le meilleur compromis vitesse/perf observé.

Itération méthode d'intégration par phase (même protocole local `2` epochs, scheduler `20/40/100`, **device=CPU**) :
- baseline (`RK4` partout) :
  - `train_time_sec=30.24`
  - `test_accuracy=0.855`
  - `final_train_accuracy=0.8528`
- variante (`Euler` en phase 1/2) :
  - `phase1_integrator_method=euler`, `phase2_integrator_method=euler`
  - `train_time_sec=7.82`
  - `test_accuracy=0.854`
  - `final_train_accuracy=0.8526`
- conclusion :
  - gain net `-22.41s` (`~74.1%`, `~3.87x`) avec écart accuracy négligeable (`-0.1 pt`).

Micro-benchmark vectorisation niveau (CPU/GPU) :
- `python experiments/level_vectorized_micro_benchmark.py --batch-size 125 --dim 784 --n-classes 10 --n-attractors 2 --device cpu --out-json experiments/diag_level_vectorized_micro_benchmark_cpu.json`

Validation robuste de la variante `Euler` phase 1/2 (run long `10` epochs, `phase1=3`, `phase2=3`, **device=CPU**) :
- baseline (`RK4` partout, `20/40/100`) :
  - `train_time_sec=292.24`
  - `test_accuracy=0.841`
  - `final_train_accuracy=0.8454`
- variante (`Euler` phase 1/2, `RK4` phase 3) :
  - `phase1_integrator_method=euler`
  - `phase2_integrator_method=euler`
  - `phase3_integrator_method=rk4`
  - `train_time_sec=230.77`
  - `test_accuracy=0.841`
  - `final_train_accuracy=0.8454`
- conclusion :
  - gain net `-61.47s` (`~21.0%`, `~1.27x`) sans perte de performance observée.

Validation sur pipeline cible Fashion-MNIST (run local `2` epochs, `n_train=5000`, `n_test=1000`, scheduler `20/40/100`, **device=CPU**) :
- baseline (`RK4` partout) :
  - `train_time_sec=250.93`
  - `test_accuracy=0.745`
  - `final_train_accuracy=0.7538`
- variante (`Euler` phase 1/2, `RK4` phase 3) :
  - `phase1_integrator_method=euler`
  - `phase2_integrator_method=euler`
  - `phase3_integrator_method=rk4`
  - `train_time_sec=55.14`
  - `test_accuracy=0.745`
  - `final_train_accuracy=0.7538`
- conclusion :
  - gain net `-195.79s` (`~78.0%`, `~4.55x`) sans régression d'accuracy sur ce protocole.

Validation robuste sur Fashion-MNIST (run local `10` epochs, `n_train=5000`, `n_test=1000`, scheduler `20/40/100`, **device=CPU**) :
- baseline (`RK4` partout) :
  - `train_time_sec=2880.86`
  - `test_accuracy=0.798`
  - `final_train_accuracy=0.8216`
- variante (`Euler` phase 1/2, `RK4` phase 3) :
  - `phase1_integrator_method=euler`
  - `phase2_integrator_method=euler`
  - `phase3_integrator_method=rk4`
  - `train_time_sec=1676.21`
  - `test_accuracy=0.798`
  - `final_train_accuracy=0.8216`
- conclusion :
  - gain net `-1204.65s` (`~41.8%`, `~1.72x`) sans perte accuracy.

Campagne multi-seeds `fast_train_cpu` (Fashion-MNIST, local CPU, `10` epochs, seeds `42..44`) :
- seed `42` : `train_time_sec=1562.67`, `test_accuracy=0.798`, `train_accuracy=0.8216`
- seed `43` : `train_time_sec=1480.52`, `test_accuracy=0.814`, `train_accuracy=0.8284`
- seed `44` : `train_time_sec=1390.66`, `test_accuracy=0.802`, `train_accuracy=0.8188`
- agrégé (`3` seeds) :
  - `train_time_sec mean = 1477.95s` (~24m38s)
  - `test_accuracy mean = 0.8047` (`std ~ 0.0068`)
- note :
  - le preset est bien appliqué (`effective_phase*=20/40/100`, `effective_methods=euler/euler/rk4` dans `profile_summary.json`).

Étape suivante - ablation diagnostics (préliminaire, seed `42`, Fashion-MNIST, CPU, `10` epochs, `fast_train_cpu`) :
- baseline (`diagnostics_every_epochs=1`) :
  - `train_time_sec=1721.36`
  - `test_accuracy=0.798`
  - `final_train_accuracy=0.8216`
- variante `diagnostics_every_epochs=2` :
  - `train_time_sec=1815.48` (plus lent)
  - `test_accuracy=0.798`
  - `final_train_accuracy=0.8216`
- variante `diagnostics_every_epochs=3` :
  - `train_time_sec=1720.71` (quasi identique baseline)
  - `test_accuracy=0.798`
  - `final_train_accuracy=0.8170`
- lecture :
  - sur ce protocole seed unique, espacer les diagnostics n'apporte pas de gain net clair.
  - la décision finale doit être confirmée en multi-seeds avant généralisation.

Étape suivante - variante agressive intégrateur (Fashion-MNIST, CPU, `10` epochs, seeds `42..44`) :
- référence `fast_train_cpu` (méthodes `euler/euler/rk4`) :
  - seed `42`: `1721.36s`, `test=0.798`
  - seed `43`: `1722.28s`, `test=0.814`
  - seed `44`: `1415.51s`, `test=0.802`
  - moyenne: `1619.72s`, `test_mean=0.8047`
- variante agressive (override `phase3_integrator_method=euler`, donc `euler/euler/euler`) :
  - seed `42`: `469.74s`, `test=0.798`
  - seed `43`: `455.09s`, `test=0.814`
  - seed `44`: `449.46s`, `test=0.802`
  - moyenne: `458.10s`, `test_mean=0.8047`
- conclusion :
  - gain net moyen `-1161.62s` (`~71.7%`, `~3.54x`) sans perte accuracy observée sur ces seeds.
  - c'est actuellement le levier runtime le plus efficace du plan.

Validation preset `ultra_fast_train_cpu` (run complet, seed `42`, Fashion-MNIST, CPU, `10` epochs) :
- `train_time_sec=471.67`
- `test_accuracy=0.798`
- `final_train_accuracy=0.8216`
- `effective_methods=euler/euler/euler`
- lecture :
  - cohérent avec l'ablation précédente (`phase3=euler`) et sans régression accuracy sur ce seed.

Validation multi-seeds preset `ultra_fast_train_cpu` (Fashion-MNIST, CPU, `10` epochs, seeds `42..44`) :
- seed `42` : `train_time_sec=471.67`, `test_accuracy=0.798`, `train_accuracy=0.8216`
- seed `43` : `train_time_sec=451.36`, `test_accuracy=0.814`, `train_accuracy=0.8286`
- seed `44` : `train_time_sec=465.70`, `test_accuracy=0.802`, `train_accuracy=0.8188`
- agrégé (`3` seeds) :
  - `train_time_sec mean = 462.91s` (~7m43s)
  - `test_accuracy mean = 0.8047` (`std ~ 0.0068`)
- lecture :
  - même accuracy moyenne que les variantes précédentes, avec un gain temps majeur.

Synthèse exécutive (Fashion-MNIST, CPU, `10` epochs, seeds `42..44`) :

| Variante | Intégrateur phase 1/2/3 | Train time moyen | Test acc moyenne | Écart vs RK4 |
|---|---|---:|---:|---:|
| RK4 baseline | `rk4/rk4/rk4` | `1844.72s` | `0.8047` | référence |
| `fast_train_cpu` | `euler/euler/rk4` | `1619.72s` | `0.8047` | `-12.2%` temps (`~1.14x`) |
| `ultra_fast_train_cpu` | `euler/euler/euler` | `462.91s` | `0.8047` | `-74.9%` temps (`~3.98x`) |

Outil d'agrégation des campagnes runtime :
- script : `experiments/aggregate_profile_runs.py`
- usage :
  - `python experiments/aggregate_profile_runs.py --run-dirs <dir1> <dir2> <dir3> --out-json <agg.json>`
- sorties agrégées générées pour les campagnes CPU Fashion-MNIST (`10` epochs, seeds `42..44`) :
  - `experiments/profile_runtime_fashion_local_long_rk4_20_40_100_cpu_agg.json`
  - `experiments/profile_runtime_fashion_local_long_fastpreset_cpu_agg.json`
  - `experiments/profile_runtime_fashion_local_long_ultra_fastpreset_cpu_agg.json`

Ablations qualité/modèle (B1/B2/B3/B5/B23) :
- script dédié : `experiments/quality_ablation_runtime.py`
- référence figée pour campagnes Fashion-MNIST CPU dans ce runner :
  - `sigma_init_mode='sqrt_d_std'`
  - `learn_projections=True`
  - les variantes d'ablation (`B1/B2/B3/B23`) restent prioritaires quand elles
    explicitent un override.
- leviers supportés :
  - `--ablation b1` : `repulsion_mode=global` vs `inter_class_only`
  - `--ablation b2` : `sigma_init_mode=sqrt_d_std` vs `median_pairwise`
  - `--ablation b3` : `learn_projections=False` vs `True`
  - `--ablation b5` : `level_score_weighting=exponential` vs `uniform`
  - `--ablation b23` : croisement `sigma_init_mode` x `learn_projections` (2x2)
- protocole recommandé (Fashion-MNIST CPU, seeds `42 43 44`) :
  - `python experiments/quality_ablation_runtime.py --ablation b1 --dataset fashion_mnist --seeds 42 43 44 --n-train 5000 --n-test 1000 --n-epochs 10 --phase1-epochs 3 --phase2-epochs 3 --training-preset ultra_fast_train_cpu --device cpu --out-json experiments/quality_ablation_b1_fashion_cpu.json`
  - `python experiments/quality_ablation_runtime.py --ablation b2 --dataset fashion_mnist --seeds 42 43 44 --n-train 5000 --n-test 1000 --n-epochs 10 --phase1-epochs 3 --phase2-epochs 3 --training-preset ultra_fast_train_cpu --device cpu --out-json experiments/quality_ablation_b2_fashion_cpu.json`
  - `python experiments/quality_ablation_runtime.py --ablation b3 --dataset fashion_mnist --seeds 42 43 44 --n-train 5000 --n-test 1000 --n-epochs 10 --phase1-epochs 3 --phase2-epochs 3 --training-preset ultra_fast_train_cpu --device cpu --out-json experiments/quality_ablation_b3_fashion_cpu.json`
  - `python experiments/quality_ablation_runtime.py --ablation b5 --dataset fashion_mnist --seeds 42 43 44 --n-train 5000 --n-test 1000 --n-epochs 10 --phase1-epochs 3 --phase2-epochs 3 --training-preset fashion_cpu_accuracy --device cpu --out-json experiments/quality_ablation_b5_fashion_cpu.json`
  - `python experiments/quality_ablation_runtime.py --ablation b23 --dataset fashion_mnist --seeds 42 43 44 --n-train 5000 --n-test 1000 --n-epochs 10 --phase1-epochs 3 --phase2-epochs 3 --training-preset ultra_fast_train_cpu --device cpu --out-json experiments/quality_ablation_b23_fashion_cpu.json`
- reprise/continuité de run (sans recalcul des seeds déjà terminés) :
  - ajouter `--resume` avec le même `--out-json` et la même config CLI.
  - le runner reprend uniquement les seeds manquants et met à jour le JSON
    de manière incrémentale (safe pour interruption/reprise).

Résultat B1 (Fashion-MNIST, CPU, seeds `42..44`, `ultra_fast_train_cpu`) :
- `repulsion_mode=global` :
  - `train_time_mean=1484.18s` (`std=578.54s`)
  - `test_accuracy_mean=0.8047`
- `repulsion_mode=inter_class_only` :
  - `train_time_mean=3324.96s` (`std=1042.98s`)
  - `test_accuracy_mean=0.8047`
- conclusion B1 :
  - pas de gain accuracy observé avec `inter_class_only`,
  - surcoût compute très élevé (`~2.24x` plus lent en moyenne),
  - `global` est retenu comme meilleur compromis opérationnel.

Résultat B2 (Fashion-MNIST, CPU, seeds `42..44`, `ultra_fast_train_cpu`) :
- `sigma_init_mode=sqrt_d_std` :
  - `train_time_mean=551.53s` (`std=2.96s`)
  - `test_accuracy_mean=0.8047` (`std=0.0068`)
  - `final_train_accuracy_mean=0.8229`
- `sigma_init_mode=median_pairwise` :
  - `train_time_mean=528.14s` (`std=16.15s`)
  - `test_accuracy_mean=0.8047` (`std=0.0052`)
  - `final_train_accuracy_mean=0.8241`
- conclusion B2 :
  - accuracy test moyenne identique entre les deux variantes (`0.8047`),
  - gain runtime modeste en faveur de `median_pairwise` (`-23.38s`, `~4.2%`),
  - `median_pairwise` est un candidat raisonnable pour le défaut opérationnel.

Résultat B3 (Fashion-MNIST, CPU, seeds `42..44`, `ultra_fast_train_cpu`) :
- `learn_projections=False` :
  - `train_time_mean=491.11s` (`std=51.09s`)
  - `test_accuracy_mean=0.8047` (`std=0.0068`)
  - `final_train_accuracy_mean=0.8229`
- `learn_projections=True` :
  - `train_time_mean=355.04s` (`std=14.10s`)
  - `test_accuracy_mean=0.8187` (`std=0.0061`)
  - `final_train_accuracy_mean=0.8863`
- conclusion B3 :
  - gain accuracy test net avec projections apprises (`+1.4 points`),
  - gain runtime significatif (`-136.08s`, `~27.7%`, `~1.38x` plus rapide),
  - `learn_projections=True` est recommandé sur ce protocole.

Validation robuste B3 (Fashion-MNIST, CPU, seeds `42..51`, `ultra_fast_train_cpu`) :
- `learn_projections=False` :
  - `train_time_mean=539.45s` (`std=53.72s`)
  - `test_accuracy_mean=0.8003` (`std=0.0103`)
  - `final_train_accuracy_mean=0.8176`
- `learn_projections=True` :
  - `train_time_mean=373.31s` (`std=33.31s`)
  - `test_accuracy_mean=0.8144` (`std=0.0103`)
  - `final_train_accuracy_mean=0.8929`
- deltas (`True` vs `False`) :
  - `test_accuracy`: `+1.41 points`
  - `train_time`: `-166.14s` (`~30.8%`, `~1.45x` plus rapide)
  - `final_train_accuracy`: `+7.53 points`
- conclusion :
  - le gain B3 est confirmé sur `10` seeds,
  - `learn_projections=True` est retenu comme choix opérationnel robuste.

Résultat B23 (Fashion-MNIST, CPU, seeds `42..44`, `ultra_fast_train_cpu`) :
- `sigma_sqrt_d_std + learn_projections=False` :
  - `train_time_mean=588.73s`
  - `test_accuracy_mean=0.8047`
- `sigma_median_pairwise + learn_projections=False` :
  - `train_time_mean=561.94s`
  - `test_accuracy_mean=0.8047`
- `sigma_sqrt_d_std + learn_projections=True` :
  - `train_time_mean=379.60s`
  - `test_accuracy_mean=0.8187`
- `sigma_median_pairwise + learn_projections=True` :
  - `train_time_mean=418.60s`
  - `test_accuracy_mean=0.8163`
- conclusion B23 :
  - le levier dominant est `learn_projections=True` (gain net qualité et runtime),
  - avec projections apprises, `sigma_sqrt_d_std` domine `median_pairwise`
    (meilleur score test et temps plus faible),
  - recommandation opérationnelle sur ce protocole :
    `sigma_init_mode='sqrt_d_std'` + `learn_projections=True`.

Validation robuste B23 (Fashion-MNIST, CPU, seeds `42..51`, `ultra_fast_train_cpu`) :
- `sigma_sqrt_d_std + learn_projections=False` :
  - `train_time_mean=560.15s` (`std=56.13s`)
  - `test_accuracy_mean=0.8003` (`std=0.0103`)
- `sigma_median_pairwise + learn_projections=False` :
  - `train_time_mean=521.46s` (`std=22.22s`)
  - `test_accuracy_mean=0.8016` (`std=0.0103`)
- `sigma_sqrt_d_std + learn_projections=True` :
  - `train_time_mean=382.40s` (`std=42.53s`)
  - `test_accuracy_mean=0.8130` (`std=0.0106`)
- `sigma_median_pairwise + learn_projections=True` :
  - `train_time_mean=399.17s` (`std=15.76s`)
  - `test_accuracy_mean=0.8189` (`std=0.0114`)
- conclusion robuste B23 :
  - `learn_projections=True` reste le levier principal (qualité + runtime),
  - en régime `learn_projections=True`, `median_pairwise` devient meilleur en
    accuracy (`+0.59 point`) mais plus lent (`+16.76s`, `~4.4%`) que `sqrt_d_std`,
  - choix opérationnel selon objectif :
    - accuracy max : `sigma_init_mode='median_pairwise'` + `learn_projections=True`
    - runtime max : `sigma_init_mode='sqrt_d_std'` + `learn_projections=True`.

Validation preset `fashion_cpu_accuracy` (Fashion-MNIST, CPU, seeds `42..51`) :
- run : `experiments/quality_ablation_b23_fashion_cpu_seeds42_51_accuracy_preset.json`
- meilleur compromis observé :
  - `sigma_median_pairwise + learn_projections=True`
  - `test_accuracy_mean=0.8191` (`std=0.0081`)
  - `train_time_mean=378.41s` (`std=25.21s`)
- lecture :
  - `learn_projections=True` reste le levier principal,
  - sur cette campagne, `median_pairwise+learned` domine aussi `sqrt_d_std+learned`
    en runtime (`378.41s` vs `406.52s`) avec une accuracy légèrement supérieure
    (`0.8191` vs `0.8188`).

Validation preset `fashion_cpu_runtime` (Fashion-MNIST, CPU, seeds `42..51`) :
- run : `experiments/quality_ablation_b23_fashion_cpu_seeds42_51_runtime_preset.json`
- meilleur compromis observé :
  - `sigma_sqrt_d_std + learn_projections=True`
  - `test_accuracy_mean=0.8146` (`std=0.0107`)
  - `train_time_mean=366.09s` (`std=45.33s`)
- comparaison directe vs preset `fashion_cpu_accuracy` (meilleur variant) :
  - `fashion_cpu_runtime` : `366.09s`, `0.8146`
  - `fashion_cpu_accuracy` : `378.41s`, `0.8191`
  - delta (`runtime` vs `accuracy`) :
    - temps : `-12.32s` (`~3.3%`)
    - accuracy : `-0.45 point`
- décision opérationnelle :
  - priorité accuracy : `fashion_cpu_accuracy`
  - priorité runtime strict : `fashion_cpu_runtime`.

Validation B5 (pondération inter-niveaux, Fashion-MNIST, CPU, seeds `42..51`,
preset `fashion_cpu_accuracy`) :
- `level_score_weighting='exponential'` :
  - `train_time_mean=359.04s` (`std=24.46s`)
  - `test_accuracy_mean=0.8206` (`std=0.0079`)
- `level_score_weighting='uniform'` :
  - `train_time_mean=349.63s` (`std=18.03s`)
  - `test_accuracy_mean=0.7547` (`std=0.0447`)
- conclusion B5 :
  - `uniform` perd `-6.59 points` d'accuracy moyenne vs `exponential`,
  - `uniform` est nettement moins stable (écart-type test fortement accru),
  - le léger gain temps (`~2.6%`) ne compense pas la régression qualité,
  - `exponential` est confirmé comme défaut opérationnel.

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

Sous-test B2 - Résultats multi-seeds consolidés (`n_runs=5`, seeds `42..46`) :
- indépendant : `67.30% ± 4.37`
- couplé tuned (baseline) : `71.20% ± 6.43`
- delta moyen : `+3.90 points`
- lecture :
  - le gain couplé moyen reste positif mais non garanti run par run
  - la variance du couplé est plus élevée que l'indépendant, ce qui motive un mécanisme anti-collapse
  - ce constat est la motivation directe des itérations de stabilisation `E7`/`E8` (garde-fou level-aware)
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

Sous-test E7 - Diagnostic seed-wise + patch anti-cascade LR :
- script dédié: `experiments/seedwise_coupling_diagnostics.py`
- traces exportées par seed: `level_divergence`, `mu_sep`, `lr`, `level_accuracy`, états attracteurs
- ajout de traces géométriques par niveau: `level_attractor_diagnostics`
  (`intra_spread_mean`, `inter_centroid_dist_min`, `separation_ratio`)
- script d'analyse mécanistique: `experiments/analyze_seedwise_collapse.py`
  (isolation des niveaux bloqués en phase 3 avec timeline `div/lr/mu_sep/level_acc/separation_ratio`)
- seeds testées: `42..46`, `n_samples=1200`
- constat stable avant patch:
  - un niveau reste fréquemment proche du hasard en fin de run (`level_acc ~ 0.500`)
  - en phase 3 tardive, décays LR répétés jusqu'au plancher (`1e-4`), variance élevée entre seeds
- correctif implémenté:
  - limitation des décays LR de stabilité (cooldown + plafond d'événements)
  - réaction de stabilité bornée à la phase 3
  - réactivation mesurée de `mu_sep` adaptatif en phase 3 dans le protocole de diagnostic
- résultats après patch (même protocole, seeds `42..46`):
  - couplé (avant): `68.60% ± 7.54`, min `56.67%`
  - couplé (après): `69.33% ± 7.10`, min `57.33%`
  - indépendant: `59.73% ± 5.93`
  - delta moyen couplé(après) - indépendant: `+9.60 points`
- run dégradé toujours présent (seed `42`), mais queue basse légèrement réduite
- commande de diagnostic mécanistique ciblé (`42..43`) :
  - `python experiments/seedwise_coupling_diagnostics.py --mode coupled_tuned --base-seed 42 --n-runs 2 --n-samples 1200 --json-out experiments/diag_seed42_43_mechanistic.json`
  - `python experiments/analyze_seedwise_collapse.py --input-json experiments/diag_seed42_43_mechanistic.json`

Sous-test E8 - Garde-fou level-aware (phase 3 uniquement) :
- objectif: traiter les runs où un niveau reste proche du hasard (`~0.500`) en phase couplée active
- déclenchement: uniquement en phase 3, sur divergence suffisante, avec condition d'accuracy globale basse
- action: recovery locale douce des attracteurs du niveau bloqué + ajustement modéré `mu_sep`/LR
- validation multi-seeds (`42..46`, `n_samples=1200`) :
  - couplé baseline (avant): `68.60% ± 7.54`, min `56.67%`
  - couplé level-aware (filtré): `70.80% ± 5.54`, min `60.67%`
  - indépendant: `59.73% ± 5.93`
  - delta moyen couplé(level-aware) - indépendant: `+11.07 points`
- point clé:
  - seed `42` passe de `-11.67 pts` (couplé vs indép.) à `+0.67 pt`
  - les seeds forts restent positifs; légère baisse sur seed `46` (`-2.00 pts` vs baseline couplée)

### Campagne F - Validation réelle sur Fashion-MNIST (Colab)

Objectif :
- Vérifier la stabilité et la progression du protocole 3 phases sur données réelles haute dimension (`784D`, `10` classes).

Protocole :
- Dataset : Fashion-MNIST (`n_train=5000`, `n_test=1000`)
- Seed : `42`
- Entraînement : `10` epochs, phases `(2,3,5)`
- Config recovery : seuil différentiel multi-classes (`other_levels_strong_threshold=0.20`), détection blocage calibrée `chance + epsilon = 0.22`.

Résultats observés :
- Accuracy train finale : `82.3%`
- Accuracy test : `80.1%`
- Progression de phase :
  - fin phase 1 : `75.36%`
  - fin phase 2 : `78.86%`
  - fin phase 3 : `82.26%`
- Accuracy niveaux en phase 3 : progression conjointe (`~0.79 -> ~0.82`) avec divergence inter-niveaux faible (`level_div` proche de `0`).
- Événements de stabilisation/recovery : aucun trigger (`events=[]`), run naturellement stable.
- Coût : `6979s` (~`1h56`) pour 10 epochs sur ce sous-ensemble.

Confirmation Kaggle (reproductibilité inter-environnements) :
- Même protocole reproduit sur Kaggle GPU :
  - accuracy test : `80.1%` (identique)
  - accuracy train finale : `82.3%` (identique)
  - temps total : `6222s` (~`1h44`)
- Lecture :
  - comportement stable et déterministe sur `seed=42`
  - coût compute toujours élevé malgré un runtime légèrement plus rapide que Colab.

Extension multi-seeds (mode `coupled_tuned`, seeds `42..46`) :
- Scores test :
  - seed `42`: `80.1%`
  - seed `43`: `81.6%`
  - seed `44`: `80.3%`
  - seed `45`: `80.6%`
  - seed `46`: `78.8%`
- Agrégé (5 seeds) :
  - moyenne : `80.28%`
  - écart-type : `~1.0%`
  - minimum : `78.8%`
  - maximum : `81.6%`
- Stabilité :
  - aucun trigger `recovery`/`stability` observé (`events=[]` sur les runs reportés)
  - progression 3 phases cohérente sur les 5 seeds.
- Signal mécanistique à surveiller :
  - sur seeds `44` et `46`, le niveau `L0` décroche en phase 3 (jusqu'à `~0.74`)
    alors que `L1/L2` montent vers `~0.81`, avec `level_div` croissante (jusqu'à `~0.075`).
  - ce pattern reste compensé au niveau test, mais confirme que `L0` est le niveau le plus sensible sous couplage fort.

Comparatif final couplé vs indépendant (Fashion-MNIST, seeds `42..46`, `n_train=5000`, `10` epochs) :

| Seed | Couplé | Indépendant | Delta |
|---|---:|---:|---:|
| 42 | 80.1% | 80.2% | -0.1 pt |
| 43 | 81.6% | 81.4% | +0.2 pt |
| 44 | 80.3% | 80.2% | +0.1 pt |
| 45 | 80.6% | 80.8% | -0.2 pt |
| 46 | 78.8% | 78.9% | -0.1 pt |
| **Moyenne** | **80.28%** | **80.30%** | **-0.02 pt** |
| **Std** | **±1.00%** | **±0.95%** | |

Conclusion de délimitation (campagne F) :
- Sur ce protocole, le delta couplé vs indépendant est statistiquement nul.
- Ce résultat est cohérent avec l'ablation MNIST courte (delta proche de `0`).
- Interprétation retenue :
  - soit le protocole (`10` epochs, `2` attracteurs/classe) est trop court pour laisser émerger un gain couplé,
  - soit Fashion-MNIST n'expose pas la structure topologique où le couplage apporte un avantage net.
- Positionnement scientifique :
  - ce n'est pas un échec du modèle, mais une délimitation de domaine d'application du gain couplé,
  - le bénéfice du couplage apparaît conditionnel à la structure du problème (ambiguïté locale + besoin de contexte global).

Test de réduction dimensionnelle (levier vitesse) :
- Variante testée : `784 -> 64 -> 32` (même protocole, même seed, même budget epochs)
- Résultat :
  - test : `77.3%` (vs `80.1%` en `784 -> 392 -> 196`)
  - temps : `6872s` (vs `6979s`)
- Lecture :
  - perte de performance `-2.8 pts` sans gain temps significatif
  - la réduction `level_dims` n'est pas un levier d'accélération utile dans ce protocole.

Signification :
- Validation positive du comportement du modèle sur données réelles (pas uniquement synthétiques).
- La stratégie 3 phases reste cohérente en `784D` et multi-classes sans instabilité apparente.
- Le verrou principal devient l'infrastructure de calcul (temps), plus que la stabilité algorithmique sur ce protocole.
- Les leviers simples testés (`max_steps` raisonnable, réduction de dimensions latentes) ne débloquent pas le coût de calcul.

Conséquence pratique :
- À infrastructure constante, une campagne multi-seeds complète Fashion-MNIST reste coûteuse (ordre de grandeur `~2h/run` sur ce protocole).
- Deux chemins opérationnels :
  - assumer une validation réelle en seed unique (`seed=42`) et documenter explicitement la limite compute
  - migrer vers une infra plus puissante (GPU haut de gamme / HPC) avant campagne complète.

Commande de reproduction (Colab GPU) :
- `PYTHONIOENCODING=utf-8 python experiments/fashion_mnist_short_probe.py --device cuda --seed 42 --n-train 5000 --n-test 1000 --n-epochs 10 --phase1-epochs 2 --phase2-epochs 3 --json-out experiments/diag_fashion_mnist_seed42_n5000_e10_mutex.json`

Sous-test E9 - Patch cause-oriented (déblocage L1 en phase 3) :
- diagnostic mécanique seed `42` (phase 3, avant patch) :
  - `L1` reste au hasard pendant `12` epochs (streak), malgré couplage actif
  - `separation_ratio` non nul (`~0.15 -> ~0.26`) : blocage d'apprentissage plutôt qu'effondrement géométrique total
- correctif implémenté dans `HAMLTrainer` :
  - re-anchor supervisé du niveau bloqué sur prototypes de classe en espace latent de niveau
  - atténuation top-down temporaire après recovery (`td_cooldown_epochs`, `td_scale_during_cooldown`)
- contrôle seed `42` (run unitaire post-patch) :
  - `L1` se débloque immédiatement après recovery (`0.500 -> 0.602 -> 0.644`)
  - test run observé : `65.0%` (amélioration mécanique confirmée, calibration perf encore à stabiliser)

Sous-test E10 - Test d'échelle (protocole inchangé, `n_samples=3200`) :
- objectif: trancher "plafond du modèle" vs "plafond du protocole court (`n_samples=1200`)"
- protocole: seeds `42..46`, mêmes hyperparamètres que E9 (streak ungated, recovery ciblé L1)
- résultats:
  - couplé: `78.75% ± 7.63`, min `68.25%`, max `87.38%`
  - comparaison E8 (`n_samples=1200`): moyenne `+7.95 pts`, min `+7.58 pts`, variance plus large (`+2.09` de std)
- lecture seed-wise extrêmes:
  - seed `42` (`87.38%`): pas de déclenchement recovery L1; run naturellement fort, puis divergence tardive de `L0` en fin de phase 3
  - seed `45` (`68.25%`): pas de déclenchement recovery L1; `L2` reste proche hasard (`~0.51`) et devient le facteur limitant
- conclusion:
  - le plafond `70-71%` observé sur `n_samples=1200` venait du protocole court
  - la variance inter-seeds reste le verrou principal à `n_samples=3200`.

Sous-test E10b - Recovery multi-niveaux dynamique (`target_level_idx=None`) :
- objectif: vérifier si la généralisation "tout niveau bloqué" permet de débloquer les runs faibles (`n_samples=3200`) sans régression sur les runs forts
- changement minimal: retrait de la restriction `target_level_idx=1` dans `experiments/seedwise_coupling_diagnostics.py`
- validation ciblée:
  - seed `42`: `85.75%` (vs `87.38%` avant), aucun déclenchement `level-recovery` observé
  - seed `43`: `81.75%` (vs `86.50%` avant), aucun déclenchement `level-recovery` observé
  - seed `45`: `67.88%` (vs `68.25%` avant), aucun déclenchement `level-recovery` observé
- conclusion:
  - la détection dynamique ne suffit pas seule: le recovery ne s'active pas sur ces runs
  - le verrou n'est pas la restriction L1 mais les conditions de déclenchement sur ce protocole `n_samples=3200`.

Sous-test E10c - Recalibrage de gate `max_train_accuracy_to_trigger=0.71` :
- objectif: garder la gate ouverte plus longtemps sur `n_samples=3200` pour récupérer les runs avec niveau bloqué
- validation ciblée:
  - seed `45`: trigger phase 3 observé, score `68.88%` (au-dessus de `68.25%` de référence seed 45)
  - seed `42`: aucun trigger observé sur ce run ciblé, score `77.63%`
- campagne complète `42..46`:
  - couplé: `74.68% ± 5.49`, min `67.25%`, max `83.38%`
  - comparaison E10 (`threshold` précédent): moyenne `-4.08 pts`, max `-4.00 pts`, min `-1.00 pt`
- conclusion:
  - le seuil `0.71` améliore le cas ciblé seed `45` mais dégrade la performance globale multi-seeds
  - ce réglage n'est pas retenu comme nouveau défaut.

Validation cross-dataset (baseline mutex figée) :
- Protocole: seeds `42..46`, `n_samples=3200`, même runner seed-wise, même format JSON.
- `noisy_moons` (`noise=0.30`) :
  - scores: `88.12%`, `91.00%`, `89.88%`, `86.25%`, `90.38%`
  - agrégé: `89.13% ± 1.73`, min `86.25%`
  - lecture: transfert très stable sur frontières non convexes 2D.
- `make_classification` (`4` classes, features redondantes) :
  - scores: `79.38%`, `75.50%`, `76.88%`, `71.50%`, `67.12%`
  - agrégé: `74.08% ± 4.31`, min `67.12%`
  - lecture: transfert partiel sur régime plus difficile, variance maîtrisée mais borne basse plus faible.
- Fichiers:
  - `experiments/diag_noisy_moons_seed42_46_n3200_diffgate_mutex.json`
  - `experiments/diag_make_classification_seed42_46_n3200_diffgate_mutex.json`

Condition d'initialisation obligatoire (I1) :
- Pour éviter la dégénérescence en haute dimension, imposer
  `sigma_init^(l) = sqrt(d_l) * sigma_data^(l)`.
- Sans cette condition, les kernels gaussiens saturent et les gradients s'annulent en grande dimension.

### Version 0.2.2 (fix haute dimension)

**CORRECTION CRITIQUE** : Saturation gaussienne en haute dimension

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

**Attention :** Sans ce fix, HAML ne fonctionne PAS au-delà de ~20 dimensions.

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

Frédérick MURAT


