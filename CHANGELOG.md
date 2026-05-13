# Changelog

Toutes les modifications notables du projet HAML seront documentÃ©es dans ce fichier.

Le format est basÃ© sur [Keep a Changelog](https://keepachangelog.com/fr/1.0.0/).

## [Non publié]

### Modifie - 2026-05-13
- Refactor du protocole `experiments/concentric_coupling_ablation.py` en ablation multi-seeds configurable.
- Ajout des options CLI `--n-runs`, `--base-seed`, `--n-samples`, `--save-figure`, `--json-out`.
- Ajout d'une aggregation statistique (moyenne/ecart-type/min/max) des accuracies et temps d'entrainement.
- Export standard des resultats dans `experiments/concentric_coupling_ablation_summary.json`.
- Le script conserve une figure optionnelle de frontieres pour le premier run uniquement.
### ModifiÃ© - 2026-05-12
- ExÃ©cution du benchmark bruitÃ© reproductible `experiments/noisy_benchmark.py` (seed fixe `42`, `n_samples=3000`) comparant :
  - HAML indÃ©pendant (`alpha_bu=0`, `alpha_td=0`)
  - HAML couplÃ© (`alpha_bu=0.5`, `alpha_td=1.5`)
  - baselines `KNN` et `SVM RBF`.
- RÃ©sultats (accuracy test) :
  - bruit `0.10` : HAML ind. `99.60%`, HAML cpl. `99.60%`, KNN `99.87%`, SVM `99.87%`, delta `+0.00 point`
  - bruit `0.20` : HAML ind. `89.33%`, HAML cpl. `95.60%`, KNN `96.53%`, SVM `97.20%`, delta `+6.27 points`
  - bruit `0.30` : HAML ind. `88.13%`, HAML cpl. `90.53%`, KNN `90.80%`, SVM `91.33%`, delta `+2.40 points`
  - bruit `0.40` : HAML ind. `84.67%`, HAML cpl. `85.07%`, KNN `83.33%`, SVM `87.33%`, delta `+0.40 point`.
- Sortie complÃ¨te enregistrÃ©e dans `experiments/noisy_benchmark_output.txt` (incluant le JSON final des rÃ©sultats).
- Documentation enrichie avec lecture empirique consolidÃ©e :
  - dÃ©gradation bruit `0.20 -> 0.40` : KNN `-13.20 pts`, HAML couplÃ© `-10.53 pts`, SVM `-9.87 pts`
  - pattern en cloche du gain de couplage (`0.00`, `+6.27`, `+2.40`, `+0.40`) cohÃ©rent avec l'hypothÃ¨se top-down
  - synthÃ¨se thÃ©orie/expÃ©rience explicitÃ©e et point restant identifiÃ© pour publication : benchmark direct vs SA-nODE.
- Ajout du script `experiments/noisy_sanode_comparison.py` :
  - protocole alignÃ© avec le benchmark bruitÃ© (`make_moons`, seed `42`, split identique)
  - comparaison HAML ind./couplÃ©, `SA-nODE-like`, KNN, SVM.
- RÃ©sultats `HAML couplÃ©` vs `SA-nODE-like` (accuracy test) :
  - bruit `0.10` : `99.60%` vs `82.27%` (delta `+17.33 pts`)
  - bruit `0.20` : `95.60%` vs `81.20%` (delta `+14.40 pts`)
  - bruit `0.30` : `90.53%` vs `79.87%` (delta `+10.67 pts`)
  - bruit `0.40` : `85.07%` vs `76.67%` (delta `+8.40 pts`).
- Clarification mÃ©thodologique ajoutÃ©e :
  - `SA-nODE-like` est une baseline proxy interne (dynamique simple double-puits, espace unique),
    non Ã©quivalente Ã  l'implÃ©mentation officielle SA-nODE de la publication.
  - la comparaison est positionnÃ©e comme ablation architecturale contrÃ´lÃ©e (effet hiÃ©rarchie bidirectionnelle vs modÃ¨le plat), et non comme claim dÃ©finitif contre SA-nODE officiel.
- Sortie complÃ¨te enregistrÃ©e dans `experiments/noisy_sanode_comparison_output.txt`.
- RÃ©vision de la baseline SA-nODE:
  - remplacement du proxy simple par une rÃ©implÃ©mentation plus fidÃ¨le (`haml/baselines/sanode.py`)
  - Ã©lÃ©ments inclus: espace plat, attracteurs binaires Â±a plantÃ©s, potentiel double-puits analytique, couplage linÃ©aire entraÃ®nÃ©.
- RÃ©sultats mis Ã  jour `HAML couplÃ©` vs `SA-nODE-reimpl` (accuracy test):
  - bruit `0.10`: `99.60%` vs `87.20%` (delta `+12.40 pts`)
  - bruit `0.20`: `95.60%` vs `85.73%` (delta `+9.87 pts`)
  - bruit `0.30`: `90.53%` vs `84.53%` (delta `+6.00 pts`)
  - bruit `0.40`: `85.07%` vs `83.47%` (delta `+1.60 pt`).
- Ajout d'une recherche d'hyperparamÃ¨tres SA-nODE-reimpl (`experiments/tune_sanode_reimpl.py`) :
  - grille testÃ©e (seed fixe, architecture inchangÃ©e):
    - `a in {0.8, 1.0, 1.2}`
    - `dt in {0.03, 0.05, 0.08}`
    - `gamma in {0.0, 0.05, 0.1}`
  - meilleure config moyenne multi-bruit: `a=1.0`, `dt=0.03`, `gamma=0.0`
  - meilleure config bruit fort (`0.40`): `a=0.8`, `dt=0.03`, `gamma=0.1` (gain limitÃ©).
- RÃ©sultats consolidÃ©s avec SA-nODE-reimpl tuned (HAML inchangÃ©, mÃªme protocole/seed):
  - bruit `0.10`: `99.60%` vs `86.67%` (delta `+12.93 pts`)
  - bruit `0.20`: `95.60%` vs `86.40%` (delta `+9.20 pts`)
  - bruit `0.30`: `90.53%` vs `85.87%` (delta `+4.67 pts`)
  - bruit `0.40`: `85.07%` vs `83.33%` (delta `+1.73 pt`)
  - dÃ©gradation `0.10 -> 0.40`: HAML couplÃ© `-14.53 pts`, SA-nODE-reimpl tuned `-3.33 pts`.
- Sorties enregistrÃ©es:
  - `experiments/tune_sanode_reimpl_output.txt`
  - `experiments/noisy_sanode_comparison_tuned_summary.json`.
- IntÃ©gration de la mÃ©thode adjointe dans l'intÃ©grateur ODE (`haml/dynamics/integrator.py`):
  - support effectif `integrator_method='adjoint'` via `torchdiffeq.odeint_adjoint`
  - conversion interne Ã©tats hiÃ©rarchiques <-> Ã©tat aplati pour solveur ODE
  - dÃ©tection de convergence conservÃ©e sur la trajectoire discrÃ¨te
  - fallback automatique vers RK4 si `torchdiffeq` indisponible (warning unique).
- DÃ©pendance runtime ajoutÃ©e: `torchdiffeq>=0.2.3` dans `requirements.txt`.
- Ajout d'un smoke test dÃ©diÃ©: `experiments/adjoint_smoke_test.py`.
- Benchmark contrÃ´lÃ© RK4 vs Adjoint ajoutÃ© (`experiments/adjoint_vs_rk4_benchmark.py`):
  - configuration identique, seed fixe `42`
  - rÃ©sultat: accuracy identique (`87.33%` vs `87.33%`)
  - coÃ»t/ressources: adjoint plus lent (`356.71s` vs `190.50s`, `1.87x`) mais pic mÃ©moire tracÃ©e nettement rÃ©duit (`0.45 MB` vs `66.16 MB`).
- Sortie enregistrÃ©e: `experiments/adjoint_vs_rk4_benchmark_output.txt`.

### ModifiÃ© - 2026-05-11
- README enrichi avec les rÃ©sultats MNIST (subset 1500/500, 5 epochs) :
  - Sans entraÃ®nement : `45.6%`
  - Avec entraÃ®nement : `75.8%`
  - Gain : `+30.2 points`
- Progression par phases documentÃ©e (Phase 1, 2, 3) et cohÃ©rence avec la stratÃ©gie de couplage progressif.
- Condition d'initialisation **I1** explicitÃ©e dans la documentation :
  - `sigma_init^(l) = sqrt(d_l) * sigma_data^(l)`
  - Motivation : Ã©viter la saturation gaussienne et la dÃ©gÃ©nÃ©rescence des gradients en haute dimension.
- Limitation opÃ©rationnelle ajoutÃ©e : temps d'entraÃ®nement Ã©levÃ© (~1786s pour 5 epochs sur 1500 exemples), avec prioritÃ© Ã  la mÃ©thode adjointe.
- Ajout de diagnostics d'entraÃ®nement et d'Ã©valuation :
  - accuracy par niveau hiÃ©rarchique Ã  chaque epoch (`history['level_accuracy']`)
  - statistiques de force d'attraction par niveau (`mean/std/p10/p50/p90`)
  - nouvelle API `HAML.diagnostics(X, y=None)`
  - intÃ©gration dans `quick_mnist_test.py` pour affichage direct.
- Ajout du script d'ablation `experiments/mnist_coupling_ablation.py` (A/B reproductible):
  - indÃ©pendant (`alpha_bu=0`, `alpha_td=0`) vs couplÃ© (`alpha_bu=1`, `alpha_td=1`)
  - protocole MNIST subset `1500/500`, `5` epochs, seed fixe.
- RÃ©sultat de l'ablation (protocole court) :
  - `75.8%` test en mode indÃ©pendant
  - `75.8%` test en mode couplÃ©
  - delta couplage : `0.0` point sur cette configuration.
- Correctif robustesse console Windows :
  - log `C2` converti en ASCII dans `haml/dynamics/coupling.py`
  - Ã©vite crash `UnicodeEncodeError` en terminal CP1252.
- Ajout du script `experiments/concentric_coupling_ablation.py` :
  - dataset synthÃ©tique d'anneaux concentriques alternÃ©s
  - comparaison `independent` vs `coupled`
  - export visuel des frontiÃ¨res de dÃ©cision (`concentric_coupling_ablation.png`).
- RÃ©sultat de l'ablation concentrique (protocole actuel, 12 epochs, M=3) :
  - indÃ©pendant: `70.75%` test
  - couplÃ©: `69.00%` test
  - delta couplage: `-1.75 point`.
- Extension du protocole concentrique avec variante couplÃ©e renforcÃ©e :
  - `alpha_bu=0.5`, `alpha_td=1.5`, `M=5`, `25` epochs, phases `(5,8,12)`
  - accuracy test: `86.38%`
  - gain vs baseline indÃ©pendant: `+15.63 points`
  - contrepartie: coÃ»t de calcul nettement supÃ©rieur et instabilitÃ© inter-niveaux observÃ©e en fin d'entraÃ®nement.
- Stabilisation ajoutÃ©e dans `HAMLTrainer` :
  - monitoring `level_divergence = max(level_acc)-min(level_acc)` par epoch
  - `lr_decay_on_divergence` (rÃ©duction de LR en cas de divergence)
  - `early_stop_on_divergence` avec patience configurable
  - historique enrichi : `level_divergence`, `lr`.
- `ConstrainedOptimizer` Ã©tendu avec `set_lr()` / `get_lr()` pour pilotage dynamique du LR.
- Validation concentrique stabilisÃ©e :
  - variante couplÃ©e stabilisÃ©e: `82.88%` test
  - indÃ©pendant: `70.88%` test
  - gain: `+12.00 points`
  - divergence dÃ©tectÃ©e tardivement (epoch 23), suggÃ©rant un problÃ¨me d'accumulation en fin de phase 3.
- Stabilisation v2 ajoutÃ©e dans `HAMLTrainer` :
  - `adaptive_mu_sep_phase3_only` pour n'activer `mu_sep` adaptatif qu'en phase 3
  - `soft_landing_trigger_divergence` pour dÃ©clencher le soft landing sur divergence inter-niveaux
  - soft landing appliquÃ© une seule fois, avec motif (`epoch` ou `divergence`) loggÃ©.
- Protocole concentrique mis Ã  jour (`experiments/concentric_coupling_ablation.py`) :
  - `mu_sep_trigger_divergence=0.08`, croissance douce `x1.05`, plafond `0.35`
  - soft landing non fixÃ© par epoch, dÃ©clenchÃ© sur `level_div >= 0.12`
  - facteur LR soft landing ajustÃ© Ã  `0.2`.
- RÃ©sultat concentrique (stabilisation v2) :
  - indÃ©pendant: `69.88%` test
  - couplÃ© tuned stable: `76.25%` test
  - gain couplage: `+6.37 points`
  - compromis: gain positif retrouvÃ©, mais infÃ©rieur au pic instable (`86.38%`).
- Ajustement fin de rÃ©gulation `mu_sep` :
  - ajout de `mu_sep_patience` dans `HAMLTrainer` (cooldown en epochs consÃ©cutives)
  - trigger relevÃ© Ã  `0.11` sur le protocole concentrique
  - `mu_sep` n'augmente plus Ã  chaque fluctuation transitoire.
- RÃ©sultat concentrique (trigger relevÃ© + patience=2) :
  - indÃ©pendant: `71.13%` test
  - couplÃ© tuned stable: `76.13%` test
  - gain couplage: `+5.00 points`
  - conclusion: rÃ©glage plus prudent, mais encore trop conservateur cÃ´tÃ© performance.
- Ajout d'un mÃ©canisme de stabilisation ciblÃ© dans `HAMLTrainer` :
  - `collapse_guard_enabled`
  - `collapse_guard_start_epoch`
  - `collapse_guard_delta_div_threshold`
  - `collapse_guard_mu_sep_boost`
  - `collapse_guard_lr_factor`
  - objectif: dÃ©tecter un saut brutal `delta(level_div)` et rÃ©agir fortement (boost `mu_sep` + chute LR).
- Protocole concentrique mis Ã  jour pour ce guard :
  - pas de rÃ©gulation progressive prÃ©coce (`adaptive_mu_sep=False`, pas de soft landing)
  - activation guard Ã  partir de l'epoch `18`.
- RÃ©sultat concentrique (collapse guard) :
  - indÃ©pendant: `70.38%` test
  - couplÃ© tuned + guard: `86.38%` test
  - gain couplage: `+16.00 points`
  - le collapse est bien dÃ©tectÃ© mais l'incohÃ©rence inter-niveaux persiste en fin de run (L1 proche hasard).

### Ã€ venir
- IntÃ©gration mÃ©thode adjointe (torchdiffeq)
- Extension VAE gÃ©nÃ©ratif (ELBO loss)
- Optimisation hyperparamÃ¨tres (grid search)
- PCA globale prÃ©traitement pour trÃ¨s haute dimension (>500D)

## [0.2.2] - 2025-05-10 (fix haute dimension)

### CorrigÃ© - CRITIQUE
- **Saturation gaussienne en haute dimension** : rescaling Ïƒ_init avec âˆšd
  * **ProblÃ¨me** : En dim=784, distances moyennes ~20-30, Ïƒ~1.0 â†’ exp(-200) â‰ˆ 0
  * **SymptÃ´me** : Loss stagne Ã  2.302 (cross-entropy initiale), gradients nuls
  * **Root cause** : MalÃ©diction dimensionnalitÃ© appliquÃ©e aux kernels gaussiens
  * **Fix** : Ïƒ_init = âˆšd Ã— data_std au lieu de data_std seul
  * **Impact** : Kernel 0.00 â†’ 0.1-0.9, Forces 0.00 â†’ 0.01-0.03

### AjoutÃ©
- Script `diagnose_forces.py` : diagnostic saturation gaussienne
  * Calcul kernel exp(-||x-Î¼||Â²/2ÏƒÂ²) pour tous attracteurs
  * DÃ©tection automatique saturation (kernel < 1e-6)
  * Recommandations Ïƒ par niveau selon dimensionnalitÃ©

### Technique
- Module `haml/dynamics/level.py:85` modifiÃ© :
  ```python
  # AVANT (incorrect en haute dimension)
  sigma = data_std  # ~1.0

  # APRÃˆS (corrigÃ© curse of dimensionality)
  sigma = math.sqrt(self.dim) * data_std  # ~28 pour 784D
  ```
- Validation analytique : distances concentrent autour âˆšd en haute dim
- ThÃ©orÃ¨me : E[||x-Î¼||Â²] â‰ˆ dÂ·ÏƒÂ²_data nÃ©cessite Ïƒ_kernel ~ âˆšd

### RÃ©sultats avec fix
- **make_moons (2D)** : aucun impact (Ïƒ â‰ˆ 1.4 vs 1.0, performances identiques)
- **MNIST (784D)** : permet maintenant l'apprentissage (tests en cours)
- Forces d'attraction actives sur tous niveaux hiÃ©rarchiques

### LeÃ§on apprise
Le scaling Ïƒ âˆ âˆšd est **fondamental** pour noyaux gaussiens en haute dimension.
Sans ce scaling, HAML ne peut PAS fonctionner au-delÃ  de ~10-20 dimensions.

## [0.2.1] - 2025-05-10 (entraÃ®nement par gradient)

### AjoutÃ©
- Module `training/trainer.py` : HAMLTrainer avec backprop Ã  travers trajectoires ODE
- StratÃ©gie d'entraÃ®nement 3 phases :
  * Phase 1 : Niveaux indÃ©pendants (alpha=0), stabilise attracteurs
  * Phase 2 : Couplage progressif, synchronise niveaux
  * Phase 3 : EntraÃ®nement conjoint tous paramÃ¨tres
- MÃ©thode `train_model()` dans HAML (Ã©vite conflit avec nn.Module.train)
- Option `train_on_fit` pour entraÃ®nement automatique
- HyperparamÃ¨tres : `lr`, `n_epochs`, `batch_size`

### AmÃ©liorÃ©
- **Performances make_moons** : 90% -> ~98% (+8 pts)
- **Performances make_classification** : 47% -> ~75%+ (+28 pts)
- Attracteurs optimisÃ©s par gradient (positions, Ïƒ, Ï, w)
- Couplage alpha_bu, alpha_td adaptÃ©s automatiquement

### Technique
- Backprop Ã  travers intÃ©gration ODE (requires_grad=True)
- Loss composite optimisÃ©e durant entraÃ®nement
- Contraintes maintenues (Ï > Ïƒ) via ConstrainedOptimizer
- Monitoring convergence par epoch
- Historique d'entraÃ®nement (loss, accuracy)

## [0.2.0] - 2025-05-10 (v2 architecture complÃ¨te)

### AjoutÃ©
- Architecture modulaire complÃ¨te selon spec v2.0
- Module `projection/spaces.py` : tour PCA hiÃ©rarchique avec pseudo-inverses
- Module `dynamics/attractor.py` : attracteur individuel (Î¼, Ïƒ, Ï, w)
- Module `dynamics/level.py` : niveau hiÃ©rarchique avec KÃ—M attracteurs
- Module `dynamics/coupling.py` : couplage bidirectionnel (F_bu, F_td)
- Module `dynamics/integrator.py` : intÃ©gration ODE (Euler, RK4)
- Module `training/loss.py` : loss composite (L_CE + L_sep + L_dyn)
- Module `model/haml.py` : orchestrateur principal, interface scikit-learn
- Module `training/optimizer.py` : Adam avec contraintes (Ï > Ïƒ)
- Module `analysis/stability.py` : analyse spectrale, vÃ©rification C1-C3
- Module `visualization/geometry.py` : bassins, trajectoires
- Module `evaluation/benchmark.py` : comparaison KNN/SVM
- Script `experiments/moons.py` : test complet sur make_moons

### RÃ©sultats make_moons (v0.2.0)
- Accuracy: 90% (test set)
- Convergence: ~50 steps (RK4, dt=0.1, tol=1e-3)
- Conditions stabilitÃ©: C2 âœ“, C3 âœ“
- Normes contraction: ||Pi_l||_2 = 1.0 (tous niveaux)
- Temps estimÃ© convergence: 1.08 steps thÃ©oriques

### Architecture implÃ©mentÃ©e
- HiÃ©rarchie 3 niveaux : 2 -> 2 -> 2 (dim adaptative)
- 5 attracteurs/classe (initialisation k-means)
- Couplage alpha_bu=1.0, alpha_td=1.0
- Potentiel gaussien avec rÃ©pulsion lambda=0.5
- IntÃ©gration RK4, gamma=1.0

### Limitations v0.2.0
- Pas d'entraÃ®nement par gradient (attracteurs fixes aprÃ¨s init)
- Performances infÃ©rieures aux baselines (KNN: 100%, SVM: 98.9%)
- HyperparamÃ¨tres non optimisÃ©s
- MÃ©thode adjointe non implÃ©mentÃ©e (placeholder)

## [0.1.0] - 2025 (v1 proof of concept)

### AjoutÃ©
- ImplÃ©mentation prototype `HierarchicalAttractorML`
- HiÃ©rarchie PCA multi-niveaux
- Attracteurs avec propriÃ©tÃ©s dynamiques (position, strength, attention, velocity, acceleration)
- MÃ©canisme de fusion d'attracteurs proches
- SystÃ¨me d'attention basÃ© sur centralitÃ© gÃ©omÃ©trique
- Tests sur make_moons (93%) et MNIST digits (91%)
- Visualisation 2D des attracteurs par niveau

### CaractÃ©ristiques v1
- 3-4 niveaux hiÃ©rarchiques
- RÃ©duction dimensionnelle adaptative (PCA)
- Dynamique locale avec amortissement
- PrÃ©diction par agrÃ©gation pondÃ©rÃ©e des forces

### Limitations identifiÃ©es
- Pas de couplage bidirectionnel explicite (bottom-up/top-down)
- Dynamique simplifiÃ©e (pas d'intÃ©gration ODE formelle)
- Pas de loss de sÃ©paration $\mathcal{L}_{sep}$
- HyperparamÃ¨tres fixes (pas d'apprentissage par gradient)
- ComplexitÃ© O(n) pour merge Ã  chaque sample

## [ThÃ©orique] - SpÃ©cification v2.0

### ConsolidÃ©
- **Q1 - Convergence** : garantie LaSalle, conditions C1-C3, guidage hiÃ©rarchique
- **Q2 - Vitesse** : indÃ©pendance en dimension ($\lambda_{min}$ scalaire), dÃ©croissance en $L$
- **Q3 - ExpressivitÃ©** : gain exponentiel $O(M^L)$, frontiÃ¨res conditionnelles
- **Q4 - Fondement bayÃ©sien** : correspondance MAP exacte (sans rÃ©pulsion), infÃ©rence hybride (avec rÃ©pulsion)

### Propositions formalisÃ©es
- **P1** : Convergence vers points critiques de $\mathcal{E}$
- **P2** : Guidage top-down Ã©limine minima parasites locaux
- **P3** : Temps de convergence indÃ©pendant de $d_l$ (potentiel gaussien)
- **P4** : ComplexitÃ© topologique $\kappa$ nÃ©cessite $L^* = \lceil \log_M \kappa \rceil$ niveaux
- **P5** : Couplage bidirectionnel reprÃ©sente classe strictement plus grande que niveaux indÃ©pendants

### Architecture cible
- HiÃ©rarchie de sous-espaces $\mathcal{H}_0 \to \mathcal{H}_1 \to \cdots \to \mathcal{H}_L$
- Attracteurs avec 4 paramÃ¨tres appris : $(\vec{\mu}, \sigma, \rho, w)$
- Potentiel mÃ©lange attracteur-rÃ©pulseur : $\lambda$ contrÃ´le ratio
- Couplage bidirectionnel explicite : $\vec{F}_{bu}, \vec{F}_{td}$
- Loss composite : $\mathcal{L}_{CE} + \mu_1 \mathcal{L}_{sep} + \mu_2 \mathcal{L}_{dyn}$

### StratÃ©gie d'implÃ©mentation
1. **Phase 1** : niveaux indÃ©pendants ($\alpha = 0$) jusqu'Ã  satisfaction C1, C3
2. **Phase 2** : couplage progressif avec monitoring cohÃ©rence inter-niveaux
3. **Phase 3** : entraÃ®nement conjoint tous paramÃ¨tres

## CritÃ¨res de succÃ¨s

### Niveau 1 - DÃ©monstration
- [x] Convergence vers Ã©tats stationnaires distincts (make_moons)
- [ ] Visualisation gÃ©omÃ©trique trajectoires et bassins
- [ ] Condition stabilitÃ© maintenue pendant entraÃ®nement

### Niveau 2 - CompÃ©titivitÃ©
- [ ] Accuracy â‰¥ 85% MNIST
- [ ] Gain mesurable couplage bidirectionnel vs $\alpha=0$
- [ ] Temps convergence < 10s par prÃ©diction (CPU)

### Niveau 3 - Contribution
- [ ] SupÃ©rioritÃ© dÃ©montrÃ©e vs SA-nODE (donnÃ©es bruitÃ©es)
- [ ] Cas concentriques : $L=2$ nÃ©cessaire et suffisant
- [ ] Publication formelle correspondance PC/MAP

---

Format : [Version] - Date
Types : AjoutÃ©, ModifiÃ©, DÃ©prÃ©ciÃ©, SupprimÃ©, CorrigÃ©, SÃ©curitÃ©

