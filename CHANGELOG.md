# Changelog

Toutes les modifications notables du projet HAML seront documentées dans ce fichier.

Le format est basé sur [Keep a Changelog](https://keepachangelog.com/fr/1.0.0/).

## [Non publié]

### Modifié - 2026-05-18
- Ajout d'un script de profilage runtime `experiments/profile_training_runtime.py` pour mesurer objectivement les coûts avant optimisation :
  - profilage global via `cProfile` (`.prof` + tops `cumtime`/`tottime`),
  - profilage ligne-par-ligne optionnel via `line_profiler` (si disponible),
  - export d'un résumé JSON des artefacts et métriques run.
- Le script supporte deux modes dataset :
  - `make_moons` (local, sans dépendance réseau) pour smoke-tests rapides,
  - `fashion_mnist` (OpenML) pour profilage du pipeline cible quand le réseau est disponible.

### Modifié - 2026-05-17
- Campagne F (Fashion-MNIST) complétée avec comparaison `coupled_tuned` vs `independent` sur seeds `42..46` :
  - couplé: `80.28% ± 1.00`
  - indépendant: `80.30% ± 0.95`
  - delta moyen: `-0.02 pt` (statistiquement nul sur ce protocole).
- README mis à jour pour expliciter la délimitation :
  - absence de gain couplé mesurable sur Fashion-MNIST en `10` epochs (`2` attracteurs/classe),
  - positionné comme condition d'applicabilité (gain couplé dépendant de la structure du problème).

### Modifié - 2026-05-16
- README: ajout de la référence d'archive Zenodo (DOI) `10.5281/zenodo.20238600`.
- Validation réelle ajoutée sur Fashion-MNIST via `experiments/fashion_mnist_short_probe.py` :
  - protocole: seed `42`, `n_train=5000`, `n_test=1000`, `10` epochs, phases `(2,3,5)`
  - résultat: accuracy test `80.1%`, accuracy train finale `82.3%`
  - progression de phase: `75.36%` -> `78.86%` -> `82.26%`
  - stabilité: aucun trigger recovery/stability (`events=[]`), divergence inter-niveaux faible.
- Test d'accélération par réduction dimensionnelle documenté (Fashion-MNIST, seed `42`) :
  - variante `level_dims=[64,32]` vs baseline `784->392->196`
  - résultat: `77.3%` test vs `80.1%` et temps quasi inchangé (`6872s` vs `6979s`)
  - conclusion: pas de gain compute exploitable via réduction de dimensions dans ce protocole.
- Compatibilité de lancement GPU améliorée:
  - ajout de l'option `--device {auto,cpu,cuda}` dans
    `experiments/mnist_short_probe.py` et `experiments/fashion_mnist_short_probe.py`
  - résolution de device `auto` dans `haml/model/haml.py` (CUDA si disponible, sinon CPU).
- Paramètres de speed-test exposés dans `experiments/fashion_mnist_short_probe.py` :
  - ajout des options CLI `--batch-size`, `--max-steps`, `--tol`
  - objectif : mesurer rapidement l'impact compute (2 epochs) avant lancement de campagnes longues.

### Modifié - 2026-05-15
- Nettoyage des artefacts d'expériences pour réduire le bruit opérationnel dans `github_app/experiments` :
  - conservation en clair des baselines actives :
    - `diag_seed42_46_n3200_diffgate_mutex.json`
    - `diag_noisy_moons_seed42_46_n3200_diffgate_mutex.json`
    - `diag_make_classification_seed42_46_n3200_diffgate_mutex.json`
  - archivage réversible des anciens artefacts (`.json/.log/.txt/.md`) vers :
    - `experiments/_archive_results_2026-05-15/`
- Dédoublonnage du workspace au profit de `github_app` :
  - déplacement des anciennes copies racine (`haml`, `experiments`, `v1`, scripts legacy) vers :
    - `_archive_legacy_2026-05-15/`
  - objectif : une seule base active du modèle, sans perte d'historique.
- Correctif packaging `pip` dans `setup.py` :
  - lecture BOM-safe de `requirements.txt` via `encoding="utf-8-sig"`
  - filtrage robuste des commentaires en tête de fichier
  - correction URL du projet vers `ikobootloader/HIERARCHICAL-ATTRACTOR-MACHINE-LEARNING`
  - validation locale: `python setup.py egg_info` OK.

### Modifié - 2026-05-15
- Extension minimale de `experiments/seedwise_coupling_diagnostics.py` :
  - ajout d'un sélecteur `--dataset` pour réutiliser le même protocole et le même format JSON sur plusieurs distributions;
  - jeux supportés: `concentric`, `noisy_moons`, `make_classification`.
- Baseline de stabilisation figée dans `haml/training/config.py` :
  - nouveau paramètre `StabilityConfig.skip_lr_decay_if_recovery_triggered=True`
  - objectif: officialiser le mutex intra-epoch (pas de double réduction LR quand `level-recovery` et `stability` se chevauchent).
- Validation transfert `noisy_moons` (`noise=0.30`, seeds `42..46`, `n_samples=3200`) :
  - scores test: `88.12%`, `91.00%`, `89.88%`, `86.25%`, `90.38%`
  - agrégé: `mean=89.13%`, `std=1.73`, `min=86.25%`, `max=91.00%`
  - sortie: `github_app/experiments/diag_noisy_moons_seed42_46_n3200_diffgate_mutex.json`.
- Validation transfert `make_classification` (`4` classes, features redondantes, seeds `42..46`, `n_samples=3200`) :
  - scores test: `79.38%`, `75.50%`, `76.88%`, `71.50%`, `67.12%`
  - agrégé: `mean=74.08%`, `std=4.31`, `min=67.12%`, `max=79.38%`
  - sortie: `github_app/experiments/diag_make_classification_seed42_46_n3200_diffgate_mutex.json`.

### Modifié - 2026-05-14
- Recalibrage du trigger level-recovery sur protocole `n_samples=3200` :
  - `max_train_accuracy_to_trigger` ajusté à `0.71` dans `experiments/seedwise_coupling_diagnostics.py`
  - test ciblé seeds `42`/`45`: trigger confirmé sur seed `45` (score `68.88%`), pas de trigger sur seed `42` (score `77.63%`)
  - campagne complète couplée seeds `42..46`: `74.68% ± 5.49`, min `67.25%`, max `83.38%`
  - comparaison à E10 (`78.75% ± 7.63`): régression moyenne `-4.08 pts`; seuil `0.71` non retenu comme configuration par défaut.
- Campagne seed-wise `coupled_tuned` étendue à `n_samples=3200` (seeds `42..46`) :
  - résultat global: `78.75% ± 7.63`, min `68.25%`, max `87.38%`
  - gain net vs E8 (`n_samples=1200`): moyenne `+7.95 pts`, min `+7.58 pts`
  - variance inter-seeds en hausse (`std +2.09`).
- Diagnostic des seeds extrêmes (`n_samples=3200`) :
  - seed `42` (`87.38%`) : aucun déclenchement recovery L1; run fort sans garde-fou level-recovery
  - seed `45` (`68.25%`) : aucun déclenchement recovery L1; niveau `L2` proche hasard en phase 3
  - implication: le recovery calibré sur `n_samples=1200` n'est pas le levier principal des runs faibles à `n_samples=3200`.
- Test ciblé de généralisation multi-niveaux (`target_level_idx=None`) sur `n_samples=3200` :
  - seeds `42,43`: `85.75%`, `81.75%` (aucun déclenchement `level-recovery` observé)
  - seed `45`: `67.88%` (aucun déclenchement `level-recovery` observé)
  - conclusion: retirer la restriction L1 ne suffit pas; le gating de déclenchement reste le facteur bloquant.
- Recovery level-aware: correction de design du streak dans `haml/training/trainer.py` :
  - le streak de blocage niveau est désormais accumulé indépendamment des gates globales (`divergence`, `train_accuracy`)
  - les gates globales contrôlent uniquement le déclenchement effectif du recovery
  - effet visé: éviter les déclenchements trop tardifs dus à une fenêtre de gate intermittente.
- Ajout d'un garde-fou level-aware en entraînement (`haml/training/config.py`, `haml/training/trainer.py`) :
  - nouveau `LevelRecoveryConfig` (stagnation near-chance en phase 3)
  - déclenchement borné: phase 3 + divergence minimale + budget de déclenchement
  - recovery locale douce des attracteurs du niveau bloqué (de-collapse ciblé).
- Itération de robustification du trigger level-aware:
  - ajout d'une condition `max_train_accuracy_to_trigger` pour éviter les faux positifs sur runs déjà sains
  - garde-fou activé dans le protocole seed-wise `coupled_tuned` de `experiments/seedwise_coupling_diagnostics.py`.
- Revalidation seed-wise (`42..46`, `n_samples=1200`) après garde-fou level-aware filtré :
  - couplé baseline: moyenne `68.60%`, std `7.54`, min `56.67%`
  - couplé level-aware: moyenne `70.80%`, std `5.54`, min `60.67%`
  - indépendant: moyenne `59.73%`, std `5.93`, min `51.67%`
  - seed `42` corrigé: delta couplé-vs-indépendant `-11.67 pts` -> `+0.67 pt`.
- Formalisation théorique dans `README.md`:
  - ajout de la proposition `P6` (détection de collapse niveau-aware en phase 3)
  - condition de déclenchement et effet attendu explicités.
- README complété sur la campagne concentrique :
  - ajout d'un sous-test `B2` avec résultats multi-seeds consolidés (`n_runs=5`)
  - chiffres documentés : indépendant `67.30% ± 4.37`, couplé `71.20% ± 6.43`, delta `+3.90 pts`
  - lien explicite entre instabilité observée en `B2` et réponse méthodologique `E7/E8`.
- Instrumentation mécanistique niveau-aware ajoutée dans `HAMLTrainer` (`haml/training/trainer.py`) :
  - nouvel historique `level_attractor_diagnostics` par epoch et par niveau
  - métriques exportées: `intra_spread_mean/min/max`, `inter_centroid_dist_min/mean`, `separation_ratio`.
- Script de diagnostic ciblé ajouté `experiments/analyze_seedwise_collapse.py` :
  - lecture des traces seed-wise et isolation des niveaux bloqués en phase 3
  - timeline mécanistique consolidée (`level_divergence`, `mu_sep`, `lr`, `level_acc`, géométrie attracteurs).
- Run de contrôle mécanistique `coupled_tuned` sur seeds `42..43` (`n_samples=1200`) :
  - seed `42`: `69.0%` test, niveau `L1` bloqué proche hasard en phase 3 (`max streak=12`), `L2` également stagnant
  - seed `43`: `75.0%` test, `L1` apprend (`0.669` final) alors que `L2` reste proche hasard
  - conclusion opérationnelle: le run dégradé est corrélé à un blocage de niveau intermédiaire en phase 3 (pas seulement à une cascade LR).
- Patch cause-oriented de récupération niveau (`haml/training/trainer.py`, `haml/training/config.py`) :
  - récupération locale changée: re-anchor supervisé sur prototypes de classe en espace latent du niveau
  - ajout d'une atténuation top-down temporaire post-recovery (`td_cooldown_epochs`, `td_scale_during_cooldown`)
  - réglage par défaut conservateur: `td_cooldown_epochs=1`, `td_scale_during_cooldown=0.7`.
- Contrôle seed `42` post-patch causale (`experiments/diag_seed42_post_patch.json`) :
  - déblocage mécanique de `L1` confirmé en phase 3 (`0.500 -> 0.602 -> 0.644` juste après recovery)
  - accuracy test du run de contrôle: `65.0%` (effet mécanique validé, calibration performance à consolider en multi-seeds).

- Ajout d'un script de diagnostic seed-wise dédié `experiments/seedwise_coupling_diagnostics.py`:
  - export JSON par seed des traces `level_divergence`, `mu_sep`, `lr`, `level_accuracy`
  - export des états attracteurs (sigma/rho/poids/positions)
  - support des modes `independent` et `coupled_tuned`.
- Diagnostic reproduit sur seeds `42..46` (`n_samples=1200`) :
  - couplé avant patch: moyenne `68.60%`, écart-type `7.54`, min `56.67%`
  - indépendant: moyenne `59.73%`, écart-type `5.93`, min `51.67%`
  - confirmation d'une instabilité tardive (phase 3) avec décays LR en cascade et niveau bloqué proche hasard sur certains runs.
- Stabilisation entraînement renforcée (`haml/training/config.py`, `haml/training/trainer.py`) :
  - nouveaux paramètres `StabilityConfig`:
    - `phase3_only`
    - `lr_decay_cooldown_epochs`
    - `max_lr_decay_events`
  - logique trainer ajustée pour éviter la cascade de décays LR (cooldown + budget max).
- Protocole `coupled_tuned` du diagnostic mis à jour:
  - `adaptive_mu_sep` réactivé en phase 3 (croissance modérée)
  - stabilité conservatrice bornée (cooldown + plafond d'événements).
- Revalidation seed-wise après patch (même protocole seeds `42..46`, `n_samples=1200`) :
  - couplé après patch: moyenne `69.33%`, écart-type `7.10`, min `57.33%`
  - amélioration modérée de la queue basse, seed dégradé encore présent (seed `42`).

- Refactor de l'orchestration d'entraînement avec configurations métier dédiées :
  - nouveau module `haml/training/config.py` avec
    `PhaseConfig`, `StabilityConfig`, `AdaptiveMuSepConfig`,
    `SoftLandingConfig`, `CollapseGuardConfig`
  - `HAMLTrainer` accepte désormais ces objets de configuration
    au lieu d'une longue liste de paramètres plats.
- Intégration des nouvelles configs côté orchestration :
  - `haml/model/haml.py` instancie `PhaseConfig` dans `train_model()`
  - `experiments/concentric_coupling_ablation.py` migre vers les configs
    groupées sans changement de protocole.
- Réduction de duplication dans le scoring hiérarchique :
  - extraction `_compute_hierarchical_scores()` et `_level_weight()`
    dans `HAML` pour centraliser la logique utilisée par `predict`
    et `predict_proba`.
- Paramétrisation explicite de l'initialisation `rho` :
  - ajout de `rho_sigma_ratio` (`HAML` et `Level`)
  - remplacement de la constante en dur `rho_init = sigma * 2.0`.
- Robustesse d'initialisation des attracteurs :
  - validation explicite si une classe ne contient aucun échantillon
    lors de `initialize_attractors`.
- Ajout de tests unitaires ciblés (`tests/test_refactor_invariants.py`) :
  - pondération de niveau (`uniform` / `exponential`)
  - agrégation hiérarchique centralisée des scores
  - application de `rho_sigma_ratio`
  - erreur explicite si une classe est absente à l'initialisation
  - injection des configs groupées dans `HAMLTrainer`.
- README restructuré pour segmenter clairement les campagnes de tests :
  - organisation en blocs `Campagne A..E` avec objectif/protocole/résultats/reproduction
  - séparation explicite des sous-tests de stabilisation concentrique (`E1..E6`).
- README enrichi avec un index rapide des campagnes :
  - tableau synthétique `campagne -> objectif -> script(s) -> indicateur clé`
  - navigation accélérée pour distinguer les protocoles de test.
- Correction d'encodage dans `README.md` :
  - suppression d'un artefact mojibake sur l'intitulé "CORRECTION CRITIQUE"
  - restauration des symboles grecs `μ`, `σ`, `ρ` dans la section entraînement.
  - correction de la ligne d'avertissement haute dimension (suppression de `⚠ï¸` corrompu).

### Modifie - 2026-05-13
- Refactor du protocole `experiments/concentric_coupling_ablation.py` en ablation multi-seeds configurable.
- Ajout des options CLI `--n-runs`, `--base-seed`, `--n-samples`, `--save-figure`, `--json-out`.
- Ajout d'une aggregation statistique (moyenne/ecart-type/min/max) des accuracies et temps d'entrainement.
- Export standard des resultats dans `experiments/concentric_coupling_ablation_summary.json`.
- Le script conserve une figure optionnelle de frontieres pour le premier run uniquement.
- README enrichi avec l'analyse concentrique qualitative + quantitative :
  - ajout d'un résultat multi-seeds observé (`n_runs=2`, seeds `42,43`) : delta moyen couplé vs indépendant `+13.00 points`
  - explicitation du signal qualitatif de frontière (topologie annulaire capturée en mode couplé, non capturée en mode indépendant)
  - réserve documentée sur les lobes parasites et l'instabilité inter-niveaux tardive.
- Clarification scientifique ajoutée sur la comparaison bruitée HAML vs SA-nODE-reimpl :
  - la dégradation plus faible de SA-nODE-reimpl sous bruit croissant est maintenue comme tension ouverte à expliquer (pas masquée comme simple trade-off descriptif).
- Reformulation de la claim mémoire adjointe :
  - le ratio `0.45 MB` vs `66.16 MB` est précisé comme métrique `tracemalloc` (mémoire Python tracée), proxy partiel et non mesure de mémoire globale processus/GPU.
- Ajout d'une prédiction falsifiable explicite pour l'hypothèse "ambiguïté intermédiaire" :
  - le profil en cloche du gain de couplage doit se reproduire en multi-seeds sur d'autres datasets.
### Modifié - 2026-05-12
- Exécution du benchmark bruité reproductible `experiments/noisy_benchmark.py` (seed fixe `42`, `n_samples=3000`) comparant :
  - HAML indépendant (`alpha_bu=0`, `alpha_td=0`)
  - HAML couplé (`alpha_bu=0.5`, `alpha_td=1.5`)
  - baselines `KNN` et `SVM RBF`.
- Résultats (accuracy test) :
  - bruit `0.10` : HAML ind. `99.60%`, HAML cpl. `99.60%`, KNN `99.87%`, SVM `99.87%`, delta `+0.00 point`
  - bruit `0.20` : HAML ind. `89.33%`, HAML cpl. `95.60%`, KNN `96.53%`, SVM `97.20%`, delta `+6.27 points`
  - bruit `0.30` : HAML ind. `88.13%`, HAML cpl. `90.53%`, KNN `90.80%`, SVM `91.33%`, delta `+2.40 points`
  - bruit `0.40` : HAML ind. `84.67%`, HAML cpl. `85.07%`, KNN `83.33%`, SVM `87.33%`, delta `+0.40 point`.
- Sortie complète enregistrée dans `experiments/noisy_benchmark_output.txt` (incluant le JSON final des résultats).
- Documentation enrichie avec lecture empirique consolidée :
  - dégradation bruit `0.20 -> 0.40` : KNN `-13.20 pts`, HAML couplé `-10.53 pts`, SVM `-9.87 pts`
  - pattern en cloche du gain de couplage (`0.00`, `+6.27`, `+2.40`, `+0.40`) cohérent avec l'hypothèse top-down
  - synthèse théorie/expérience explicitée et point restant identifié pour publication : benchmark direct vs SA-nODE.
- Ajout du script `experiments/noisy_sanode_comparison.py` :
  - protocole aligné avec le benchmark bruité (`make_moons`, seed `42`, split identique)
  - comparaison HAML ind./couplé, `SA-nODE-like`, KNN, SVM.
- Résultats `HAML couplé` vs `SA-nODE-like` (accuracy test) :
  - bruit `0.10` : `99.60%` vs `82.27%` (delta `+17.33 pts`)
  - bruit `0.20` : `95.60%` vs `81.20%` (delta `+14.40 pts`)
  - bruit `0.30` : `90.53%` vs `79.87%` (delta `+10.67 pts`)
  - bruit `0.40` : `85.07%` vs `76.67%` (delta `+8.40 pts`).
- Clarification méthodologique ajoutée :
  - `SA-nODE-like` est une baseline proxy interne (dynamique simple double-puits, espace unique),
    non équivalente à l'implémentation officielle SA-nODE de la publication.
  - la comparaison est positionnée comme ablation architecturale contrôlée (effet hiérarchie bidirectionnelle vs modèle plat), et non comme claim définitif contre SA-nODE officiel.
- Sortie complète enregistrée dans `experiments/noisy_sanode_comparison_output.txt`.
- Révision de la baseline SA-nODE:
  - remplacement du proxy simple par une réimplémentation plus fidèle (`haml/baselines/sanode.py`)
  - éléments inclus: espace plat, attracteurs binaires ±a plantés, potentiel double-puits analytique, couplage linéaire entraîné.
- Résultats mis à jour `HAML couplé` vs `SA-nODE-reimpl` (accuracy test):
  - bruit `0.10`: `99.60%` vs `87.20%` (delta `+12.40 pts`)
  - bruit `0.20`: `95.60%` vs `85.73%` (delta `+9.87 pts`)
  - bruit `0.30`: `90.53%` vs `84.53%` (delta `+6.00 pts`)
  - bruit `0.40`: `85.07%` vs `83.47%` (delta `+1.60 pt`).
- Ajout d'une recherche d'hyperparamètres SA-nODE-reimpl (`experiments/tune_sanode_reimpl.py`) :
  - grille testée (seed fixe, architecture inchangée):
    - `a in {0.8, 1.0, 1.2}`
    - `dt in {0.03, 0.05, 0.08}`
    - `gamma in {0.0, 0.05, 0.1}`
  - meilleure config moyenne multi-bruit: `a=1.0`, `dt=0.03`, `gamma=0.0`
  - meilleure config bruit fort (`0.40`): `a=0.8`, `dt=0.03`, `gamma=0.1` (gain limité).
- Résultats consolidés avec SA-nODE-reimpl tuned (HAML inchangé, même protocole/seed):
  - bruit `0.10`: `99.60%` vs `86.67%` (delta `+12.93 pts`)
  - bruit `0.20`: `95.60%` vs `86.40%` (delta `+9.20 pts`)
  - bruit `0.30`: `90.53%` vs `85.87%` (delta `+4.67 pts`)
  - bruit `0.40`: `85.07%` vs `83.33%` (delta `+1.73 pt`)
  - dégradation `0.10 -> 0.40`: HAML couplé `-14.53 pts`, SA-nODE-reimpl tuned `-3.33 pts`.
- Sorties enregistrées:
  - `experiments/tune_sanode_reimpl_output.txt`
  - `experiments/noisy_sanode_comparison_tuned_summary.json`.
- Intégration de la méthode adjointe dans l'intégrateur ODE (`haml/dynamics/integrator.py`):
  - support effectif `integrator_method='adjoint'` via `torchdiffeq.odeint_adjoint`
  - conversion interne états hiérarchiques <-> état aplati pour solveur ODE
  - détection de convergence conservée sur la trajectoire discrète
  - fallback automatique vers RK4 si `torchdiffeq` indisponible (warning unique).
- Dépendance runtime ajoutée: `torchdiffeq>=0.2.3` dans `requirements.txt`.
- Ajout d'un smoke test dédié: `experiments/adjoint_smoke_test.py`.
- Benchmark contrôlé RK4 vs Adjoint ajouté (`experiments/adjoint_vs_rk4_benchmark.py`):
  - configuration identique, seed fixe `42`
  - résultat: accuracy identique (`87.33%` vs `87.33%`)
  - coût/ressources: adjoint plus lent (`356.71s` vs `190.50s`, `1.87x`) mais pic mémoire tracée nettement réduit (`0.45 MB` vs `66.16 MB`).
- Sortie enregistrée: `experiments/adjoint_vs_rk4_benchmark_output.txt`.

### Modifié - 2026-05-11
- README enrichi avec les résultats MNIST (subset 1500/500, 5 epochs) :
  - Sans entraînement : `45.6%`
  - Avec entraînement : `75.8%`
  - Gain : `+30.2 points`
- Progression par phases documentée (Phase 1, 2, 3) et cohérence avec la stratégie de couplage progressif.
- Condition d'initialisation **I1** explicitée dans la documentation :
  - `sigma_init^(l) = sqrt(d_l) * sigma_data^(l)`
  - Motivation : éviter la saturation gaussienne et la dégénérescence des gradients en haute dimension.
- Limitation opérationnelle ajoutée : temps d'entraînement élevé (~1786s pour 5 epochs sur 1500 exemples), avec priorité à la méthode adjointe.
- Ajout de diagnostics d'entraînement et d'évaluation :
  - accuracy par niveau hiérarchique à chaque epoch (`history['level_accuracy']`)
  - statistiques de force d'attraction par niveau (`mean/std/p10/p50/p90`)
  - nouvelle API `HAML.diagnostics(X, y=None)`
  - intégration dans `quick_mnist_test.py` pour affichage direct.
- Ajout du script d'ablation `experiments/mnist_coupling_ablation.py` (A/B reproductible):
  - indépendant (`alpha_bu=0`, `alpha_td=0`) vs couplé (`alpha_bu=1`, `alpha_td=1`)
  - protocole MNIST subset `1500/500`, `5` epochs, seed fixe.
- Résultat de l'ablation (protocole court) :
  - `75.8%` test en mode indépendant
  - `75.8%` test en mode couplé
  - delta couplage : `0.0` point sur cette configuration.
- Correctif robustesse console Windows :
  - log `C2` converti en ASCII dans `haml/dynamics/coupling.py`
  - évite crash `UnicodeEncodeError` en terminal CP1252.
- Ajout du script `experiments/concentric_coupling_ablation.py` :
  - dataset synthétique d'anneaux concentriques alternés
  - comparaison `independent` vs `coupled`
  - export visuel des frontières de décision (`concentric_coupling_ablation.png`).
- Résultat de l'ablation concentrique (protocole actuel, 12 epochs, M=3) :
  - indépendant: `70.75%` test
  - couplé: `69.00%` test
  - delta couplage: `-1.75 point`.
- Extension du protocole concentrique avec variante couplée renforcée :
  - `alpha_bu=0.5`, `alpha_td=1.5`, `M=5`, `25` epochs, phases `(5,8,12)`
  - accuracy test: `86.38%`
  - gain vs baseline indépendant: `+15.63 points`
  - contrepartie: coût de calcul nettement supérieur et instabilité inter-niveaux observée en fin d'entraînement.
- Stabilisation ajoutée dans `HAMLTrainer` :
  - monitoring `level_divergence = max(level_acc)-min(level_acc)` par epoch
  - `lr_decay_on_divergence` (réduction de LR en cas de divergence)
  - `early_stop_on_divergence` avec patience configurable
  - historique enrichi : `level_divergence`, `lr`.
- `ConstrainedOptimizer` étendu avec `set_lr()` / `get_lr()` pour pilotage dynamique du LR.
- Validation concentrique stabilisée :
  - variante couplée stabilisée: `82.88%` test
  - indépendant: `70.88%` test
  - gain: `+12.00 points`
  - divergence détectée tardivement (epoch 23), suggérant un problème d'accumulation en fin de phase 3.
- Stabilisation v2 ajoutée dans `HAMLTrainer` :
  - `adaptive_mu_sep_phase3_only` pour n'activer `mu_sep` adaptatif qu'en phase 3
  - `soft_landing_trigger_divergence` pour déclencher le soft landing sur divergence inter-niveaux
  - soft landing appliqué une seule fois, avec motif (`epoch` ou `divergence`) loggé.
- Protocole concentrique mis à jour (`experiments/concentric_coupling_ablation.py`) :
  - `mu_sep_trigger_divergence=0.08`, croissance douce `x1.05`, plafond `0.35`
  - soft landing non fixé par epoch, déclenché sur `level_div >= 0.12`
  - facteur LR soft landing ajusté à `0.2`.
- Résultat concentrique (stabilisation v2) :
  - indépendant: `69.88%` test
  - couplé tuned stable: `76.25%` test
  - gain couplage: `+6.37 points`
  - compromis: gain positif retrouvé, mais inférieur au pic instable (`86.38%`).
- Ajustement fin de régulation `mu_sep` :
  - ajout de `mu_sep_patience` dans `HAMLTrainer` (cooldown en epochs consécutives)
  - trigger relevé à `0.11` sur le protocole concentrique
  - `mu_sep` n'augmente plus à chaque fluctuation transitoire.
- Résultat concentrique (trigger relevé + patience=2) :
  - indépendant: `71.13%` test
  - couplé tuned stable: `76.13%` test
  - gain couplage: `+5.00 points`
  - conclusion: réglage plus prudent, mais encore trop conservateur côté performance.
- Ajout d'un mécanisme de stabilisation ciblé dans `HAMLTrainer` :
  - `collapse_guard_enabled`
  - `collapse_guard_start_epoch`
  - `collapse_guard_delta_div_threshold`
  - `collapse_guard_mu_sep_boost`
  - `collapse_guard_lr_factor`
  - objectif: détecter un saut brutal `delta(level_div)` et réagir fortement (boost `mu_sep` + chute LR).
- Protocole concentrique mis à jour pour ce guard :
  - pas de régulation progressive précoce (`adaptive_mu_sep=False`, pas de soft landing)
  - activation guard à partir de l'epoch `18`.
- Résultat concentrique (collapse guard) :
  - indépendant: `70.38%` test
  - couplé tuned + guard: `86.38%` test
  - gain couplage: `+16.00 points`
  - le collapse est bien détecté mais l'incohérence inter-niveaux persiste en fin de run (L1 proche hasard).

### À venir
- Intégration méthode adjointe (torchdiffeq)
- Extension VAE génératif (ELBO loss)
- Optimisation hyperparamètres (grid search)
- PCA globale prétraitement pour très haute dimension (>500D)

## [0.2.2] - 2025-05-10 (fix haute dimension)

### Corrigé - CRITIQUE
- **Saturation gaussienne en haute dimension** : rescaling Ïƒ_init avec √d
  * **Problème** : En dim=784, distances moyennes ~20-30, Ïƒ~1.0 → exp(-200) ≈ 0
  * **Symptôme** : Loss stagne à 2.302 (cross-entropy initiale), gradients nuls
  * **Root cause** : Malédiction dimensionnalité appliquée aux kernels gaussiens
  * **Fix** : Ïƒ_init = √d × data_std au lieu de data_std seul
  * **Impact** : Kernel 0.00 → 0.1-0.9, Forces 0.00 → 0.01-0.03

### Ajouté
- Script `diagnose_forces.py` : diagnostic saturation gaussienne
  * Calcul kernel exp(-||x-Î¼||²/2Ïƒ²) pour tous attracteurs
  * Détection automatique saturation (kernel < 1e-6)
  * Recommandations Ïƒ par niveau selon dimensionnalité

### Technique
- Module `haml/dynamics/level.py:85` modifié :
  ```python
  # AVANT (incorrect en haute dimension)
  sigma = data_std  # ~1.0

  # APRÈS (corrigé curse of dimensionality)
  sigma = math.sqrt(self.dim) * data_std  # ~28 pour 784D
  ```
- Validation analytique : distances concentrent autour √d en haute dim
- Théorème : E[||x-Î¼||²] ≈ d·Ïƒ²_data nécessite Ïƒ_kernel ~ √d

### Résultats avec fix
- **make_moons (2D)** : aucun impact (Ïƒ ≈ 1.4 vs 1.0, performances identiques)
- **MNIST (784D)** : permet maintenant l'apprentissage (tests en cours)
- Forces d'attraction actives sur tous niveaux hiérarchiques

### Leçon apprise
Le scaling σ ∝ √d est **fondamental** pour noyaux gaussiens en haute dimension.
Sans ce scaling, HAML ne peut PAS fonctionner au-delà de ~10-20 dimensions.

## [0.2.1] - 2025-05-10 (entraînement par gradient)

### Ajouté
- Module `training/trainer.py` : HAMLTrainer avec backprop à travers trajectoires ODE
- Stratégie d'entraînement 3 phases :
  * Phase 1 : Niveaux indépendants (alpha=0), stabilise attracteurs
  * Phase 2 : Couplage progressif, synchronise niveaux
  * Phase 3 : Entraînement conjoint tous paramètres
- Méthode `train_model()` dans HAML (évite conflit avec nn.Module.train)
- Option `train_on_fit` pour entraînement automatique
- Hyperparamètres : `lr`, `n_epochs`, `batch_size`

### Amélioré
- **Performances make_moons** : 90% -> ~98% (+8 pts)
- **Performances make_classification** : 47% -> ~75%+ (+28 pts)
- Attracteurs optimisés par gradient (positions, Ïƒ, Ï, w)
- Couplage alpha_bu, alpha_td adaptés automatiquement

### Technique
- Backprop à travers intégration ODE (requires_grad=True)
- Loss composite optimisée durant entraînement
- Contraintes maintenues (Ï > Ïƒ) via ConstrainedOptimizer
- Monitoring convergence par epoch
- Historique d'entraînement (loss, accuracy)

## [0.2.0] - 2025-05-10 (v2 architecture complète)

### Ajouté
- Architecture modulaire complète selon spec v2.0
- Module `projection/spaces.py` : tour PCA hiérarchique avec pseudo-inverses
- Module `dynamics/attractor.py` : attracteur individuel (Î¼, Ïƒ, Ï, w)
- Module `dynamics/level.py` : niveau hiérarchique avec K×M attracteurs
- Module `dynamics/coupling.py` : couplage bidirectionnel (F_bu, F_td)
- Module `dynamics/integrator.py` : intégration ODE (Euler, RK4)
- Module `training/loss.py` : loss composite (L_CE + L_sep + L_dyn)
- Module `model/haml.py` : orchestrateur principal, interface scikit-learn
- Module `training/optimizer.py` : Adam avec contraintes (Ï > Ïƒ)
- Module `analysis/stability.py` : analyse spectrale, vérification C1-C3
- Module `visualization/geometry.py` : bassins, trajectoires
- Module `evaluation/benchmark.py` : comparaison KNN/SVM
- Script `experiments/moons.py` : test complet sur make_moons

### Résultats make_moons (v0.2.0)
- Accuracy: 90% (test set)
- Convergence: ~50 steps (RK4, dt=0.1, tol=1e-3)
- Conditions stabilité: C2 ✓, C3 ✓
- Normes contraction: ||Pi_l||_2 = 1.0 (tous niveaux)
- Temps estimé convergence: 1.08 steps théoriques

### Architecture implémentée
- Hiérarchie 3 niveaux : 2 -> 2 -> 2 (dim adaptative)
- 5 attracteurs/classe (initialisation k-means)
- Couplage alpha_bu=1.0, alpha_td=1.0
- Potentiel gaussien avec répulsion lambda=0.5
- Intégration RK4, gamma=1.0

### Limitations v0.2.0
- Pas d'entraînement par gradient (attracteurs fixes après init)
- Performances inférieures aux baselines (KNN: 100%, SVM: 98.9%)
- Hyperparamètres non optimisés
- Méthode adjointe non implémentée (placeholder)

## [0.1.0] - 2025 (v1 proof of concept)

### Ajouté
- Implémentation prototype `HierarchicalAttractorML`
- Hiérarchie PCA multi-niveaux
- Attracteurs avec propriétés dynamiques (position, strength, attention, velocity, acceleration)
- Mécanisme de fusion d'attracteurs proches
- Système d'attention basé sur centralité géométrique
- Tests sur make_moons (93%) et MNIST digits (91%)
- Visualisation 2D des attracteurs par niveau

### Caractéristiques v1
- 3-4 niveaux hiérarchiques
- Réduction dimensionnelle adaptative (PCA)
- Dynamique locale avec amortissement
- Prédiction par agrégation pondérée des forces

### Limitations identifiées
- Pas de couplage bidirectionnel explicite (bottom-up/top-down)
- Dynamique simplifiée (pas d'intégration ODE formelle)
- Pas de loss de séparation $\mathcal{L}_{sep}$
- Hyperparamètres fixes (pas d'apprentissage par gradient)
- Complexité O(n) pour merge à chaque sample

## [Théorique] - Spécification v2.0

### Consolidé
- **Q1 - Convergence** : garantie LaSalle, conditions C1-C3, guidage hiérarchique
- **Q2 - Vitesse** : indépendance en dimension ($\lambda_{min}$ scalaire), décroissance en $L$
- **Q3 - Expressivité** : gain exponentiel $O(M^L)$, frontières conditionnelles
- **Q4 - Fondement bayésien** : correspondance MAP exacte (sans répulsion), inférence hybride (avec répulsion)

### Propositions formalisées
- **P1** : Convergence vers points critiques de $\mathcal{E}$
- **P2** : Guidage top-down élimine minima parasites locaux
- **P3** : Temps de convergence indépendant de $d_l$ (potentiel gaussien)
- **P4** : Complexité topologique $\kappa$ nécessite $L^* = \lceil \log_M \kappa \rceil$ niveaux
- **P5** : Couplage bidirectionnel représente classe strictement plus grande que niveaux indépendants

### Architecture cible
- Hiérarchie de sous-espaces $\mathcal{H}_0 \to \mathcal{H}_1 \to \cdots \to \mathcal{H}_L$
- Attracteurs avec 4 paramètres appris : $(\vec{\mu}, \sigma, \rho, w)$
- Potentiel mélange attracteur-répulseur : $\lambda$ contrôle ratio
- Couplage bidirectionnel explicite : $\vec{F}_{bu}, \vec{F}_{td}$
- Loss composite : $\mathcal{L}_{CE} + \mu_1 \mathcal{L}_{sep} + \mu_2 \mathcal{L}_{dyn}$

### Stratégie d'implémentation
1. **Phase 1** : niveaux indépendants ($\alpha = 0$) jusqu'à satisfaction C1, C3
2. **Phase 2** : couplage progressif avec monitoring cohérence inter-niveaux
3. **Phase 3** : entraînement conjoint tous paramètres

## Critères de succès

### Niveau 1 - Démonstration
- [x] Convergence vers états stationnaires distincts (make_moons)
- [ ] Visualisation géométrique trajectoires et bassins
- [ ] Condition stabilité maintenue pendant entraînement

### Niveau 2 - Compétitivité
- [ ] Accuracy ≥ 85% MNIST
- [ ] Gain mesurable couplage bidirectionnel vs $\alpha=0$
- [ ] Temps convergence < 10s par prédiction (CPU)

### Niveau 3 - Contribution
- [ ] Supériorité démontrée vs SA-nODE (données bruitées)
- [ ] Cas concentriques : $L=2$ nécessaire et suffisant
- [ ] Publication formelle correspondance PC/MAP

---

Format : [Version] - Date
Types : Ajouté, Modifié, Déprécié, Supprimé, Corrigé, Sécurité

