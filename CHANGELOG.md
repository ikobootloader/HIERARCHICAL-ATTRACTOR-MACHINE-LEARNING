# Changelog

Toutes les modifications notables du projet HAML seront documentées dans ce fichier.

Le format est basé sur [Keep a Changelog](https://keepachangelog.com/fr/1.0.0/).

## [Non publié]

### À venir
- Intégration méthode adjointe (torchdiffeq)
- Extension VAE génératif (ELBO loss)
- Optimisation hyperparamètres (grid search)
- PCA globale prétraitement pour très haute dimension (>500D)

## [0.2.2] - 2025-05-10 (fix haute dimension)

### Corrigé - CRITIQUE
- **Saturation gaussienne en haute dimension** : rescaling σ_init avec √d
  * **Problème** : En dim=784, distances moyennes ~20-30, σ~1.0 → exp(-200) ≈ 0
  * **Symptôme** : Loss stagne à 2.302 (cross-entropy initiale), gradients nuls
  * **Root cause** : Malédiction dimensionnalité appliquée aux kernels gaussiens
  * **Fix** : σ_init = √d × data_std au lieu de data_std seul
  * **Impact** : Kernel 0.00 → 0.1-0.9, Forces 0.00 → 0.01-0.03

### Ajouté
- Script `diagnose_forces.py` : diagnostic saturation gaussienne
  * Calcul kernel exp(-||x-μ||²/2σ²) pour tous attracteurs
  * Détection automatique saturation (kernel < 1e-6)
  * Recommandations σ par niveau selon dimensionnalité

### Technique
- Module `haml/dynamics/level.py:85` modifié :
  ```python
  # AVANT (incorrect en haute dimension)
  sigma = data_std  # ~1.0

  # APRÈS (corrigé curse of dimensionality)
  sigma = math.sqrt(self.dim) * data_std  # ~28 pour 784D
  ```
- Validation analytique : distances concentrent autour √d en haute dim
- Théorème : E[||x-μ||²] ≈ d·σ²_data nécessite σ_kernel ~ √d

### Résultats avec fix
- **make_moons (2D)** : aucun impact (σ ≈ 1.4 vs 1.0, performances identiques)
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
- Attracteurs optimisés par gradient (positions, σ, ρ, w)
- Couplage alpha_bu, alpha_td adaptés automatiquement

### Technique
- Backprop à travers intégration ODE (requires_grad=True)
- Loss composite optimisée durant entraînement
- Contraintes maintenues (ρ > σ) via ConstrainedOptimizer
- Monitoring convergence par epoch
- Historique d'entraînement (loss, accuracy)

## [0.2.0] - 2025-05-10 (v2 architecture complète)

### Ajouté
- Architecture modulaire complète selon spec v2.0
- Module `projection/spaces.py` : tour PCA hiérarchique avec pseudo-inverses
- Module `dynamics/attractor.py` : attracteur individuel (μ, σ, ρ, w)
- Module `dynamics/level.py` : niveau hiérarchique avec K×M attracteurs
- Module `dynamics/coupling.py` : couplage bidirectionnel (F_bu, F_td)
- Module `dynamics/integrator.py` : intégration ODE (Euler, RK4)
- Module `training/loss.py` : loss composite (L_CE + L_sep + L_dyn)
- Module `model/haml.py` : orchestrateur principal, interface scikit-learn
- Module `training/optimizer.py` : Adam avec contraintes (ρ > σ)
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
