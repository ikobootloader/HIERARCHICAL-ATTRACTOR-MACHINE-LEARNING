# Spécification Théorique & Plan de Projet
## Hierarchical Attractor ML (HAML) — Système dynamique hiérarchique bidirectionnel
### pour la classification supervisée

---

> **Statut** : Spécification théorique de référence — v2.0 (consolidée)
> **Positionnement** : Contribution originale à l'intersection des systèmes dynamiques, de l'apprentissage automatique et des théories hiérarchiques du traitement de l'information

---

## Partie I — Fondements théoriques

### 1. Idée centrale et positionnement

#### 1.1 La thèse du projet

Ce projet soutient qu'un **système dynamique hiérarchique bidirectionnel**, dont les attracteurs sont appris par gradient, constitue un modèle de classification supervisée à la fois compétitif, interprétable géométriquement, biologiquement plausible, et formellement équivalent à une inférence MAP bayésienne hiérarchique.

La classification n'est pas un calcul direct (forward pass) mais le **résultat de la convergence** d'un système physique multi-échelles vers un bassin d'attraction stable — la classe prédite étant identifiée par l'attracteur atteint en régime stationnaire.

Quatre propriétés théoriques, établies dans les sections 3 à 6, fondent cette architecture :

- **Convergence garantie** (§3) : le système converge toujours vers un état stationnaire, et la hiérarchie guide cette convergence vers le bon bassin
- **Scalabilité** (§4) : le temps de convergence ne croît ni avec la dimension ni avec le nombre de niveaux
- **Expressivité strictement supérieure** (§5) : la hiérarchie bidirectionnelle représente une classe de fonctions de décision inaccessible aux modèles à niveau unique ou à niveaux indépendants
- **Fondement bayésien exact** (§6) : le régime stationnaire est un estimateur MAP dans un modèle génératif hiérarchique précisément caractérisé

#### 1.2 Distinction fondamentale avec SA-nODE (Marino et al., 2024)

SA-nODE est le travail le plus proche dans la littérature. Les différences architecturales sont fondamentales :

| Dimension | SA-nODE | HAML |
|---|---|---|
| Espace d'état | Dimension = entrée brute (784 pixels) | Hiérarchie de sous-espaces de dimension décroissante |
| Attracteurs | Eigenvecteurs binaires (±a) fixés avant entraînement | Vecteurs continus à positions libres, appris par gradient |
| Potentiel | Double puits analytique fixe | Mélange attracteur-répulseur explicite, paramétrique |
| Structure | Un seul niveau dynamique | $L$ niveaux couplés bidirectionnellement |
| Couplage inter-niveaux | Absent | Top-down et bottom-up simultanés |
| Temps de convergence vs $d$ | Dépend du potentiel polynomial | Indépendant de $d_l$ (potentiel gaussien) |
| Ancrage théorique | Physique des spins, mécanique statistique | Systèmes dynamiques + Predictive Coding (Friston) |
| Interprétabilité | Spectrale (espace réciproque) | Géométrique (espace des features) |
| Équivalence bayésienne | Non établie | MAP exact sous conditions M1–M3 (§6) |

#### 1.3 Ancrage dans la littérature

- **SA-nODE** (Marino et al., 2024) — référence directe et point de différenciation principal
- **Neural ODEs** (Chen et al., NeurIPS 2018) — cadre d'intégration différentiable, méthode adjointe
- **Modern Hopfield Networks** (Ramsauer et al., ICLR 2021) — convergence en une mise à jour, lien attention/énergie
- **Predictive Coding** (Rao & Ballard, 1999 ; Friston, 2005) — fondement théorique du couplage bidirectionnel
- **Active Inference** (Friston, 2017–2023) — cadre d'énergie libre variationnelle
- **GLVQ** (Sato & Yamada, 1996) — cas limite Option A, baseline de comparaison
- **Contrastive / Discriminative Priors** (Larochelle & Bengio, 2008) — interprétation bayésienne des termes répulsifs

---

### 2. Architecture mathématique

#### 2.1 Espace d'états hiérarchique

Soit $\mathcal{X} \subset \mathbb{R}^d$ l'espace des données brutes. On définit une tour de projections :

$$\mathcal{X} = \mathcal{H}_0 \xrightarrow{\pi_1} \mathcal{H}_1 \xrightarrow{\pi_2} \cdots \xrightarrow{\pi_L} \mathcal{H}_L$$

où $\mathcal{H}_l \subset \mathbb{R}^{d_l}$ avec $d_0 > d_1 > \cdots > d_L$.

Les projections $\pi_l : \mathcal{H}_{l-1} \to \mathcal{H}_l$ sont des projections linéaires initialisées par PCA et optionnellement affinées pendant l'entraînement. Les injections remontantes $\pi_l^\dagger : \mathcal{H}_l \to \mathcal{H}_{l-1}$ sont leurs pseudo-inverses de Moore-Penrose.

**Propriété clé** : les projections PCA normalisées sont des contractions, i.e. $\|\Pi_l\|_2 \leq \sqrt{d_{l+1}/d_l} < 1$. Cette propriété est exploitée dans l'analyse de convergence (§4).

L'état du système au niveau $l$ au temps $t$ est noté $\vec{x}^{(l)}(t) \in \mathcal{H}_l$.

#### 2.2 Attracteurs et potentiel

Au niveau $l$, on dispose de $K \times M$ attracteurs, où $K$ est le nombre de classes et $M$ le nombre d'attracteurs par classe :

$$\mathcal{A}^{(l)} = \left\{ \vec{\mu}^{(l)}_{c,m} \in \mathcal{H}_l \;\middle|\; c \in \{1,\ldots,K\},\; m \in \{1,\ldots,M\} \right\}$$

Chaque attracteur est caractérisé par quatre paramètres appris :
- **Position** $\vec{\mu}^{(l)}_{c,m} \in \mathcal{H}_l$
- **Portée d'attraction** $\sigma^{(l)}_{c,m} > 0$ — initialisée à l'écart-type local des données de la classe $c$
- **Portée de répulsion** $\rho^{(l)}_{c,m} > 0$ — avec $\rho > \sigma$ typiquement
- **Poids** $w^{(l)}_{c,m} > 0$ — importance relative de l'attracteur dans la décision

Le potentiel d'énergie au niveau $l$ est un mélange attracteur-répulseur explicite :

$$E^{(l)}(\vec{x}) = -\sum_{c,m} w^{(l)}_{c,m} \left[ \exp\!\left(-\frac{\|\vec{x} - \vec{\mu}^{(l)}_{c,m}\|^2}{2(\sigma^{(l)}_{c,m})^2}\right) - \lambda \sum_{c' \neq c,\, m'} \exp\!\left(-\frac{\|\vec{x} - \vec{\mu}^{(l)}_{c',m'}\|^2}{2(\rho^{(l)}_{c',m'})^2}\right) \right]$$

où $\lambda > 0$ est le rapport attraction/répulsion (hyperparamètre global appris).

**Interprétation bayésienne** (anticipant §6) : sans les termes répulsifs, $-E^{(l)}$ est proportionnel au log d'un mélange de gaussiennes — i.e. le log-prior $\log p(\vec{x}^{(l)})$ dans le modèle génératif implicite. Les termes répulsifs correspondent à un prior discriminatif contrastif.

#### 2.3 Dynamique intra-niveau

La dynamique au niveau $l$ en isolation est le flot de gradient de $E^{(l)}$ :

$$\dot{\vec{x}}^{(l)} = -\nabla_{\vec{x}} E^{(l)}(\vec{x}^{(l)}) \equiv \vec{F}^{(l)}_{intra}(\vec{x}^{(l)})$$

En développant :

$$\vec{F}^{(l)}_{intra}(\vec{x}) = \sum_{c,m} \frac{w^{(l)}_{c,m}}{(\sigma^{(l)}_{c,m})^2} \left(\vec{\mu}^{(l)}_{c,m} - \vec{x}\right) \exp\!\left(-\frac{\|\vec{x} - \vec{\mu}^{(l)}_{c,m}\|^2}{2(\sigma^{(l)}_{c,m})^2}\right) - \lambda \sum_{c' \neq c,\, m'} \frac{w^{(l)}_{c',m'}}{(\rho^{(l)}_{c',m'})^2} \left(\vec{\mu}^{(l)}_{c',m'} - \vec{x}\right) \exp\!\left(-\frac{\|\vec{x} - \vec{\mu}^{(l)}_{c',m'}\|^2}{2(\rho^{(l)}_{c',m'})^2}\right)$$

#### 2.4 Couplage bidirectionnel entre niveaux

Chaque niveau reçoit simultanément un signal ascendant et un signal descendant.

**Signal bottom-up** — erreur de prédiction ascendante :

$$\vec{F}^{(l)}_{bu} = \alpha_{bu} \cdot \pi_l\!\left(\vec{x}^{(l-1)}(t) - \pi_l^\dagger\!\left(\vec{x}^{(l)}(t)\right)\right)$$

C'est la différence entre l'état réel du niveau inférieur et la reconstruction que le niveau courant en ferait. C'est exactement le signal d'erreur de prédiction du Predictive Coding (Rao & Ballard, 1999).

**Signal top-down** — prédiction descendante :

$$\vec{F}^{(l)}_{td} = \alpha_{td} \cdot \left(\pi_l^\dagger\!\left(\vec{x}^{(l+1)}(t)\right) - \vec{x}^{(l)}(t)\right)$$

C'est la force de rappel vers la représentation prédite par le niveau supérieur — la prédiction générée top-down dans le cadre du Predictive Coding.

**Équation du mouvement complète au niveau $l$** :

$$\dot{\vec{x}}^{(l)}(t) = \vec{F}^{(l)}_{intra}(\vec{x}^{(l)}) + \vec{F}^{(l)}_{bu} + \vec{F}^{(l)}_{td} - \gamma\, \dot{\vec{x}}^{(l)}$$

où $\gamma > 0$ est le coefficient d'amortissement visqueux garantissant la dissipation d'énergie.

Conditions aux bords : niveau 0 ($\vec{F}^{(0)}_{bu} = 0$), niveau $L$ ($\vec{F}^{(L)}_{td} = 0$).

#### 2.5 Système couplé complet

$$\begin{cases}
\dot{\vec{x}}^{(0)} = \vec{F}^{(0)}_{intra} + \vec{F}^{(0)}_{td} - \gamma\dot{\vec{x}}^{(0)} \\
\dot{\vec{x}}^{(l)} = \vec{F}^{(l)}_{intra} + \vec{F}^{(l)}_{bu} + \vec{F}^{(l)}_{td} - \gamma\dot{\vec{x}}^{(l)} \quad \forall\; 0 < l < L \\
\dot{\vec{x}}^{(L)} = \vec{F}^{(L)}_{intra} + \vec{F}^{(L)}_{bu} - \gamma\dot{\vec{x}}^{(L)}
\end{cases}$$

**Conditions initiales** : $\vec{x}^{(l)}(0) = \pi_l \circ \cdots \circ \pi_1(\vec{x}_{input})$.

#### 2.6 Prédiction

Après intégration jusqu'au régime stationnaire (critère : $\|\dot{\vec{x}}^{(l)}\| < \epsilon\; \forall l$) :

$$\hat{c} = \arg\max_c \sum_l \beta_l \cdot \sum_m \exp\!\left(-\frac{\|\vec{x}^{(l)}(T) - \vec{\mu}^{(l)}_{c,m}\|^2}{2(\sigma^{(l)}_{c,m})^2}\right)$$

où $\beta_l$ sont des poids par niveau (appris ou fixés à $2^l$).

---

### 3. Convergence et unicité du régime stationnaire (Q1)

#### 3.1 Convergence garantie — Théorème de LaSalle

Dans le régime sur-amorti ($\gamma$ grand, terme d'inertie négligeable), le système se réduit à un flot de gradient pur :

$$\dot{\vec{X}} = -\frac{1}{\gamma}\nabla_{\vec{X}} \mathcal{E}(\vec{X})$$

où $\vec{X} = (\vec{x}^{(0)}, \ldots, \vec{x}^{(L)})$ et $\mathcal{E}$ est la fonctionnelle d'énergie totale :

$$\mathcal{E}(\vec{X}) = \sum_l E^{(l)}(\vec{x}^{(l)}) + \frac{\alpha_{bu} + \alpha_{td}}{2} \sum_l \|\vec{x}^{(l)} - \pi_l^\dagger(\vec{x}^{(l+1)})\|^2$$

**Proposition 1 (Convergence)** : $\mathcal{E}$ est bornée inférieurement et $\dot{\mathcal{E}} = -\frac{1}{\gamma}\|\nabla \mathcal{E}\|^2 \leq 0$ le long des trajectoires. Par le théorème de LaSalle, le système converge vers l'ensemble des points critiques de $\mathcal{E}$. Le système ne peut ni osciller indéfiniment ni diverger.

#### 3.2 Nature des points fixes

Les minima de $\mathcal{E}$ sont de trois types :

**Type 1 — Minima cohérents** (souhaitables) : $\vec{x}^{(l)*} \approx \vec{\mu}^{(l)}_{c,m}$ pour la même classe $c$ à tous les niveaux, avec cohérence inter-niveaux ($\vec{x}^{(l)*} \approx \pi_l^\dagger(\vec{x}^{(l+1)*})$).

**Type 2 — Minima incohérents** (pathologiques) : classes différentes à différents niveaux. Apparaissent quand $\alpha_{bu} + \alpha_{td}$ est trop faible.

**Type 3 — Minima parasites** : créés par interactions attraction/répulsion sans correspondre à un attracteur planté. Disparaissent quand la séparation inter-classes est suffisante.

#### 3.3 Conditions suffisantes de convergence vers le bon bassin

Soit $c^*$ la vraie classe de $\vec{x}_{input}$.

**C1 — Dominance locale** : $\|\pi_l(\vec{x}_{input}) - \vec{\mu}^{(l)}_{c^*}\| < \|\pi_l(\vec{x}_{input}) - \vec{\mu}^{(l)}_{c'}\|\; \forall c' \neq c^*, \forall l$

**C2 — Couplage suffisant** : $\alpha_{bu} + \alpha_{td} > \alpha_{min}(\mathcal{A}, \pi)$ — borne calculable par analyse spectrale du système linéarisé

**C3 — Séparation suffisante** : les bassins d'attraction inter-classes ne se chevauchent pas — garantie par $\mathcal{L}_{sep}$ pendant l'entraînement

#### 3.4 Rôle stabilisateur de la hiérarchie

**Proposition 2 (Guidage hiérarchique)** : avec couplage top-down actif, le niveau $L$ (faible dimension, peu de minima) converge en premier et rapidement vers la bonne classe globale. Le signal top-down contraint ensuite les niveaux inférieurs à chercher dans le bon bassin, éliminant les minima parasites locaux qui auraient pu piéger un système à niveau unique.

La hiérarchie rend le problème de convergence *plus facile*, pas plus difficile.

#### 3.5 Stratégie de couplage progressif (implication pour l'implémentation)

- **Phase 1** : entraîner avec $\alpha_{bu} = \alpha_{td} = 0$ (niveaux indépendants) jusqu'à satisfaction de C1 et C3
- **Phase 2** : augmenter $\alpha_{bu}$ et $\alpha_{td}$ progressivement avec monitoring de cohérence inter-niveaux
- **Phase 3** : entraînement conjoint de tous les paramètres

---

### 4. Vitesse de convergence (Q2)

#### 4.1 Spectre de relaxation

Dans le régime linéarisé autour d'un point fixe $\vec{X}^*$, la perturbation $\delta\vec{X}(t)$ évolue comme :

$$\delta\vec{X}(t) = \sum_k c_k \vec{v}_k \exp\!\left(-\frac{\lambda_k}{\gamma} t\right)$$

Le temps de convergence est dominé par la plus petite valeur propre positive de la Hessienne $\mathbf{H} = \nabla^2_{\vec{X}} \mathcal{E}(\vec{X}^*)$ :

$$T \sim \frac{\gamma}{\lambda_{min}(\mathbf{H})}$$

#### 4.2 Structure bloc-tridiagonale de $\mathbf{H}$

$$\mathbf{H} = \begin{pmatrix} \mathbf{H}^{(0)} + \mathbf{C}^{(0)} & -\alpha_{td}\Pi_1^T & 0 & \cdots \\ -\alpha_{td}\Pi_1 & \mathbf{H}^{(1)} + \mathbf{C}^{(1)} & -\alpha_{td}\Pi_2^T & \cdots \\ 0 & -\alpha_{td}\Pi_2 & \ddots & \vdots \\ \vdots & \vdots & \vdots & \ddots \end{pmatrix}$$

avec $\mathbf{C}^{(l)} = (\alpha_{bu} + \alpha_{td})\mathbf{I}_{d_l}$ — le couplage ajoute $(\alpha_{bu}+\alpha_{td})$ à toutes les valeurs propres diagonales.

#### 4.3 Trois régimes

**Régime découplé** ($\alpha = 0$) : convergence parallèle indépendante, $T = \max_l \gamma/\lambda_{min}^{(l)}$. La hiérarchie n'apporte rien.

**Régime modéré** : par la théorie de Weyl et la propriété de contraction $\|\Pi_l\|_2 < 1$ :

$$\lambda_{min}(\mathbf{H}) \geq \lambda_{min}(\mathbf{H}_{diag}) + (\alpha_{bu} + \alpha_{td})(1 - \|\Pi\|_2) > \lambda_{min}(\mathbf{H}_{diag})$$

Le couplage **accélère** la convergence.

**Régime fort** ($\alpha \gg \lambda_{min}^{(l)}$) : domination par le couplage, la hiérarchie perd son sens. Convergence rapide mais sans discrimination multi-échelles.

Il existe un **couplage optimal** $\alpha^*$ minimisant $T$ — valeur que l'entraînement doit découvrir.

#### 4.4 Indépendance en dimension — Résultat central

**Proposition 3** : au point fixe $\vec{x}^{(l)*} = \vec{\mu}^{(l)}_{c,m}$, la Hessienne intra-niveau se réduit à :

$$\mathbf{H}^{(l)}\big|_{\vec{\mu}} = \kappa^{(l)} \cdot \mathbf{I}_{d_l}$$

où $\kappa^{(l)} = \frac{w^{(l)}}{(\sigma^{(l)})^2} - \lambda \sum_{c' \neq c} \frac{w^{(l)}_{c'}}{(\rho^{(l)}_{c'})^2} \exp\!\left(-\frac{\|\vec{\mu}^{(l)}_c - \vec{\mu}^{(l)}_{c'}\|^2}{2\rho^{(l)2}}\right)$

est un **scalaire indépendant de $d_l$**. Le temps de convergence intra-niveau ne croît pas avec la dimension — propriété spécifique au potentiel gaussien, absente du double puits polynomial de SA-nODE.

#### 4.5 Indépendance en nombre de niveaux

Par décomposition de Schur de $\mathbf{H}$ (structure tridiagonale par blocs) :

$$\lambda_{min}(\mathbf{H}) \geq \min_l \kappa^{(l)} - 2\max_l \|\alpha_{td}\Pi_l^T\|_2$$

La borne inférieure est **indépendante de $L$**. Chaque niveau supplémentaire ajoute une force top-down qui accélère les niveaux inférieurs — convergence monotonement plus rapide avec $L$ croissant jusqu'au régime fort.

#### 4.6 Tableau récapitulatif

| Paramètre | Effet sur $T$ | Mécanisme |
|---|---|---|
| $\gamma$ | $T \propto \gamma$ | Linéaire direct |
| $\sigma^{(l)}$ | $T \propto (\sigma^{(l)})^2$ | Hessienne plus plate |
| $\alpha_{bu} + \alpha_{td}$ | Décroissant jusqu'à $\alpha^*$, puis croissant | Régime optimal |
| $d_l$ | **Aucun effet** | $\lambda_{min}$ scalaire indépendant de $d_l$ |
| $L$ | **Décroissant** (jusqu'au régime fort) | Top-down accélère les niveaux fins |

---

### 5. Expressivité de la hiérarchie (Q3)

#### 5.1 Expressivité d'un niveau unique

La frontière de décision à un niveau est une combinaison de noyaux gaussiens — appartient à la famille des classifieurs RBF. Avec $M \to \infty$ attracteurs : universelle. Mais le nombre d'attracteurs nécessaires pour une géométrie complexe peut être prohibitif.

#### 5.2 Cas analytique nécessitant $L \geq 2$

**Problème des anneaux concentriques alternés** : classes $c=1$ et $c=2$ définies par $r \in [0,1] \cup [2,3]$ et $r \in [1,2] \cup [3,4]$ respectivement.

- **Avec $L=1$** : $M = O(1/\sigma^2)$ attracteurs nécessaires pour couvrir chaque anneau.
- **Avec $L=2$** : niveau 1 (projection radiale, $d_1=1$) résout la structure globale avec 2 attracteurs par classe. Le signal top-down contraint le niveau 0 à affiner localement. **Deux niveaux suffisent** là où un niveau nécessite $O(1/\sigma^2)$ attracteurs.

#### 5.3 Formalisation — Complexité topologique

**Définition** : la complexité topologique $\kappa(\mathcal{P})$ d'un problème de classification est le nombre minimum de composantes connexes de la frontière de décision optimale.

**Proposition 4 (Gain exponentiel)** : un problème de complexité topologique $\kappa$ nécessite :
- Avec $L=1$ niveau : $O(\kappa)$ attracteurs
- Avec $L$ niveaux et $M$ attracteurs par niveau : $O(ML)$ attracteurs pour représenter $O(M^L)$ configurations

Le gain en nombre d'attracteurs est **exponentiel en $L$** — analogue du gain de profondeur dans les réseaux de neurones.

#### 5.4 Ce que le couplage bidirectionnel ajoute

Avec des niveaux indépendants, la décision de chaque niveau ne dépend pas des autres — contribution additive.

Avec le couplage bidirectionnel, la frontière de décision effective au niveau $l$ est **conditionnelle** au contexte des autres niveaux :

$$\mathcal{F}^{(l)}_{\text{effective}} = \mathcal{F}^{(l)}(\vec{x}^{(l)} \mid \vec{x}^{(l+1)*}, \vec{x}^{(l-1)*})$$

**Proposition 5** : il existe des problèmes de classification (exemple : ambiguïté contextuelle locale résolue par le contexte global) représentables par la hiérarchie bidirectionnelle mais **non représentables** par des niveaux indépendants avec le même nombre d'attracteurs.

#### 5.5 Complexité de Rademacher

Pour un classifieur à $L$ niveaux couplés :

$$\mathcal{R}_n(\mathcal{F}_L) = O\!\left(\sqrt{\frac{KML}{n}} \cdot \prod_l (1 + \alpha_l \|\Pi_l\|_F)\right)$$

La complexité croît en $O(\sqrt{L})$ — **sous-linéaire** en $L$. Le rapport expressivité/risque de sur-apprentissage est favorable à la hiérarchie.

#### 5.6 Cas limites instructifs

- $\alpha_{td} \to \infty$ : classifieur purement abstrait opérant dans $\mathcal{H}_L$ — perd l'information locale
- $\alpha_{td} \to 0$ : niveaux indépendants — perd l'expressivité conditionnelle
- $\alpha^*_{td}$ optimal : dépend de la structure du problème — appris par gradient

---

### 6. Correspondance avec le Predictive Coding et fondement bayésien (Q4)

#### 6.1 Le modèle génératif du Predictive Coding

Le Predictive Coding (Rao & Ballard, 1999 ; Friston, 2005) postule un modèle génératif hiérarchique :

$$p(\vec{x}^{(0:L)}) = p(\vec{x}^{(L)}) \prod_{l=0}^{L-1} p(\vec{x}^{(l)} \mid \vec{x}^{(l+1)})$$

Dans l'approximation de champ moyen ponctuel, l'énergie libre variationnelle se réduit à :

$$\mathcal{F}\big|_{ponctuel} = \sum_l \frac{1}{2}\|\vec{\varepsilon}^{(l)}\|^2_{\Sigma^{(l)}} - \log p(\vec{x}^{(L)})$$

où $\vec{\varepsilon}^{(l)} = \vec{x}^{(l)} - g^{(l+1)}(\vec{x}^{(l+1)})$ sont les erreurs de prédiction. La dynamique d'inférence est :

$$\dot{\vec{x}}^{(l)} = -\frac{\partial \mathcal{F}}{\partial \vec{x}^{(l)}} = -\vec{\varepsilon}^{(l)} + \left(\frac{\partial g^{(l+1)}}{\partial \vec{x}^{(l)}}\right)^T \vec{\varepsilon}^{(l+1)}$$

C'est exactement la structure du couplage bidirectionnel de HAML.

#### 6.2 Identification terme à terme

Avec des générateurs linéaires $g^{(l)}(\vec{x}^{(l)}) = \Pi_l^\dagger \vec{x}^{(l)}$ et des covariances isotropes $\Sigma^{(l)} = \frac{1}{\alpha_{bu}+\alpha_{td}}\mathbf{I}$, l'énergie libre variationnelle devient :

$$\mathcal{F}\big|_{linéaire} = \frac{\alpha_{bu}+\alpha_{td}}{2} \sum_l \|\vec{x}^{(l)} - \Pi_l^\dagger \vec{x}^{(l+1)}\|^2 - \log p(\vec{x}^{(L)})$$

Les termes de couplage sont **identiques** à ceux de $\mathcal{E}$. Il reste à identifier $E^{(l)}$ avec $-\log p(\vec{x}^{(l)})$ :

$$-E^{(l)}(\vec{x})\big|_{\lambda=0} = \log \sum_{c,m} w^{(l)}_{c,m}\, \mathcal{N}(\vec{x} \mid \vec{\mu}^{(l)}_{c,m},\, \sigma^{(l)2}_{c,m}\mathbf{I}) + \text{cst}$$

Le potentiel intra-niveau est le log d'un **mélange de gaussiennes** — exactement la structure d'un prior de mélange bayésien.

#### 6.3 Correspondance exacte vs approximative

**Sans répulsion ($\lambda = 0$)** :

$$\mathcal{E} \equiv \mathcal{F}\big|_{ponctuel}$$

La correspondance est **exacte**. HAML implémente l'inférence MAP dans un modèle génératif hiérarchique à prior de mélange gaussien et vraisemblance conditionnelle gaussienne linéaire.

**Avec répulsion ($\lambda > 0$)** :

$$\mathcal{E} = \mathcal{F}_{PC} + \lambda\, \mathcal{F}_{discriminatif}$$

où $\mathcal{F}_{discriminatif}$ est un **prior répulsif discriminatif** (Larochelle & Bengio, 2008), pénalisant les états proches des attracteurs des mauvaises classes. HAML implémente une **inférence hybride** : inférence variationnelle hiérarchique (PC) augmentée d'un prior discriminatif contrastif. Cette combinaison dans un cadre dynamique continu est originale.

#### 6.4 Conditions de validité (MAP exact)

| Condition | Formulation | Statut dans HAML |
|---|---|---|
| M1 — Linéarité des générateurs | $g^{(l)}(\vec{x}^{(l)}) = \Pi_l^\dagger \vec{x}^{(l)}$ | Satisfaite par construction |
| M2 — Covariances isotropes | $\Sigma^{(l)} = \frac{1}{\alpha_{bu}+\alpha_{td}}\mathbf{I}$ | Satisfaite si couplage uniforme ; relaxable |
| M3 — Répulsion nulle ou absorbée | $\lambda = 0$ (exacte) ou $\lambda > 0$ (étendue) | Choix de régime |

#### 6.5 Modèle génératif implicite complet

Le modèle génératif que HAML implémente implicitement est :

$$p(\vec{x}^{(l)}) = \sum_{c,m} w^{(l)}_{c,m}\, \mathcal{N}(\vec{x}^{(l)} \mid \vec{\mu}^{(l)}_{c,m},\, \sigma^{(l)2}_{c,m}\mathbf{I})$$

$$p(\vec{x}^{(l-1)} \mid \vec{x}^{(l)}) = \mathcal{N}\!\left(\vec{x}^{(l-1)} \;\middle|\; \Pi_l^\dagger \vec{x}^{(l)},\; \frac{1}{\alpha_{bu}+\alpha_{td}}\mathbf{I}\right)$$

#### 6.6 Extension vers un modèle génératif complet

La correspondance bayésienne suggère une loss alternative plus rigoureuse — l'ELBO :

$$\mathcal{L}_{ELBO} = \mathbb{E}_q\!\left[\log p(\vec{x}^{(0)} \mid \vec{x}^{(1:L)})\right] - \sum_l KL\!\left[q^{(l)} \,\|\, p^{(l)}\right]$$

Cela ouvre la voie à un **VAE dynamique hiérarchique** capable non seulement de classifier mais aussi de *générer* de nouveaux exemples de chaque classe — extension naturelle du projet.

---

### 7. Loss d'entraînement

#### 7.1 Loss composite

$$\mathcal{L} = \mathcal{L}_{CE} + \mu_1\, \mathcal{L}_{sep} + \mu_2\, \mathcal{L}_{dyn}$$

**Loss de classification** :

$$\mathcal{L}_{CE} = -\frac{1}{N}\sum_{n=1}^N \log\, P(\hat{c} = y_n \mid \{\vec{x}^{(l)}(T)\}_l)$$

où $P(c \mid \cdot)$ est un softmax sur les forces d'attraction agrégées par niveau.

**Loss de séparation** (maintien de la condition de stabilité C3) :

$$\mathcal{L}_{sep} = \sum_l \sum_{c \neq c'} \sum_{m,m'} \exp\!\left(-\frac{\|\vec{\mu}^{(l)}_{c,m} - \vec{\mu}^{(l)}_{c',m'}\|^2}{2\sigma^2_{min}}\right)$$

**Loss de cohérence dynamique** (encourage la convergence rapide) :

$$\mathcal{L}_{dyn} = \frac{1}{T}\int_0^T \sum_l \|\dot{\vec{x}}^{(l)}(t)\|^2\, dt$$

#### 7.2 Différentiation à travers la trajectoire

Le gradient $\partial\mathcal{L}/\partial\theta$ est calculé par :
- **Méthode directe** (déroulage Euler/RK4) : simple, coûteux en mémoire — pour $T$ court
- **Méthode adjointe** (Pontryagin, implémentée via `torchdiffeq`) : efficace en mémoire — pour $T$ long

#### 7.3 Initialisation

Les positions $\vec{\mu}^{(l)}_{c,m}$ sont initialisées par **k-means par classe** sur $\pi_l(\mathcal{X}_{train})$, garantissant séparation initiale et satisfaction de C1 et C3 dès le départ.

---

## Partie II — Plan de projet

### 8. Modules

| Module | Fichier | Responsabilité principale |
|---|---|---|
| 1 | `dynamics/attractor.py` | Attracteur individuel, forces att./rép., condition stabilité locale |
| 2 | `dynamics/level.py` | Niveau hiérarchique, $\vec{F}^{(l)}_{intra}$, énergie $E^{(l)}$ |
| 3 | `dynamics/coupling.py` | Forces $\vec{F}^{(l)}_{bu}$ et $\vec{F}^{(l)}_{td}$, conditions aux bords |
| 4 | `dynamics/integrator.py` | Euler, RK4, critère d'arrêt, interface méthode adjointe |
| 5 | `model/haml.py` | Classe principale, interface scikit-learn, orchestration |
| 6 | `training/loss.py` | $\mathcal{L}_{CE}$, $\mathcal{L}_{sep}$, $\mathcal{L}_{dyn}$, agrégation |
| 7 | `training/optimizer.py` | Adam + projection gradient (contrainte stabilité), scheduler |
| 8 | `projection/spaces.py` | Tour de sous-espaces, PCA, $\pi_l$ et $\pi_l^\dagger$ |
| 9 | `analysis/stability.py` | Vérification Hessienne, $\lambda_{min}$, monitoring entraînement |
| 10 | `visualization/geometry.py` | Bassins d'attraction 2D, trajectoires, cartes d'énergie |
| 11 | `evaluation/benchmark.py` | Baselines (KNN, SVM, GLVQ), métriques, cross-validation |
| 12 | `experiments/` | Scripts reproductibles, configs YAML |

### 9. Structure des fichiers

```
haml/
│
├── dynamics/
│   ├── attractor.py
│   ├── level.py
│   ├── coupling.py
│   └── integrator.py
│
├── model/
│   └── haml.py
│
├── training/
│   ├── loss.py
│   └── optimizer.py
│
├── projection/
│   └── spaces.py
│
├── analysis/
│   └── stability.py
│
├── visualization/
│   └── geometry.py
│
├── evaluation/
│   └── benchmark.py
│
├── experiments/
│   ├── configs/
│   ├── moons.py
│   ├── classification.py
│   ├── mnist.py
│   └── fashion_mnist.py
│
├── tests/
│   ├── test_attractor.py
│   ├── test_coupling.py
│   ├── test_integrator.py
│   ├── test_stability.py
│   └── test_haml.py
│
├── notebooks/
│   ├── 01_theory_illustration.ipynb
│   ├── 02_stability_analysis.ipynb
│   └── 03_benchmark_results.ipynb
│
├── THEORY.md
├── README.md
├── requirements.txt
└── setup.py
```

### 10. Stack technique

| Composant | Choix | Justification |
|---|---|---|
| Framework | PyTorch | Autograd pour différentiation à travers les trajectoires |
| Intégration EDO | Custom Euler + RK4 | Contrôle total sur le déroulage |
| Méthode adjointe | `torchdiffeq` | Implémentation de référence (Chen et al.) |
| Optimisation | Adam + projection gradient custom | Stabilité + contrainte de séparation |
| Analyse spectrale | NumPy / SciPy | Calcul $\lambda_{min}$ pour vérification stabilité |
| Visualisation | Matplotlib + Seaborn | Standard |
| Configuration | YAML + `omegaconf` | Reproductibilité expériences |
| Tests | pytest | Couverture unitaire |
| Versioning | Git, tags sémantiques | `v0.1-scaffold`, `v1.0-stable`, `v2.0-bidirectional` |

### 11. Risques techniques

| Risque | Probabilité | Impact | Mitigation |
|---|---|---|---|
| Non-convergence du système couplé | Moyen | Élevé | Couplage progressif (§3.5) |
| Instabilité numérique | Faible | Moyen | Adaptive step size, RK4 |
| Violation conditions stabilité pendant entraînement | Moyen | Élevé | Projection gradient + monitoring $\lambda_{min}$ |
| Coût computationnel prohibitif | Élevé | Moyen | Méthode adjointe, early stopping EDO |
| Mauvaise séparation des bassins | Moyen | Élevé | $\mathcal{L}_{sep}$ + initialisation k-means |

### 12. Critères de succès

**Niveau 1 — Démonstration de concept**
- Convergence vers états stationnaires distincts sur `make_moons` et `make_classification`
- Visualisation géométrique interprétable des trajectoires et bassins
- Condition de stabilité maintenue pendant l'entraînement

**Niveau 2 — Compétitivité**
- Accuracy ≥ 85% sur digits MNIST
- Gain mesurable du couplage bidirectionnel vs niveaux indépendants ($\alpha=0$)
- Temps de convergence < 10s par prédiction sur CPU

**Niveau 3 — Contribution originale**
- Démonstration empirique de la supériorité sur données bruitées vs SA-nODE
- Cas synthétique concentriques : $L=2$ nécessaire et suffisant là où $L=1$ échoue
- Note théorique formalisée sur la correspondance PC/MAP (§6) — base d'une publication

---

### 13. Références

- Marino, R. et al. (2024). *SA-nODE*. arXiv:2311.10387.
- Chen, R. T. Q. et al. (2018). *Neural Ordinary Differential Equations*. NeurIPS.
- Ramsauer, H. et al. (2021). *Hopfield Networks is All You Need*. ICLR.
- Rao, R. P. N., & Ballard, D. H. (1999). *Predictive coding in the visual cortex*. Nature Neuroscience.
- Friston, K. (2005). *A theory of cortical responses*. Phil. Trans. R. Soc. B.
- Friston, K. (2017). *Active inference: A process theory*. Neural Computation.
- Larochelle, H., & Bengio, Y. (2008). *Classification using discriminative restricted Boltzmann machines*. ICML.
- Sato, A., & Yamada, K. (1996). *Generalized Learning Vector Quantization*. NeurIPS.
- Pontryagin, L. S. (1962). *Mathematical Theory of Optimal Processes*. Wiley.
- Hawkins, J., & George, D. (2006). *Hierarchical Temporal Memory*. Numenta.
- Weyl, H. (1912). *Das asymptotische Verteilungsgesetz der Eigenwerte linearer partieller Differentialgleichungen*. Math. Ann.

---

*Spécification théorique v2.0 — consolidée après analyse complète de Q1–Q4.
Les quatre propositions (§3–§6) constituent la base théorique d'une contribution publiable.*

*Projet intitié par curiosité - FM*
