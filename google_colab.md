

````markdown
# Lancer des runs HAML sur Google Colab

Ce guide explique comment installer le dépôt HAML dans Google Colab, mettre à jour le code depuis GitHub, lancer des expériences, sauvegarder les résultats JSON et récupérer les fichiers générés.

Repository utilisé :

```text
https://github.com/ikobootloader/HIERARCHICAL-ATTRACTOR-MACHINE-LEARNING
```

---

## 1. Préparer l’environnement Colab

Dans une cellule Colab, commencer par cloner le dépôt :

```python
!rm -rf HIERARCHICAL-ATTRACTOR-MACHINE-LEARNING
!git clone https://github.com/ikobootloader/HIERARCHICAL-ATTRACTOR-MACHINE-LEARNING.git
%cd HIERARCHICAL-ATTRACTOR-MACHINE-LEARNING
```

Installer ensuite le projet en mode éditable :

```python
!pip install -e .
```

Le mode éditable `-e .` permet à Python d’utiliser directement les fichiers du dépôt cloné.

---

## 2. Redémarrer la session Colab si nécessaire

Après une installation ou une mise à jour importante, Colab peut afficher un message du type :

```text
You must restart the runtime in order to use newly installed versions.
```

Dans ce cas, cliquer sur :

```text
Runtime > Restart runtime
```

ou exécuter :

```python
import os
os.kill(os.getpid(), 9)
```

Après le redémarrage, revenir dans le dossier du projet :

```python
%cd /content/HIERARCHICAL-ATTRACTOR-MACHINE-LEARNING
```

---

## 3. Vérifier que le projet est bien installé

Tester l’import principal :

```python
from haml import HAML
```

Vérifier aussi que le script d’expérience existe :

```python
!ls experiments
```

Pour vérifier spécifiquement le script Fashion-MNIST :

```python
!ls experiments | grep fashion
```

Le fichier attendu est :

```text
fashion_mnist_short_probe.py
```

---

## 4. Mettre à jour le dépôt depuis GitHub

Si le dépôt GitHub a été modifié, mettre à jour la copie Colab avec :

```python
%cd /content/HIERARCHICAL-ATTRACTOR-MACHINE-LEARNING
!git pull
!pip install -e .
```

Si des modules Python ont déjà été importés avant la mise à jour, redémarrer ensuite le runtime :

```python
import os
os.kill(os.getpid(), 9)
```

Puis revenir dans le projet :

```python
%cd /content/HIERARCHICAL-ATTRACTOR-MACHINE-LEARNING
```

---

## 5. Lancer un run Fashion-MNIST court

Commande recommandée dans Colab :

```python
!PYTHONIOENCODING=utf-8 python experiments/fashion_mnist_short_probe.py \
  --seed 42 \
  --n-train 3000 \
  --n-test 600 \
  --n-epochs 8 \
  --phase1-epochs 2 \
  --phase2-epochs 2 \
  --json-out experiments/diag_fashion_mnist_seed42_n3000_e8_mutex.json
```

Version en une seule ligne :

```python
!PYTHONIOENCODING=utf-8 python experiments/fashion_mnist_short_probe.py --seed 42 --n-train 3000 --n-test 600 --n-epochs 8 --phase1-epochs 2 --phase2-epochs 2 --json-out experiments/diag_fashion_mnist_seed42_n3000_e8_mutex.json
```

---

## 6. Lancer plusieurs runs avec plusieurs seeds

Exemple avec trois seeds :

```python
!PYTHONIOENCODING=utf-8 python experiments/fashion_mnist_short_probe.py \
  --seed 42 \
  --n-train 3000 \
  --n-test 600 \
  --n-epochs 8 \
  --phase1-epochs 2 \
  --phase2-epochs 2 \
  --json-out experiments/diag_fashion_mnist_seed42_n3000_e8.json

!PYTHONIOENCODING=utf-8 python experiments/fashion_mnist_short_probe.py \
  --seed 123 \
  --n-train 3000 \
  --n-test 600 \
  --n-epochs 8 \
  --phase1-epochs 2 \
  --phase2-epochs 2 \
  --json-out experiments/diag_fashion_mnist_seed123_n3000_e8.json

!PYTHONIOENCODING=utf-8 python experiments/fashion_mnist_short_probe.py \
  --seed 2026 \
  --n-train 3000 \
  --n-test 600 \
  --n-epochs 8 \
  --phase1-epochs 2 \
  --phase2-epochs 2 \
  --json-out experiments/diag_fashion_mnist_seed2026_n3000_e8.json
```

---

## 7. Lancer automatiquement une série de runs

Il est possible d’utiliser une boucle Bash dans Colab :

```python
!for SEED in 42 123 2026; do \
  PYTHONIOENCODING=utf-8 python experiments/fashion_mnist_short_probe.py \
    --seed $SEED \
    --n-train 3000 \
    --n-test 600 \
    --n-epochs 8 \
    --phase1-epochs 2 \
    --phase2-epochs 2 \
    --json-out experiments/diag_fashion_mnist_seed${SEED}_n3000_e8.json; \
done
```

---

## 8. Run très court pour vérifier que tout fonctionne

Avant de lancer une expérience plus longue, il est conseillé de faire un petit test rapide :

```python
!PYTHONIOENCODING=utf-8 python experiments/fashion_mnist_short_probe.py \
  --seed 42 \
  --n-train 500 \
  --n-test 100 \
  --n-epochs 2 \
  --phase1-epochs 1 \
  --phase2-epochs 1 \
  --json-out experiments/test_fashion_mnist_quick.json
```

Ce run sert uniquement à vérifier que le script, les imports et les dépendances fonctionnent correctement.

---

## 9. Run intermédiaire

Exemple de run un peu plus sérieux, mais encore raisonnable pour Colab :

```python
!PYTHONIOENCODING=utf-8 python experiments/fashion_mnist_short_probe.py \
  --seed 42 \
  --n-train 5000 \
  --n-test 1000 \
  --n-epochs 12 \
  --phase1-epochs 3 \
  --phase2-epochs 3 \
  --json-out experiments/diag_fashion_mnist_seed42_n5000_e12.json
```

---

## 10. Run plus long

Exemple de run plus coûteux :

```python
!PYTHONIOENCODING=utf-8 python experiments/fashion_mnist_short_probe.py \
  --seed 42 \
  --n-train 10000 \
  --n-test 2000 \
  --n-epochs 20 \
  --phase1-epochs 5 \
  --phase2-epochs 5 \
  --json-out experiments/diag_fashion_mnist_seed42_n10000_e20.json
```

À utiliser de préférence avec un runtime GPU.

---

## 11. Vérifier le GPU Colab

Pour vérifier si un GPU est disponible :

```python
!nvidia-smi
```

Si aucun GPU n’apparaît, aller dans :

```text
Runtime > Change runtime type > Hardware accelerator > GPU
```

Puis redémarrer la session.

---

## 12. Lire les fichiers JSON générés

Afficher la liste des résultats :

```python
!ls experiments/*.json
```

Afficher le contenu brut d’un fichier JSON :

```python
!cat experiments/diag_fashion_mnist_seed42_n3000_e8_mutex.json
```

Lire le JSON avec Python :

```python
import json

path = "experiments/diag_fashion_mnist_seed42_n3000_e8_mutex.json"

with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

data
```

---

## 13. Télécharger un fichier JSON depuis Colab

Pour télécharger un fichier sur son ordinateur :

```python
from google.colab import files

files.download("experiments/diag_fashion_mnist_seed42_n3000_e8_mutex.json")
```

---

## 14. Sauvegarder les résultats dans Google Drive

Monter Google Drive :

```python
from google.colab import drive
drive.mount('/content/drive')
```

Créer un dossier de résultats :

```python
!mkdir -p /content/drive/MyDrive/HAML_runs
```

Copier les JSON générés :

```python
!cp experiments/*.json /content/drive/MyDrive/HAML_runs/
```

Vérifier la copie :

```python
!ls /content/drive/MyDrive/HAML_runs
```

---

## 15. Exemple de workflow complet Colab

Voici une séquence complète, de l’installation au lancement du run :

```python
!rm -rf HIERARCHICAL-ATTRACTOR-MACHINE-LEARNING
!git clone https://github.com/ikobootloader/HIERARCHICAL-ATTRACTOR-MACHINE-LEARNING.git
%cd HIERARCHICAL-ATTRACTOR-MACHINE-LEARNING
!pip install -e .
```

Redémarrer ensuite le runtime si Colab le demande.

Puis :

```python
%cd /content/HIERARCHICAL-ATTRACTOR-MACHINE-LEARNING

!PYTHONIOENCODING=utf-8 python experiments/fashion_mnist_short_probe.py \
  --seed 42 \
  --n-train 3000 \
  --n-test 600 \
  --n-epochs 8 \
  --phase1-epochs 2 \
  --phase2-epochs 2 \
  --json-out experiments/diag_fashion_mnist_seed42_n3000_e8_mutex.json
```

Enfin, télécharger le résultat :

```python
from google.colab import files
files.download("experiments/diag_fashion_mnist_seed42_n3000_e8_mutex.json")
```

---

## 16. Différence entre commande Windows et commande Colab

Sur Windows PowerShell, on peut écrire :

```powershell
$env:PYTHONIOENCODING='utf-8'; python script.py
```

Mais cette syntaxe ne fonctionne pas dans Colab.

Dans Colab, il faut utiliser la syntaxe Linux :

```python
!PYTHONIOENCODING=utf-8 python script.py
```

Le `!` indique à Colab qu’il faut exécuter une commande système, et non du code Python.

---

## 17. Erreurs fréquentes

### Erreur : `SyntaxError: invalid syntax`

Cause probable : une commande shell ou PowerShell a été exécutée comme du Python.

Mauvais exemple :

```python
$env:PYTHONIOENCODING='utf-8'; python script.py
```

Bon exemple Colab :

```python
!PYTHONIOENCODING=utf-8 python script.py
```

---

### Erreur : `No such file or directory`

Exemple :

```text
python3: can't open file '/content/HIERARCHICAL-ATTRACTOR-MACHINE-LEARNING/github_app/experiments/fashion_mnist_short_probe.py'
```

Cause probable : le chemin du script est incorrect.

Mauvais chemin :

```text
github_app/experiments/fashion_mnist_short_probe.py
```

Bon chemin :

```text
experiments/fashion_mnist_short_probe.py
```

Commande correcte :

```python
!PYTHONIOENCODING=utf-8 python experiments/fashion_mnist_short_probe.py
```

---

### Erreur après `pip install`

Si Colab indique qu’il faut redémarrer le runtime, c’est normal.

Après redémarrage :

```python
%cd /content/HIERARCHICAL-ATTRACTOR-MACHINE-LEARNING
```

Puis relancer l’expérience.

---

## 18. Conseils pour nommer les fichiers de sortie

Utiliser des noms explicites :

```text
diag_fashion_mnist_seed42_n3000_e8.json
diag_fashion_mnist_seed123_n3000_e8.json
diag_fashion_mnist_seed42_n5000_e12.json
diag_fashion_mnist_seed42_n10000_e20.json
```

Convention proposée :

```text
diag_<dataset>_seed<seed>_n<n_train>_e<epochs>.json
```

Exemple :

```text
diag_fashion_mnist_seed42_n3000_e8.json
```

---

## 19. Bonnes pratiques expérimentales

Pour des résultats plus sérieux :

1. lancer plusieurs seeds ;
2. conserver tous les fichiers JSON ;
3. noter les paramètres de chaque run ;
4. éviter de comparer un seul run isolé ;
5. sauvegarder les résultats dans Google Drive ;
6. vérifier que le même script et le même commit GitHub sont utilisés.

Pour connaître le commit Git utilisé dans Colab :

```python
!git rev-parse HEAD
```

Pour voir l’état du dépôt :

```python
!git status
```

---

## 20. Exemple de tableau de runs

| Nom du run | Seed | Train | Test | Epochs | Phase 1 | Phase 2 | Sortie JSON |
|---|---:|---:|---:|---:|---:|---:|---|
| quick-test | 42 | 500 | 100 | 2 | 1 | 1 | `test_fashion_mnist_quick.json` |
| baseline-3k | 42 | 3000 | 600 | 8 | 2 | 2 | `diag_fashion_mnist_seed42_n3000_e8.json` |
| baseline-3k | 123 | 3000 | 600 | 8 | 2 | 2 | `diag_fashion_mnist_seed123_n3000_e8.json` |
| baseline-5k | 42 | 5000 | 1000 | 12 | 3 | 3 | `diag_fashion_mnist_seed42_n5000_e12.json` |
| long-10k | 42 | 10000 | 2000 | 20 | 5 | 5 | `diag_fashion_mnist_seed42_n10000_e20.json` |

---

## 21. Commande recommandée actuelle

Commande principale recommandée pour un run Fashion-MNIST court :

```python
%cd /content/HIERARCHICAL-ATTRACTOR-MACHINE-LEARNING

!PYTHONIOENCODING=utf-8 python experiments/fashion_mnist_short_probe.py \
  --seed 42 \
  --n-train 3000 \
  --n-test 600 \
  --n-epochs 8 \
  --phase1-epochs 2 \
  --phase2-epochs 2 \
  --json-out experiments/diag_fashion_mnist_seed42_n3000_e8_mutex.json
```
````
