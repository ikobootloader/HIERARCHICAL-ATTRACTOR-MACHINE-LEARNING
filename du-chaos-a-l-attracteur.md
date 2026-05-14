---

## HAML : quand la physique des attracteurs réinvente l'apprentissage automatique

*Une expérience de pensée à la frontière des systèmes dynamiques et du machine learning — et ce qu'elle m'a appris en chemin.*

---

Tout a commencé avec un livre. *Le chaos, la complexité et l'émergence de la vie*, de John Gribbin — un de ces ouvrages qu'on relit plusieurs fois non par obligation, mais parce que chaque lecture en extrait quelque chose de nouveau. La dernière fois que je l'ai parcouru, une idée m'a traversé avec cette clarté particulière qui précède parfois les projets qu'on n'avait pas prévu de lancer.

Et si les bassins d'attraction — ces régions de l'espace vers lesquelles convergent spontanément les systèmes dynamiques — pouvaient servir de structure fondamentale à un modèle d'apprentissage automatique ?

### La thèse, en clair

Les réseaux de neurones actuels classifient par calcul direct : une donnée entre, des opérations matriciales s'enchaînent, une prédiction sort. C'est efficace, mais c'est aussi, d'une certaine façon, un peu brutal — un forward pass sans mémoire d'état, sans dynamique, sans géométrie.

L'idée que j'ai voulu explorer est différente. Plutôt qu'un calcul, une **convergence**. La donnée en entrée initialise un système physique qui, soumis à des forces d'attraction et de répulsion, évolue dans le temps jusqu'à se stabiliser dans un bassin — et c'est l'identité de ce bassin qui constitue la prédiction. La classe n'est pas calculée : elle est *atteinte*, comme une bille qui roule naturellement vers le creux d'un paysage énergétique.

Ce projet s'appelle **HAML** — *Hierarchical Attractor ML* — et sa spécification théorique repose sur quatre propositions : convergence garantie vers un état stationnaire, temps de convergence indépendant de la dimension des données, expressivité strictement supérieure aux modèles à niveau unique, et équivalence formelle avec une inférence bayésienne de type MAP. Sur le papier, c'est élégant. Dans le code, c'est une autre histoire — j'y reviendrai.

### Des fondements dans la physique et les neurosciences

Ce qui rend le projet intellectuellement solide c'est qu'il ne s'appuie pas sur une seule intuition, mais sur plusieurs corpus convergents.

Des **systèmes dynamiques** d'abord : la théorie des attracteurs, les bassins de stabilité, la notion d'auto-organisation par laquelle un système dissipatif peut converger naturellement vers des structures de faible entropie. J'ai beaucoup pensé à ce principe de néguentropie — un organisme vivant qui capte de l'énergie pour maintenir sa structure dans le temps — et c'est exactement ce que j'ai voulu transposer à l'apprentissage.

Des **neurosciences computationnelles** ensuite. HAML est organisé en niveaux hiérarchiques couplés dans les deux sens : chaque niveau reçoit à la fois un signal ascendant (les données brutes, traitées progressivement) et un signal descendant (les prédictions des niveaux supérieurs). C'est le cadre du *Predictive Coding* formalisé par Rao, Ballard et Friston — le cerveau ne traite pas passivement les signaux sensoriels, il génère en permanence des prédictions top-down et les ajuste à la lumière des erreurs bottom-up. HAML implémente cette logique dans un système différentiable et entraînable par gradient.

Et des **mathématiques**, enfin — le théorème de LaSalle pour la convergence, la théorie spectrale pour la stabilité des attracteurs, la méthode adjointe de Pontryagin pour différencier à travers les trajectoires d'intégration numérique.

Je n'ai absolument pas la prétention d'être expert dans ces domaines, je les explore au fil de l'eau grâce à l'IA.

### Un projet qui communique avec lui-même — via deux IA

La mise en œuvre a pris une forme un peu inhabituelle. J'ai orchestré un dialogue entre deux intelligences artificielles : Codex d'OpenAI pour la génération du code, Claude d'Anthropic pour l'analyse critique des résultats. Je fournissais à chacune les éléments du projet — spécification théorique, changelogs, résultats de runs — et les retours de l'une alimentaient les instructions données à l'autre. Un pipeline de feedback humain-médié entre deux systèmes, qu'on aurait pu automatiser mais qui, faute d'être allé jusqu'au bout, est resté semi-manuel. C'était faisable, je ne l'ai juste pas fait.

### À mi-parcours : des bugs, et ce qu'ils révèlent

Soyons honnêtes : à ce stade, les résultats sont mitigés. Des bugs structurels dans le code ont fait diverger certains runs — des runs longs, très longs, qu'il faut maintenant reprendre depuis le début pour évaluer correctement le modèle corrigé. Les données générées jusqu'ici sont probablement de faible qualité, et les modèles utilisés pour le codage ne sont pas les plus puissants du marché. C'est, disons, une approximation sérieuse plutôt qu'une expérience rigoureuse.

Mais c'est précisément là où le projet devient intéressant pour moi, indépendamment de ses résultats.

### Ce que j'apprends en le faisant

En creusant HAML, je suis tombé sur les travaux de Marino et al. (2024) sur les SA-nODE — *Self-Attractor Neural ODEs* — qui constituent le point de comparaison le plus proche dans la littérature. Les différences sont fondamentales : là où SA-nODE opère dans un espace d'état unique, HAML déploie une hiérarchie de sous-espaces de dimension décroissante ; là où SA-nODE fixe ses attracteurs avant l'entraînement, HAML les apprend librement par gradient. La ressemblance de fond reste réelle — utiliser des équations différentielles ordinaires comme substrat dynamique d'un modèle de classification — mais l'architecture diverge assez pour que la comparaison soit honnête.

Et c'est l'un des enseignements constants de ce genre de projets : les idées qu'on croit originales ont presque toujours déjà été explorées, souvent mieux formalisées, parfois réfutées. On a une petite idée, on se dit qu'on est un génie — et en fait non, d'autres l'ont imaginée avant, l'ont structurée mieux, et ont déjà prouvé que ça marchait ou pas. Ce n'est pas décourageant. C'est simplement la condition normale de qui réfléchit dans un domaine vivant peuplé de gens intelligents et bien financés.

### Pourquoi continuer, alors ?

Parce que ce projet ne vise pas la publication dans Nature. Il vise quelque chose de plus modeste et, à sa façon, de plus durable : comprendre en faisant, évaluer comment les IA raisonnent quand on leur soumet un problème ouvert, et transformer des abonnements sous-utilisés en apprentissages réels. Je me suis dit : j'ai deux abonnements, autant que ça serve à quelque chose — même si c'est un peu pour faire n'importe quoi.

Les IA consultées trouvent toutes le projet « extraordinaire » tout en relevant consciencieusement ses lacunes. C'est une réponse qui m'a appris autant sur le fonctionnement de ces systèmes que sur la validité de HAML lui-même.

Le projet continue. Les runs reprennent. Et si les résultats sont médiocres, au moins la question posée était belle.

---

*HAML — Hierarchical Attractor ML — est un système de classification supervisée dans lequel la prédiction émerge de la convergence d'un système dynamique hiérarchique vers un bassin d'attraction stable. La spécification théorique complète (v2.0) couvre les fondements mathématiques, les conditions de convergence et de stabilité, l'équivalence bayésienne, et le plan d'implémentation.*
