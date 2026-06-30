# Propositions d'amélioration du rapport VisionAI

*Analyse de la branche `ML_finition` — état du code vs état du rapport (`rapport_visionai.docx`)*
*Généré le 28 juin 2026*

---

## 1. Diagnostic principal

Le rapport actuel est **un livrable intermédiaire daté de la semaine du 6 avril 2026**. Il décrit un projet « en cours », avec beaucoup de travail annoncé comme « à faire » ou « prévu la semaine prochaine ».

Or, depuis cette date, le code a fortement progressé (voir l'historique git : intégration de RBF, SVM, optimiseurs, métriques, notebooks). **Le rapport ne reflète plus du tout l'état réel du projet et sous-évalue massivement le travail accompli.**

C'est le problème n°1 pour le rendu final : un correcteur qui lit ce rapport pensera que le projet s'arrête à « linéaire + MLP sur données synthétiques », alors que la réalité est bien plus complète.

### Tableau des écarts rapport / code

| Sujet | Ce que dit le rapport | Réalité du code (`ML_finition`) |
|---|---|---|
| Modèles | Linéaire + MLP | **4 modèles** : Linéaire, MLP, RBF, SVM |
| SVM | Absent | Implémenté : noyaux linéaire / RBF / polynomial, hinge loss (Pegasos), multi-classe (460 lignes) |
| RBF | Absent | Implémenté : centres + moindres carrés (Gauss-Jordan) + raffinement gradient (300 lignes) |
| Optimiseurs | SGD seul | descente de gradient, SGD momentum (+ Nesterov), Adam, trait `Optimizer` commun |
| Métriques | MSE seulement | accuracy, matrice de confusion, précision, rappel, F1 + F1 macro, MSE, MAE, R², k-fold |
| Tests | « 40 tests » | **69 tests d'intégration + ~54 tests unitaires (~123 au total)** |
| Dataset | « à faire, données à télécharger » | Dataset réel : 3 classes (aucun / humain / animal), images 64×64×3 = 12288 entrées |
| Notebooks | « à faire » | `analyse_dataset`, `analyse_hyperparametres`, `comparaison_modeles` écrits |
| Analyse phénomènes ML | « à faire (Ali) » | Notebook hyperparamètres (RBF/MLP/SVM) existant |
| Cadre temporel | « livrable intermédiaire » | Doit devenir le **rendu final** |

> Le `README.md` est nettement plus à jour et plus juste que le rapport : il liste les 4 modèles, les 3 optimiseurs et toutes les métriques. **Utiliser le README comme source de vérité** lors de la réécriture.

### Point de vigilance technique

Les notebooks `comparaison_modeles.ipynb` et `analyse_hyperparametres.ipynb` **n'ont aucune sortie sauvegardée** (cellules non exécutées, 0 graphe enregistré). Avant d'intégrer des résultats au rapport, il faut **les exécuter** et récupérer les graphes. Seul `analyse_dataset.ipynb` contient des sorties (3 images).

---

## 2. Propositions classées par priorité

### Priorité haute (impact fort sur la note)

**P1 — Reframe « rendu final » au lieu de « livrable intermédiaire ».**
Supprimer partout le ton « pas fini / prévu / à faire » :
- Page de garde : retirer « Livrable intermédiaire — semaine du 6 avril ».
- Section 1 (intro) : réécrire au passé accompli.
- Section 2.1, 15.2, 17 : supprimer ou transformer profondément.
Sinon le rapport donne l'impression d'un projet inachevé alors qu'il est largement abouti.

**P2 — Ajouter une section RBF et une section SVM.**
C'est le manque le plus grave : deux modèles entièrement codés et testés sont absents. Ce sont des livrables typiquement attendus du sujet (modèles non-linéaires au-delà du MLP).

**P3 — Remplacer la section « Résultats » (actuellement limitée au XOR) par de vrais résultats expérimentaux.**
Entraînement sur le dataset réel, comparaison des 4 modèles, étude d'hyperparamètres, matrices de confusion, discussion sur/sous-apprentissage. (Nécessite d'exécuter les notebooks — voir point de vigilance.)

**P4 — Corriger les chiffres faux.**
« 40 tests » → ~123 ; tableau d'avancement (section 2) à passer entièrement en « Fait ».

### Priorité moyenne

**P5 — Ajouter une section Optimiseurs** (trait commun, momentum/Nesterov, Adam ; intuition de pourquoi Adam converge mieux).

**P6 — Étoffer la section Évaluation/Métriques** (matrice de confusion, précision/rappel/F1, R², k-fold cross-validation).

**P7 — Corriger les annexes** : répartition du travail (A) et arborescence (B) doivent inclure RBF, SVM, optimiseurs, métriques et les nouveaux notebooks.

### Priorité basse (finition)

**P8 — Atteindre/dépasser les 20 pages** : avec P2/P3/P5/P6, le seuil de 20 pages exigé sera atteint naturellement.
**P9 — Ajouter des captures** : graphes des notebooks, matrices de confusion, decision boundaries, capture du client web.
**P10 — Relire l'orthographe et les accents** (le rapport actuel n'a pas d'accents — à uniformiser).

---

## 3. Plan section par section du rapport révisé

Structure proposée pour le rendu final (les sections **en gras** sont nouvelles ou fortement remaniées) :

1. Introduction *(réécrire : projet abouti, pas intermédiaire)*
2. Le projet et le cahier des charges *(mettre le tableau d'avancement entièrement en « Fait »)*
3. Pourquoi Rust *(conserver — bonne section)*
4. Architecture du projet *(ajouter metrics/ et optim/ dans la description des crates)*
5. Algèbre linéaire : Vector et Matrix *(conserver)*
6. Fonctions d'activation *(conserver)*
7. Le modèle linéaire *(conserver ; préciser régression + classification one-vs-rest)*
8. Le MLP *(conserver)*
9. La backpropagation *(conserver — point fort)*
10. **Le réseau RBF** *(NOUVEAU : principe du noyau gaussien, choix des centres, résolution par moindres carrés régularisés + raffinement gradient, ce qu'on a appris)*
11. **Le SVM** *(NOUVEAU : marge maximale, hinge loss, Pegasos/SGD, kernel trick — noyaux linéaire/RBF/polynomial, extension multi-classe)*
12. **Les optimiseurs** *(NOUVEAU : trait Optimizer, descente de gradient, momentum + Nesterov, Adam ; comparaison de convergence)*
13. Sérialisation *(conserver)*
14. **Évaluation et métriques** *(NOUVEAU/étoffé : accuracy, matrice de confusion, précision/rappel/F1, MSE/MAE/R², k-fold cross-validation)*
15. API REST *(conserver ; vérifier que les endpoints couvrent les 4 modèles)*
16. Bindings Python *(conserver ; mentionner PyRBF, PySVM, LinearRegression exposés)*
17. Application cliente web *(conserver)*
18. Tests *(corriger : ~123 tests)*
19. **Résultats expérimentaux** *(REMANIÉ : dataset réel 3 classes, comparaison des 4 modèles, courbes loss/accuracy, matrices de confusion, étude hyperparamètres, sur/sous-apprentissage)*
20. Difficultés rencontrées *(conserver — point fort)*
21. Conclusion *(réécrire : bilan d'un projet abouti + ouverture)*

Annexes : A. Répartition du travail *(à corriger)* · B. Arborescence *(à corriger)* · C. Glossaire *(ajouter RBF, SVM, kernel trick, Adam, momentum, cross-validation)* · D. Références.

---

## 4. Prochaines étapes suggérées

1. **Exécuter** `comparaison_modeles.ipynb` et `analyse_hyperparametres.ipynb` pour générer les graphes et résultats chiffrés.
2. **Réécrire** le rapport selon le plan ci-dessus (peut être fait section par section).
3. **Intégrer** les visuels (graphes, matrices de confusion, capture client).
4. **Relire** : chiffres, accents, cohérence avec le README.
