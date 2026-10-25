# Comprendre les expériences ajoutées

## Qu'est-ce qui change ?
Le modèle reste une régression linéaire avec sept coefficients.
On compare différentes façons de trouver ces sept nombres.

PSO : 40 solutions se déplacent et partagent la meilleure information.
Gradient : une solution suit la direction donnée par les dérivées.
Recherche aléatoire : on tire des solutions au hasard, sans les améliorer.
OLS / moindres carrés : NumPy calcule une solution de référence.
Moyenne : on prédit toujours le même prix, calculé sur l'apprentissage.

## Pourquoi plusieurs graines ?
Une graine fixe les nombres aléatoires. Vingt graines donnent vingt départs
pour le PSO et la recherche aléatoire. Le test reste identique.
Cela vérifie la stabilité du calcul des coefficients, pas l'incertitude du prix.

## Pourquoi le même nombre de coûts ?
40 particules au départ puis 400 déplacements par particule donnent
40 + 40*400 = 16 040 coûts calculés. La recherche aléatoire essaie le même
nombre de solutions. Pour le gradient et OLS, une itération ne représente pas
la même quantité de travail : il faut éviter une comparaison trompeuse.

## Pourquoi les scores PSO, gradient et OLS sont presque identiques ?
Le modèle et l'objectif sont les mêmes. Quand les méthodes trouvent les bons
coefficients, les prédictions sont les mêmes. Le PSO n'améliore pas ici le modèle.

## Pourquoi la recherche aléatoire est mauvaise dans ce test ?
Le domaine contient beaucoup de mauvaises solutions et sept dimensions.
Tirer des candidats indépendants ne concentre pas progressivement la recherche
près des bonnes solutions. Cette observation dépend du budget et des bornes.

## Que montre la sensibilité ?
On diminue le budget à 1 600 évaluations pour observer la convergence avant sa fin.
Un grand essaim a plus de positions mais moins de déplacements successifs.
Les points montrent dix essais ; le trait noir est la médiane.
On observe le coût d'apprentissage seulement. On ne règle pas le PSO sur le test.

## Que montrent les autres partages ?
Les logements de test changent. La RMSE peut changer sensiblement alors que
PSO et OLS restent presque identiques. La difficulté vient donc aussi du test,
pas seulement de l'optimiseur. Ces partages se recouvrent : ce ne sont pas cinq
expériences statistiquement indépendantes.

## Comment lire les nouveaux graphiques ?
Distribution : combien de logements se trouvent dans chaque intervalle.
Corrélations : associations linéaires, sans preuve de causalité.
Géographie : position des points, couleur selon le prix ; pas une carte complète.
Convergence : baisse du coût ou de son écart avec OLS.
Prédictions : la diagonale correspond à une prédiction exacte.
Résidus : prix observé moins prédit. Positif = sous-estimation.
RMSE : plus petite = meilleure prédiction sur ce test.
Stabilité : variation entre graines ; attention aux échelles verticales.
Coefficients : poids standardisés des variables, pas des effets causaux.

## Ordre de lecture du code complémentaire
1. preparer : séparer les lignes et standardiser sans utiliser le test.
2. descente_gradient : calculer le gradient puis corriger les coefficients.
3. recherche_aleatoire : essayer des candidats et garder le meilleur.
4. main : lancer les répétitions et les expériences, sauvegarder puis dessiner.
5. tables_rapport.py : lire le JSON et créer les tables numériques du rapport.

Les références décrivent les méthodes et les outils. L'article de Yeh et Hsu
n'est pas reproduit : nous utilisons la base associée avec notre propre protocole.
