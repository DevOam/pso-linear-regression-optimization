# Questions simples pour la présentation

Quel est le but ?
Prédire le prix d'un logement par unité de surface avec six informations.

Quel modèle utilise-t-on ?
Une régression linéaire : une constante et six coefficients, soit sept paramètres.

Que fait le PSO ?
Il essaie différentes valeurs de ces sept coefficients pour réduire la MSE.

Que représente une particule ?
Une solution possible avec sept nombres. Elle ne représente pas un logement.

Quels sont les réglages ?
40 particules et 400 itérations. La version courte utilise une graine ;
l’article compare 20 graines pour le PSO et la recherche aléatoire. Les poids restent entre
-5 et 5 après standardisation. Ces réglages n'ont pas été choisis sur le test.

Que sont pbest et gbest ?
La meilleure solution personnelle et la meilleure solution de tout le groupe.

D'où viennent les données ?
Du fichier Excel UCI Real Estate Valuation de I-Cheng Yeh (2018).
414 observations de New Taipei à Taiwan ; transactions de 2012 à 2013.

Quelles variables utilise-t-on ?
Date de transaction, âge, distance au métro, commerces, latitude et longitude.
Le numéro No est exclu : c'est seulement un identifiant.

Quelle unité pour la sortie ?
10 000 TWD par ping. Par exemple 40 veut dire 400 000 TWD par ping.
Un ping vaut environ 3,3 m². Ce n'est pas le prix total d'un logement.

Pourquoi 331 et 83 ?
Environ 80 % pour apprendre et 20 % pour évaluer sur des lignes non vues.
Le partage est aléatoire avec graine 42, et identique pour toutes les méthodes.

Pourquoi comparer avec les moindres carrés ?
Cette méthode donne la solution de référence de la régression linéaire.

Quel résultat ?
PSO : RMSE environ 8,21 ; MAE environ 6,16 ; R² environ 0,5632 sur 83 lignes.
Les deux erreurs sont en 10 000 TWD par ping. Les moindres carrés donnent
presque les mêmes résultats. Le PSO n'est pas meilleur dans cette expérience.

Peut-on l'utiliser pour estimer une maison au Maroc aujourd'hui ?
Non. Ce modèle décrit une petite base historique d'une zone de Taiwan.

Que peut-on améliorer ?
Tester plusieurs graines et des séparations par date ou zone, ajouter une
validation pour les réglages, utiliser des données plus récentes.

Pour la démonstration, lancer l’interface avec la commande du README.

Pour les méthodes supplémentaires, les budgets et les nouveaux graphiques,
lire ETUDE_EXPLIQUEE.md. L’article compare le PSO, le gradient, la recherche
aléatoire et les moindres carrés, avec une prédiction constante comme référence.
