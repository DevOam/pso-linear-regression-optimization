# Comprendre le code simplement

## Les mots à connaître
- X : tableau avec les informations des maisons.
- y : tableau avec les prix.
- coefficients : les sept nombres cherchés par le PSO.
- train : les 331 observations utilisées pour apprendre.
- test : les 83 observations utilisées pour évaluer.
- MSE : moyenne des erreurs au carré ; le PSO veut la réduire.
- RMSE : racine carrée de la MSE, ici exprimée en 10 000 TWD par ping.
- MAE : moyenne des erreurs sans leur signe.
- pbest : meilleure position personnelle d'une particule.
- gbest : meilleure position de tout le groupe.

## Quelques écritures Python
- def : crée une fonction.
- return : renvoie son résultat.
- for : répète une action.
- if : exécute une action si une condition est vraie.
- X @ coefficients : calcule les prix prédits pour toutes les lignes.
- erreurs ** 2 : met chaque erreur au carré.
- np.mean(...) : calcule la moyenne.
- np.zeros(...) : crée un tableau de zéros.
- np.argmin(...) : donne l'indice de la plus petite valeur.
- .copy() : copie le tableau pour garder une ancienne solution.
- np.clip(v, -1, 1) : ramène les valeurs trop petites à -1 et trop grandes à 1.
- indices[:nombre_train] : les 331 premiers indices.
- indices[nombre_train:] : les 83 indices qui restent.
- axis=0 : effectue le calcul pour chaque colonne.
- np.column_stack(...) : place les colonnes côte à côte.
- np.linalg.lstsq(...) : calcule la solution de référence des moindres carrés.
- Path : permet de trouver les fichiers même si on lance le script d'ailleurs.
- json : format texte utilisé pour enregistrer les résultats et coefficients.

## Comment lire experience.py
1. calculer_erreur : prédit les valeurs et calcule la MSE.
2. pso : crée les particules et déplace chacune dans une boucle.
3. mesurer_resultats : calcule RMSE, MAE et R².
4. main : lit le CSV réel, sépare les données, les transforme, appelle le PSO,
   compare les méthodes, sauvegarde les résultats et dessine les graphiques.
5. if __name__ == "__main__" : lance main quand on exécute ce fichier.

## Pourquoi transformer les données ?
Une distance peut être de 3000 mètres alors que le nombre de commerces est 3.
On soustrait la moyenne et on divise par l'écart-type pour rapprocher
les échelles. On calcule ces valeurs sur l'apprentissage seulement.
On remet ensuite les prédictions dans l'unité originale, 10 000 TWD par ping.

## Pourquoi deux boucles dans le PSO ?
La première parcourt les 400 itérations.
La deuxième parcourt les 40 particules.
Une petite boucle sur les 7 coefficients vérifie les bornes.

## Pourquoi le même résultat à chaque lancement ?
Les graines aléatoires sont fixées : 42 pour le partage et 0 pour le PSO. Cela permet de reproduire l'expérience.
Changer la graine change les tirages ; il faut relancer le code pour voir
les nouveaux résultats. Une seule graine ne prouve pas que le PSO est stable.

## L'application
Elle lit modele.json. Pour un logement, elle additionne la constante et les six produits :
prix = constante + b1*date + b2*age + b3*distance_metro + b4*commerces
       + b5*latitude + b6*longitude.
La sortie est un prix par unité de surface, pas le prix total du logement.
Elle utilise les coefficients en unités de départ : elle n'a donc pas
besoin de standardiser à nouveau les champs saisis.

## Colonnes du CSV
0 = identifiant ; 1 à 6 = variables ; 7 = prix par unité de surface.
X = donnees[:, 1:7] sélectionne les colonnes 1 à 6. La borne 7 est exclue.
prix = donnees[:, 7] sélectionne la dernière colonne.
Il n’y a pas de génération de maisons fictives dans cette version.
