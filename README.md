# SAM : régression linéaire et PSO sur données réelles

Version simple pour Master 1. Le PSO cherche 7 coefficients : une constante
et les poids des 6 variables. La cible est le prix par unité de surface.

## 1. Installer les bibliothèques
Ouvrir un terminal dans le dossier pso-linear-regression-optimization :

    pip install -r requirements.txt

## 2. Exécuter l'expérience

    python src/experience.py

Le programme lit data/real_estate.csv, utilise 331 observations pour apprendre
et 83 pour le test, puis exécute le PSO (40 particules, 400 itérations).
Il compare avec les moindres carrés et la moyenne d'apprentissage.
Les graines sont fixées : 42 pour le partage et 0 pour le PSO.
Ce partage 80/20 est celui de ce projet, pas le partage 2/3–1/3 mentionné par UCI.
Le test n'intervient pas dans l'entraînement. Aucun réglage sur le test.

## 3. Ouvrir l'application

    streamlit run application/app.py

Saisir : date décimale, âge, distance au métro en mètres, nombre de commerces,
latitude et longitude. Exemple : 2013.5 ; 20 ; 500 ; 5 ; 24.98 ; 121.54.
L'application lit les coefficients du fichier results/modele.json fourni.
Le résultat est en 10 000 TWD par ping, pas un prix total ni un montant en MAD.
Un ping vaut environ 3,3 m² selon UCI. L'interface donne aussi une conversion
indicative en TWD/m². La date décimale suit la convention de la source :
2013.5 correspond à juin 2013. Cette application utilise des données anciennes,
pas des prix actuels. Pas d'import de nouveaux CSV dans cette version simple.

## 4. Rapport LaTeX
Le PDF : rapport/rapport.pdf.
Le texte : rapport/rapport.tex.
Les noms des auteurs et de l’enseignant sont définis dans rapport/identite.tex.

    cd rapport
    pdflatex -interaction=nonstopmode -halt-on-error rapport.tex
    pdflatex -interaction=nonstopmode -halt-on-error rapport.tex

Sur Overleaf, importer le dossier rapport et choisir rapport.tex comme document principal.

## Source et préparation des données
I-Cheng Yeh (2018). Real Estate Valuation. UCI Machine Learning Repository.
DOI : https://doi.org/10.24432/C5J30W
Page : https://archive.ics.uci.edu/dataset/477/real+estate+valuation+data+set
Licence : CC BY 4.0, https://creativecommons.org/licenses/by/4.0/

Le fichier Excel original fourni par l'utilisateur est inclus sans modification
sous data/real_estate_original.xlsx. Le CSV contient exactement les mêmes valeurs,
avec des noms de colonnes plus courts. 414 lignes, 6 variables, un identifiant
et une cible. Aucune ligne supprimée, aucune donnée inventée ou complétée.
L'identifiant est exclu de l'apprentissage. La standardisation est calculée sur
l'apprentissage seulement. Les dates vont de 2012.6666667 à 2013.5833333.
La base ne contient ni surface totale ni nombre de chambres.
Voir data/provenance.json pour les détails et l'empreinte du fichier original.

## Fichiers de résultats
metrics.json : les trois mesures sur le test, en unité originale.
modele.json : constante, six poids, ordre des variables et bornes observées.
convergence.csv : 401 meilleurs coûts (départ et 400 itérations).
predictions_test.csv : identifiants, prix observés et prédictions des 83 lignes.
ids_apprentissage.csv : identifiants des 331 lignes d'apprentissage.

## Comprendre et présenter
Lire LIRE_LE_CODE.md puis SOUTENANCE.md. Respecter les consignes de votre
établissement sur les aides utilisées. Les résultats sont recalculés avec le
code fourni. La version courte utilise un seul partage ; l’étude complète
ajoute les répétitions et cinq autres partages décrits plus bas.
Les données concernent une zone de Taiwan en 2012–2013 : ne pas présenter le
modèle comme un outil de prix actuels ou de prix au Maroc.

## Format scientifique IEEE
Le rapport utilise la classe officielle IEEEtran en mode conference A4,
avec deux colonnes, résumé, mots-clés et références numérotées.
Le fichier IEEEtran.cls est fourni dans rapport/ pour compiler sans installation
supplémentaire de cette classe. Garder ce fichier à côté de rapport.tex.
Source de la classe : https://ctan.org/pkg/ieeetran (licence LPPL, voir le fichier).
Structure suivie : https://conferences.ieeeauthorcenter.ieee.org/write-your-paper/structure-your-paper/
Le rapport reste en français et constitue un travail pédagogique au format IEEE.

## Étude complète et article enrichi
Le fichier experience.py garde une version courte pour comprendre le PSO.
L'article enrichi utilise en plus l'étude comparative suivante :

    python src/etude_complete.py
    python src/tables_rapport.py

Puis recompiler rapport/rapport.tex deux fois avec pdflatex.
Les calculs peuvent prendre plusieurs dizaines de secondes selon le PC.
Pour reproduire les durées dans les mêmes conditions, définir éventuellement
OPENBLAS_NUM_THREADS=1 avant de lancer Python ; les temps restent indicatifs.

L'étude complète ajoute :
- Descente de gradient projetée : 400 pas, pas calculé sur l'apprentissage.
- Recherche aléatoire : 16 040 candidats, comme les 16 040 coûts du PSO principal.
- 20 initialisations pour PSO et recherche aléatoire (graines 0 à 19).
- 20 chronométrages pour les méthodes déterministes, sans nouvelles données.
- Sensibilité : 10/20/40/80 particules, budget de 1 600 coûts, dix graines.
  Les nombres d'itérations sont 159/79/39/19. Seul le coût d'apprentissage sert
  à cette comparaison ; elle ne choisit pas un modèle final selon le test.
- Cinq autres partages (graines 10/20/30/40/50) : PSO graine 0 et OLS.

Les nouveaux CSV sont dans results/ : comparaison_20.csv, sensibilite.csv,
autres_partages.csv. Le résumé est etude_complete.json ; les courbes brutes
sont dans historiques.npz. Les figures vectorielles en couleur sont dans
rapport/figures/. Les deux tables LaTeX générées restent dans rapport/.
L'application continue d'utiliser le modèle PSO du partage 42 et de graine 0.
L'étude étendue ne sélectionne pas un meilleur modèle pour l'application.

Le PSO, le gradient, la recherche aléatoire et OLS cherchent les coefficients
du même modèle. Il ne s'agit pas de quatre architectures de machine learning.
Lire ETUDE_EXPLIQUEE.md pour préparer les nouvelles parties de la soutenance.
