"""Recréer les deux tables LaTeX à partir des vrais résultats de l'étude."""
from pathlib import Path
import json

DOSSIER = Path(__file__).resolve().parents[1]
r = json.loads((DOSSIER / 'results/etude_complete.json').read_text())

def francais(valeur, chiffres=4):
    return f'{valeur:.{chiffres}f}'.replace('.', ',')

s = r'''\begin{table*}[t]\centering\small
\caption{Comparaison sur le partage principal : moyennes et écarts-types de 20 exécutions. Erreurs en $10^4$ TWD par ping. OLS, gradient et moyenne sont déterministes ; leurs répétitions servent au chronométrage.}
\label{tab:resultats}
\begin{tabular}{lrrrrr}\toprule
Méthode & RMSE & Écart-type RMSE & MAE & $R^2$ & Temps moyen (ms)\\\midrule
'''
for nom, libelle in [('Moyenne','Moyenne'),('Aleatoire','Recherche aléatoire'),('Gradient','Gradient'),('PSO','PSO'),('OLS','Moindres carrés')]:
    m = r['methodes'][nom]
    ecart = m['RMSE']['ecart_type']
    texte_ecart = francais(ecart) if ecart > .00001 else ('$<10^{-6}$' if nom == 'PSO' else '0')
    s += libelle + ' & ' + francais(m['RMSE']['moyenne']) + ' & ' + texte_ecart
    s += ' & ' + francais(m['MAE']['moyenne']) + ' & ' + francais(m['R2']['moyenne'])
    s += ' & ' + francais(1000*m['secondes']['moyenne'],3) + r'\\' + '\n'
s += '\\bottomrule\n\\end{tabular}\n\\end{table*}\n'
(DOSSIER / 'rapport/table_resultats.tex').write_text(s)

s = r'''\begin{table}[t]\centering\small
\caption{Autres partages : RMSE du PSO de graine 0 et d'OLS, sans sélection du meilleur partage.}
\label{tab:partages}
\begin{tabular}{lrr}\toprule
Graine du partage & PSO & OLS\\\midrule
'''
for seed in [10,20,30,40,50]:
    valeurs = [x['RMSE'] for x in r['partages'] if x['partage'] == seed]
    s += str(seed) + ' & ' + francais(valeurs[0]) + ' & ' + francais(valeurs[1]) + r'\\' + '\n'
s += '\\bottomrule\n\\end{tabular}\n\\end{table}\n'
(DOSSIER / 'rapport/table_partages.tex').write_text(s)
