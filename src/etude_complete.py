"""Expériences supplémentaires. Lancer : python src/etude_complete.py.
Le fichier experience.py reste la version de départ, plus courte.
"""
from pathlib import Path
import json
import time
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experience import pso, calculer_erreur, mesurer_resultats

DOSSIER = Path(__file__).resolve().parents[1]
FIGURES = DOSSIER / 'rapport/figures'
FIGURES.mkdir(exist_ok=True)
COULEURS = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#777777']
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.grid': True,
                     'grid.alpha': .18, 'savefig.facecolor': 'white'})


def preparer(donnees, graine):
    """Le partage et la standardisation sont refaits pour chaque expérience."""
    indices = np.random.default_rng(graine).permutation(len(donnees))
    apprentissage = indices[:331]
    test = indices[331:]
    X = donnees[:, 1:7]
    y = donnees[:, 7]
    moyenne_X = X[apprentissage].mean(axis=0)
    ecart_X = X[apprentissage].std(axis=0)
    moyenne_y = y[apprentissage].mean()
    ecart_y = y[apprentissage].std()
    A = np.column_stack([np.ones(331), (X[apprentissage] - moyenne_X) / ecart_X])
    B = np.column_stack([np.ones(83), (X[test] - moyenne_X) / ecart_X])
    cible = (y[apprentissage] - moyenne_y) / ecart_y
    return A, B, cible, y[test], moyenne_y, ecart_y, test


def descente_gradient(X, y):
    """400 pas ; le pas est fixé avec la courbure de l'objectif, sans test."""
    coefficients = np.zeros(X.shape[1])
    hessienne = 2 * X.T @ X / len(y)
    pas = 1 / np.linalg.eigvalsh(hessienne).max()
    historique = [float(calculer_erreur(X, y, coefficients))]
    for iteration in range(400):
        gradient = 2 * X.T @ (X @ coefficients - y) / len(y)
        coefficients = coefficients - pas * gradient
        coefficients = np.clip(coefficients, -5, 5)
        historique.append(float(calculer_erreur(X, y, coefficients)))
    return coefficients, historique, float(pas)


def recherche_aleatoire(X, y, graine):
    """16 040 candidats uniformes : même nombre de coûts que le PSO standard."""
    hasard = np.random.default_rng(graine)
    meilleur = np.zeros(X.shape[1])
    meilleure_erreur = float('inf')
    historique = []
    # 401 groupes de 40 candidats, y compris le groupe initial.
    for groupe in range(401):
        for candidat in range(40):
            coefficients = hasard.uniform(-5, 5, X.shape[1])
            erreur = calculer_erreur(X, y, coefficients)
            if erreur < meilleure_erreur:
                meilleure_erreur = erreur
                meilleur = coefficients.copy()
        historique.append(float(meilleure_erreur))
    return meilleur, historique


def sauver_figure(nom):
    plt.tight_layout()
    plt.savefig(FIGURES / (nom + '.pdf'))
    plt.close()


def sauver_lignes(nom, lignes):
    with open(DOSSIER / 'results' / nom, 'w', newline='') as fichier:
        writer = csv.DictWriter(fichier, fieldnames=list(lignes[0]))
        writer.writeheader()
        writer.writerows(lignes)


def main():
    donnees = np.loadtxt(DOSSIER / 'data/real_estate.csv', delimiter=',', skiprows=1)
    A, B, u, y_test, moyenne, ecart, test = preparer(donnees, 42)
    ols = np.linalg.lstsq(A, u, rcond=None)[0]
    cout_ols = float(calculer_erreur(A, u, ols))
    assert np.max(np.abs(ols)) < 5
    lignes = []
    historiques_pso = []
    historiques_hasard = []
    predictions = {}
    coefficients_zero = {}

    # 1. Répéter PSO et recherche aléatoire avec 20 graines, même partage.
    for nom in ['PSO', 'Aleatoire']:
        for graine in range(20):
            debut = time.perf_counter()
            if nom == 'PSO':
                coefficients, historique = pso(A, u, graine)
                historiques_pso.append(historique)
            else:
                coefficients, historique = recherche_aleatoire(A, u, graine)
                historiques_hasard.append(historique)
            secondes = time.perf_counter() - debut
            prediction = moyenne + ecart * (B @ coefficients)
            mesures = mesurer_resultats(y_test, prediction)
            lignes.append(dict(methode=nom, graine=graine, **mesures,
                               MSE_train=historique[-1], secondes=secondes))
            assert np.all(np.diff(historique) <= 1e-10)
            if graine == 0:
                predictions[nom] = prediction
                coefficients_zero[nom] = coefficients
        print(nom, ': 20 exécutions terminées', flush=True)

    # 2. Références déterministes, chronométrées 20 fois sans refaire le partage.
    for nom in ['OLS', 'Gradient', 'Moyenne']:
        for repetition in range(20):
            debut = time.perf_counter()
            if nom == 'OLS':
                coefficients = np.linalg.lstsq(A, u, rcond=None)[0]
            elif nom == 'Gradient':
                coefficients, historique_gradient, pas = descente_gradient(A, u)
            else:
                coefficients = np.zeros(7)
            secondes = time.perf_counter() - debut
            prediction = moyenne + ecart * (B @ coefficients)
            lignes.append(dict(methode=nom, graine=repetition,
                               **mesurer_resultats(y_test, prediction),
                               MSE_train=float(calculer_erreur(A, u, coefficients)), secondes=secondes))
        predictions[nom] = prediction
        coefficients_zero[nom] = coefficients
    sauver_lignes('comparaison_20.csv', lignes)
    np.savez(DOSSIER / 'results/historiques.npz',
             pso=np.array(historiques_pso), aleatoire=np.array(historiques_hasard),
             gradient=np.array(historique_gradient))

    # 3. Taille de l'essaim, budget court identique de 1 600 évaluations.
    sensibilite = []
    for nombre in [10, 20, 40, 80]:
        iterations = 1600 // nombre - 1
        for graine in range(10):
            coefficients, historique = pso(A, u, graine, nombre, iterations)
            sensibilite.append(dict(particules=nombre, iterations=iterations,
                                    graine=graine, evaluations=nombre*(iterations+1),
                                    ecart_OLS=float(max(0, historique[-1] - cout_ols))))
        print('Sensibilité', nombre, 'particules terminée', flush=True)
    sauver_lignes('sensibilite.csv', sensibilite)

    # 4. Autres partages prédéfinis : aucun choix du meilleur partage.
    partages = []
    for graine in [10, 20, 30, 40, 50]:
        AA, BB, uu, yy, mm, ss, tt = preparer(donnees, graine)
        cp, hh = pso(AA, uu, 0)
        co = np.linalg.lstsq(AA, uu, rcond=None)[0]
        for nom, coefficients in [('PSO', cp), ('OLS', co)]:
            partages.append(dict(partage=graine, methode=nom,
                                **mesurer_resultats(yy, mm + ss * (BB @ coefficients))))
    sauver_lignes('autres_partages.csv', partages)

    # Résumé JSON : tous les nombres du rapport viennent de ces calculs.
    resume = {'cout_OLS': cout_ols, 'pas_gradient': pas, 'methodes': {},
              'sensibilite': {}, 'partages': partages}
    for nom in ['Moyenne', 'Aleatoire', 'Gradient', 'PSO', 'OLS']:
        selection = [r for r in lignes if r['methode'] == nom]
        resume['methodes'][nom] = {}
        for mesure in ['RMSE', 'MAE', 'R2', 'MSE_train', 'secondes']:
            valeurs = [r[mesure] for r in selection]
            resume['methodes'][nom][mesure] = {'moyenne': float(np.mean(valeurs)),
                'ecart_type': float(np.std(valeurs, ddof=1)),
                'minimum': float(np.min(valeurs)), 'maximum': float(np.max(valeurs))}
    for nombre in [10, 20, 40, 80]:
        valeurs = [r['ecart_OLS'] for r in sensibilite if r['particules'] == nombre]
        resume['sensibilite'][str(nombre)] = {'mediane': float(np.median(valeurs)),
                                             'maximum': float(np.max(valeurs))}
    with open(DOSSIER / 'results/etude_complete.json', 'w') as fichier:
        json.dump(resume, fichier, indent=2)

    # 5. Figures en couleur, export vectoriel PDF.
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3))
    axes[0].hist(donnees[:, 7], bins=20, color=COULEURS[0], edgecolor='white')
    axes[0].set(xlabel='Prix (10 000 TWD / ping)', ylabel='Effectif')
    axes[1].hist(donnees[:, 3], bins=20, color=COULEURS[1], edgecolor='white')
    axes[1].set(xlabel='Distance au métro (m)', ylabel='Effectif')
    sauver_figure('distribution')

    noms = ['Date', 'Âge', 'Métro', 'Commerces', 'Latitude', 'Longitude', 'Prix']
    correlation = np.corrcoef(donnees[:, 1:], rowvar=False)
    fig, ax = plt.subplots(figsize=(5.2, 4.5))
    im = ax.imshow(correlation, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.grid(False)
    ax.set_xticks(range(7), noms, rotation=45, ha='right')
    ax.set_yticks(range(7), noms)
    for i in range(7):
        for j in range(7):
            ax.text(j, i, f'{correlation[i,j]:.2f}', ha='center', va='center',
                    color='white' if abs(correlation[i,j]) > .6 else 'black', fontsize=8)
    fig.colorbar(im, ax=ax, label='Corrélation de Pearson')
    sauver_figure('correlations')

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    im = ax.scatter(donnees[:, 6], donnees[:, 5], c=donnees[:, 7],
                    cmap='viridis', s=22, alpha=.8)
    ax.set(xlabel='Longitude (degrés)', ylabel='Latitude (degrés)')
    ax.ticklabel_format(useOffset=False)
    fig.colorbar(im, ax=ax, label='Prix (10 000 TWD / ping)')
    sauver_figure('geographie')

    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    evaluations = 40 * np.arange(1, 402)
    for H, nom, couleur in [(historiques_pso, 'PSO', COULEURS[0]),
                             (historiques_hasard, 'Recherche aléatoire', COULEURS[3])]:
        H = np.array(H)
        ecarts = np.maximum(H - cout_ols, 1e-14)
        ax.semilogy(evaluations, np.median(ecarts, axis=0), label=nom, color=couleur)
        ax.fill_between(evaluations, np.quantile(ecarts,.25,axis=0),
                        np.quantile(ecarts,.75,axis=0), color=couleur, alpha=.18)
    ax.set(xlabel="Nombre d'évaluations de l'objectif", ylabel='Meilleur coût − coût OLS')
    ax.legend()
    sauver_figure('convergence_budget')

    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ax.semilogy(np.maximum(np.array(historiques_pso[0])-cout_ols,1e-14),label='PSO : graine 0',color=COULEURS[0])
    ax.semilogy(np.maximum(np.array(historique_gradient)-cout_ols,1e-14),label='Gradient',color=COULEURS[2])
    ax.set(xlabel='Itération (coûts différents selon la méthode)', ylabel='Coût − coût OLS')
    ax.legend()
    sauver_figure('convergence_gradient')

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.1), sharex=True, sharey=True)
    for ax, nom, couleur in zip(axes, ['Aleatoire', 'Gradient', 'PSO'], [COULEURS[3],COULEURS[2],COULEURS[0]]):
        ax.scatter(y_test, predictions[nom], s=18, alpha=.7, color=couleur)
        ax.plot([0,120], [0,120], '--', color='#555555')
        ax.set(title=nom, xlabel='Prix observé')
    axes[0].set_ylabel('Prix prédit')
    sauver_figure('predictions_comparees')

    residus = y_test - predictions['PSO']
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
    axes[0].scatter(predictions['PSO'], residus, color=COULEURS[0], s=18)
    axes[0].axhline(0, color=COULEURS[1], linestyle='--')
    axes[0].set(xlabel='Prix prédit', ylabel='Résidu : observé − prédit')
    axes[1].hist(residus, bins=15, color=COULEURS[2], edgecolor='white')
    axes[1].set(xlabel='Résidu', ylabel='Effectif')
    sauver_figure('residus')

    noms_methodes = ['Moyenne','Aleatoire','Gradient','PSO','OLS']
    valeurs = [resume['methodes'][n]['RMSE']['moyenne'] for n in noms_methodes]
    barres = [resume['methodes'][n]['RMSE']['ecart_type'] for n in noms_methodes]
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ax.bar(noms_methodes, valeurs, yerr=barres, color=[COULEURS[4],COULEURS[3],COULEURS[2],COULEURS[0],COULEURS[1]], capsize=4)
    ax.set_ylabel('RMSE test (10 000 TWD / ping)')
    sauver_figure('comparaison_rmse')

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3))
    for ax, nom, couleur in zip(axes,['PSO','Aleatoire'],[COULEURS[0],COULEURS[3]]):
        valeurs = [r['RMSE'] for r in lignes if r['methode']==nom]
        ax.plot(range(20), valeurs, 'o-', color=couleur)
        ax.axhline(resume['methodes']['OLS']['RMSE']['moyenne'],color=COULEURS[1],ls='--')
        ax.set(xlabel="Graine d'initialisation", ylabel='RMSE test', title=nom)
        ax.ticklabel_format(useOffset=False,axis='y')
    sauver_figure('stabilite')

    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    for i,nombre in enumerate([10,20,40,80]):
        v=[max(r['ecart_OLS'],1e-14) for r in sensibilite if r['particules']==nombre]
        ax.scatter(np.full(10,i),v,color=COULEURS[i],s=22)
        ax.plot([i-.2,i+.2],[np.median(v)]*2,color='black')
    ax.set_xticks(range(4),['10','20','40','80'])
    ax.set(xlabel="Nombre de particules (1 600 évaluations)",ylabel='Coût final − coût OLS')
    ax.set_yscale('log')
    sauver_figure('sensibilite')

    fig, ax = plt.subplots(figsize=(5.5,3.2))
    for nom,couleur in [('PSO',COULEURS[0]),('OLS',COULEURS[1])]:
        v=[r['RMSE'] for r in partages if r['methode']==nom]
        ax.plot([10,20,30,40,50],v,'o-' if nom=='PSO' else 'x--',label=nom,color=couleur)
    ax.set(xlabel='Graine du partage 80/20',ylabel='RMSE test')
    ax.legend()
    sauver_figure('partages')

    # Les coefficients standardisés permettent de comparer leurs échelles.
    fig, ax = plt.subplots(figsize=(5.5,3.2))
    x=np.arange(6)
    ax.bar(x-.18,coefficients_zero['PSO'][1:],width=.36,label='PSO',color=COULEURS[0])
    ax.bar(x+.18,ols[1:],width=.36,label='OLS',color=COULEURS[1])
    ax.set_xticks(x,noms[:6],rotation=25)
    ax.axhline(0,color='#555555',lw=.8)
    ax.set_ylabel('Coefficient standardisé (sans unité)')
    ax.legend()
    sauver_figure('coefficients')
    print(json.dumps(resume['methodes'],indent=2),flush=True)


if __name__ == '__main__':
    main()
