"""Projet simple : prix des maisons avec une régression linéaire et le PSO.
Lancer : python src/experience.py
Données réelles UCI : prix par unité de surface à New Taipei, Taiwan.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DOSSIER = Path(__file__).resolve().parents[1]


def calculer_erreur(X, y, coefficients):
    """MSE = moyenne des carrés des erreurs."""
    predictions = X @ coefficients
    erreurs = y - predictions
    return np.mean(erreurs ** 2)


def pso(X, y, graine=0, nombre_particules=40, nombre_iterations=400):
    """Chaque particule cherche 7 coefficients : b0 et les 6 poids des variables."""
    hasard = np.random.default_rng(graine)
    nombre_coefficients = X.shape[1]

    positions = hasard.uniform(-5, 5, (nombre_particules, nombre_coefficients))
    vitesses = np.zeros((nombre_particules, nombre_coefficients))
    meilleures_positions = positions.copy()
    meilleures_erreurs = np.zeros(nombre_particules)

    # On évalue les premières positions.
    for i in range(nombre_particules):
        meilleures_erreurs[i] = calculer_erreur(X, y, positions[i])

    indice = np.argmin(meilleures_erreurs)
    meilleure_position = meilleures_positions[indice].copy()
    meilleure_erreur = meilleures_erreurs[indice]
    historique = [float(meilleure_erreur)]

    for iteration in range(nombre_iterations):
        # L'inertie diminue de 0.9 au début à 0.4 à la fin.
        inertie = 0.9 - 0.5 * iteration / (nombre_iterations - 1)
        guide = meilleure_position.copy()

        for i in range(nombre_particules):
            r1 = hasard.random(nombre_coefficients)
            r2 = hasard.random(nombre_coefficients)

            # Trois parties : ancien mouvement, mémoire, meilleur de l'essaim.
            mouvement = inertie * vitesses[i]
            memoire = 1.5 * r1 * (meilleures_positions[i] - positions[i])
            groupe = 1.5 * r2 * (guide - positions[i])
            vitesses[i] = mouvement + memoire + groupe

            # Une particule ne doit pas se déplacer trop vite.
            vitesses[i] = np.clip(vitesses[i], -1, 1)
            positions[i] = positions[i] + vitesses[i]

            # Si une coordonnée dépasse une borne, on l'arrête à cette borne.
            for j in range(nombre_coefficients):
                if positions[i, j] < -5:
                    positions[i, j] = -5
                    vitesses[i, j] = 0
                if positions[i, j] > 5:
                    positions[i, j] = 5
                    vitesses[i, j] = 0

            erreur = calculer_erreur(X, y, positions[i])
            if erreur < meilleures_erreurs[i]:
                meilleures_erreurs[i] = erreur
                meilleures_positions[i] = positions[i].copy()

        indice = np.argmin(meilleures_erreurs)
        if meilleures_erreurs[indice] < meilleure_erreur:
            meilleure_erreur = meilleures_erreurs[indice]
            meilleure_position = meilleures_positions[indice].copy()
        historique.append(float(meilleure_erreur))

    return meilleure_position, historique


def mesurer_resultats(y, predictions):
    """Les trois mesures sont calculées dans l’unité originale : 10 000 TWD par ping."""
    erreurs = y - predictions
    rmse = np.sqrt(np.mean(erreurs ** 2))
    mae = np.mean(np.abs(erreurs))
    r2 = 1 - np.sum(erreurs ** 2) / np.sum((y - np.mean(y)) ** 2)
    return {"RMSE": float(rmse), "MAE": float(mae), "R2": float(r2)}


def main():
    # 1. Lire les données réelles, déjà fournies dans le dossier data.
    donnees = np.loadtxt(DOSSIER / "data/real_estate.csv", delimiter=",", skiprows=1)
    # Colonne 0 = identifiant : on ne l'utilise pas pour prédire.
    X = donnees[:, 1:7]
    prix = donnees[:, 7]
    nombre_maisons = len(prix)

    # 2. Mélanger les lignes, puis prendre 80 % pour apprendre.
    hasard_partage = np.random.default_rng(42)
    indices = hasard_partage.permutation(nombre_maisons)
    nombre_train = int(0.8 * nombre_maisons)
    nombre_test = nombre_maisons - nombre_train
    X_train = X[indices[:nombre_train]]
    X_test = X[indices[nombre_train:]]
    y_train = prix[indices[:nombre_train]]
    y_test = prix[indices[nombre_train:]]

    # 3. Calculer les moyennes et écarts-types sur l'apprentissage seulement.
    moyenne_X = X_train.mean(axis=0)
    ecart_X = X_train.std(axis=0)
    moyenne_y = y_train.mean()
    ecart_y = y_train.std()
    X_train = (X_train - moyenne_X) / ecart_X
    X_test = (X_test - moyenne_X) / ecart_X
    y_train_reduit = (y_train - moyenne_y) / ecart_y

    # La colonne de 1 permet d'ajouter le coefficient b0 au calcul.
    X_train = np.column_stack([np.ones(nombre_train), X_train])
    X_test = np.column_stack([np.ones(nombre_test), X_test])

    # 4. Chercher les coefficients avec le PSO.
    coefficients, historique = pso(X_train, y_train_reduit)
    predictions_pso = moyenne_y + ecart_y * (X_test @ coefficients)

    # 5. Comparer avec les moindres carrés : une seule ligne NumPy.
    coefficients_reference = np.linalg.lstsq(X_train, y_train_reduit, rcond=None)[0]
    predictions_reference = moyenne_y + ecart_y * (X_test @ coefficients_reference)
    predictions_moyenne = np.full(nombre_test, moyenne_y)
    resultats = {
        "PSO": mesurer_resultats(y_test, predictions_pso),
        "Moindres carres": mesurer_resultats(y_test, predictions_reference),
        "Moyenne": mesurer_resultats(y_test, predictions_moyenne)
    }
    print(json.dumps(resultats, indent=2))
    with open(DOSSIER / "results/metrics.json", "w") as fichier:
        json.dump(resultats, fichier, indent=2)
    np.savetxt(DOSSIER / "results/convergence.csv", historique,
               delimiter=",", header="meilleure_MSE", comments="")

    # Sauvegarder le partage et les prédictions pour vérifier les résultats.
    test = np.column_stack([donnees[indices[nombre_train:], 0], y_test,
                            predictions_pso, predictions_reference])
    np.savetxt(DOSSIER / "results/predictions_test.csv", test, delimiter=",",
               header="id,prix_observe,prediction_pso,prediction_moindres_carres", comments="")
    np.savetxt(DOSSIER / "results/ids_apprentissage.csv", donnees[indices[:nombre_train], 0],
               delimiter=",", header="id", comments="", fmt="%d")

    # 6. Revenir aux coefficients qui utilisent les unités de départ.
    coefficients_reels = ecart_y * coefficients[1:] / ecart_X
    constante = moyenne_y + ecart_y * coefficients[0]
    constante = constante - np.sum(coefficients_reels * moyenne_X)
    modele = {"constante": float(constante),
              "coefficients": coefficients_reels.tolist(),
              "variables": ["date_transaction", "age_ans", "distance_metro_m",
                            "commerces", "latitude", "longitude"],
              "minimums": X.min(axis=0).tolist(),
              "maximums": X.max(axis=0).tolist(),
              "unite": "10 000 TWD / ping"}
    with open(DOSSIER / "results/modele.json", "w") as fichier:
        json.dump(modele, fichier, indent=2)

    # 7. Dessiner deux graphiques pour le rapport.
    plt.figure(figsize=(7, 2.6))
    plt.plot(historique, color="#126078", label="PSO")
    erreur_reference = calculer_erreur(X_train, y_train_reduit, coefficients_reference)
    plt.axhline(erreur_reference, color="#c46b36", linestyle="--", label="Moindres carrés")
    plt.yscale("log")
    plt.xlabel("Itération")
    plt.ylabel("MSE sur apprentissage")
    plt.legend()
    plt.tight_layout()
    plt.savefig(DOSSIER / "rapport/convergence.pdf")
    plt.close()

    plt.figure(figsize=(7, 2.6))
    plt.scatter(y_test, predictions_pso,
                s=12, color="#126078", alpha=0.6)
    plt.plot([0, 120], [0, 120], "--", color="#c46b36")
    plt.xlabel("Prix observé (10 000 TWD / ping)")
    plt.ylabel("Prix prédit (10 000 TWD / ping)")
    plt.tight_layout()
    plt.savefig(DOSSIER / "rapport/predictions.pdf")
    plt.close()


if __name__ == "__main__":
    main()
