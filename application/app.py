"""Petite interface : streamlit run application/app.py"""
from pathlib import Path
import json
import streamlit as st

DOSSIER = Path(__file__).resolve().parents[1]
with open(DOSSIER / "results/modele.json") as fichier:
    modele = json.load(fichier)

st.title("SAM : prix du logement par unité de surface")
st.caption("Régression linéaire entraînée avec le PSO sur 414 observations UCI.")
st.info("Données historiques de New Taipei (Taiwan), 2012–2013. Ce modèle ne donne pas les prix actuels ni les prix au Maroc.")
st.write("La sortie est en 10 000 dollars de Taiwan par ping. Un ping vaut environ 3,3 m².")

mini = modele["minimums"]
maxi = modele["maximums"]
date = st.number_input("Date décimale (exemple : 2013.5 = juin 2013)",
                       min_value=mini[0], max_value=maxi[0], value=2013.5, format="%.4f")
age = st.number_input("Âge du logement (ans)", min_value=mini[1], max_value=maxi[1], value=20.0)
distance = st.number_input("Distance au métro (m)", min_value=mini[2], max_value=maxi[2], value=500.0)
commerces = st.number_input("Nombre de commerces proches", min_value=int(mini[3]), max_value=int(maxi[3]), value=5)
latitude = st.number_input("Latitude", min_value=mini[4], max_value=maxi[4], value=24.98, format="%.5f")
longitude = st.number_input("Longitude", min_value=mini[5], max_value=maxi[5], value=121.54, format="%.5f")

if st.button("Calculer le prix par unité de surface"):
    valeurs = [date, age, distance, commerces, latitude, longitude]
    prix = modele["constante"]
    for i in range(6):
        prix = prix + modele["coefficients"][i] * valeurs[i]
    if prix < 0:
        st.error("Prédiction négative : le modèle n'est pas adapté à cette combinaison.")
    else:
        st.success(f"Prix prédit : {prix:.2f} × 10 000 TWD par ping")
        st.write(f"Soit environ {prix * 10000 / 3.3:,.0f} TWD par m² (conversion indicative).")
        st.caption("Il s'agit d'un prix par surface, pas du prix total du logement. Les bornes individuelles des champs ne garantissent pas qu'une combinaison existe dans les données.")
