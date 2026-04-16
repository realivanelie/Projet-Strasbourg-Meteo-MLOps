# import streamlit as st
# import requests
# import pandas as pd
# import matplotlib.pyplot as plt
# import sqlite3

# # Configuration
# st.set_page_config(page_title="Monitoring Meteo Strasbourg", layout="wide")
# DB_PATH = "data/meteo_predictions.db"

# st.title("Systeme de Prevision Meteo - Dashboard MLOps")

# # --- SIDEBAR VERSIONING ---
# st.sidebar.header("Informations MLOps")
# try:
#     version_res = requests.get("http://127.0.0.1:8000/version").json()
#     st.sidebar.info(f"Logiciel : {version_res['software_version']}")
#     st.sidebar.info(f"Modele : {version_res['model_version']}")
# except:
#     st.sidebar.error("API FastAPI non detectee")

# # --- NAVIGATION ---
# tab1, tab2 = st.tabs(["🚀 Previsions (Futur)", "📊 Suivi des Erreurs (Passe)"])

# # --- TAB 1 : PREVISIONS (72H) ---
# with tab1:
#     st.header("Previsions pour les prochaines 72 heures")
#     try:
#         response = requests.get("http://127.0.0.1:8000/predictions")
#         if response.status_code == 200:
#             df_pred = pd.DataFrame(response.json())
#             df_pred['date_cible'] = pd.to_datetime(df_pred['date_cible'])
#             # On ne garde que la derniere prediction par date (nettoyage visuel)
#             df_pred = df_pred.sort_values('date_cible').drop_duplicates('date_cible', keep='last')
            
#             fig1, ax1 = plt.subplots(figsize=(10, 4))
#             ax1.plot(df_pred['date_cible'], df_pred['temperature_predite'], color='#3498db', marker='o', label='Prediction')
#             ax1.set_ylabel("Temp (°C)")
#             plt.xticks(rotation=45)
#             st.pyplot(fig1)
#             st.dataframe(df_pred[['date_cible', 'temperature_predite']], use_container_width=True)
#     except:
#         st.warning("En attente de donnees de l'API...")

# # --- TAB 2 : SUIVI DES ERREURS (REEL VS PREDIT) ---
# with tab2:
#     st.header("Comparaison : Reel vs Prediction")
#     st.write("Analyse de la performance sur les dernieres observations disponibles.")
    
#     try:
#         conn = sqlite3.connect(DB_PATH)
#         # On recupere les observations reelles et les predictions passees
#         obs = pd.read_sql("SELECT date, temperature as reel FROM observations", conn)
#         preds = pd.read_sql("SELECT date_cible as date, temperature_predite as predit FROM predictions_batch", conn)
#         conn.close()

#         # Nettoyage et Jointure
#         obs['date'] = pd.to_datetime(obs['date'])
#         preds['date'] = pd.to_datetime(preds['date'])
        
#         # On fusionne sur la date pour comparer
#         df_compare = pd.merge(obs, preds, on='date').dropna()
#         df_compare = df_compare.sort_values('date').drop_duplicates('date', keep='last')

#         if not df_compare.empty:
#             # Graphique de comparaison
#             fig2, ax2 = plt.subplots(figsize=(10, 4))
#             ax2.plot(df_compare['date'], df_compare['reel'], 'g-', label='Reel (Observations)', alpha=0.7)
#             ax2.plot(df_compare['date'], df_compare['predit'], 'r--', label='Predit (Modele)', alpha=0.7)
#             ax2.fill_between(df_compare['date'], df_compare['reel'], df_compare['predit'], color='gray', alpha=0.2, label='Erreur')
#             ax2.legend()
#             plt.xticks(rotation=45)
#             st.pyplot(fig2)

#             # Metriques d'erreur
#             df_compare['erreur_abs'] = (df_compare['reel'] - df_compare['predit']).abs()
#             mae = df_compare['erreur_abs'].mean()
            
#             c1, c2 = st.columns(2)
#             c1.metric("Erreur Moyenne (MAE) actuelle", f"{mae:.2f} °C")
#             c2.write("Le graphique montre le decalage entre la realite et les previsions passees.")
#         else:
#             st.info("Pas encore assez de donnees communes (Observations + Predictions) pour comparer.")
#     except Exception as e:
#         st.error(f"Erreur de lecture base de donnees : {e}")