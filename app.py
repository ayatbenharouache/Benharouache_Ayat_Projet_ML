import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.base import BaseEstimator, TransformerMixin

# ==============================================================================
# STRUCTURES REQUISES POUR LE FICHIER .PKL
# ==============================================================================
class CustomDataImputer(BaseEstimator, TransformerMixin): 
    def __init__(self):
        self.mode_global_purpose = None
        self.valeur_secours_approv_in_adv = "nopre"
        self.rate_median_global = None
        self.spread_median_global = None
        self.property_median_global = None
        self.upfront_median_global = None
        self.income_median_global = None
        self.dtir1_median_global = None
        self.mode_age_train = None
        self.mode_global_loan_limit = None
        self.mode_global_term = None
        self.mode_global_neg_amm = None
        self.mode_global_submission = None
        self.mode_par_nature = None
        self.map_medians_rate = None
        self.map_medians_spread = None
        self.map_medians_property = None
        self.map_medians_upfront = None
        self.map_medians_income = None
        self.map_medianes_dtir1 = None
        self.map_modes_loan_limit = None
        self.map_modes_term = None
        self.map_modes_neg_amm = None
        self.map_modes_submission = None
        self.bins_montant = None
        self.bins_income = None

    def fit(self, X, y=None): return self
    def transform(self, X):
        X_copy = X.copy()
        valeurs_conditionnelles = X_copy['business_or_commercial'].map(self.mode_par_nature).fillna(self.mode_global_purpose)
        valeurs_rate = X_copy['loan_type'].map(self.map_medians_rate).fillna(self.rate_median_global)
        valeurs_spread = X_copy['loan_type'].map(self.map_medians_spread).fillna(self.spread_median_global)
        tranche = pd.cut(X_copy['loan_amount'], bins=self.bins_montant, labels=False, include_lowest=True)
        cles_upfront = list(zip(tranche, X_copy['business_or_commercial']))
        valeurs_upfront = pd.Series([self.map_medians_upfront.get(c, self.upfront_median_global) for c in cles_upfront], index=X_copy.index)
        X_copy['Upfront_charges'] = X_copy['Upfront_charges'].fillna(valeurs_upfront)
        cles_prop = list(zip(tranche, X_copy['loan_type']))
        valeurs_prop = pd.Series([self.map_medians_property.get(c, self.property_median_global) for c in cles_prop], index=X_copy.index)
        X_copy['property_value'] = X_copy['property_value'].fillna(valeurs_prop)
        X_copy['age'] = X_copy['age'].fillna(self.mode_age_train)
        cles_inc = list(zip(X_copy['age'], tranche, X_copy['business_or_commercial']))
        valeurs_income = pd.Series([self.map_medians_income.get(c, self.income_median_global) for c in cles_inc], index=X_copy.index)
        X_copy['income'] = X_copy['income'].fillna(valeurs_income)
        tranche_inc = pd.cut(X_copy['income'], bins=self.bins_income, labels=False, include_lowest=True)
        cles_dtir1 = list(zip(tranche_inc, X_copy['business_or_commercial'], X_copy['Credit_Worthiness']))
        valeurs_dtir1 = pd.Series([self.map_medianes_dtir1.get(c, self.dtir1_median_global) for c in cles_dtir1], index=X_copy.index)
        X_copy['dtir1'] = X_copy['dtir1'].fillna(valeurs_dtir1)
        X_copy['Interest_rate_spread'] = X_copy['Interest_rate_spread'].fillna(valeurs_spread)
        X_copy['loan_purpose'] = X_copy['loan_purpose'].fillna(valeurs_conditionnelles)
        X_copy['approv_in_adv'] = X_copy['approv_in_adv'].fillna(self.valeur_secours_approv_in_adv)
        X_copy['rate_of_interest'] = X_copy['rate_of_interest'].fillna(valeurs_rate)
        X_copy['loan_limit'] = X_copy['loan_limit'].fillna(X_copy['loan_type'].map(self.map_modes_loan_limit)).fillna(self.mode_global_loan_limit)
        X_copy['term'] = X_copy['term'].fillna(X_copy['loan_type'].map(self.map_modes_term)).fillna(self.mode_global_term)
        X_copy['Neg_ammortization'] = X_copy['Neg_ammortization'].fillna(X_copy['loan_type'].map(self.map_modes_neg_amm)).fillna(self.mode_global_neg_amm)
        cles_sub = list(zip(X_copy['loan_type'], X_copy['Region']))
        valeurs_sub = pd.Series([self.map_modes_submission.get(c, self.mode_global_submission) for c in cles_sub], index=X_copy.index)
        X_copy['submission_of_application'] = X_copy['submission_of_application'].fillna(valeurs_sub)
        return X_copy

class CategoricalToStringConverter(BaseEstimator, TransformerMixin):
    def __init__(self, variables): self.variables = variables
    def fit(self, X, y=None): return self
    def transform(self, X):
        X_copy = X.copy()
        X_copy[self.variables] = X_copy[self.variables].astype(str)
        return X_copy

# ==============================================================================
# CHARGEMENT DES FICHIERS MODELES .PKL
# ==============================================================================
@st.cache_resource
def charger_modeles():
    pipeline_prepro = joblib.load(r'C:\Users\HP\Documents\Projet_Machine_Learning\pipeline_preprocessing_final.pkl')
    pack_mlp = joblib.load(r'C:\Users\HP\Documents\Projet_Machine_Learning\modele_mlp_final.pkl')
    return pipeline_prepro, pack_mlp['scaler'], pack_mlp['model']

pipeline_preprocessing, scaler_modele, modele_mlp = charger_modeles()

# ==============================================================================
# FONCTION COMMUNE DE PRÉDICTION
# ==============================================================================
def executer_prediction(df_entree):
    variables_attendues_modele = [
        'loan_limit', 'Gender', 'approv_in_adv', 'loan_type', 'loan_purpose', 
        'Credit_Worthiness', 'open_credit', 'Neg_ammortization', 'interest_only', 
        'lump_sum_payment', 'occupancy_type', 'Secured_by', 'total_units', 
        'credit_type', 'co-applicant_credit_type', 'age', 'submission_of_application', 
        'Region', 'loan_amount', 'term', 'income', 'dtir1'
    ]
    variables_cat = [
        'loan_limit', 'Gender', 'approv_in_adv', 'loan_type', 'loan_purpose', 
        'Credit_Worthiness', 'open_credit', 'business_or_commercial', 
        'Neg_ammortization', 'interest_only', 'lump_sum_payment', 
        'construction_type', 'occupancy_type', 'Secured_by', 'total_units', 
        'credit_type', 'co-applicant_credit_type', 'age',
        'submission_of_application', 'Region', 'Security_Type'
    ]
    ordre_initial_preprocessing = [
        'loan_limit', 'Gender', 'approv_in_adv', 'loan_type', 'loan_purpose',
        'Credit_Worthiness', 'open_credit', 'business_or_commercial', 'loan_amount',
        'rate_of_interest', 'Interest_rate_spread', 'Upfront_charges', 'term',
        'Neg_ammortization', 'interest_only', 'lump_sum_payment', 'property_value',
        'construction_type', 'occupancy_type', 'Secured_by', 'total_units', 'income',
        'credit_type', 'co-applicant_credit_type', 'age', 'submission_of_application',
        'LTV', 'Region', 'Security_Type', 'dtir1'
    ]

    df_clean = df_entree.copy()
    
    if 'business_or_commercial' not in df_clean.columns: df_clean['business_or_commercial'] = 'nob/c'
    if 'Neg_ammortization' not in df_clean.columns: df_clean['Neg_ammortization'] = 'not_neg'
    if 'interest_only' not in df_clean.columns: df_clean['interest_only'] = 'not_int'
    if 'lump_sum_payment' not in df_clean.columns: df_clean['lump_sum_payment'] = 'not_lpsm'
    
    cols_inutiles = ['ID', 'year', 'Credit_Score']
    df_clean = df_clean.drop(columns=[c for c in cols_inutiles if c in df_clean.columns])
    df_clean = df_clean[ordre_initial_preprocessing]
    
    if hasattr(pipeline_preprocessing, "feature_names_in_"):
        df_clean.columns = pipeline_preprocessing.feature_names_in_

    X_processed = pipeline_preprocessing.transform(df_clean)
    colonnes_intermediaires = variables_cat + [c for c in ordre_initial_preprocessing if c not in variables_cat and c not in ['property_value', 'LTV']]
    df_res = pd.DataFrame(X_processed, columns=colonnes_intermediaires)
    
    df_final = df_res[variables_attendues_modele]
    X_scaled = scaler_modele.transform(df_final.values)
    
    return modele_mlp.predict(X_scaled), modele_mlp.predict_proba(X_scaled)[:, 1]


# ==============================================================================
# STRUCTURE VISUELLE (IMAGE EN HAUT À GAUCHE)
# ==============================================================================
# Modifié : L'image est maintenant envoyée dans st.sidebar pour s'afficher en haut à gauche
try:
    st.sidebar.image(r"C:\Users\HP\Documents\Projet_Machine_Learning\image_app_predict.png", use_container_width=True)
except:
    pass

st.markdown("<h2 style='color: #1E3A8A;'>🏛️ Application d'Évaluation du Risque de Crédit</h2>", unsafe_allow_html=True)
st.write("Cette application permet de prédire la solvabilité d'un dossier emprunteur à l'aide d'un modèle d'IA.")

# Sélecteur d'option sur la page centrale
option = st.selectbox(
     "Comment souhaitez-vous utiliser le modèle de prédiction ?",
     ('', 'Saisir les paramètres manuellement', 'Charger un fichier de données CSV'))

st.write("---")

# ==============================================================================
# GESTION DU MODE 1 : SAISIE MANUELLE AU CENTRE (FORMULAIRE)
# ==============================================================================
if option == 'Saisir les paramètres manuellement':
    st.subheader("📝 Formulaire de saisie des paramètres")
    
    # Création du formulaire AU CENTRE (en 3 colonnes) grâce à st.form
    with st.form("formulaire_unitaire_centre"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**👤 Profil Emprunteur**")
            gender = st.selectbox("Genre", ['Male', 'Female', 'Joint', 'Sex Not Available'])
            age = st.selectbox("Classe d'âge", ['25-34', '35-44', '45-54', '55-64', '65-74', '>74', '<25'])
            income = st.number_input("Revenu mensuel ($)", min_value=0, value=5500)
            credit_worthiness = st.selectbox("Solvabilité (Credit Worthiness)", ['l1', 'l2'])
            credit_type = st.selectbox("Type de crédit historique", ['CIB', 'EXP', 'CRIF', 'EQUI'])
            co_applicant_credit_type = st.selectbox("Type de crédit co-demandeur", ['CIB', 'EXP'])
            
        with col2:
            st.markdown("**📉 Caractéristiques Prêt**")
            loan_type = st.selectbox("Type de prêt", ['type1', 'type2', 'type3'])
            loan_purpose = st.selectbox("But du prêt", ['p1', 'p3', 'p4', 'p2'])
            loan_amount = st.number_input("Montant du prêt ($)", min_value=1000, value=250000)
            term = st.slider("Durée du prêt (en mois)", 12, 360, 360)
            loan_limit = st.selectbox("Limite de prêt", ['cf', 'ncf'])
            approv_in_adv = st.selectbox("Approbation préalable", ['nopre', 'pre'])
            open_credit = st.selectbox("Crédit ouvert", ['nopc', 'opc'])
            
        with col3:
            st.markdown("**🏠 Propriété & Paramètres**")
            property_value = st.number_input("Valeur du bien ($)", min_value=1000, value=350000)
            ltv = st.number_input("Ratio LTV (%)", min_value=0.0, value=75.0)
            dtir1 = st.slider("Taux d'endettement DTI (%)", 1, 100, 35)
            business_or_commercial = st.selectbox("Commercial / Professionnel", ['nob/c', 'b/c'])
            occupancy_type = st.selectbox("Type d'occupation", ['pr', 'sr', 'ir'])
            secured_by = st.selectbox("Garantie par", ['home', 'land'])
            total_units = st.selectbox("Nombre d'unités", ['1U', '2U', '3U', '4U'])
            submission_of_application = st.selectbox("Soumission de la demande", ['to_inst', 'not_inst'])
            region = st.selectbox("Région d'origine", ['south', 'North', 'central', 'North-East'])
            
            # Paramètres techniques par défaut requis
            rate_of_interest = 4.0
            interest_rate_spread = 0.4
            upfront_charges = 2000.0
            construction_type = 'sb'
            security_type = 'direct'

        # Bouton de validation situé au centre, en bas du formulaire
        bouton_valider = st.form_submit_button("🚀 Valider et Prédire le Dossier")
    
    if bouton_valider:
        data = {
            'loan_limit': loan_limit, 'Gender': gender, 'approv_in_adv': approv_in_adv, 'loan_type': loan_type,
            'loan_purpose': loan_purpose, 'Credit_Worthiness': credit_worthiness, 'open_credit': open_credit,
            'business_or_commercial': business_or_commercial, 'loan_amount': loan_amount, 'rate_of_interest': rate_of_interest,
            'Interest_rate_spread': interest_rate_spread, 'Upfront_charges': upfront_charges, 'term': float(term),
            'Neg_ammortization': 'not_neg', 'interest_only': 'not_int', 'lump_sum_payment': 'not_lpsm',
            'property_value': property_value, 'construction_type': construction_type, 'occupancy_type': occupancy_type,
            'Secured_by': secured_by, 'total_units': total_units, 'income': float(income), 'credit_type': credit_type,
            'co-applicant_credit_type': co_applicant_credit_type, 'age': age, 'submission_of_application': submission_of_application,
            'LTV': ltv, 'Region': region, 'Security_Type': security_type, 'dtir1': float(dtir1)
        }
        df_manuel = pd.DataFrame(data, index=[0])
        
        st.markdown("**Paramètres configurés :**")
        st.write(df_manuel)
        
        try:
            preds, probas = executer_prediction(df_manuel)
            st.write("---")
            st.subheader("🔮 Résultat de l'Évaluation :")
            if preds[0] == 1:
                st.error(f"🚨 **Dossier Refusé : Risque de Défaut Élevé** (Probabilité : {probas[0]*100:.2f}%)")
            else:
                st.success(f"✅ **Dossier Accepté : Client Solvable** (Probabilité de risque : {probas[0]*100:.2f}%)")
            st.progress(float(probas[0]))
        except Exception as e:
            st.error(f"Erreur lors du calcul : {e}")

# ==============================================================================
# GESTION DU MODE 2 : CHARGEMENT DE FICHIER CSV (CENTRE)
# ==============================================================================
elif option == 'Charger un fichier de données CSV':
    st.subheader("📂 Analyse de fichiers groupés (Batch)")
    
    st.markdown("""
    ⚠️ **Important :** Pour que le modèle fonctionne, votre fichier CSV doit respecter strictement la structure d'origine.
    Voici un aperçu de l'en-tête et des premières colonnes requises :
    """)
    
    structure_exemple = pd.DataFrame({
        'ID': [173554], 'year': [2019], 'loan_limit': ['cf'], 'Gender': ['Joint'], 
        'approv_in_adv': ['nopre'], 'loan_type': ['type1'], 'loan_amount': [150000], 'income': [5500]
    })
    st.dataframe(structure_exemple)
    
    csv_template_content = "ID,year,loan_limit,Gender,approv_in_adv,loan_type,loan_purpose,Credit_Worthiness,open_credit,business_or_commercial,loan_amount,rate_of_interest,Interest_rate_spread,Upfront_charges,term,Neg_ammortization,interest_only,lump_sum_payment,property_value,construction_type,occupancy_type,Secured_by,total_units,income,credit_type,Credit_Score,co-applicant_credit_type,age,submission_of_application,LTV,Region,Security_Type,dtir1\n"
    
    st.download_button(
        label="📥 Télécharger le fichier modèle vide (.csv)",
        data=csv_template_content,
        file_name="modele_structure_credit.csv",
        mime="text/csv"
    )
    
    st.write("---")
    uploaded_file = st.file_uploader("Veuillez sélectionner votre fichier CSV finalisé", type="csv")
    
    if uploaded_file is not None:
        df_csv = pd.read_csv(uploaded_file)
        st.write("📋 **Aperçu du fichier importé (10 premières lignes) :**")
        st.dataframe(df_csv.head(10))
        
        if st.button("🚀 Exécuter l'analyse automatique sur le fichier"):
            with st.spinner("Analyse et calcul des scores en cours..."):
                try:
                    preds, probas = executer_prediction(df_csv)
                    df_final = df_csv.copy()
                    df_final['Prediction_Status'] = preds
                    df_final['Probabilite_Defaut'] = probas
                    df_final['Diagnostic'] = df_final['Prediction_Status'].map({1: "Risque de Défaut", 0: "Sain"})
                    
                    st.write("---")
                    st.subheader("📊 Résultats Généraux")
                    nb_defauts = int((preds == 1).sum())
                    
                    c1, c2 = st.columns(2)
                    c1.metric("Profils Sains Validés", f"{len(preds) - nb_defauts}")
                    c2.metric("Profils Suspects (Rejet)", f"{nb_defauts}", delta=f"{(nb_defauts/len(preds))*100:.1f}%")
                    
                    st.write("📋 **Rapport détaillé (Aperçu) :**")
                    st.dataframe(df_final[['ID', 'loan_amount', 'income', 'Probabilite_Defaut', 'Diagnostic']].head(10))
                    
                    csv_data = df_final.to_csv(index=False).encode('utf-8')
                    st.download_button(label="📥 Télécharger le rapport global complet (.csv)", data=csv_data, file_name="resultats_predictions_credit.csv", mime="text/csv")
                except Exception as e:
                    st.error(f"Erreur lors de l'analyse du fichier : {e}")