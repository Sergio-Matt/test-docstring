import streamlit as st


#importer le fichier : kaggle_survey_2020_responses.csv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

df = pd.read_csv(r'C:\Users\Professionnel\OneDrive\Dokumenty\DIPLOMES\DATA ANALYST\PROJET\Dataset\kaggle_survey_2020_responses.csv', header=0, low_memory=False)

# definir la premiere ligne comme nom des colonnes et supprimer la deuxieme ligne
df = df.drop(index=0).reset_index(drop=True)
print(df.head())

# Configurer la page
st.set_page_config(
    page_title="Présentation du Projet Data Job",
    layout="wide"
)

# Titre principal
st.title("Projet DataJob")
st.write("Datascientest - Formation Data Analyst - Bootcamp NOV2023")

# Barre latérale
st.sidebar.title("Sommaire")
sections = ["Introduction", "Exploration des données", "Definition du projet", "Nettoyage et Preprocessing", "Analyse PCA", "Modélisations supervisées", "Exploitation et conclusion"]
section = st.sidebar.radio("Aller vers la section :", sections)


############## Section : Introduction
if section == "Introduction":
  st.header("Introduction")
  st.markdown("""
  Le projet datajob s'appuie sur une enquete Kaggle de 2020. Cette enquete s'interresse aux differents metiers de la data et recolte des informations sur le profil des repondants
  """)
  st.code("""
  # Afficher shape et df
  df.shape
  df.head(10)""")
  st.write(df.shape)
  st.dataframe(df.head(10))
  st.markdown(""" <br><br> """, unsafe_allow_html=True)

  st.markdown("""
  On remarque rapidement que le jeu de données contient beaucoup de NA""")
  st.dataframe(df.isna().sum())
  st.markdown(""" <br><br> """, unsafe_allow_html=True)

  st.markdown(""" On s'interresse au cheminement du questionnaire et au sens des questions """)
  st.image("Schema du questionnaire.jpg", caption="Schéma du déroulé du questionnaire", use_column_width=True)
  st.markdown(""" <br><br> """, unsafe_allow_html=True)




########## Section : Exploration des données
elif section == "Exploration des données":
    st.header("Exploration des Données")
    st.markdown(""" <br> """, unsafe_allow_html=True)

    # ---------------------- Profil des Répondants -----------------------
    st.markdown("""
<div style="background-color: #1E90FF; padding: 10px; border-radius: 5px; text-align: center;">
    <h3 style="color: white;">Profil des Répondants</h3>
</div>
""", unsafe_allow_html=True)
    st.markdown(""" <br><br> """, unsafe_allow_html=True)

    # Ligne 1 : Distribution des Métiers & Profil des Répondants
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Distribution des Métiers (Q5)")
        df_filtré = df[~df['Q5'].isin(['Student', 'Other', 'Currently not employed'])]
        q5_graph = df_filtré['Q5'].value_counts()
        plt.figure(figsize=(5, 4))
        sns.barplot(x=q5_graph.index, y=q5_graph.values)
        plt.xticks(rotation=90)
        plt.xlabel('Les métiers')
        plt.ylabel('Nombre')
        plt.title('Distribution des Métiers (Q5)')
        st.pyplot(plt.gcf())
        st.write("**Interprétation :** Les métiers les plus représentés sont liés à la science des données.")

    with col2:
        st.subheader("Profil des Répondants")
        age_categories = ['18-21', '22-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60-69', '70+']
        df['Q1'] = pd.Categorical(df['Q1'], categories=age_categories, ordered=True)
        plt.figure(figsize=(5, 4))
        sns.countplot(x='Q1', hue='Q2', data=df, order=age_categories)
        plt.xlabel('Catégorie d\'âge')
        plt.ylabel('Nombre de réponses')
        plt.title('Âge et Genre des Répondants')
        plt.xticks(rotation=45)
        st.pyplot(plt.gcf())
        st.write("**Interprétation :** Le graphique montre une répartition équilibrée entre les genres pour la plupart des catégories d'âge.")

    # Ligne 2 : Tranche d'âge par Métier
    st.markdown(""" <br><br> """, unsafe_allow_html=True)
    st.subheader("Tranche d'âge par Métier")
    df_filtré['Q1'] = pd.Categorical(df_filtré['Q1'], categories=age_categories, ordered=True)
    plt.figure(figsize=(10, 4))
    sns.countplot(x='Q5', hue='Q1', data=df_filtré, order=df_filtré['Q5'].value_counts().index)
    plt.xticks(rotation=90)
    plt.xlabel('Métiers')
    plt.ylabel('Nombre de réponses')
    plt.title('Tranche d\'âge par métier')
    plt.legend(title='Âge', bbox_to_anchor=(1.05, 1), loc='upper left')
    st.pyplot(plt.gcf())
    st.write("**Interprétation :** La répartition par âge montre une concentration de jeunes professionnels dans les métiers de la data.")

    # Ligne 3 : Niveau d'Études par Métier
    st.markdown(""" <br><br> """, unsafe_allow_html=True)
    st.subheader("Niveau d'Études par Métier")
    df_filtré = df.dropna(subset=['Q5', 'Q4'])
    plt.figure(figsize=(10, 4))
    sns.countplot(x='Q5', hue='Q4', data=df_filtré, order=df_filtré['Q5'].value_counts().index)
    plt.xticks(rotation=90)
    plt.xlabel('Métiers')
    plt.ylabel('Nombre de réponses')
    plt.title('Niveau d\'Études par Métier')
    st.pyplot(plt.gcf())
    st.write("**Interprétation :** La majorité des professionnels dans les métiers de la data possèdent un niveau d'études élevé.")

    
    
    # Section : Articles
    st.markdown(""" <br><br> """, unsafe_allow_html=True)
    st.markdown("## Les métiers Data plaisent aux jeunes : quelques articles à lire")
    # Présentation des articles avec des liens cliquables
    st.markdown("""
    <div style="background-color: #f9f9f9; padding: 10px; border-radius: 8px; margin-bottom: 20px;">
        <h4><a href="https://www.jobirl.com/conseils-et-infos/eugenie-data-scientist-metier-d-avenir" target="_blank" style="color: #2c5fbb;">Eugénie, Data Scientist : Un Métier d'Avenir</a></h4>
        <p style="color: #555;">Découvrez le parcours d'Eugénie, une Data Scientist passionnée, et les compétences nécessaires pour réussir dans ce domaine.</p>
    </div>
    <div style="background-color: #f9f9f9; padding: 10px; border-radius: 8px; margin-bottom: 20px;">
        <h4><a href="https://www.emploi-store.fr/portail/accueil/espace-jeunes/metiers-du-futur/article/data-scientist-un-metier-du-futur" target="_blank" style="color: #2c5fbb;">Data Scientist : Un Métier du Futur</a></h4>
        <p style="color: #555;">Un focus sur les opportunités et les débouchés dans les métiers de la Data pour les années à venir.</p>
    </div>
    <div style="background-color: #f9f9f9; padding: 10px; border-radius: 8px; margin-bottom: 20px;">
        <h4><a href="https://speaknact.fr/fr/blog/article/224--les-metiers-de-la-data-ont-le-vent-en-poupe" target="_blank" style="color: #2c5fbb;">Les métiers de la data ont le vent en poupe</a></h4>
        <p style="color: #555;">Un article détaillant pourquoi les métiers de la data sont en forte croissance et leur impact dans le monde professionnel.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(""" <br><br><br><br> """, unsafe_allow_html=True)

    
    # ---------------------- Connaissances et Compétences -----------------------
    st.markdown("""
<div style="background-color: #1E90FF; padding: 10px; border-radius: 5px; text-align: center;">
    <h3 style="color: white;">Connaissances et Compétences</h3>
</div>
""", unsafe_allow_html=True)
    st.markdown(""" <br> """, unsafe_allow_html=True)

    # Ligne 1 : Langages utilisés dans les Métiers de la Data
    st.subheader("Langages utilisés dans les Métiers de la Data (Q7)")
    q7_columns = [col for col in df.columns if "Q7" in col]
    plt.figure(figsize=(10, 4))
    for col in q7_columns:
        sns.countplot(x=col, data=df, label=col)
    plt.xlabel("Langages")
    plt.xticks(rotation=90)
    plt.ylabel("Nombre de réponses")
    plt.title("Langages utilisés (Q7)")
    st.pyplot(plt.gcf())
    st.write("**Interprétation :** Les langages les plus couramment utilisés incluent Python, suivi de R et SQL.")
    st.markdown(""" <br><br> """, unsafe_allow_html=True)

    # Ligne 2 : Environnements de Travail
    st.subheader("Les Environnements de Travail Utilisés (Q9)")
    q9_columns = [col for col in df.columns if "Q9" in col]
    plt.figure(figsize=(10, 4))
    for col in q9_columns:
        sns.countplot(x=col, data=df, label=col)
    plt.xlabel("Environnements de Travail")
    plt.xticks(rotation=90)
    plt.ylabel("Nombre de réponses")
    plt.title("Environnements de Travail (Q9)")
    st.pyplot(plt.gcf())
    st.write("**Interprétation :** Les environnements populaires incluent Jupyter Notebook, Visual Studio Code, et Google Colab.")
    st.markdown(""" <br><br> """, unsafe_allow_html=True)

    # Ligne 3 : Outils de DataViz
    st.subheader("Les Outils de DataViz les Plus Utilisés (Q14)")
    q14_columns = [col for col in df.columns if "Q14" in col]
    plt.figure(figsize=(10, 4))
    for col in q14_columns:
        sns.countplot(x=col, data=df, label=col)
    plt.xlabel("Outils de DataViz")
    plt.xticks(rotation=90)
    plt.ylabel("Nombre de réponses")
    plt.title("Outils de DataViz (Q14)")
    st.pyplot(plt.gcf())
    st.write("**Interprétation :** Tableau, Power BI, et Matplotlib sont parmi les outils les plus utilisés.")


    st.markdown("""<br><br><br><br>""", unsafe_allow_html=True)
    st.markdown("## Les métiers data: Des connaissances et compétences partagées")
    st.image("data-profils.jpg", caption="data-profils")
    st.markdown("""<br><br><br><br>""", unsafe_allow_html=True)


############## Section suivante : Definition du projet
elif section == "Definition du projet":
    st.header("Definiton du Projet")
    st.markdown("""
  Les points relevés :
  - Une hype des metiers de la data, notamment chez les jeunes
  - Des connaissances et competences partagées
  - Des missions spécifiques en fonction du metiers
  - Une meconnaissance du public des differents metiers
    """)
    st.markdown("""<br>""", unsafe_allow_html=True)
    st.markdown(""" Notre proposition : Entrainer un modele de Machine Learning pour aider les étudiants à s'orienter dans les metiers de la data """)
    st.markdown("""<br>""", unsafe_allow_html=True)
    st.image("image1.jpg")


############### Section : Nettoyage et Preprocessing
elif section == "Nettoyage et Preprocessing":
  st.header("Nettoyage et Préparation des Données")
  st.markdown("""
  Les étapes réalisées :
  - Suppression des colonnes inutiles (Times, Q1, Q2, Q3 et colonnes du theme "Environnement de travail et metier")
  - Suppression des colonnes des jeux A et B car trop de valeurs manquantes
  - Nettoyage de la colonne Q5 (suppression des NaN, Student, Other et Currently not employed)
  - Encodage manuel des variables "choix unique" aprés scission des colonnes en deux listes "choix unique" et "choix multiple"
  - Séparation du jeu de données pour eviter le surapprentissage (75/25)
  - Encodage des variables "choix multiple" par OneHotEncoder
  - Sur echantillonnage RandomOverSampler pour rééquilibrage des variables de Q5
  - Standardisation des données pour mise à l'echelle
  """)
  st.code("""
  # Afficher df initial
  df.head(10)""")
  st.dataframe(df.head(10))
  st.markdown("""<br><br>""", unsafe_allow_html=True)
  


  df.info()
  # 20036 lignes pour 355 colonnes au format "object"

  # total des Nan dans le df
  print(df.isna().sum().sum())

  # Nombre de modalités pour chaque colonne
  df.nunique()

  # Supprimer les colonnes spécifiées pour éviter un ML disciminant
  columns_to_drop = ['Time from Start to Finish (seconds)', 'Q1', 'Q2', 'Q3', 'Q20', 'Q21', 'Q22', 'Q24', 'Q25']
  df = df.drop(columns=columns_to_drop, errors='ignore')

  # Supprimer toutes les colonnes contenant "_A" ou "_B" dans leur nom pour simplifier le traitement
  columns_to_drop = [col for col in df.columns if '_A' in col or '_B' in col]
  df = df.drop(columns=columns_to_drop, errors='ignore')

  # supprimer les lignes de Q5 qui contiennenet student ou other ou Currently not employed
  df = df[(df['Q5'] != 'Student') & (df['Q5'] != 'Other') & (df['Q5'] != 'Currently not employed')]
  print(df['Q5'].value_counts())

  # Supprimer les lignes avec des NaN dans la colonne 'Q5'
  df = df.dropna(subset=['Q5'])

  # nombre de NaN dans df
  print(df.isna().sum().sum())

  

  # Afficher toutes les modalités de toutes les colonnes de df
  for i in df:
    print(str(i),':',df[str(i)].unique())

  # diviser le df en 2: df_choix_unique et df_choix_multiple
  df_choix_unique = df.columns[df.nunique()<=3]   # 3 = moalité x + modalité y + Nan
  df_choix_multiple = df.columns[df.nunique()>3]

  # Afficher les colonnes
  print(df_choix_multiple.to_list())

  # pour chaque colonne de la liste "df_choix_unique", remplacer le Nan par "0" et remplacer la  modalité par "1"

  # Parcourir chaque colonne de df_choix_unique
  for col in df_choix_unique:
    # Remplacer les valeurs NaN par 0
    df[col] = df[col].fillna(0)
    # Obtenir la liste des modalités uniques (hors NaN et 0)
    modalites = df[col].unique()
    modalites = [modalite for modalite in modalites if modalite != 0 and not pd.isna(modalite)]

    # Remplacer chaque modalité par 1
    for modalite in modalites:
        df[col] = df[col].replace(modalite, 1)

  st.code("""
  # Afficher df apres encodage des colonne "choix multiple"
  df.head(10)""")
  st.dataframe(df.head(10))
  st.markdown("""<br><br>""", unsafe_allow_html=True)

  # separer le df en X et Y pour eviter la fuite de données du onehotencoder
  from sklearn.model_selection import train_test_split

  y = df['Q5']
  X = df.drop('Q5', axis=1)

  # prompt: reequilibre les classes  de X et y avec oversampling

  from imblearn.over_sampling import RandomOverSampler

  oversampler = RandomOverSampler(random_state=42)
  X_resampled, y_resampled = oversampler.fit_resample(X, y)

  
  X = X_resampled
  y = y_resampled

  #train test split
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=48)

  print("Train Set:", X_train.shape)
  print("Test Set:", X_test.shape)

  # encoder la variable cible avec un onehotencoder puis reduire la dimension à 1 pour les modelisations

  from sklearn.preprocessing import OneHotEncoder
  import numpy as np

  # Initialiser le OneHotEncoder
  enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

  # Ajuster et transformer la variable cible y_train
  enc.fit(np.array(y_train).reshape(-1, 1))
  y_train_encoded = enc.transform(np.array(y_train).reshape(-1, 1))

  # Transformer y_test avec le même encodeur
  y_test_encoded = enc.transform(np.array(y_test).reshape(-1, 1))

  # Réduire la dimension à 1 (en utilisant argmax pour obtenir l'index de la classe)
  y_train_encoded = np.argmax(y_train_encoded, axis=1)
  y_test_encoded = np.argmax(y_test_encoded, axis=1)

  print("Encoded y_train shape:", y_train_encoded.shape)
  print("Encoded y_test shape:", y_test_encoded.shape)
  
  # afficher encodage de Q5
  mapping = {index: label for index, label in enumerate(enc.categories_[0])}
  
  
  # pour les colonnes  'Q4', 'Q6', 'Q8', 'Q11', 'Q13', 'Q15', 'Q30', 'Q32', 'Q38'  de X_train et X_test appliquer un onehotencoder

  from sklearn.preprocessing import OneHotEncoder

  # definir enc
  enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

  # les colonnes ciblées
  columns_to_encode = ['Q4', 'Q6', 'Q8', 'Q11', 'Q13', 'Q15', 'Q30', 'Q32', 'Q38']


  enc.fit(X_train[columns_to_encode])


  X_train_encoded = enc.transform(X_train[columns_to_encode])
  X_test_encoded = enc.transform(X_test[columns_to_encode])


  encoded_column_names = list(enc.get_feature_names_out(columns_to_encode))


  # creer dataframe à partir des arrays
  X_train_encoded_df = pd.DataFrame(X_train_encoded, columns=encoded_column_names, index=X_train.index)
  X_test_encoded_df = pd.DataFrame(X_test_encoded, columns=encoded_column_names, index=X_test.index)


  # supprimer les colonnes d'origine
  X_train = X_train.drop(columns=columns_to_encode)
  X_test = X_test.drop(columns=columns_to_encode)

  # definir X_train et X_test
  X_train = pd.concat([X_train, X_train_encoded_df], axis=1)
  X_test = pd.concat([X_test, X_test_encoded_df], axis=1)
 
  st.code("""
  # Afficher X_train apres One Hot Encoding des colonnes "choix unique"
  X_train.head(10)""")
  st.dataframe(X_train.head(10))
  st.markdown("""<br><br>""", unsafe_allow_html=True)
  # normaliser avec standarscaler X_train et X_test

  from sklearn.preprocessing import StandardScaler


  scaler = StandardScaler()

  X_train = scaler.fit_transform(X_train)
  X_test = scaler.transform(X_test)

  st.code("""
  # Afficher encodage de Q5
  mapping """)
  st.dataframe(mapping)



########################### Section : Analyse PCA
elif section == "Analyse PCA":
  st.header("Analyse PCA")
  st.markdown("""Objectif principal: comprendre la structure du jeu de données et évaluer ou se situe l'information""")
  st.markdown("""<br><br>""", unsafe_allow_html=True)
###
  st.markdown("""
  L’analyse de la variance expliquée cumulée montre que les 200 premières composantes principales préservent près de 100 % de la variance totale """)
  st.image("Vaiance cumulée par nombre de composants.jpg", caption="Variance cumulée")
  st.markdown("""<br><br>""", unsafe_allow_html=True)

###
  st.markdown("""
  la méthode du coude montre une forte diminution de l’inertie lorsque le nombre de clusters augmente, mais cette diminution ralentit nettement à partir de k = 3
  """)
  st.image("Methode du coude.jpg", caption="Methode du coude")
  st.markdown("""<br><br>""", unsafe_allow_html=True)

###
  st.markdown("""
  la structure des données, réduite par la PCA, ne permet pas de capturer suffisamment d’information pour différencier clairement les métiers
  """)
  st.image("Graphique de clustering.jpg", caption="Graphique de clustering")
  st.markdown("""<br><br>""", unsafe_allow_html=True)






######################### Section : Modélisations supervisées
elif section == "Modélisations supervisées":
  st.header("Modélisations Supervisées")
  st.markdown("""
  Quatre algorithmes ont été testés :
  - **Decision Tree**
  - **Random Forest**
  - **XGBoost**
  - **SVM**
  """)

  ###
  # Définir les groupes d'images
  image_groups = {
    "Decision Tree": ("Accuracy decision tree.jpg", "Matrice de confusion decision tree.jpg"),
    "Random Forest": ("Accuracy random forest.jpg", "Matrice de confusion random forest.jpg"),
    "XGBoost": ("Accuracy xgboost.jpg", "Matrice de confusion xgboost.jpg"),
    "SVM": ("Accuracy svm.jpg", "Matrice de confusion svm.jpg")
  }

  # Sélecteur pour choisir un groupe
  selected_group = st.selectbox(
    "Choix de l'algorithme",
    list(image_groups.keys())
  )

  # Récupérer les deux images du groupe sélectionné
  image1, image2 = image_groups[selected_group]

  # Afficher les deux images côte à côte
  col1, col2 = st.columns(2)

  with col1:
    st.image(image1, caption=f"{selected_group} - Image 1", use_column_width=True)

  with col2:
    st.image(image2, caption=f"{selected_group} - Image 2", use_column_width=True)

  
  st.markdown("""<br><br><br>""", unsafe_allow_html=True)
  st.markdown("""### Comparatif des 4 algorithmes""")

# Définir les précisions des modèles
  accuracy_dt = 0.8123
  accuracy_rf = 0.8613
  accuracy_xgb = 0.7955
  accuracy_svm = 0.7296

# Données pour le graphique
  models = ['Decision Tree', 'Random Forest', 'XGBoost', 'SVM']
  accuracies = [accuracy_dt, accuracy_rf, accuracy_xgb, accuracy_svm]

# Créer le graphique avec annotations
  plt.figure(figsize=(6, 4))  # Taille ajustée
  bars = plt.bar(models, accuracies, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])

# Ajouter les résultats au-dessus des barres
  for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{acc:.2f}",
             ha='center', va='bottom', fontsize=10, color='black')

# Ajouter des labels et un titre
  plt.xlabel("Modèles", fontsize=12)
  plt.ylabel("Précision", fontsize=12)
  plt.ylim(0, 1)  # entre 0% et 100%
  plt.grid(axis='y', linestyle='--', alpha=0.7)

# Afficher le graphique dans Streamlit
  st.pyplot(plt.gcf())
  st.markdown("""<br><br><br>""", unsafe_allow_html=True)



  # hyperparametarge
  

  # Définir les groupes d'images avec une seule image par groupe
  image_groups2 = {
    "Decision Tree hyperparametrage": "hyperparametrage decision tree.jpg",
    "Random Forest hyperparametrage": "hyperparametrage random forest.jpg",
  }

  # Sélecteur pour choisir un groupe
  selected_group2 = st.selectbox(
    "Choisir l'algorithme",
    list(image_groups2.keys())
  )

  # Récupérer l'image du groupe sélectionné
  image = image_groups2[selected_group2]

  # Afficher l'image
  st.image(image, caption=f"{selected_group2}", use_column_width=True)





######################## Section : Exploitation et conclusion
elif section == "Exploitation et conclusion" : 
  st.header("Exploitation")
  st.markdown("""
  Un prototype est présenté pour collecter les réponses des utilisateurs, appliquer le modèle prédictif 
  et fournir des recommandations sur les métiers de la data.
  """)
  st.markdown("""<br><br><br>""", unsafe_allow_html=True)

  ###
  st.markdown(""" Une exemple de questionnaire google forms """)
  st.image("POC1.jpg", caption="POC1")
  st.markdown("""<br><br><br>""", unsafe_allow_html=True)

  ###
  st.markdown(""" Les réponses sont collectées dans un tableur exportable en fichier CSV """)
  st.image("POC2.jpg", caption="POC2")
  st.markdown("""<br><br><br>""", unsafe_allow_html=True)

  ###
  st.markdown(""" Q6 encodée via OHE """)
  st.image("POC3.jpg", caption="POC3")
  st.markdown("""<br><br><br>""", unsafe_allow_html=True)

  ###
  st.markdown(""" Code permettant d'appliquer notre algorithme entrainé sur notre nouveau jeu """)
  st.image("POC4.jpg", caption="POC4")
  st.markdown("""<br><br><br>""", unsafe_allow_html=True)

  ###
  st.markdown(""" Template de POC """) 
  st.image("POC5.jpg", caption="POC5")
  st.markdown("""<br><br><br>""", unsafe_allow_html=True)

  st.header("Conclusion")
  st.markdown("""
  **Limites identifiées :**
   - **Manque de temps et/ou de personnes**
  - **Limite materiel et puissance de calcul**
  - **Parcours de formation vs developpement du projet**
  """)
  st.markdown("""
  **Perspectives :**
  - **Ajouter des nouvelles colonnes de soft skills**
  - **Developper une application fonctionnelle**
  - **Faire le lien avec d'autres projet pour aller plus loin**
  """)



st.sidebar.info("Utilisez le menu pour naviguer entre les sections.")
