# Separate Python module version of your Streamlit ML App using modular functions

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    silhouette_score, davies_bouldin_score, roc_curve
)
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from scipy.stats import randint
import streamlit as st
import matplotlib as mpl

mpl.rcParams.update({
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16
})

# Load and preprocess data

def load_and_preprocess(file_path):
    df = pd.read_csv(file_path)
    df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})
    X = df.drop(columns=['Attrition'])
    y = df['Attrition']
    return df, X, y

# Create pipeline

def create_pipeline(X):
    cat_cols = X.select_dtypes(include='object').columns.tolist()
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    pipeline = ColumnTransformer([
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ]), cat_cols),
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ]), num_cols)
    ])
    return pipeline

# Train supervised models

def train_supervised_models(X_train, X_test, y_train, y_test):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(),
        "Gradient Boosting": GradientBoostingClassifier(),
        "SVM": SVC(probability=True),
        "k-NN": KNeighborsClassifier()
    }
    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        results.append({
            'Model': name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1': f1_score(y_test, y_pred),
            'AUC': roc_auc_score(y_test, y_proba) if y_proba is not None else None
        })
    return pd.DataFrame(results), models

# Train unsupervised models

def train_unsupervised_models(X):
    models = {
        'KMeans': KMeans(n_clusters=2),
        'DBSCAN': DBSCAN(eps=2.5, min_samples=5),
        'Agglomerative': AgglomerativeClustering(n_clusters=2),
        'GaussianMixture': GaussianMixture(n_components=2)
    }
    results = []
    pca_outputs = {}

    for name, model in models.items():
        labels = model.fit_predict(X)
        silhouette = silhouette_score(X, labels) if len(set(labels)) > 1 else None
        db_index = davies_bouldin_score(X, labels) if len(set(labels)) > 1 else None
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        pca_outputs[name] = (X_pca, labels)
        results.append({
            'Model': name,
            'Silhouette Score': silhouette,
            'Davies-Bouldin': db_index,
            'Clusters': len(set(labels))
        })
    return pd.DataFrame(results), pca_outputs

# Hyperparameter tuning

def tune_supervised(X, y):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    log_grid = GridSearchCV(LogisticRegression(max_iter=1000), {
        'penalty': ['l1', 'l2'],
        'C': [0.01, 0.1, 1, 10],
        'solver': ['liblinear']
    }, scoring='f1', cv=cv, n_jobs=-1)
    log_grid.fit(X, y)
    log_score = cross_val_score(log_grid.best_estimator_, X, y, cv=cv, scoring='f1').mean()

    knn_search = RandomizedSearchCV(KNeighborsClassifier(), {
        'n_neighbors': randint(3, 20),
        'weights': ['uniform', 'distance'],
        'p': [1, 2]
    }, n_iter=20, cv=cv, scoring='f1', n_jobs=-1, random_state=42)
    knn_search.fit(X, y)
    knn_score = cross_val_score(knn_search.best_estimator_, X, y, cv=cv, scoring='f1').mean()

    return log_grid.best_params_, log_score, knn_search.best_params_, knn_score


def tune_unsupervised_best_only(X):
    cluster_range = range(2, 10)
    best_result = None
    best_score = -1
    for k in cluster_range:
        km = KMeans(n_clusters=k, random_state=42)
        km_labels = km.fit_predict(X)
        km_score = silhouette_score(X, km_labels)
        if km_score > best_score:
            best_score = km_score
            best_result = {"Model": "KMeans", "Clusters": k, "Silhouette Score": km_score}

        ag = AgglomerativeClustering(n_clusters=k)
        ag_labels = ag.fit_predict(X)
        ag_score = silhouette_score(X, ag_labels)
        if ag_score > best_score:
            best_score = ag_score
            best_result = {"Model": "Hierarchical", "Clusters": k, "Silhouette Score": ag_score}

    return best_result

# Streamlit app

def run_streamlit_app():
    st.set_page_config(page_title="Employee ML App", layout="wide")
    st.markdown("""
    <style>
        body {
            background-color: #f5ebff;
        }
        .stApp {
            background-color: #f0e6ff;
            color: #2c1a4c;
        }
        section[data-testid="stSidebar"] {
            background-color: #a885cf !important;
        }
        section[data-testid="stSidebar"] * {
            color: black !important;
        }
        /* Specifically for the file uploader input text */
        section[data-testid="stSidebar"] .stFileUploader label div {
            color: black !important;
        }
        section[data-testid="stSidebar"] .stFileUploader button {
            color: black !important;
            background-color: white !important;
            border: 1px solid black !important;
        }
    </style>
""", unsafe_allow_html=True)


    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", [
        "Upload & Preprocess",
        "Train Supervised Models",
        "Train Unsupervised Models",
        "Hyperparameter Tuning Supervised",
        "Hyperparameter Tuning Unsupervised"
    ])
    st.title("Employee Attrition ML Dashboard")
    file = st.sidebar.file_uploader("Upload your dataset", type=["csv"])

    if file:
        df, X, y = load_and_preprocess(file)
        pipeline = create_pipeline(X)
        X_trans = pipeline.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_trans, y, test_size=0.2, random_state=42, stratify=y)

        if selection == "Upload & Preprocess":
            st.subheader("View Options")
            option = st.radio("Choose a view:", ["Sample Data", "Missing Values", "Correlation Matrix", "Histograms", "Box Plots", "Pipeline"])
            if option == "Sample Data":
                st.dataframe(df.head())
            elif option == "Missing Values":
                fig, ax = plt.subplots()
                sns.heatmap(df.isnull(), cbar=False, cmap="YlGnBu", ax=ax)
                st.pyplot(fig)
            elif option == "Correlation Matrix":
                fig, ax = plt.subplots()
                sns.heatmap(df.select_dtypes(include=[np.number]).corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
                st.pyplot(fig)
            elif option == "Histograms":
                column = st.selectbox("Select column for histogram:", df.select_dtypes(include=[np.number]).columns)
                fig, ax = plt.subplots()
                sns.histplot(df[column], kde=True, ax=ax, color="mediumslateblue")
                st.pyplot(fig)
            elif option == "Box Plots":
                column = st.selectbox("Select column for boxplot:", df.select_dtypes(include=[np.number]).columns)
                fig, ax = plt.subplots()
                sns.boxplot(y=df[column], ax=ax, color="orchid")
                st.pyplot(fig)
            elif option == "Pipeline":
                st.write(pipeline)

        elif selection == "Train Supervised Models":
            st.subheader("Supervised Models Trained")
            results_df, trained_models = train_supervised_models(X_train, X_test, y_train, y_test)
            st.dataframe(results_df)
            st.subheader("ROC Curves")
            for name, model in trained_models.items():
                if hasattr(model, "predict_proba"):
                    y_proba = model.predict_proba(X_test)[:, 1]
                    fpr, tpr, _ = roc_curve(y_test, y_proba)
                    plt.figure()
                    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc_score(y_test, y_proba):.2f})")
                    plt.xlabel("False Positive Rate")
                    plt.ylabel("True Positive Rate")
                    plt.title("ROC Curve")
                    plt.legend()
                    st.pyplot(plt)

        elif selection == "Train Unsupervised Models":
            st.subheader("Unsupervised Models Trained")
            unsup_results, pca_outputs = train_unsupervised_models(X_trans)
            st.dataframe(unsup_results)
            st.subheader("Cluster Visualization (PCA)")
            for model_name, (pca_data, labels) in pca_outputs.items():
                fig, ax = plt.subplots()
                scatter = ax.scatter(pca_data[:, 0], pca_data[:, 1], c=labels, cmap='viridis')
                ax.set_title(f"{model_name} Clusters")
                st.pyplot(fig)

        elif selection == "Hyperparameter Tuning Supervised":
            st.subheader("Hyperparameter Tuning - Supervised")
            log_params, log_score, knn_params, knn_score = tune_supervised(X_train, y_train)
            st.write("**Logistic Regression Best Params:**", log_params)
            st.write("**Logistic Regression CV F1 Score:**", log_score)
            st.write("**k-NN Best Params:**", knn_params)
            st.write("**k-NN CV F1 Score:**", knn_score)

        elif selection == "Hyperparameter Tuning Unsupervised":
            st.subheader("Hyperparameter Tuning - Unsupervised")
            best_unsup = tune_unsupervised_best_only(X_trans)
            st.write("**Best Unsupervised Tuning Result:**")
            st.json(best_unsup)

if __name__ == '__main__':
    run_streamlit_app()
