# ============================================
# PIPELINE DE ANÁLISIS - BREAST CANCER DATASET
# ============================================

import itertools
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

warnings.filterwarnings("ignore")


# =========================================================
# 1. IMPORTAR DATASET
# =========================================================
data = load_breast_cancer()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="target")

# Crear un DataFrame completo
df = X.copy()
df["target"] = y
df["class_name"] = df["target"].map({0: data.target_names[0], 1: data.target_names[1]})


# =========================================================
# 2. REVISAR CARACTERÍSTICAS GENERALES
# =========================================================
print("\n" + "=" * 60)
print("INFORMACIÓN GENERAL DEL DATASET")
print("=" * 60)

print("\nDescripción breve del dataset:")
print(data.DESCR[:1500], "...")  # solo una parte para no saturar la salida

print("\nNombres de las características:")
for i, feature in enumerate(data.feature_names, start=1):
    print(f"{i:2d}. {feature}")

print("\nTamaño del dataset:")
print(f"Filas (muestras): {X.shape[0]}")
print(f"Columnas (features): {X.shape[1]}")

print("\nTipo de información:")
print(X.dtypes)

print("\nResumen tipo info():")

# info() imprime directamente, así que usamos un buffer
from io import StringIO
buffer = StringIO()
df.info(buf=buffer)
print(buffer.getvalue())

print("Clases:")
print(f"Etiquetas numéricas: {np.unique(y)}")
print(f"Nombres de clases: {data.target_names}")
print("\nConteo por clase:")
print(df["class_name"].value_counts())

print("\nHuecos / valores faltantes:")
print(df.isnull().sum())
print(f"\nTotal de valores faltantes en todo el dataset: {df.isnull().sum().sum()}")


# =========================================================
# 3. REVISAR CAPACIDAD DE CLASIFICACIÓN UNIVARIABLE
# =========================================================
print("\n" + "=" * 60)
print("CAPACIDAD DE CLASIFICACIÓN UNIVARIABLE")
print("=" * 60)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
model = LogisticRegression(max_iter=5000)

univ_results = []

for feature in X.columns:
    X_uni = X[[feature]]

    auc_scores = cross_val_score(
        model, X_uni, y, cv=cv, scoring="roc_auc"
    )
    acc_scores = cross_val_score(
        model, X_uni, y, cv=cv, scoring="accuracy"
    )

    univ_results.append({
        "feature": feature,
        "mean_auc": auc_scores.mean(),
        "std_auc": auc_scores.std(),
        "mean_accuracy": acc_scores.mean(),
        "std_accuracy": acc_scores.std()
    })

univ_df = pd.DataFrame(univ_results).sort_values(by="mean_auc", ascending=False)

print("\nTop 10 variables univariables con mejor AUC:")
print(univ_df.head(10).to_string(index=False))


# =========================================================
# 4. REVISAR CAPACIDAD DE CLASIFICACIÓN BIVARIABLE
# =========================================================
print("\n" + "=" * 60)
print("CAPACIDAD DE CLASIFICACIÓN BIVARIABLE")
print("=" * 60)

biv_results = []

feature_pairs = list(itertools.combinations(X.columns, 2))

for f1, f2 in feature_pairs:
    X_bi = X[[f1, f2]]

    auc_scores = cross_val_score(
        model, X_bi, y, cv=cv, scoring="roc_auc"
    )
    acc_scores = cross_val_score(
        model, X_bi, y, cv=cv, scoring="accuracy"
    )

    biv_results.append({
        "feature_1": f1,
        "feature_2": f2,
        "mean_auc": auc_scores.mean(),
        "std_auc": auc_scores.std(),
        "mean_accuracy": acc_scores.mean(),
        "std_accuracy": acc_scores.std()
    })

biv_df = pd.DataFrame(biv_results).sort_values(by="mean_auc", ascending=False)

print("\nTop 10 pares de variables con mejor AUC:")
print(biv_df.head(10).to_string(index=False))


# =========================================================
# 5. GRÁFICAS DE DENSIDAD
#    Usaremos las 6 mejores variables univariables
# =========================================================
top_uni_features = univ_df.head(6)["feature"].tolist()

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for ax, feature in zip(axes, top_uni_features):
    df[df["target"] == 0][feature].plot(kind="kde", ax=ax, label=data.target_names[0], linewidth=2)
    df[df["target"] == 1][feature].plot(kind="kde", ax=ax, label=data.target_names[1], linewidth=2)
    ax.set_title(f"Densidad: {feature}")
    ax.set_xlabel(feature)
    ax.set_ylabel("Densidad")
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.suptitle("Gráficas de densidad de las mejores variables univariables", fontsize=16)
plt.tight_layout()
plt.show()


# =========================================================
# 6. GRÁFICAS DE DISPERSIÓN
#    Usaremos los 4 mejores pares bivariables
# =========================================================
top_bi_pairs = biv_df.head(4)[["feature_1", "feature_2"]].values.tolist()

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.flatten()

for ax, (f1, f2) in zip(axes, top_bi_pairs):
    class_0 = df[df["target"] == 0]
    class_1 = df[df["target"] == 1]

    ax.scatter(class_0[f1], class_0[f2], alpha=0.6, label=data.target_names[0])
    ax.scatter(class_1[f1], class_1[f2], alpha=0.6, label=data.target_names[1])

    ax.set_title(f"Dispersión: {f1} vs {f2}")
    ax.set_xlabel(f1)
    ax.set_ylabel(f2)
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.suptitle("Gráficas de dispersión de los mejores pares de variables", fontsize=16)
plt.tight_layout()
plt.show()


# =========================================================
# 7. RESUMEN FINAL AUTOMÁTICO
# =========================================================
print("\n" + "=" * 60)
print("RESUMEN FINAL")
print("=" * 60)

print("\nMejor variable univariable:")
print(univ_df.iloc[0].to_string())

print("\nMejor par bivariable:")
print(biv_df.iloc[0].to_string())

print("\nConclusión rápida:")
print(
    "Se evaluó la capacidad de clasificación usando regresión logística con validación cruzada. "
    "Primero se analizaron variables individuales y luego combinaciones de dos variables. "
    "Las mejores variables y pares se seleccionaron con base en el AUC promedio. "
    "Finalmente, se visualizaron con gráficas de densidad y de dispersión para observar "
    "la separación entre clases."
)