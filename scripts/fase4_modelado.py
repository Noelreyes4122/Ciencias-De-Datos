"""
fase4_modelado.py
=================
CRISP-DM Fase 4: Modelado
--------------------------
Modelos aplicados al dataset integrado de agua potable:

  M1. Regresión Lineal Múltiple   → Predecir INDICE_POTABILIDAD_PCT
  M2. Random Forest Regressor     → Predecir INDICE_POTABILIDAD_PCT
  M3. K-Means Clustering          → Clasificar plantas por nivel de riesgo
  M4. Árbol de Decisión           → Clasificar ALERTA_CALIDAD

Ejecutar: python scripts/fase4_modelado.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection   import train_test_split, cross_val_score
from sklearn.linear_model      import LinearRegression
from sklearn.ensemble          import RandomForestRegressor
from sklearn.tree              import DecisionTreeClassifier, plot_tree
from sklearn.cluster           import KMeans
from sklearn.preprocessing     import StandardScaler, LabelEncoder
from sklearn.metrics           import (mean_squared_error, mean_absolute_error,
                                       r2_score, classification_report,
                                       confusion_matrix, ConfusionMatrixDisplay)
from sklearn.pipeline          import Pipeline
import matplotlib.cm as cm

from config import *
from fase2_limpieza   import ejecutar_limpieza
from fase3_integracion import integrar_datasets


# ══════════════════════════════════════════════════════════════════════════════
# PREPARAR FEATURES PARA MODELADO
# ══════════════════════════════════════════════════════════════════════════════

def preparar_features(df_final):
    """Codifica variables categóricas y construye X, y."""
    df = df_final.dropna(subset=["INDICE_POTABILIDAD_PCT",
                                  "PRODUCCION_MILLONES_M3",
                                  "CLORO_RESIDUAL_MGL"]).copy()

    # Encode PLANTA (One-Hot)
    df_encoded = pd.get_dummies(df, columns=["PLANTA"], prefix="PLT", drop_first=False)

    # Encode MES como número
    df_encoded["MES_NUM"] = df_encoded["MES"].map(MESES_ES)

    # Features para regresión
    feat_cols = (
        ["PRODUCCION_MILLONES_M3", "LOG_PRODUCCION",
         "CLORO_RESIDUAL_MGL", "NUM_MUESTRAS", "AÑO", "MES_NUM"]
        + [c for c in df_encoded.columns if c.startswith("PLT_")]
    )
    feat_cols = [c for c in feat_cols if c in df_encoded.columns]

    X = df_encoded[feat_cols]
    y = df_encoded["INDICE_POTABILIDAD_PCT"]

    # Features para clustering (sin encode)
    X_cluster = df[["PRODUCCION_MILLONES_M3", "INDICE_POTABILIDAD_PCT",
                     "CLORO_RESIDUAL_MGL", "INDICE_EFICIENCIA"]].dropna()

    return X, y, X_cluster, feat_cols, df


# ══════════════════════════════════════════════════════════════════════════════
# M1. REGRESIÓN LINEAL MÚLTIPLE
# ══════════════════════════════════════════════════════════════════════════════

def modelo_regresion_lineal(X_train, X_test, y_train, y_test, feat_cols):
    print("\n── M1: Regresión Lineal Múltiple ──────────────────────────────")

    modelo = LinearRegression()
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)

    print(f"   R²:    {r2:.4f}")
    print(f"   RMSE:  {rmse:.4f}")
    print(f"   MAE:   {mae:.4f}")

    # Coeficientes más importantes
    coefs = pd.Series(modelo.coef_, index=feat_cols).abs().sort_values(ascending=False)
    print(f"   Top 5 variables (por |coeficiente|):")
    for var, val in coefs.head(5).items():
        print(f"     {var:<35} {val:.4f}")

    # Gráfico: Predicho vs Real
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(y_test, y_pred, alpha=0.65, color=COLOR_AZUL2,
               edgecolors="white", linewidth=0.4, s=50, label="Observaciones")
    lim_min = min(y_test.min(), y_pred.min()) - 2
    lim_max = max(y_test.max(), y_pred.max()) + 2
    ax.plot([lim_min, lim_max], [lim_min, lim_max],
            color=COLOR_ROJO, lw=1.8, ls="--", label="Línea perfecta")
    ax.set_xlabel("IP Real (%)", fontsize=12)
    ax.set_ylabel("IP Predicho (%)", fontsize=12)
    ax.set_title(f"M1: Regresión Lineal – IP Real vs Predicho\n(R²={r2:.4f}  RMSE={rmse:.4f})",
                 fontsize=12, fontweight="bold", color=COLOR_AZUL)
    ax.legend(fontsize=10)
    fig.tight_layout()
    ruta = os.path.join(GRAFICOS_DIR, "13_regresion_lineal_real_vs_pred.png")
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   ✓ Gráfico guardado: {ruta}")

    return {"modelo": modelo, "r2": r2, "rmse": rmse, "mae": mae}


# ══════════════════════════════════════════════════════════════════════════════
# M2. RANDOM FOREST REGRESSOR
# ══════════════════════════════════════════════════════════════════════════════

def modelo_random_forest(X_train, X_test, y_train, y_test, feat_cols):
    print("\n── M2: Random Forest Regressor ────────────────────────────────")

    rf = RandomForestRegressor(
        n_estimators=200, max_depth=8,
        random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)

    print(f"   R²:    {r2:.4f}")
    print(f"   RMSE:  {rmse:.4f}")
    print(f"   MAE:   {mae:.4f}")

    # Importancia de variables
    importances = pd.Series(rf.feature_importances_, index=feat_cols).sort_values(ascending=False)
    print(f"   Top 5 variables importantes:")
    for var, val in importances.head(5).items():
        print(f"     {var:<35} {val:.4f}")

    # Gráfico: Feature Importance
    top_n = 10
    top_imp = importances.head(top_n)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.barh(top_imp.index[::-1], top_imp.values[::-1],
            color=COLOR_AZUL, height=0.6, zorder=3)
    ax.set_xlabel("Importancia", fontsize=11)
    ax.set_title(f"M2: Random Forest – Top {top_n} Variables Importantes",
                 fontsize=13, fontweight="bold", color=COLOR_AZUL)
    ax.grid(axis="x", alpha=0.3); ax.set_facecolor("#FAFAFA")
    fig.tight_layout()
    ruta = os.path.join(GRAFICOS_DIR, "14_random_forest_importancia.png")
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   ✓ Gráfico guardado: {ruta}")

    # Gráfico: Real vs Predicho
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(y_test, y_pred, alpha=0.65, color=COLOR_VERDE,
               edgecolors="white", linewidth=0.4, s=50)
    lim_min = min(y_test.min(), y_pred.min()) - 2
    lim_max = max(y_test.max(), y_pred.max()) + 2
    ax.plot([lim_min, lim_max], [lim_min, lim_max],
            color=COLOR_ROJO, lw=1.8, ls="--", label="Línea perfecta")
    ax.set_xlabel("IP Real (%)", fontsize=12)
    ax.set_ylabel("IP Predicho (%)", fontsize=12)
    ax.set_title(f"M2: Random Forest – IP Real vs Predicho\n(R²={r2:.4f}  RMSE={rmse:.4f})",
                 fontsize=12, fontweight="bold", color=COLOR_AZUL)
    ax.legend(fontsize=10)
    fig.tight_layout()
    ruta = os.path.join(GRAFICOS_DIR, "15_random_forest_real_vs_pred.png")
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   ✓ Gráfico guardado: {ruta}")

    return {"modelo": rf, "r2": r2, "rmse": rmse, "mae": mae, "importances": importances}


# ══════════════════════════════════════════════════════════════════════════════
# M3. K-MEANS CLUSTERING
# ══════════════════════════════════════════════════════════════════════════════

def modelo_kmeans(X_cluster):
    print("\n── M3: K-Means Clustering (clasificar plantas por riesgo) ─────")

    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)

    # Elbow method para elegir K
    inercias = []
    rango_k  = range(2, 8)
    for k in rango_k:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inercias.append(km.inertia_)

    # Gráfico Elbow
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(rango_k, inercias, "o-", color=COLOR_AZUL, lw=2.2, markersize=8)
    ax.axvline(3, color=COLOR_ROJO, ls="--", lw=1.5, label="K=3 seleccionado")
    ax.set_xlabel("Número de Clusters (K)", fontsize=12)
    ax.set_ylabel("Inercia (WCSS)", fontsize=12)
    ax.set_title("M3: K-Means – Método del Codo para elegir K",
                 fontsize=13, fontweight="bold", color=COLOR_AZUL)
    ax.legend(); fig.tight_layout()
    ruta = os.path.join(GRAFICOS_DIR, "16_kmeans_elbow.png")
    fig.savefig(ruta, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"   ✓ Gráfico guardado: {ruta}")

    # Modelo final K=3
    K = 3
    km_final = KMeans(n_clusters=K, random_state=42, n_init=10)
    X_cluster = X_cluster.copy()
    X_cluster["CLUSTER"] = km_final.fit_predict(X_scaled)

    # Etiquetas de riesgo basadas en IP promedio por cluster
    ip_por_cluster = X_cluster.groupby("CLUSTER")["INDICE_POTABILIDAD_PCT"].mean()
    orden = ip_por_cluster.sort_values()
    etiquetas = {}
    nombres = ["RIESGO ALTO", "RIESGO MEDIO", "RIESGO BAJO"]
    for i, (cluster, _) in enumerate(orden.items()):
        etiquetas[cluster] = nombres[i]
    X_cluster["NIVEL_RIESGO"] = X_cluster["CLUSTER"].map(etiquetas)

    print(f"   Clusters encontrados: {K}")
    for cl in range(K):
        sub = X_cluster[X_cluster["CLUSTER"] == cl]
        print(f"     Cluster {cl} ({etiquetas[cl]}): {len(sub)} registros | "
              f"IP_prom={sub['INDICE_POTABILIDAD_PCT'].mean():.1f}%  "
              f"Prod_prom={sub['PRODUCCION_MILLONES_M3'].mean():.3f} Mm³")

    # Gráfico Clusters: Scatter IP vs Producción
    colores_cl = {
        "RIESGO ALTO":  COLOR_ROJO,
        "RIESGO MEDIO": COLOR_NARANJA,
        "RIESGO BAJO":  COLOR_VERDE
    }
    fig, ax = plt.subplots(figsize=(9, 5))
    for nivel, grupo in X_cluster.groupby("NIVEL_RIESGO"):
        ax.scatter(grupo["PRODUCCION_MILLONES_M3"], grupo["INDICE_POTABILIDAD_PCT"],
                   color=colores_cl[nivel], label=nivel, s=60, alpha=0.8,
                   edgecolors="white", linewidth=0.4)
    ax.axhline(UMBRAL_IP_BAJO, color=COLOR_ROJO, lw=1.5, ls="--", alpha=0.6)
    ax.set_xlabel("Producción (Millones m³)", fontsize=12)
    ax.set_ylabel("Índice de Potabilidad (%)", fontsize=12)
    ax.set_title("M3: K-Means Clustering – Nivel de Riesgo por Registro\n(K=3)",
                 fontsize=13, fontweight="bold", color=COLOR_AZUL)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3); ax.set_facecolor("#FAFAFA")
    fig.tight_layout()
    ruta = os.path.join(GRAFICOS_DIR, "17_kmeans_clusters.png")
    fig.savefig(ruta, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"   ✓ Gráfico guardado: {ruta}")

    return {"modelo": km_final, "datos": X_cluster, "etiquetas": etiquetas}


# ══════════════════════════════════════════════════════════════════════════════
# M4. ÁRBOL DE DECISIÓN – CLASIFICACIÓN ALERTA
# ══════════════════════════════════════════════════════════════════════════════

def modelo_arbol_decision(df_final):
    print("\n── M4: Árbol de Decisión – Clasificar ALERTA_CALIDAD ──────────")

    df = df_final.dropna(subset=["INDICE_POTABILIDAD_PCT", "PRODUCCION_MILLONES_M3",
                                   "CLORO_RESIDUAL_MGL", "ALERTA_CALIDAD"]).copy()

    # Codificar target
    le = LabelEncoder()
    df["ALERTA_ENC"] = le.fit_transform(df["ALERTA_CALIDAD"])
    clases = le.classes_

    feat_arbol = ["PRODUCCION_MILLONES_M3", "CLORO_RESIDUAL_MGL",
                  "NUM_MUESTRAS", "AÑO"]
    X_a = df[feat_arbol]
    y_a = df["ALERTA_ENC"]

    # Solo entrenar si hay suficientes datos por clase
    conteo = df["ALERTA_CALIDAD"].value_counts()
    print(f"   Distribución clases: {conteo.to_dict()}")

    X_a_train, X_a_test, y_a_train, y_a_test = train_test_split(
        X_a, y_a, test_size=0.25, random_state=42, stratify=y_a
    )

    arbol = DecisionTreeClassifier(max_depth=4, random_state=42,
                                    class_weight="balanced")
    arbol.fit(X_a_train, y_a_train)
    y_a_pred = arbol.predict(X_a_test)

    print("\n   Reporte de clasificación:")
    print(classification_report(y_a_test, y_a_pred,
                                 target_names=clases, zero_division=0))

    # Gráfico Árbol de Decisión
    fig, ax = plt.subplots(figsize=(14, 5))
    plot_tree(arbol, feature_names=feat_arbol, class_names=clases,
              filled=True, rounded=True, fontsize=8, ax=ax,
              impurity=False, proportion=False)
    ax.set_title("M4: Árbol de Decisión – Clasificación ALERTA_CALIDAD",
                 fontsize=13, fontweight="bold", color=COLOR_AZUL)
    fig.tight_layout()
    ruta = os.path.join(GRAFICOS_DIR, "18_arbol_decision.png")
    fig.savefig(ruta, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"   ✓ Gráfico guardado: {ruta}")

    # Gráfico Matriz de Confusión
    fig, ax = plt.subplots(figsize=(6, 5))
    cm_display = ConfusionMatrixDisplay.from_predictions(
        y_a_test, y_a_pred, display_labels=clases,
        cmap="Blues", ax=ax
    )
    ax.set_title("M4: Matriz de Confusión – ALERTA_CALIDAD",
                 fontsize=12, fontweight="bold", color=COLOR_AZUL)
    fig.tight_layout()
    ruta = os.path.join(GRAFICOS_DIR, "19_confusion_matrix.png")
    fig.savefig(ruta, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"   ✓ Gráfico guardado: {ruta}")

    return {"modelo": arbol, "clases": clases, "le": le}


# ══════════════════════════════════════════════════════════════════════════════
# COMPARACIÓN DE MODELOS DE REGRESIÓN
# ══════════════════════════════════════════════════════════════════════════════

def comparar_modelos(res_lr, res_rf):
    """Tabla comparativa M1 vs M2."""
    print("\n" + "=" * 65)
    print("  COMPARACIÓN: M1 (Regresión Lineal) vs M2 (Random Forest)")
    print("=" * 65)
    metricas = ["R²", "RMSE", "MAE"]
    vals_lr  = [res_lr["r2"],  res_lr["rmse"],  res_lr["mae"]]
    vals_rf  = [res_rf["r2"],  res_rf["rmse"],  res_rf["mae"]]
    print(f"  {'Métrica':<8} {'Reg. Lineal':>14} {'Random Forest':>14}  {'Mejor'}")
    print("  " + "-" * 48)
    for m, lr, rf in zip(metricas, vals_lr, vals_rf):
        mejor = "✅ RF" if (rf > lr if m == "R²" else rf < lr) else "✅ LR"
        print(f"  {m:<8} {lr:>14.4f} {rf:>14.4f}  {mejor}")
    print()
    if res_rf["r2"] > res_lr["r2"]:
        print("  → Random Forest supera a la Regresión Lineal.")
        print("    Indica relaciones NO lineales entre producción y calidad.")
    else:
        print("  → Regresión Lineal es suficiente para este dataset.")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 65)
    print("  FASE 4: MODELADO – CRISP-DM")
    print("=" * 65)

    # Cargar dataset integrado
    datasets_limpios = ejecutar_limpieza()
    df_final = integrar_datasets(datasets_limpios)

    # Preparar features
    X, y, X_cluster, feat_cols, df_enc = preparar_features(df_final)
    print(f"\n  Dataset para modelado: {X.shape[0]} registros × {X.shape[1]} features")
    print(f"  Target: INDICE_POTABILIDAD_PCT")
    print(f"  Distribución target: min={y.min():.1f}%  max={y.max():.1f}%  media={y.mean():.2f}%")

    # Train / Test split 75% / 25%
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    print(f"\n  Train: {len(X_train)} | Test: {len(X_test)}")

    # Modelos
    res_lr  = modelo_regresion_lineal(X_train, X_test, y_train, y_test, feat_cols)
    res_rf  = modelo_random_forest(X_train, X_test, y_train, y_test, feat_cols)
    res_km  = modelo_kmeans(X_cluster)
    res_arb = modelo_arbol_decision(df_final)

    # Comparar
    comparar_modelos(res_lr, res_rf)

    print("\n✅  Fase 4 (Modelado) completada.")
    print(f"   Gráficos guardados en: {GRAFICOS_DIR}")
