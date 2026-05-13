"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   💧 ANÁLISIS DE AGUA POTABLE – REPÚBLICA DOMINICANA                        ║
║   CRISP-DM Fase 5: Evaluación                                                ║
║   INF-379-02 Introducción a la Ciencia de Datos                              ║
║   Noel Reyes | 21-2021 | UNPHU 2026                                          ║
╠══════════════════════════════════════════════════════════════════════════════╣
║   CONTENIDO DE ESTA FASE:                                                    ║
║     E1. Validación Cruzada K-Fold (k=10) para M1 y M2                       ║
║     E2. Análisis de Residuos (M1 Reg. Lineal vs M2 Random Forest)           ║
║     E3. Dashboard comparativo visual de todos los modelos                    ║
║     E4. Evaluación frente a objetivos de negocio                             ║
║     E5. Conclusiones y recomendaciones finales                               ║
╠══════════════════════════════════════════════════════════════════════════════╣
║   PREREQUISITO:                                                              ║
║     Ejecutar primero las fases anteriores o tener el dataset integrado:      ║
║     outputs/dataset_agua_integrado_final.csv                                 ║
╠══════════════════════════════════════════════════════════════════════════════╣
║   USO:                                                                       ║
║     python fase5_evaluacion.py                                               ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import sys
import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.model_selection   import train_test_split, cross_val_score, KFold
from sklearn.linear_model      import LinearRegression
from sklearn.ensemble          import RandomForestRegressor
from sklearn.tree              import DecisionTreeClassifier
from sklearn.cluster           import KMeans
from sklearn.preprocessing     import StandardScaler, LabelEncoder
from sklearn.metrics           import (mean_squared_error, mean_absolute_error,
                                       r2_score, classification_report,
                                       accuracy_score, f1_score,
                                       ConfusionMatrixDisplay)


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════════════════════

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR   = os.path.join(BASE_DIR, "outputs")
GRAFICOS_DIR = os.path.join(OUTPUT_DIR, "graficos")
DATASET_PATH = os.path.join(OUTPUT_DIR, "dataset_agua_integrado_final.csv")

os.makedirs(OUTPUT_DIR,   exist_ok=True)
os.makedirs(GRAFICOS_DIR, exist_ok=True)

# Paleta de colores del proyecto
COLOR_AZUL    = "#1B4F8A"
COLOR_AZUL2   = "#2E75B6"
COLOR_ROJO    = "#C0392B"
COLOR_VERDE   = "#1A7A4A"
COLOR_NARANJA = "#D4730A"

# Umbrales de calidad
UMBRAL_IP_BAJO       = 95
UMBRAL_CLORO_MIN     = 0.2
UMBRAL_CLORO_MAX     = 5.0

MESES_ES = {
    "ENERO": 1, "FEBRERO": 2, "MARZO": 3, "ABRIL": 4,
    "MAYO": 5, "JUNIO": 6, "JULIO": 7, "AGOSTO": 8,
    "SEPTIEMBRE": 9, "OCTUBRE": 10, "NOVIEMBRE": 11, "DICIEMBRE": 12
}

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "#FAFAFA",
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "font.family":      "sans-serif",
})


# ══════════════════════════════════════════════════════════════════════════════
# CARGAR Y PREPARAR DATASET INTEGRADO
# ══════════════════════════════════════════════════════════════════════════════

def cargar_dataset():
    """Carga el dataset integrado generado en la Fase 3."""
    if not os.path.exists(DATASET_PATH):
        print(f"❌ No se encontró el dataset en: {DATASET_PATH}")
        print("   Ejecuta primero las fases 2, 3 y 4 del pipeline.")
        sys.exit(1)

    df = pd.read_csv(DATASET_PATH, parse_dates=["FECHA"])
    print(f"✅ Dataset cargado: {df.shape[0]} filas × {df.shape[1]} columnas")
    print(f"   Período: {df['AÑO'].min()} – {df['AÑO'].max()}")
    print(f"   Plantas: {sorted(df['PLANTA'].unique())}\n")
    return df


def preparar_features(df):
    """Prepara las matrices X, y y X_cluster para los modelos."""
    df_modelo = df.dropna(subset=["INDICE_POTABILIDAD_PCT",
                                   "PRODUCCION_MILLONES_M3",
                                   "CLORO_RESIDUAL_MGL"]).copy()

    # One-Hot Encoding de PLANTA
    df_enc = pd.get_dummies(df_modelo, columns=["PLANTA"], prefix="PLT", drop_first=False)
    df_enc["MES_NUM"] = df_enc["MES"].map(MESES_ES)

    feat_cols = (
        ["PRODUCCION_MILLONES_M3", "LOG_PRODUCCION",
         "CLORO_RESIDUAL_MGL", "NUM_MUESTRAS", "AÑO", "MES_NUM"]
        + [c for c in df_enc.columns if c.startswith("PLT_")]
    )
    feat_cols = [c for c in feat_cols if c in df_enc.columns]

    X = df_enc[feat_cols]
    y = df_enc["INDICE_POTABILIDAD_PCT"]

    X_cluster = df_modelo[["PRODUCCION_MILLONES_M3", "INDICE_POTABILIDAD_PCT",
                             "CLORO_RESIDUAL_MGL", "INDICE_EFICIENCIA"]].dropna()
    return X, y, X_cluster, feat_cols, df_modelo


# ══════════════════════════════════════════════════════════════════════════════
# REENTRENAR MODELOS (necesarios para la evaluación)
# ══════════════════════════════════════════════════════════════════════════════

def entrenar_modelos(X, y, X_cluster, feat_cols, df_modelo):
    """Entrena los 4 modelos del proyecto para su evaluación."""
    print("=" * 65)
    print("  ENTRENANDO MODELOS (base para evaluación)")
    print("=" * 65)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    print(f"  Train: {len(X_train)} | Test: {len(X_test)}\n")

    # ── M1: Regresión Lineal ─────────────────────────────────────────────────
    print("── M1: Regresión Lineal Múltiple")
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    yp_lr  = lr.predict(X_test)
    res_lr = {
        "modelo":  lr,
        "y_pred":  yp_lr,
        "y_test":  y_test,
        "r2":      r2_score(y_test, yp_lr),
        "rmse":    np.sqrt(mean_squared_error(y_test, yp_lr)),
        "mae":     mean_absolute_error(y_test, yp_lr),
    }
    print(f"   R²={res_lr['r2']:.4f}  RMSE={res_lr['rmse']:.4f}  MAE={res_lr['mae']:.4f}")

    # ── M2: Random Forest ────────────────────────────────────────────────────
    print("── M2: Random Forest Regressor")
    rf = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    yp_rf  = rf.predict(X_test)
    res_rf = {
        "modelo":       rf,
        "y_pred":       yp_rf,
        "y_test":       y_test,
        "r2":           r2_score(y_test, yp_rf),
        "rmse":         np.sqrt(mean_squared_error(y_test, yp_rf)),
        "mae":          mean_absolute_error(y_test, yp_rf),
        "importances":  pd.Series(rf.feature_importances_, index=feat_cols).sort_values(ascending=False),
    }
    print(f"   R²={res_rf['r2']:.4f}  RMSE={res_rf['rmse']:.4f}  MAE={res_rf['mae']:.4f}")

    # ── M3: K-Means ──────────────────────────────────────────────────────────
    print("── M3: K-Means Clustering (K=3)")
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)
    km       = KMeans(n_clusters=3, random_state=42, n_init=10)
    X_cluster = X_cluster.copy()
    X_cluster["CLUSTER"] = km.fit_predict(X_scaled)
    ip_por_cluster = X_cluster.groupby("CLUSTER")["INDICE_POTABILIDAD_PCT"].mean()
    orden = ip_por_cluster.sort_values()
    etiquetas = {c: n for (c, _), n in zip(orden.items(),
                 ["RIESGO ALTO", "RIESGO MEDIO", "RIESGO BAJO"])}
    X_cluster["NIVEL_RIESGO"] = X_cluster["CLUSTER"].map(etiquetas)
    res_km = {"modelo": km, "datos": X_cluster, "etiquetas": etiquetas, "scaler": scaler}
    for cl in range(3):
        sub = X_cluster[X_cluster["CLUSTER"] == cl]
        print(f"   Cluster {cl} ({etiquetas[cl]}): {len(sub)} registros  "
              f"IP_prom={sub['INDICE_POTABILIDAD_PCT'].mean():.1f}%")

    # ── M4: Árbol de Decisión ─────────────────────────────────────────────────
    print("── M4: Árbol de Decisión (ALERTA_CALIDAD)")
    df_arb = df_modelo.dropna(subset=["INDICE_POTABILIDAD_PCT", "PRODUCCION_MILLONES_M3",
                                       "CLORO_RESIDUAL_MGL", "ALERTA_CALIDAD"]).copy()
    le = LabelEncoder()
    df_arb["ALERTA_ENC"] = le.fit_transform(df_arb["ALERTA_CALIDAD"])
    clases = le.classes_
    feat_arbol = ["PRODUCCION_MILLONES_M3", "CLORO_RESIDUAL_MGL", "NUM_MUESTRAS", "AÑO"]
    Xa_train, Xa_test, ya_train, ya_test = train_test_split(
        df_arb[feat_arbol], df_arb["ALERTA_ENC"],
        test_size=0.25, random_state=42, stratify=df_arb["ALERTA_ENC"]
    )
    arbol = DecisionTreeClassifier(max_depth=4, random_state=42, class_weight="balanced")
    arbol.fit(Xa_train, ya_train)
    ya_pred = arbol.predict(Xa_test)
    res_arb = {"modelo": arbol, "clases": clases, "le": le,
               "y_test": ya_test, "y_pred": ya_pred}
    acc = accuracy_score(ya_test, ya_pred)
    f1  = f1_score(ya_test, ya_pred, average="macro", zero_division=0)
    print(f"   Accuracy={acc:.4f}  F1-macro={f1:.4f}")

    print(f"\n  ✅ Modelos listos para evaluación\n")
    return res_lr, res_rf, res_km, res_arb, X_train, X_test, y_train, y_test


# ══════════════════════════════════════════════════════════════════════════════
# E1: VALIDACIÓN CRUZADA K-FOLD
# ══════════════════════════════════════════════════════════════════════════════

def evaluacion_crossval(X, y):
    """E1: Validación cruzada K-Fold (k=10) para Regresión Lineal y Random Forest."""
    print("=" * 65)
    print("  E1: VALIDACIÓN CRUZADA K-FOLD (k=10)")
    print("=" * 65)

    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    # Regresión Lineal
    cv_r2_lr   = cross_val_score(LinearRegression(), X, y, cv=kf, scoring="r2")
    cv_rmse_lr = np.sqrt(-cross_val_score(LinearRegression(), X, y, cv=kf,
                                           scoring="neg_mean_squared_error"))
    cv_mae_lr  = -cross_val_score(LinearRegression(), X, y, cv=kf,
                                   scoring="neg_mean_absolute_error")

    # Random Forest
    rf_cv = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
    cv_r2_rf   = cross_val_score(rf_cv, X, y, cv=kf, scoring="r2")
    cv_rmse_rf = np.sqrt(-cross_val_score(rf_cv, X, y, cv=kf,
                                           scoring="neg_mean_squared_error"))
    cv_mae_rf  = -cross_val_score(rf_cv, X, y, cv=kf,
                                   scoring="neg_mean_absolute_error")

    print(f"\n  {'Métrica':<10} {'LR Media':>10} {'LR ±std':>10} {'RF Media':>10} {'RF ±std':>10}")
    print("  " + "─" * 55)
    for metrica, lr_m, lr_s, rf_m, rf_s in [
        ("R²",   cv_r2_lr.mean(),   cv_r2_lr.std(),   cv_r2_rf.mean(),   cv_r2_rf.std()),
        ("RMSE", cv_rmse_lr.mean(), cv_rmse_lr.std(), cv_rmse_rf.mean(), cv_rmse_rf.std()),
        ("MAE",  cv_mae_lr.mean(),  cv_mae_lr.std(),  cv_mae_rf.mean(),  cv_mae_rf.std()),
    ]:
        print(f"  {metrica:<10} {lr_m:>10.4f} {lr_s:>10.4f} {rf_m:>10.4f} {rf_s:>10.4f}")

    # Interpretación estabilidad
    print(f"\n  Estabilidad M1 (R² std): {cv_r2_lr.std():.4f} "
          f"→ {'✅ Estable' if cv_r2_lr.std() < 0.1 else '⚠️ Variable'}")
    print(f"  Estabilidad M2 (R² std): {cv_r2_rf.std():.4f} "
          f"→ {'✅ Estable' if cv_r2_rf.std() < 0.1 else '⚠️ Variable'}")

    # Gráfico boxplot CV
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    datos_pairs = [
        (cv_r2_lr,   cv_r2_rf,   "R²",   "mayor = mejor"),
        (cv_rmse_lr, cv_rmse_rf, "RMSE", "menor = mejor"),
        (cv_mae_lr,  cv_mae_rf,  "MAE",  "menor = mejor"),
    ]
    for ax, (datos_lr, datos_rf, metrica, nota) in zip(axes, datos_pairs):
        bp = ax.boxplot([datos_lr, datos_rf], patch_artist=True,
                        labels=["Reg. Lineal\n(M1)", "Random Forest\n(M2)"],
                        notch=False, widths=0.4)
        bp["boxes"][0].set_facecolor(COLOR_AZUL2); bp["boxes"][0].set_alpha(0.75)
        bp["boxes"][1].set_facecolor(COLOR_VERDE);  bp["boxes"][1].set_alpha(0.75)
        for med in bp["medians"]:
            med.set_color("black"); med.set_linewidth(2.2)
        for elem in ["whiskers", "caps", "fliers"]:
            for item in bp[elem]: item.set_color("#555555")
        ax.set_title(f"{metrica}\n({nota})", fontsize=11, fontweight="bold", color=COLOR_AZUL)
        ax.set_ylabel(metrica, fontsize=10)

    fig.suptitle("E1: Validación Cruzada K-Fold (k=10) – M1 Reg. Lineal vs M2 Random Forest\n"
                 "Proyecto Agua Potable RD | Noel Reyes | UNPHU 2026",
                 fontsize=13, fontweight="bold", color=COLOR_AZUL)
    plt.tight_layout()
    ruta = os.path.join(GRAFICOS_DIR, "E1_cv_kfold_comparacion.png")
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  ✓ Gráfico guardado: {ruta}\n")

    return {
        "cv_r2_lr": cv_r2_lr, "cv_rmse_lr": cv_rmse_lr, "cv_mae_lr": cv_mae_lr,
        "cv_r2_rf": cv_r2_rf, "cv_rmse_rf": cv_rmse_rf, "cv_mae_rf": cv_mae_rf,
    }


# ══════════════════════════════════════════════════════════════════════════════
# E2: ANÁLISIS DE RESIDUOS
# ══════════════════════════════════════════════════════════════════════════════

def evaluacion_residuos(res_lr, res_rf):
    """E2: Análisis de residuos para M1 y M2."""
    print("=" * 65)
    print("  E2: ANÁLISIS DE RESIDUOS")
    print("=" * 65)

    y_test   = res_lr["y_test"].values
    y_pred_lr = res_lr["y_pred"]
    y_pred_rf = res_rf["y_pred"]

    residuos_lr = y_test - y_pred_lr
    residuos_rf = y_test - y_pred_rf

    print(f"\n  M1 Regresión Lineal:")
    print(f"     Media residuos:   {residuos_lr.mean():.4f}")
    print(f"     Std  residuos:    {residuos_lr.std():.4f}")
    print(f"     Mín / Máx:        {residuos_lr.min():.4f} / {residuos_lr.max():.4f}")
    print(f"     Evaluación:       "
          f"{'✅ Bien centrado (sin sesgo)' if abs(residuos_lr.mean()) < 1 else '⚠️ Sesgo detectado'}")

    print(f"\n  M2 Random Forest:")
    print(f"     Media residuos:   {residuos_rf.mean():.4f}")
    print(f"     Std  residuos:    {residuos_rf.std():.4f}")
    print(f"     Mín / Máx:        {residuos_rf.min():.4f} / {residuos_rf.max():.4f}")
    print(f"     Evaluación:       "
          f"{'✅ Bien centrado (sin sesgo)' if abs(residuos_rf.mean()) < 1 else '⚠️ Sesgo detectado'}")

    # Comparación de varianza de residuos
    mejora_std = ((residuos_lr.std() - residuos_rf.std()) / residuos_lr.std()) * 100
    print(f"\n  Reducción de varianza (M1→M2): {mejora_std:.1f}%"
          f"  {'✅ RF mejora la dispersión' if mejora_std > 0 else '⚠️ LR tiene menor dispersión'}")

    # Figura 2×2: scatter + histograma para cada modelo
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    for row, (residuos, y_pred, color, nombre) in enumerate([
        (residuos_lr, y_pred_lr, COLOR_AZUL2, "M1: Regresión Lineal"),
        (residuos_rf, y_pred_rf, COLOR_VERDE,  "M2: Random Forest"),
    ]):
        # Scatter residuos vs predicho
        axes[row, 0].scatter(y_pred, residuos, alpha=0.6, color=color,
                              edgecolors="white", linewidth=0.3, s=55)
        axes[row, 0].axhline(0, color=COLOR_ROJO, lw=1.8, ls="--", label="Residuo=0")
        axes[row, 0].axhline(residuos.mean(), color=COLOR_NARANJA, lw=1.4, ls=":",
                              label=f"Media={residuos.mean():.3f}")
        axes[row, 0].set_xlabel("IP Predicho (%)", fontsize=10)
        axes[row, 0].set_ylabel("Residuo (Real – Predicho)", fontsize=10)
        axes[row, 0].set_title(f"{nombre}\nResiduos vs Predicho",
                                fontweight="bold", color=COLOR_AZUL, fontsize=11)
        axes[row, 0].legend(fontsize=9)

        # Histograma de residuos
        axes[row, 1].hist(residuos, bins=25, color=color, edgecolor="white", alpha=0.82)
        axes[row, 1].axvline(0, color=COLOR_ROJO, lw=1.8, ls="--", label="Residuo=0")
        axes[row, 1].axvline(residuos.mean(), color=COLOR_NARANJA, lw=1.4, ls=":",
                              label=f"Media={residuos.mean():.3f}")
        axes[row, 1].set_xlabel("Residuo", fontsize=10)
        axes[row, 1].set_ylabel("Frecuencia", fontsize=10)
        axes[row, 1].set_title(f"{nombre}\nDistribución de Residuos  (std={residuos.std():.3f})",
                                fontweight="bold", color=COLOR_AZUL, fontsize=11)
        axes[row, 1].legend(fontsize=9)

    fig.suptitle("E2: Análisis de Residuos – M1 Regresión Lineal vs M2 Random Forest\n"
                 "Proyecto Agua Potable RD | Noel Reyes | UNPHU 2026",
                 fontsize=13, fontweight="bold", color=COLOR_AZUL)
    plt.tight_layout()
    ruta = os.path.join(GRAFICOS_DIR, "E2_analisis_residuos.png")
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  ✓ Gráfico guardado: {ruta}\n")

    return {"residuos_lr": residuos_lr, "residuos_rf": residuos_rf}


# ══════════════════════════════════════════════════════════════════════════════
# E3: DASHBOARD COMPARATIVO DE MODELOS
# ══════════════════════════════════════════════════════════════════════════════

def evaluacion_dashboard(res_lr, res_rf, res_km, res_arb, cv_results):
    """E3: Dashboard visual comparativo de todos los modelos."""
    print("=" * 65)
    print("  E3: DASHBOARD COMPARATIVO DE MODELOS")
    print("=" * 65)

    cv_r2_lr = cv_results["cv_r2_lr"]
    cv_r2_rf = cv_results["cv_r2_rf"]

    acc_arb = accuracy_score(res_arb["y_test"], res_arb["y_pred"])
    f1_arb  = f1_score(res_arb["y_test"], res_arb["y_pred"], average="macro", zero_division=0)
    f1_por_clase = f1_score(res_arb["y_test"], res_arb["y_pred"],
                             average=None, zero_division=0,
                             labels=range(len(res_arb["clases"])))

    fig = plt.figure(figsize=(16, 11))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.50, wspace=0.42)

    # ── Panel 1: R² comparativo ──────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    nombres = ["Reg. Lineal\n(M1)", "Random Forest\n(M2)"]
    r2_vals = [res_lr["r2"], res_rf["r2"]]
    bars = ax1.bar(nombres, r2_vals, color=[COLOR_AZUL2, COLOR_VERDE], width=0.45, zorder=3)
    for b, v in zip(bars, r2_vals):
        ax1.text(b.get_x() + b.get_width() / 2, v + 0.008,
                 f"{v:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax1.set_ylim(0, 1.1); ax1.set_ylabel("R²", fontsize=11)
    ax1.set_title("R² en Test Set\n(mayor = mejor)", fontweight="bold", color=COLOR_AZUL)

    # ── Panel 2: RMSE y MAE ───────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    x    = np.arange(2)
    ancho = 0.35
    rmse_vals = [res_lr["rmse"], res_rf["rmse"]]
    mae_vals  = [res_lr["mae"],  res_rf["mae"]]
    b1 = ax2.bar(x - ancho / 2, rmse_vals, ancho, color=COLOR_AZUL2, label="RMSE", alpha=0.85, zorder=3)
    b2 = ax2.bar(x + ancho / 2, mae_vals,  ancho, color=COLOR_NARANJA, label="MAE",  alpha=0.85, zorder=3)
    for b, v in list(zip(b1, rmse_vals)) + list(zip(b2, mae_vals)):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.02,
                 f"{v:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax2.set_xticks(x); ax2.set_xticklabels(["Reg. Lineal\n(M1)", "Random Forest\n(M2)"])
    ax2.set_ylabel("Error", fontsize=11); ax2.legend(fontsize=9)
    ax2.set_title("RMSE y MAE en Test Set\n(menor = mejor)", fontweight="bold", color=COLOR_AZUL)

    # ── Panel 3: CV R² con barras de error ────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    cv_medias = [cv_r2_lr.mean(), cv_r2_rf.mean()]
    cv_stds   = [cv_r2_lr.std(),  cv_r2_rf.std()]
    bars_cv = ax3.bar(nombres, cv_medias, yerr=cv_stds, capsize=10,
                       color=[COLOR_AZUL2, COLOR_VERDE], width=0.45, zorder=3,
                       error_kw={"elinewidth": 2.2, "ecolor": COLOR_ROJO, "capthick": 2})
    for b, v in zip(bars_cv, cv_medias):
        ax3.text(b.get_x() + b.get_width() / 2, v + 0.015,
                 f"{v:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax3.set_ylim(0, 1.15); ax3.set_ylabel("R² CV (±std)", fontsize=11)
    ax3.set_title("R² Validación Cruzada K-Fold\n(k=10, barras de error = ±std)",
                   fontweight="bold", color=COLOR_AZUL)

    # ── Panel 4: Distribución clusters K-Means ────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    conteo_clusters = res_km["datos"]["NIVEL_RIESGO"].value_counts()
    colores_km = [{"RIESGO BAJO": COLOR_VERDE, "RIESGO MEDIO": COLOR_NARANJA,
                    "RIESGO ALTO": COLOR_ROJO}.get(k, COLOR_AZUL)
                  for k in conteo_clusters.index]
    wedges, texts, autotexts = ax4.pie(
        conteo_clusters.values, labels=conteo_clusters.index,
        autopct="%1.1f%%", colors=colores_km, startangle=90, pctdistance=0.72
    )
    for t in autotexts: t.set_fontsize(10); t.set_fontweight("bold")
    ax4.set_title("M3: K-Means (K=3)\nDistribución por Nivel de Riesgo",
                   fontweight="bold", color=COLOR_AZUL)

    # ── Panel 5: F1-Score por clase (Árbol de Decisión) ──────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    clases = res_arb["clases"]
    colores_arb = [COLOR_ROJO, COLOR_NARANJA, COLOR_VERDE][:len(clases)]
    bars_arb = ax5.bar(clases, f1_por_clase, color=colores_arb, width=0.45, zorder=3)
    for b, v in zip(bars_arb, f1_por_clase):
        ax5.text(b.get_x() + b.get_width() / 2, v + 0.012,
                 f"{v:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax5.set_ylim(0, 1.2); ax5.set_ylabel("F1-Score", fontsize=11)
    ax5.set_title(f"M4: Árbol de Decisión\nF1 por clase  (Acc={acc_arb:.3f} | F1-macro={f1_arb:.3f})",
                   fontweight="bold", color=COLOR_AZUL)

    # ── Panel 6: Matriz de confusión del árbol ────────────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    ConfusionMatrixDisplay.from_predictions(
        res_arb["y_test"], res_arb["y_pred"],
        display_labels=clases, cmap="Blues", ax=ax6, colorbar=False
    )
    ax6.set_title("M4: Matriz de Confusión\nÁrbol de Decisión – ALERTA_CALIDAD",
                   fontweight="bold", color=COLOR_AZUL, fontsize=10)

    fig.suptitle(
        "E3: Dashboard de Evaluación – Todos los Modelos\n"
        "Proyecto Análisis de Agua Potable RD | Noel Reyes 21-2021 | UNPHU 2026",
        fontsize=14, fontweight="bold", color=COLOR_AZUL
    )
    ruta = os.path.join(GRAFICOS_DIR, "E3_dashboard_evaluacion.png")
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Gráfico guardado: {ruta}\n")


# ══════════════════════════════════════════════════════════════════════════════
# E4: EVALUACIÓN FRENTE A OBJETIVOS DE NEGOCIO
# ══════════════════════════════════════════════════════════════════════════════

def evaluacion_objetivos(res_lr, res_rf, res_km, res_arb, df):
    """E4: Verifica si los modelos cumplen los objetivos del negocio."""
    print("=" * 65)
    print("  E4: EVALUACIÓN FRENTE A OBJETIVOS DE NEGOCIO")
    print("=" * 65)

    mejor_r2   = max(res_lr["r2"], res_rf["r2"])
    mejor_rmse = min(res_lr["rmse"], res_rf["rmse"])
    pct_oms    = df["CUMPLE_OMS_CLORO"].mean() * 100
    ip_global  = df["INDICE_POTABILIDAD_PCT"].mean()
    pct_normal = (df["ALERTA_CALIDAD"] == "NORMAL").mean() * 100
    acc_arb    = accuracy_score(res_arb["y_test"], res_arb["y_pred"])
    f1_arb     = f1_score(res_arb["y_test"], res_arb["y_pred"], average="macro", zero_division=0)

    objetivos = [
        ("OBJ-1", "Predecir IP con R² > 0.70",
         mejor_r2 > 0.70,
         f"Mejor R²={mejor_r2:.4f}  "
         f"({'Random Forest' if res_rf['r2'] > res_lr['r2'] else 'Reg. Lineal'})"),

        ("OBJ-2", "RMSE de predicción < 5.0%",
         mejor_rmse < 5.0,
         f"Mejor RMSE={mejor_rmse:.4f}%"),

        ("OBJ-3", "Cumplimiento norma OMS de Cloro > 70%",
         pct_oms > 70,
         f"{pct_oms:.1f}% de registros dentro de norma OMS (0.2–5.0 mg/L)"),

        ("OBJ-4", "Segmentar plantas en 3 niveles de riesgo",
         len(res_km["etiquetas"]) == 3,
         f"Niveles: {list(res_km['etiquetas'].values())}"),

        ("OBJ-5", "Clasificar alertas con Accuracy > 0.70",
         acc_arb > 0.70,
         f"Accuracy={acc_arb:.4f}  F1-macro={f1_arb:.4f}"),

        ("OBJ-6", "IP promedio histórica ≥ 95% (norma NORMAL)",
         ip_global >= UMBRAL_IP_BAJO,
         f"IP global = {ip_global:.2f}%  "
         f"({'✅ Dentro de norma' if ip_global >= UMBRAL_IP_BAJO else '⚠️ Por debajo del umbral'})"),

        ("OBJ-7", "Más del 80% de registros en estado NORMAL",
         pct_normal >= 80,
         f"{pct_normal:.1f}% de registros en estado NORMAL"),
    ]

    print(f"\n  {'Código':<8} {'Objetivo':<48} {'Estado'}")
    print("  " + "─" * 85)
    cumplidos = 0
    for cod, desc, cumple, detalle in objetivos:
        estado = "✅ CUMPLIDO    " if cumple else "❌ NO CUMPLIDO"
        if cumple: cumplidos += 1
        print(f"  {cod:<8} {desc:<48} {estado}")
        print(f"           └─ {detalle}")

    pct_exito = (cumplidos / len(objetivos)) * 100
    print(f"\n  {'─'*55}")
    print(f"  Objetivos cumplidos: {cumplidos} / {len(objetivos)}  →  Éxito: {pct_exito:.0f}%")

    return objetivos, cumplidos, pct_exito


# ══════════════════════════════════════════════════════════════════════════════
# E5: CONCLUSIONES Y RECOMENDACIONES
# ══════════════════════════════════════════════════════════════════════════════

def evaluacion_conclusiones(res_lr, res_rf, res_km, res_arb,
                             cv_results, df, objetivos, cumplidos, pct_exito):
    """E5: Conclusiones finales y recomendaciones del proyecto."""
    print("\n" + "=" * 65)
    print("  E5: CONCLUSIONES Y RECOMENDACIONES FINALES")
    print("=" * 65)

    modelo_ganador = "Random Forest (M2)" if res_rf["r2"] > res_lr["r2"] else "Regresión Lineal (M1)"
    r2_ganador     = max(res_lr["r2"], res_rf["r2"])
    rmse_ganador   = min(res_lr["rmse"], res_rf["rmse"])
    ip_global      = df["INDICE_POTABILIDAD_PCT"].mean()
    pct_oms        = df["CUMPLE_OMS_CLORO"].mean() * 100
    pct_normal     = (df["ALERTA_CALIDAD"] == "NORMAL").mean() * 100
    acc_arb        = accuracy_score(res_arb["y_test"], res_arb["y_pred"])
    cv_r2_lr       = cv_results["cv_r2_lr"]
    cv_r2_rf       = cv_results["cv_r2_rf"]

    texto = f"""
  ┌─────────────────────────────────────────────────────────────┐
  │         HALLAZGOS PRINCIPALES DEL PROYECTO                  │
  └─────────────────────────────────────────────────────────────┘

  1. MODELO DE PREDICCIÓN DE IP (M1 vs M2):
     ─────────────────────────────────────────
     El {modelo_ganador} fue el mejor modelo para predecir el
     Índice de Potabilidad (R²={r2_ganador:.4f}, RMSE={rmse_ganador:.4f}%).
     {"Los patrones no lineales entre producción y calidad justifican el Random Forest."
      if "Forest" in modelo_ganador else
      "Las relaciones lineales son suficientes para explicar el IP."}

     Validación cruzada K-Fold (k=10):
       M1 R² = {cv_r2_lr.mean():.4f} ± {cv_r2_lr.std():.4f}
       M2 R² = {cv_r2_rf.mean():.4f} ± {cv_r2_rf.std():.4f}
     {"→ M2 generaliza mejor y con mayor estabilidad." 
      if cv_r2_rf.mean() > cv_r2_lr.mean() else 
      "→ M1 es competitivo y más interpretable."}

  2. CALIDAD DEL AGUA – CLORO RESIDUAL (M3 K-Means):
     ─────────────────────────────────────────────────
     El {pct_oms:.1f}% de los registros cumple la norma OMS de Cloro
     Residual (0.2–5.0 mg/L). Se corrigió un outlier crítico de 100 mg/L
     durante la preparación de datos.
     El clustering K-Means identificó 3 niveles de riesgo operativo:
     {list(res_km['etiquetas'].values())}, lo que permite
     priorizar intervenciones en plantas con mayor riesgo.

  3. ÍNDICE DE POTABILIDAD HISTÓRICO:
     ─────────────────────────────────
     La IP promedio global es {ip_global:.2f}%.
     El {pct_normal:.1f}% de los registros mensuales está en estado NORMAL.
     {"⚠️  La IP promedio está por debajo del umbral NORMAL (95%)."
      if ip_global < UMBRAL_IP_BAJO else
      "✅ La IP promedio histórica está dentro del rango aceptable (≥ 95%)."}

  4. CLASIFICACIÓN DE ALERTAS (M4 Árbol de Decisión):
     ──────────────────────────────────────────────────
     El árbol de profundidad 4 clasifica las alertas NORMAL / BAJO / CRÍTICO
     con una Accuracy de {acc_arb:.4f}. Las variables más influyentes
     son el Cloro Residual, el Año y la Producción mensual.

  5. RESUMEN DE OBJETIVOS:
     ──────────────────────
     Se cumplieron {cumplidos}/{len(objetivos)} objetivos de negocio ({pct_exito:.0f}% de éxito).

  ┌─────────────────────────────────────────────────────────────┐
  │              RECOMENDACIONES OPERATIVAS                     │
  └─────────────────────────────────────────────────────────────┘

  R1. Implementar el {modelo_ganador} como sistema de alerta
      temprana mensual del IP por planta.

  R2. Establecer monitoreo semanal de Cloro Residual en las plantas
      clasificadas como RIESGO ALTO por el modelo K-Means.

  R3. Ampliar la integración de datos con INAPA e INDRHI para
      enriquecer el modelo en futuras iteraciones CRISP-DM.

  R4. Desarrollar un dashboard interactivo para que los operadores
      de CORAASAN consulten alertas de calidad en tiempo real.

  R5. Incorporar variables de mantenimiento de planta y condiciones
      climáticas como predictores en el próximo ciclo CRISP-DM.

  R6. Reducir el tiempo de respuesta de análisis INDRHI: el
      {df['INDICE_EFICIENCIA'].describe()['count']:.0f} registros
      integrados muestran que la eficiencia varía significativamente
      entre plantas y semestres.
"""
    print(texto)

    # Guardar informe en texto
    ruta_informe = os.path.join(OUTPUT_DIR, "informe_fase5_evaluacion.txt")
    with open(ruta_informe, "w", encoding="utf-8") as f:
        f.write("=" * 65 + "\n")
        f.write("  INFORME FASE 5 CRISP-DM – EVALUACIÓN\n")
        f.write("  Análisis de Agua Potable RD\n")
        f.write("  Noel Reyes | 21-2021 | UNPHU 2026\n")
        f.write("=" * 65 + "\n\n")
        f.write("MÉTRICAS DE MODELOS\n")
        f.write("─" * 40 + "\n")
        f.write(f"M1 Regresión Lineal : R²={res_lr['r2']:.4f}  RMSE={res_lr['rmse']:.4f}  MAE={res_lr['mae']:.4f}\n")
        f.write(f"M2 Random Forest    : R²={res_rf['r2']:.4f}  RMSE={res_rf['rmse']:.4f}  MAE={res_rf['mae']:.4f}\n")
        f.write(f"M3 K-Means (K=3)    : Niveles → {list(res_km['etiquetas'].values())}\n")
        f.write(f"M4 Árbol Decisión   : Accuracy={acc_arb:.4f}  "
                f"F1-macro={f1_score(res_arb['y_test'], res_arb['y_pred'], average='macro', zero_division=0):.4f}\n\n")
        f.write("VALIDACIÓN CRUZADA K-FOLD (k=10)\n")
        f.write("─" * 40 + "\n")
        f.write(f"M1 R² CV: {cv_r2_lr.mean():.4f} ± {cv_r2_lr.std():.4f}\n")
        f.write(f"M2 R² CV: {cv_r2_rf.mean():.4f} ± {cv_r2_rf.std():.4f}\n\n")
        f.write("OBJETIVOS DE NEGOCIO\n")
        f.write("─" * 40 + "\n")
        for cod, desc, cumple, detalle in objetivos:
            f.write(f"{cod}: [{'CUMPLIDO' if cumple else 'NO CUMPLIDO'}] {desc}\n"
                    f"      → {detalle}\n")
        f.write(f"\nÉXITO: {cumplidos}/{len(objetivos)} ({pct_exito:.0f}%)\n\n")
        f.write(texto)

    print(f"  💾 Informe guardado: {ruta_informe}\n")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import time
    t0 = time.time()

    print("=" * 65)
    print("  💧 FASE 5 CRISP-DM: EVALUACIÓN")
    print("     Análisis de Agua Potable – República Dominicana")
    print("     Noel Reyes | 21-2021 | UNPHU 2026")
    print("=" * 65)
    print()

    # ── 0. Cargar dataset integrado ──────────────────────────────────────────
    df = cargar_dataset()

    # ── 1. Preparar features ─────────────────────────────────────────────────
    X, y, X_cluster, feat_cols, df_modelo = preparar_features(df)
    print(f"  Features: {X.shape[1]} | Registros: {X.shape[0]}\n")

    # ── 2. Entrenar modelos ───────────────────────────────────────────────────
    res_lr, res_rf, res_km, res_arb, X_train, X_test, y_train, y_test = \
        entrenar_modelos(X, y, X_cluster, feat_cols, df_modelo)

    # ── E1: Validación cruzada ────────────────────────────────────────────────
    cv_results = evaluacion_crossval(X, y)

    # ── E2: Análisis de residuos ──────────────────────────────────────────────
    evaluacion_residuos(res_lr, res_rf)

    # ── E3: Dashboard comparativo ─────────────────────────────────────────────
    evaluacion_dashboard(res_lr, res_rf, res_km, res_arb, cv_results)

    # ── E4: Objetivos de negocio ──────────────────────────────────────────────
    objetivos, cumplidos, pct_exito = evaluacion_objetivos(res_lr, res_rf, res_km, res_arb, df_modelo)

    # ── E5: Conclusiones ──────────────────────────────────────────────────────
    evaluacion_conclusiones(res_lr, res_rf, res_km, res_arb,
                             cv_results, df_modelo, objetivos, cumplidos, pct_exito)

    print("=" * 65)
    print("  ✅ FASE 5 (EVALUACIÓN) COMPLETADA")
    print(f"     ⏱  Tiempo: {time.time() - t0:.1f}s")
    print(f"     📊 Gráficos → {GRAFICOS_DIR}/")
    print(f"          E1_cv_kfold_comparacion.png")
    print(f"          E2_analisis_residuos.png")
    print(f"          E3_dashboard_evaluacion.png")
    print(f"     📄 Informe  → {OUTPUT_DIR}/informe_fase5_evaluacion.txt")
    print(f"     🎯 Éxito    → {cumplidos}/{len(objetivos)} objetivos ({pct_exito:.0f}%)")
    print("=" * 65)
