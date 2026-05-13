"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   💧 ANÁLISIS DE AGUA POTABLE – REPÚBLICA DOMINICANA                        ║
║   CRISP-DM – Pipeline Completo (Fases 2 a 5)                                ║
║   INF-379-02 Introducción a la Ciencia de Datos                              ║
║   Noel Reyes | 21-2021 | UNPHU 2026                                          ║
╠══════════════════════════════════════════════════════════════════════════════╣
║   DATASET:                                                                   ║
║   C:/Users/noelr/Documents/Unphu class/Ene-april 2026/                      ║
║   Ciencias De Datos/Proyecto fInal/proyecto_agua/outputs/                    ║
║   dataset_agua_integrado_final.csv                                           ║
╠══════════════════════════════════════════════════════════════════════════════╣
║   USO:                                                                       ║
║     python pipeline_completo.py                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")          # ← evita el error de tkinter en Windows

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.model_selection  import train_test_split, cross_val_score, KFold
from sklearn.linear_model     import LinearRegression
from sklearn.ensemble         import RandomForestRegressor
from sklearn.tree             import DecisionTreeClassifier, plot_tree
from sklearn.cluster          import KMeans
from sklearn.preprocessing    import StandardScaler, LabelEncoder
from sklearn.metrics          import (mean_squared_error, mean_absolute_error,
                                      r2_score, classification_report,
                                      accuracy_score, f1_score,
                                      ConfusionMatrixDisplay)


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN – RUTAS
# ══════════════════════════════════════════════════════════════════════════════

DATASET_PATH = (
    r"C:\Users\noelr\Documents\Unphu class\Ene-april 2026"
    r"\Ciencias De Datos\Proyecto fInal\proyecto_agua\outputs"
    r"\dataset_agua_integrado_final.csv"
)

# Carpeta de gráficos (junto al dataset)
GRAFICOS_DIR = os.path.join(os.path.dirname(DATASET_PATH), "graficos")
os.makedirs(GRAFICOS_DIR, exist_ok=True)

# ── Paleta de colores ─────────────────────────────────────────────────────────
COLOR_AZUL    = "#1B4F8A"
COLOR_AZUL2   = "#2E75B6"
COLOR_ROJO    = "#C0392B"
COLOR_VERDE   = "#1A7A4A"
COLOR_NARANJA = "#D4730A"

# ── Umbrales de calidad ───────────────────────────────────────────────────────
UMBRAL_IP_BAJO       = 95
UMBRAL_IP_CRITICO    = 80
UMBRAL_CLORO_MIN     = 0.2
UMBRAL_CLORO_MAX     = 5.0

# ── Orden de meses ────────────────────────────────────────────────────────────
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
# 0. CARGA DEL DATASET
# ══════════════════════════════════════════════════════════════════════════════

def cargar_dataset():
    """Carga y prepara el dataset integrado final."""
    if not os.path.exists(DATASET_PATH):
        print(f"❌  No se encontró el archivo:\n    {DATASET_PATH}")
        sys.exit(1)

    df = pd.read_csv(DATASET_PATH, parse_dates=["FECHA"])
    df["MES_NUM"] = df["MES"].map(MESES_ES)

    print(f"✅  Dataset cargado: {df.shape[0]} filas × {df.shape[1]} columnas")
    print(f"    Plantas : {sorted(df['PLANTA'].unique())}")
    print(f"    Período : {df['AÑO'].min()} – {df['AÑO'].max()}")
    print(f"    Alertas : {df['ALERTA_CALIDAD'].value_counts().to_dict()}\n")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# FASE 2 – COMPRENSIÓN DE LOS DATOS
# ══════════════════════════════════════════════════════════════════════════════

def fase2_comprension(df):
    print("\n" + "━" * 65)
    print("  FASE 2 CRISP-DM: Comprensión de los Datos")
    print("━" * 65)

    # ── Estadísticas descriptivas ─────────────────────────────────────────────
    vars_num = ["PRODUCCION_MILLONES_M3", "INDICE_POTABILIDAD_PCT",
                "CLORO_RESIDUAL_MGL", "NUM_MUESTRAS", "INDICE_EFICIENCIA"]

    print("\n📊 Estadísticas descriptivas:")
    print(df[vars_num].describe().round(4).to_string())

    print(f"\n📋 Valores nulos por columna:")
    nulos = df.isnull().sum()
    for col, n in nulos[nulos > 0].items():
        print(f"    {col:<25} {n} nulos")

    print(f"\n🌿 IP promedio por planta:")
    for planta, val in df.groupby("PLANTA")["INDICE_POTABILIDAD_PCT"].mean().items():
        estado = "✅ OK" if val >= UMBRAL_IP_BAJO else ("⚠️ BAJO" if val >= UMBRAL_IP_CRITICO else "❌ CRÍTICO")
        print(f"    {planta:<22} {val:.2f}%  {estado}")

    print(f"\n📅 IP promedio por año:")
    for año, val in df.groupby("AÑO")["INDICE_POTABILIDAD_PCT"].mean().items():
        print(f"    {año}: {val:.2f}%  {'✅' if val >= UMBRAL_IP_BAJO else '⚠️'}")

    print(f"\n💧 Cumplimiento norma OMS Cloro: "
          f"{df['CUMPLE_OMS_CLORO'].mean()*100:.1f}% de registros")

    # ── Gráficos ──────────────────────────────────────────────────────────────
    print("\n  Generando gráficos Fase 2...")

    # Fig 1: IP promedio por año
    fig, ax = plt.subplots(figsize=(10, 4))
    ip_año = df.groupby("AÑO")["INDICE_POTABILIDAD_PCT"].mean()
    colores = [COLOR_ROJO if v < UMBRAL_IP_BAJO else COLOR_AZUL for v in ip_año.values]
    bars = ax.bar(ip_año.index, ip_año.values, color=colores, width=0.6, zorder=3)
    ax.axhline(UMBRAL_IP_BAJO, color=COLOR_ROJO, lw=1.8, ls="--", label=f"Umbral {UMBRAL_IP_BAJO}%")
    for b, v in zip(bars, ip_año.values):
        ax.text(b.get_x() + b.get_width()/2, v + 0.3, f"{v:.1f}%",
                ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_ylim(80, 103)
    ax.set_xlabel("Año", fontsize=11); ax.set_ylabel("IP Promedio (%)", fontsize=11)
    ax.set_title("Índice de Potabilidad Promedio por Año – CORAASAN (2018–2026)",
                 fontsize=13, fontweight="bold", color=COLOR_AZUL)
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(GRAFICOS_DIR, "F2_01_ip_por_ano.png"), dpi=150, bbox_inches="tight")
    plt.close(fig); print("  ✓ F2_01_ip_por_ano.png")

    # Fig 2: IP promedio por planta
    fig, ax = plt.subplots(figsize=(10, 4))
    ip_planta = df.groupby("PLANTA")["INDICE_POTABILIDAD_PCT"].mean().sort_values()
    colores_p = [COLOR_ROJO if v < UMBRAL_IP_CRITICO else
                 (COLOR_NARANJA if v < UMBRAL_IP_BAJO else COLOR_AZUL2)
                 for v in ip_planta.values]
    hbars = ax.barh(ip_planta.index, ip_planta.values, color=colores_p, height=0.55, zorder=3)
    ax.axvline(UMBRAL_IP_BAJO,     color=COLOR_ROJO,    lw=1.8, ls="--", label=f"Umbral {UMBRAL_IP_BAJO}%")
    ax.axvline(UMBRAL_IP_CRITICO,  color=COLOR_NARANJA, lw=1.4, ls=":",  label=f"Umbral {UMBRAL_IP_CRITICO}%")
    for b, v in zip(hbars, ip_planta.values):
        ax.text(v + 0.3, b.get_y() + b.get_height()/2, f"{v:.1f}%",
                va="center", fontsize=9, fontweight="bold")
    ax.set_xlim(0, 108)
    ax.set_xlabel("IP Promedio (%)", fontsize=11)
    ax.set_title("IP Promedio por Planta – CORAASAN",
                 fontsize=13, fontweight="bold", color=COLOR_AZUL)
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(GRAFICOS_DIR, "F2_02_ip_por_planta.png"), dpi=150, bbox_inches="tight")
    plt.close(fig); print("  ✓ F2_02_ip_por_planta.png")

    # Fig 3: Producción mensual por planta
    fig, ax = plt.subplots(figsize=(10, 4))
    prod_planta = df.groupby("PLANTA")["PRODUCCION_MILLONES_M3"].mean().sort_values(ascending=False)
    bars = ax.bar(prod_planta.index, prod_planta.values,
                  color=[COLOR_AZUL if i == 0 else COLOR_AZUL2 for i in range(len(prod_planta))],
                  width=0.55, zorder=3)
    for b, v in zip(bars, prod_planta.values):
        ax.text(b.get_x() + b.get_width()/2, v + 0.03, f"{v:.2f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_ylabel("Producción Media (Millones m³)", fontsize=11)
    ax.set_title("Producción Media Mensual por Planta – CORAASAN",
                 fontsize=13, fontweight="bold", color=COLOR_AZUL)
    ax.tick_params(axis="x", rotation=15); fig.tight_layout()
    fig.savefig(os.path.join(GRAFICOS_DIR, "F2_03_produccion_planta.png"), dpi=150, bbox_inches="tight")
    plt.close(fig); print("  ✓ F2_03_produccion_planta.png")

    # Fig 4: Boxplot Cloro por planta
    fig, ax = plt.subplots(figsize=(10, 4))
    plantas_ord = (df.groupby("PLANTA")["CLORO_RESIDUAL_MGL"].median().sort_values().index)
    data_box = [df[df["PLANTA"] == p]["CLORO_RESIDUAL_MGL"].dropna().values for p in plantas_ord]
    bp = ax.boxplot(data_box, patch_artist=True, notch=False)
    for patch in bp["boxes"]:
        patch.set_facecolor(COLOR_AZUL2); patch.set_alpha(0.65)
    for med in bp["medians"]:
        med.set_color(COLOR_AZUL); med.set_linewidth(2)
    ax.axhline(UMBRAL_CLORO_MIN, color=COLOR_VERDE, ls="--", lw=1.5, label=f"Mín OMS {UMBRAL_CLORO_MIN} mg/L")
    ax.axhline(UMBRAL_CLORO_MAX, color=COLOR_ROJO,  ls="--", lw=1.5, label=f"Máx OMS {UMBRAL_CLORO_MAX} mg/L")
    ax.set_xticklabels(plantas_ord, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Cloro Residual (mg/L)", fontsize=11)
    ax.set_title("Distribución de Cloro Residual por Planta",
                 fontsize=13, fontweight="bold", color=COLOR_AZUL)
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(GRAFICOS_DIR, "F2_04_cloro_boxplot.png"), dpi=150, bbox_inches="tight")
    plt.close(fig); print("  ✓ F2_04_cloro_boxplot.png")

    # Fig 5: Distribución de alertas
    fig, ax = plt.subplots(figsize=(6, 5))
    conteo = df["ALERTA_CALIDAD"].value_counts()
    colores_alerta = {"NORMAL": COLOR_VERDE, "BAJO": COLOR_NARANJA, "CRITICO": COLOR_ROJO}
    colores = [colores_alerta.get(k, COLOR_AZUL) for k in conteo.index]
    wedges, texts, autotexts = ax.pie(
        conteo.values, labels=conteo.index, autopct="%1.1f%%",
        colors=colores, startangle=140, pctdistance=0.75
    )
    for t in autotexts: t.set_fontsize(11); t.set_fontweight("bold")
    ax.set_title("Distribución de Alertas de Calidad\n(CORAASAN 2018–2026)",
                 fontsize=13, fontweight="bold", color=COLOR_AZUL)
    plt.tight_layout()
    fig.savefig(os.path.join(GRAFICOS_DIR, "F2_05_alertas_calidad.png"), dpi=150, bbox_inches="tight")
    plt.close(fig); print("  ✓ F2_05_alertas_calidad.png")

    # Fig 6: Serie temporal IP por planta
    fig, ax = plt.subplots(figsize=(13, 5))
    colores_lineas = [COLOR_AZUL, COLOR_ROJO, COLOR_VERDE, COLOR_NARANJA, COLOR_AZUL2, "#9B59B6"]
    df_sorted = df.sort_values("FECHA")
    for i, planta in enumerate(sorted(df["PLANTA"].unique())):
        sub = df_sorted[df_sorted["PLANTA"] == planta]
        ax.plot(sub["FECHA"], sub["INDICE_POTABILIDAD_PCT"],
                marker="o", markersize=3.5, linewidth=1.8,
                color=colores_lineas[i % len(colores_lineas)], label=planta, alpha=0.85)
    ax.axhline(UMBRAL_IP_BAJO, color=COLOR_ROJO, lw=1.5, ls="--", alpha=0.6,
               label=f"Umbral {UMBRAL_IP_BAJO}%")
    ax.set_ylabel("Índice de Potabilidad (%)", fontsize=11)
    ax.set_xlabel("Fecha", fontsize=11)
    ax.set_title("Evolución Mensual del Índice de Potabilidad por Planta (2018–2026)",
                 fontsize=13, fontweight="bold", color=COLOR_AZUL)
    ax.legend(fontsize=8, loc="lower left", ncol=3); fig.tight_layout()
    fig.savefig(os.path.join(GRAFICOS_DIR, "F2_06_evolucion_ip.png"), dpi=150, bbox_inches="tight")
    plt.close(fig); print("  ✓ F2_06_evolucion_ip.png")

    # Fig 7: Heatmap Planta × Año
    fig, ax = plt.subplots(figsize=(11, 4))
    pivot = df.pivot_table(index="PLANTA", columns="AÑO",
                            values="INDICE_POTABILIDAD_PCT", aggfunc="mean").round(1)
    im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto", vmin=70, vmax=100)
    plt.colorbar(im, ax=ax, label="IP Promedio (%)")
    ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels(pivot.columns, fontsize=9)
    ax.set_yticks(range(len(pivot.index)));   ax.set_yticklabels(pivot.index, fontsize=9)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                        fontsize=8, fontweight="bold",
                        color="white" if val < 82 else "black")
    ax.set_title("Heatmap – IP (%) por Planta y Año",
                 fontsize=13, fontweight="bold", color=COLOR_AZUL)
    ax.set_xlabel("Año"); ax.set_ylabel("Planta"); fig.tight_layout()
    fig.savefig(os.path.join(GRAFICOS_DIR, "F2_07_heatmap_ip.png"), dpi=150, bbox_inches="tight")
    plt.close(fig); print("  ✓ F2_07_heatmap_ip.png")

    print()


# ══════════════════════════════════════════════════════════════════════════════
# FASE 3 – PREPARACIÓN DE LOS DATOS
# ══════════════════════════════════════════════════════════════════════════════

def fase3_preparacion(df):
    print("\n" + "━" * 65)
    print("  FASE 3 CRISP-DM: Preparación de los Datos")
    print("━" * 65)

    # ── Reporte de limpieza (ya aplicada en el dataset) ───────────────────────
    print("\n📋 Estado del dataset integrado:")
    print(f"   Filas          : {len(df)}")
    print(f"   Columnas       : {len(df.columns)}")
    print(f"   Nulos totales  : {df.isnull().sum().sum()}")
    print(f"   Duplicados     : {df.duplicated().sum()}")

    print("\n📊 Variables disponibles:")
    for col in df.columns:
        print(f"   {col:<28} {df[col].dtype}   nulos={df[col].isnull().sum()}")

    # ── Estadísticas de las variables clave ───────────────────────────────────
    print("\n📈 Resumen estadístico – variables numéricas clave:")
    vars_key = ["PRODUCCION_MILLONES_M3", "INDICE_POTABILIDAD_PCT",
                "CLORO_RESIDUAL_MGL", "INDICE_EFICIENCIA"]
    print(df[vars_key].describe().round(4).to_string())

    # ── Distribución de ALERTA_CALIDAD ────────────────────────────────────────
    print("\n⚠️  Distribución ALERTA_CALIDAD:")
    for alerta, cnt in df["ALERTA_CALIDAD"].value_counts().items():
        pct = cnt / len(df) * 100
        print(f"   {alerta:<10} {cnt:>4} registros ({pct:.1f}%)")

    print(f"\n✅ Cumplimiento norma OMS Cloro: {df['CUMPLE_OMS_CLORO'].mean()*100:.1f}%")
    print(f"   Índice de Eficiencia promedio: {df['INDICE_EFICIENCIA'].mean():.4f}")

    # ── Gráficos ──────────────────────────────────────────────────────────────
    print("\n  Generando gráficos Fase 3...")

    # Fig: Scatter IP vs Producción coloreado por Cloro
    fig, ax = plt.subplots(figsize=(10, 5))
    sc = ax.scatter(df["PRODUCCION_MILLONES_M3"], df["INDICE_POTABILIDAD_PCT"],
                    c=df["CLORO_RESIDUAL_MGL"], cmap="coolwarm_r",
                    s=65, alpha=0.75, edgecolors="white", linewidth=0.4)
    plt.colorbar(sc, ax=ax, label="Cloro Residual (mg/L)")
    ax.axhline(UMBRAL_IP_BAJO, color=COLOR_ROJO, lw=1.5, ls="--", label=f"Umbral IP {UMBRAL_IP_BAJO}%")
    ax.set_xlabel("Producción Mensual (Millones m³)", fontsize=12)
    ax.set_ylabel("Índice de Potabilidad (%)", fontsize=12)
    ax.set_title("IP vs Producción – Dataset Integrado CORAASAN\n(Color = Cloro Residual mg/L)",
                 fontsize=13, fontweight="bold", color=COLOR_AZUL)
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(GRAFICOS_DIR, "F3_08_scatter_ip_produccion.png"), dpi=150, bbox_inches="tight")
    plt.close(fig); print("  ✓ F3_08_scatter_ip_produccion.png")

    # Fig: Índice de Eficiencia por planta
    fig, ax = plt.subplots(figsize=(9, 4))
    ef = df.groupby("PLANTA")["INDICE_EFICIENCIA"].mean().sort_values(ascending=False)
    bars = ax.bar(ef.index, ef.values,
                  color=[COLOR_AZUL if i == 0 else COLOR_AZUL2 for i in range(len(ef))],
                  width=0.55, zorder=3)
    for b, v in zip(bars, ef.values):
        ax.text(b.get_x() + b.get_width()/2, v + 0.005, f"{v:.3f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_ylabel("Índice Eficiencia (IP/100 × Producción Mm³)", fontsize=10)
    ax.set_title("Índice de Eficiencia por Planta",
                 fontsize=13, fontweight="bold", color=COLOR_AZUL)
    ax.tick_params(axis="x", rotation=15); fig.tight_layout()
    fig.savefig(os.path.join(GRAFICOS_DIR, "F3_09_eficiencia_planta.png"), dpi=150, bbox_inches="tight")
    plt.close(fig); print("  ✓ F3_09_eficiencia_planta.png")

    # Fig: Correlación entre variables numéricas
    fig, ax = plt.subplots(figsize=(8, 6))
    vars_corr = ["PRODUCCION_MILLONES_M3", "INDICE_POTABILIDAD_PCT",
                 "CLORO_RESIDUAL_MGL", "NUM_MUESTRAS", "INDICE_EFICIENCIA"]
    corr = df[vars_corr].corr().round(2)
    im = ax.imshow(corr.values, cmap="RdBu", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, label="Correlación de Pearson")
    ax.set_xticks(range(len(vars_corr))); ax.set_xticklabels(vars_corr, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(vars_corr))); ax.set_yticklabels(vars_corr, fontsize=8)
    for i in range(len(vars_corr)):
        for j in range(len(vars_corr)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center",
                    fontsize=9, fontweight="bold",
                    color="white" if abs(corr.values[i, j]) > 0.6 else "black")
    ax.set_title("Matriz de Correlación – Variables Numéricas",
                 fontsize=13, fontweight="bold", color=COLOR_AZUL)
    fig.tight_layout()
    fig.savefig(os.path.join(GRAFICOS_DIR, "F3_10_correlacion.png"), dpi=150, bbox_inches="tight")
    plt.close(fig); print("  ✓ F3_10_correlacion.png")

    print()


# ══════════════════════════════════════════════════════════════════════════════
# PREPARAR FEATURES PARA MODELADO
# ══════════════════════════════════════════════════════════════════════════════

def preparar_features(df):
    """Codifica variables y construye X, y para los modelos."""
    df_m = df.dropna(subset=["INDICE_POTABILIDAD_PCT",
                               "PRODUCCION_MILLONES_M3",
                               "CLORO_RESIDUAL_MGL"]).copy()

    df_enc = pd.get_dummies(df_m, columns=["PLANTA"], prefix="PLT", drop_first=False)

    feat_cols = (
        ["PRODUCCION_MILLONES_M3", "LOG_PRODUCCION",
         "CLORO_RESIDUAL_MGL", "NUM_MUESTRAS", "AÑO", "MES_NUM"]
        + [c for c in df_enc.columns if c.startswith("PLT_")]
    )
    feat_cols = [c for c in feat_cols if c in df_enc.columns]

    X = df_enc[feat_cols]
    y = df_enc["INDICE_POTABILIDAD_PCT"]

    X_cluster = df_m[["PRODUCCION_MILLONES_M3", "INDICE_POTABILIDAD_PCT",
                        "CLORO_RESIDUAL_MGL", "INDICE_EFICIENCIA"]].dropna()
    return X, y, X_cluster, feat_cols, df_m


# ══════════════════════════════════════════════════════════════════════════════
# FASE 4 – MODELADO
# ══════════════════════════════════════════════════════════════════════════════

def fase4_modelado(df):
    print("\n" + "━" * 65)
    print("  FASE 4 CRISP-DM: Modelado")
    print("━" * 65)

    X, y, X_cluster, feat_cols, df_m = preparar_features(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    print(f"\n  Dataset modelado : {X.shape[0]} registros × {X.shape[1]} features")
    print(f"  Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"  Target → INDICE_POTABILIDAD_PCT")
    print(f"  Rango  → min={y.min():.1f}%  max={y.max():.1f}%  media={y.mean():.2f}%\n")

    # ── M1: Regresión Lineal Múltiple ─────────────────────────────────────────
    print("── M1: Regresión Lineal Múltiple ────────────────────────────────")
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    yp_lr = lr.predict(X_test)
    r2_lr   = r2_score(y_test, yp_lr)
    rmse_lr = np.sqrt(mean_squared_error(y_test, yp_lr))
    mae_lr  = mean_absolute_error(y_test, yp_lr)
    print(f"   R²={r2_lr:.4f}  RMSE={rmse_lr:.4f}  MAE={mae_lr:.4f}")

    coefs = pd.Series(lr.coef_, index=feat_cols).abs().sort_values(ascending=False)
    print("   Top 5 variables (|coeficiente|):")
    for v, c in coefs.head(5).items():
        print(f"     {v:<35} {c:.4f}")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(y_test, yp_lr, alpha=0.65, color=COLOR_AZUL2,
               edgecolors="white", linewidth=0.4, s=55, label="Observaciones")
    lim = [min(y_test.min(), yp_lr.min())-2, max(y_test.max(), yp_lr.max())+2]
    ax.plot(lim, lim, color=COLOR_ROJO, lw=1.8, ls="--", label="Línea perfecta")
    ax.set_xlabel("IP Real (%)"); ax.set_ylabel("IP Predicho (%)")
    ax.set_title(f"M1: Regresión Lineal – IP Real vs Predicho\n(R²={r2_lr:.4f}  RMSE={rmse_lr:.4f})",
                 fontsize=12, fontweight="bold", color=COLOR_AZUL)
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(GRAFICOS_DIR, "F4_11_m1_reg_lineal.png"), dpi=150, bbox_inches="tight")
    plt.close(fig); print("   ✓ F4_11_m1_reg_lineal.png")

    res_lr = {"modelo": lr, "y_pred": yp_lr, "y_test": y_test,
              "r2": r2_lr, "rmse": rmse_lr, "mae": mae_lr}

    # ── M2: Random Forest ─────────────────────────────────────────────────────
    print("\n── M2: Random Forest Regressor ──────────────────────────────────")
    rf = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    yp_rf = rf.predict(X_test)
    r2_rf   = r2_score(y_test, yp_rf)
    rmse_rf = np.sqrt(mean_squared_error(y_test, yp_rf))
    mae_rf  = mean_absolute_error(y_test, yp_rf)
    print(f"   R²={r2_rf:.4f}  RMSE={rmse_rf:.4f}  MAE={mae_rf:.4f}")

    importances = pd.Series(rf.feature_importances_, index=feat_cols).sort_values(ascending=False)
    print("   Top 5 variables importantes:")
    for v, c in importances.head(5).items():
        print(f"     {v:<35} {c:.4f}")

    top_imp = importances.head(10)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.barh(top_imp.index[::-1], top_imp.values[::-1], color=COLOR_AZUL, height=0.6, zorder=3)
    ax.set_xlabel("Importancia"); ax.grid(axis="x", alpha=0.3)
    ax.set_title("M2: Random Forest – Top 10 Variables Importantes",
                 fontsize=13, fontweight="bold", color=COLOR_AZUL)
    fig.tight_layout()
    fig.savefig(os.path.join(GRAFICOS_DIR, "F4_12_m2_rf_importancia.png"), dpi=150, bbox_inches="tight")
    plt.close(fig); print("   ✓ F4_12_m2_rf_importancia.png")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(y_test, yp_rf, alpha=0.65, color=COLOR_VERDE,
               edgecolors="white", linewidth=0.4, s=55)
    ax.plot(lim, lim, color=COLOR_ROJO, lw=1.8, ls="--", label="Línea perfecta")
    ax.set_xlabel("IP Real (%)"); ax.set_ylabel("IP Predicho (%)")
    ax.set_title(f"M2: Random Forest – IP Real vs Predicho\n(R²={r2_rf:.4f}  RMSE={rmse_rf:.4f})",
                 fontsize=12, fontweight="bold", color=COLOR_AZUL)
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(GRAFICOS_DIR, "F4_13_m2_rf_predicho.png"), dpi=150, bbox_inches="tight")
    plt.close(fig); print("   ✓ F4_13_m2_rf_predicho.png")

    res_rf = {"modelo": rf, "y_pred": yp_rf, "y_test": y_test,
              "r2": r2_rf, "rmse": rmse_rf, "mae": mae_rf, "importances": importances}

    # ── M3: K-Means Clustering ────────────────────────────────────────────────
    print("\n── M3: K-Means Clustering (K=3, nivel de riesgo) ────────────────")
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)

    # Método del codo
    inercias = []
    for k in range(2, 8):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inercias.append(km.inertia_)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(2, 8), inercias, "o-", color=COLOR_AZUL, lw=2.2, markersize=8)
    ax.axvline(3, color=COLOR_ROJO, ls="--", lw=1.5, label="K=3 seleccionado")
    ax.set_xlabel("Número de Clusters (K)"); ax.set_ylabel("Inercia (WCSS)")
    ax.set_title("M3: K-Means – Método del Codo",
                 fontsize=13, fontweight="bold", color=COLOR_AZUL)
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(GRAFICOS_DIR, "F4_14_m3_elbow.png"), dpi=150, bbox_inches="tight")
    plt.close(fig); print("   ✓ F4_14_m3_elbow.png")

    K = 3
    km_final = KMeans(n_clusters=K, random_state=42, n_init=10)
    X_cl = X_cluster.copy()
    X_cl["CLUSTER"] = km_final.fit_predict(X_scaled)
    ip_por_cluster  = X_cl.groupby("CLUSTER")["INDICE_POTABILIDAD_PCT"].mean()
    orden = ip_por_cluster.sort_values()
    etiquetas = {c: n for (c, _), n in zip(orden.items(),
                 ["RIESGO ALTO", "RIESGO MEDIO", "RIESGO BAJO"])}
    X_cl["NIVEL_RIESGO"] = X_cl["CLUSTER"].map(etiquetas)

    for cl in range(K):
        sub = X_cl[X_cl["CLUSTER"] == cl]
        print(f"   Cluster {cl} ({etiquetas[cl]}): {len(sub)} registros | "
              f"IP={sub['INDICE_POTABILIDAD_PCT'].mean():.1f}%  "
              f"Prod={sub['PRODUCCION_MILLONES_M3'].mean():.3f} Mm³")

    colores_cl = {"RIESGO ALTO": COLOR_ROJO, "RIESGO MEDIO": COLOR_NARANJA, "RIESGO BAJO": COLOR_VERDE}
    fig, ax = plt.subplots(figsize=(9, 5))
    for nivel, grupo in X_cl.groupby("NIVEL_RIESGO"):
        ax.scatter(grupo["PRODUCCION_MILLONES_M3"], grupo["INDICE_POTABILIDAD_PCT"],
                   color=colores_cl[nivel], label=nivel, s=65, alpha=0.8,
                   edgecolors="white", linewidth=0.4)
    ax.axhline(UMBRAL_IP_BAJO, color=COLOR_ROJO, lw=1.5, ls="--", alpha=0.6)
    ax.set_xlabel("Producción (Millones m³)"); ax.set_ylabel("Índice de Potabilidad (%)")
    ax.set_title("M3: K-Means – Nivel de Riesgo por Registro (K=3)",
                 fontsize=13, fontweight="bold", color=COLOR_AZUL)
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(GRAFICOS_DIR, "F4_15_m3_clusters.png"), dpi=150, bbox_inches="tight")
    plt.close(fig); print("   ✓ F4_15_m3_clusters.png")

    res_km = {"modelo": km_final, "datos": X_cl, "etiquetas": etiquetas, "scaler": scaler}

    # ── M4: Árbol de Decisión ─────────────────────────────────────────────────
    print("\n── M4: Árbol de Decisión – Clasificar ALERTA_CALIDAD ────────────")
    df_arb = df_m.dropna(subset=["PRODUCCION_MILLONES_M3", "CLORO_RESIDUAL_MGL",
                                   "NUM_MUESTRAS", "ALERTA_CALIDAD"]).copy()
    le = LabelEncoder()
    df_arb["ALERTA_ENC"] = le.fit_transform(df_arb["ALERTA_CALIDAD"])
    clases = le.classes_

    feat_arbol = ["PRODUCCION_MILLONES_M3", "CLORO_RESIDUAL_MGL", "NUM_MUESTRAS", "AÑO"]
    Xa_tr, Xa_te, ya_tr, ya_te = train_test_split(
        df_arb[feat_arbol], df_arb["ALERTA_ENC"],
        test_size=0.25, random_state=42, stratify=df_arb["ALERTA_ENC"]
    )
    arbol = DecisionTreeClassifier(max_depth=4, random_state=42, class_weight="balanced")
    arbol.fit(Xa_tr, ya_tr)
    ya_pred = arbol.predict(Xa_te)

    print(f"   Distribución clases: {df_arb['ALERTA_CALIDAD'].value_counts().to_dict()}")
    print(f"   Accuracy: {accuracy_score(ya_te, ya_pred):.4f}")
    print(f"   F1-macro: {f1_score(ya_te, ya_pred, average='macro', zero_division=0):.4f}")
    print("\n   Reporte de clasificación:")
    print(classification_report(ya_te, ya_pred, target_names=clases, zero_division=0))

    fig, ax = plt.subplots(figsize=(14, 5))
    plot_tree(arbol, feature_names=feat_arbol, class_names=clases,
              filled=True, rounded=True, fontsize=8, ax=ax, impurity=False)
    ax.set_title("M4: Árbol de Decisión – Clasificación ALERTA_CALIDAD",
                 fontsize=13, fontweight="bold", color=COLOR_AZUL)
    fig.tight_layout()
    fig.savefig(os.path.join(GRAFICOS_DIR, "F4_16_m4_arbol.png"), dpi=150, bbox_inches="tight")
    plt.close(fig); print("   ✓ F4_16_m4_arbol.png")

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(ya_te, ya_pred,
                                             display_labels=clases, cmap="Blues", ax=ax)
    ax.set_title("M4: Matriz de Confusión – ALERTA_CALIDAD",
                 fontsize=12, fontweight="bold", color=COLOR_AZUL)
    fig.tight_layout()
    fig.savefig(os.path.join(GRAFICOS_DIR, "F4_17_m4_confusion.png"), dpi=150, bbox_inches="tight")
    plt.close(fig); print("   ✓ F4_17_m4_confusion.png")

    res_arb = {"modelo": arbol, "clases": clases, "le": le,
               "y_test": ya_te, "y_pred": ya_pred}

    # ── Comparación M1 vs M2 ──────────────────────────────────────────────────
    print("\n── Comparación M1 vs M2 ─────────────────────────────────────────")
    print(f"  {'Métrica':<8} {'Reg. Lineal':>14} {'Random Forest':>14}  Mejor")
    print("  " + "─" * 48)
    for m, lr_v, rf_v in [("R²", r2_lr, r2_rf), ("RMSE", rmse_lr, rmse_rf), ("MAE", mae_lr, mae_rf)]:
        mejor = "✅ RF" if (rf_v > lr_v if m == "R²" else rf_v < lr_v) else "✅ LR"
        print(f"  {m:<8} {lr_v:>14.4f} {rf_v:>14.4f}  {mejor}")

    print()
    return res_lr, res_rf, res_km, res_arb, X, y, feat_cols


# ══════════════════════════════════════════════════════════════════════════════
# FASE 5 – EVALUACIÓN
# ══════════════════════════════════════════════════════════════════════════════

def fase5_evaluacion(df, res_lr, res_rf, res_km, res_arb, X, y):
    print("\n" + "━" * 65)
    print("  FASE 5 CRISP-DM: Evaluación")
    print("━" * 65)

    # ── E1: Validación Cruzada K-Fold ─────────────────────────────────────────
    print("\n── E1: Validación Cruzada K-Fold (k=10) ─────────────────────────")
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    cv_r2_lr   = cross_val_score(LinearRegression(), X, y, cv=kf, scoring="r2")
    cv_rmse_lr = np.sqrt(-cross_val_score(LinearRegression(), X, y, cv=kf,
                                           scoring="neg_mean_squared_error"))
    cv_r2_rf   = cross_val_score(RandomForestRegressor(n_estimators=100, max_depth=8,
                                                         random_state=42, n_jobs=-1),
                                  X, y, cv=kf, scoring="r2")
    cv_rmse_rf = np.sqrt(-cross_val_score(RandomForestRegressor(n_estimators=100, max_depth=8,
                                                                   random_state=42, n_jobs=-1),
                                            X, y, cv=kf, scoring="neg_mean_squared_error"))

    print(f"  {'Métrica':<10} {'LR Media':>10} {'LR ±std':>10} {'RF Media':>10} {'RF ±std':>10}")
    print("  " + "─" * 50)
    for m, lm, ls, rm, rs in [
        ("R²",   cv_r2_lr.mean(),   cv_r2_lr.std(),   cv_r2_rf.mean(),   cv_r2_rf.std()),
        ("RMSE", cv_rmse_lr.mean(), cv_rmse_lr.std(), cv_rmse_rf.mean(), cv_rmse_rf.std()),
    ]:
        print(f"  {m:<10} {lm:>10.4f} {ls:>10.4f} {rm:>10.4f} {rs:>10.4f}")

    print(f"\n  Estabilidad M1 (R² std): {cv_r2_lr.std():.4f}  "
          f"{'✅ Estable' if cv_r2_lr.std() < 0.1 else '⚠️ Variable'}")
    print(f"  Estabilidad M2 (R² std): {cv_r2_rf.std():.4f}  "
          f"{'✅ Estable' if cv_r2_rf.std() < 0.1 else '⚠️ Variable'}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, datos_lr, datos_rf, metrica in [
        (axes[0], cv_r2_lr,   cv_r2_rf,   "R² (mayor = mejor)"),
        (axes[1], cv_rmse_lr, cv_rmse_rf, "RMSE (menor = mejor)"),
    ]:
        bp = ax.boxplot([datos_lr, datos_rf], patch_artist=True,
                        labels=["Reg. Lineal\n(M1)", "Random Forest\n(M2)"],
                        notch=False, widths=0.4)
        bp["boxes"][0].set_facecolor(COLOR_AZUL2); bp["boxes"][0].set_alpha(0.75)
        bp["boxes"][1].set_facecolor(COLOR_VERDE);  bp["boxes"][1].set_alpha(0.75)
        for med in bp["medians"]: med.set_color("black"); med.set_linewidth(2.2)
        ax.set_title(f"CV K-Fold 10 – {metrica}", fontsize=11, fontweight="bold", color=COLOR_AZUL)
        ax.set_ylabel(metrica.split(" ")[0])
    fig.suptitle("E1: Validación Cruzada K-Fold (k=10)",
                 fontsize=13, fontweight="bold", color=COLOR_AZUL)
    plt.tight_layout()
    fig.savefig(os.path.join(GRAFICOS_DIR, "F5_18_e1_cv_kfold.png"), dpi=150, bbox_inches="tight")
    plt.close(fig); print("\n   ✓ F5_18_e1_cv_kfold.png")

    # ── E2: Análisis de Residuos ──────────────────────────────────────────────
    print("\n── E2: Análisis de Residuos ──────────────────────────────────────")
    y_test     = res_lr["y_test"].values
    res_lr_arr = y_test - res_lr["y_pred"]
    res_rf_arr = y_test - res_rf["y_pred"]

    print(f"  M1 → media={res_lr_arr.mean():.4f}  std={res_lr_arr.std():.4f}  "
          f"{'✅ Sin sesgo' if abs(res_lr_arr.mean()) < 1 else '⚠️ Sesgo'}")
    print(f"  M2 → media={res_rf_arr.mean():.4f}  std={res_rf_arr.std():.4f}  "
          f"{'✅ Sin sesgo' if abs(res_rf_arr.mean()) < 1 else '⚠️ Sesgo'}")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for row, (res, pred, color, nombre) in enumerate([
        (res_lr_arr, res_lr["y_pred"], COLOR_AZUL2, "M1: Regresión Lineal"),
        (res_rf_arr, res_rf["y_pred"], COLOR_VERDE,  "M2: Random Forest"),
    ]):
        axes[row, 0].scatter(pred, res, alpha=0.6, color=color,
                              edgecolors="white", linewidth=0.3, s=55)
        axes[row, 0].axhline(0, color=COLOR_ROJO, lw=1.8, ls="--", label="Residuo=0")
        axes[row, 0].axhline(res.mean(), color=COLOR_NARANJA, lw=1.3, ls=":",
                              label=f"Media={res.mean():.3f}")
        axes[row, 0].set_xlabel("IP Predicho (%)"); axes[row, 0].set_ylabel("Residuo")
        axes[row, 0].set_title(f"{nombre} – Residuos vs Predicho",
                                fontweight="bold", color=COLOR_AZUL); axes[row, 0].legend(fontsize=9)

        axes[row, 1].hist(res, bins=25, color=color, edgecolor="white", alpha=0.82)
        axes[row, 1].axvline(0, color=COLOR_ROJO, lw=1.8, ls="--", label="Residuo=0")
        axes[row, 1].set_xlabel("Residuo"); axes[row, 1].set_ylabel("Frecuencia")
        axes[row, 1].set_title(f"{nombre} – Distribución Residuos (std={res.std():.3f})",
                                fontweight="bold", color=COLOR_AZUL); axes[row, 1].legend(fontsize=9)

    fig.suptitle("E2: Análisis de Residuos – M1 vs M2",
                 fontsize=13, fontweight="bold", color=COLOR_AZUL)
    plt.tight_layout()
    fig.savefig(os.path.join(GRAFICOS_DIR, "F5_19_e2_residuos.png"), dpi=150, bbox_inches="tight")
    plt.close(fig); print("   ✓ F5_19_e2_residuos.png")

    # ── E3: Dashboard Comparativo ─────────────────────────────────────────────
    print("\n── E3: Dashboard Comparativo de Modelos ─────────────────────────")
    acc_arb = accuracy_score(res_arb["y_test"], res_arb["y_pred"])
    f1_arb  = f1_score(res_arb["y_test"], res_arb["y_pred"], average="macro", zero_division=0)
    f1_clases = f1_score(res_arb["y_test"], res_arb["y_pred"],
                          average=None, zero_division=0,
                          labels=range(len(res_arb["clases"])))
    clases = res_arb["clases"]

    fig = plt.figure(figsize=(16, 11))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.50, wspace=0.42)
    nombres = ["Reg. Lineal\n(M1)", "Random Forest\n(M2)"]

    # Panel 1: R²
    ax1 = fig.add_subplot(gs[0, 0])
    r2_vals = [res_lr["r2"], res_rf["r2"]]
    bars = ax1.bar(nombres, r2_vals, color=[COLOR_AZUL2, COLOR_VERDE], width=0.45, zorder=3)
    for b, v in zip(bars, r2_vals):
        ax1.text(b.get_x() + b.get_width()/2, v + 0.008, f"{v:.4f}",
                 ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax1.set_ylim(0, 1.1); ax1.set_ylabel("R²", fontsize=11)
    ax1.set_title("R² Test Set\n(mayor = mejor)", fontweight="bold", color=COLOR_AZUL)

    # Panel 2: RMSE + MAE
    ax2 = fig.add_subplot(gs[0, 1])
    x = np.arange(2); w = 0.35
    b1 = ax2.bar(x - w/2, [res_lr["rmse"], res_rf["rmse"]], w,
                  color=COLOR_AZUL2, label="RMSE", alpha=0.85, zorder=3)
    b2 = ax2.bar(x + w/2, [res_lr["mae"],  res_rf["mae"]],  w,
                  color=COLOR_NARANJA, label="MAE",  alpha=0.85, zorder=3)
    for b, v in list(zip(b1, [res_lr["rmse"], res_rf["rmse"]])) + \
                list(zip(b2, [res_lr["mae"],  res_rf["mae"]])):
        ax2.text(b.get_x() + b.get_width()/2, v + 0.02, f"{v:.3f}",
                 ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax2.set_xticks(x); ax2.set_xticklabels(nombres)
    ax2.set_ylabel("Error"); ax2.legend(fontsize=9)
    ax2.set_title("RMSE y MAE Test Set\n(menor = mejor)", fontweight="bold", color=COLOR_AZUL)

    # Panel 3: CV R² con barras de error
    ax3 = fig.add_subplot(gs[0, 2])
    cv_medias = [cv_r2_lr.mean(), cv_r2_rf.mean()]
    cv_stds   = [cv_r2_lr.std(),  cv_r2_rf.std()]
    bars_cv = ax3.bar(nombres, cv_medias, yerr=cv_stds, capsize=10,
                       color=[COLOR_AZUL2, COLOR_VERDE], width=0.45, zorder=3,
                       error_kw={"elinewidth": 2.2, "ecolor": COLOR_ROJO, "capthick": 2})
    for b, v in zip(bars_cv, cv_medias):
        ax3.text(b.get_x() + b.get_width()/2, v + 0.015, f"{v:.4f}",
                 ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax3.set_ylim(0, 1.15); ax3.set_ylabel("R² CV (±std)")
    ax3.set_title("R² Validación Cruzada K-Fold\n(k=10, ±std)", fontweight="bold", color=COLOR_AZUL)

    # Panel 4: K-Means distribución
    ax4 = fig.add_subplot(gs[1, 0])
    conteo_cl = res_km["datos"]["NIVEL_RIESGO"].value_counts()
    col_km = [{"RIESGO BAJO": COLOR_VERDE, "RIESGO MEDIO": COLOR_NARANJA,
                "RIESGO ALTO": COLOR_ROJO}.get(k, COLOR_AZUL) for k in conteo_cl.index]
    _, _, autotexts = ax4.pie(conteo_cl.values, labels=conteo_cl.index,
                               autopct="%1.1f%%", colors=col_km, startangle=90, pctdistance=0.72)
    for t in autotexts: t.set_fontsize(10); t.set_fontweight("bold")
    ax4.set_title("M3: K-Means (K=3)\nDistribución Nivel de Riesgo", fontweight="bold", color=COLOR_AZUL)

    # Panel 5: F1-Score por clase árbol
    ax5 = fig.add_subplot(gs[1, 1])
    col_arb = [COLOR_ROJO, COLOR_NARANJA, COLOR_VERDE][:len(clases)]
    bars_arb = ax5.bar(clases, f1_clases, color=col_arb, width=0.45, zorder=3)
    for b, v in zip(bars_arb, f1_clases):
        ax5.text(b.get_x() + b.get_width()/2, v + 0.012, f"{v:.3f}",
                 ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax5.set_ylim(0, 1.2); ax5.set_ylabel("F1-Score")
    ax5.set_title(f"M4: Árbol de Decisión\nAcc={acc_arb:.3f} | F1-macro={f1_arb:.3f}",
                   fontweight="bold", color=COLOR_AZUL)

    # Panel 6: Matriz de confusión
    ax6 = fig.add_subplot(gs[1, 2])
    ConfusionMatrixDisplay.from_predictions(
        res_arb["y_test"], res_arb["y_pred"],
        display_labels=clases, cmap="Blues", ax=ax6, colorbar=False
    )
    ax6.set_title("M4: Matriz de Confusión\nÁrbol de Decisión", fontweight="bold", color=COLOR_AZUL, fontsize=10)

    fig.suptitle(
        "E3: Dashboard de Evaluación – Todos los Modelos\n"
        "Análisis Agua Potable RD | Noel Reyes 21-2021 | UNPHU 2026",
        fontsize=14, fontweight="bold", color=COLOR_AZUL
    )
    fig.savefig(os.path.join(GRAFICOS_DIR, "F5_20_e3_dashboard.png"), dpi=150, bbox_inches="tight")
    plt.close(fig); print("   ✓ F5_20_e3_dashboard.png")

    # ── E4: Objetivos de Negocio ──────────────────────────────────────────────
    print("\n── E4: Evaluación frente a Objetivos de Negocio ─────────────────")
    mejor_r2   = max(res_lr["r2"], res_rf["r2"])
    mejor_rmse = min(res_lr["rmse"], res_rf["rmse"])
    pct_oms    = df["CUMPLE_OMS_CLORO"].mean() * 100
    ip_global  = df["INDICE_POTABILIDAD_PCT"].mean()
    pct_normal = (df["ALERTA_CALIDAD"] == "NORMAL").mean() * 100

    objetivos = [
        ("OBJ-1", "Predecir IP con R² > 0.70",
         mejor_r2 > 0.70,
         f"Mejor R²={mejor_r2:.4f}"),
        ("OBJ-2", "RMSE de predicción IP < 5.0%",
         mejor_rmse < 5.0,
         f"Mejor RMSE={mejor_rmse:.4f}%"),
        ("OBJ-3", "Cumplimiento norma OMS Cloro > 70%",
         pct_oms > 70,
         f"{pct_oms:.1f}% dentro de norma OMS (0.2–5.0 mg/L)"),
        ("OBJ-4", "Segmentar en 3 niveles de riesgo (K-Means)",
         len(res_km["etiquetas"]) == 3,
         f"Niveles: {list(res_km['etiquetas'].values())}"),
        ("OBJ-5", "Clasificar alertas Accuracy > 0.70",
         acc_arb > 0.70,
         f"Accuracy={acc_arb:.4f}  F1-macro={f1_arb:.4f}"),
        ("OBJ-6", "IP promedio histórica ≥ 95%",
         ip_global >= UMBRAL_IP_BAJO,
         f"IP global = {ip_global:.2f}%"),
        ("OBJ-7", "Más del 80% de registros en NORMAL",
         pct_normal >= 80,
         f"{pct_normal:.1f}% en estado NORMAL"),
    ]

    print(f"\n  {'Código':<8} {'Objetivo':<46} Estado")
    print("  " + "─" * 80)
    cumplidos = 0
    for cod, desc, cumple, detalle in objetivos:
        estado = "✅ CUMPLIDO    " if cumple else "❌ NO CUMPLIDO"
        if cumple: cumplidos += 1
        print(f"  {cod:<8} {desc:<46} {estado}")
        print(f"           └─ {detalle}")

    pct_exito = (cumplidos / len(objetivos)) * 100
    print(f"\n  Objetivos cumplidos: {cumplidos}/{len(objetivos)} → Éxito: {pct_exito:.0f}%")

    # ── E5: Conclusiones ──────────────────────────────────────────────────────
    print("\n── E5: Conclusiones y Recomendaciones ───────────────────────────")
    modelo_ganador = "Random Forest (M2)" if res_rf["r2"] > res_lr["r2"] else "Regresión Lineal (M1)"

    print(f"""
  HALLAZGOS:
  1. Mejor modelo: {modelo_ganador}
     R²={mejor_r2:.4f}  RMSE={mejor_rmse:.4f}%
     CV R² = {cv_r2_rf.mean():.4f} ± {cv_r2_rf.std():.4f} (RF)

  2. Calidad del agua: {pct_oms:.1f}% de registros cumplen norma OMS de Cloro.

  3. IP promedio global: {ip_global:.2f}%
     {pct_normal:.1f}% de registros en estado NORMAL.
     {'✅ Dentro de norma' if ip_global >= UMBRAL_IP_BAJO else '⚠️ Por debajo del umbral 95%'}

  4. K-Means identificó 3 niveles de riesgo:
     {list(res_km['etiquetas'].values())}

  5. Árbol de Decisión: Accuracy={acc_arb:.4f}  F1-macro={f1_arb:.4f}

  RECOMENDACIONES:
  R1. Implementar {modelo_ganador} como sistema de alerta temprana del IP.
  R2. Monitoreo semanal de Cloro en plantas RIESGO ALTO.
  R3. Integrar datos INAPA e INDRHI en próxima iteración CRISP-DM.
  R4. Desarrollar dashboard interactivo para operadores de CORAASAN.
  R5. Incorporar variables de mantenimiento y datos climáticos.
""")

    # Guardar informe
    ruta_informe = os.path.join(os.path.dirname(DATASET_PATH), "informe_evaluacion.txt")
    with open(ruta_informe, "w", encoding="utf-8") as f:
        f.write("=" * 65 + "\n")
        f.write("  INFORME FASE 5 CRISP-DM – EVALUACIÓN\n")
        f.write("  Análisis Agua Potable RD | Noel Reyes 21-2021 | UNPHU 2026\n")
        f.write("=" * 65 + "\n\n")
        f.write(f"M1 Regresión Lineal : R²={res_lr['r2']:.4f}  RMSE={res_lr['rmse']:.4f}  MAE={res_lr['mae']:.4f}\n")
        f.write(f"M2 Random Forest    : R²={res_rf['r2']:.4f}  RMSE={res_rf['rmse']:.4f}  MAE={res_rf['mae']:.4f}\n")
        f.write(f"M3 K-Means (K=3)    : {list(res_km['etiquetas'].values())}\n")
        f.write(f"M4 Árbol Decisión   : Accuracy={acc_arb:.4f}  F1-macro={f1_arb:.4f}\n\n")
        f.write(f"CV K-Fold (k=10):\n")
        f.write(f"  M1 R²: {cv_r2_lr.mean():.4f} ± {cv_r2_lr.std():.4f}\n")
        f.write(f"  M2 R²: {cv_r2_rf.mean():.4f} ± {cv_r2_rf.std():.4f}\n\n")
        f.write("OBJETIVOS:\n")
        for cod, desc, cumple, detalle in objetivos:
            f.write(f"  {cod}: [{'OK' if cumple else 'FALLO'}] {desc}\n  → {detalle}\n")
        f.write(f"\nÉxito: {cumplidos}/{len(objetivos)} ({pct_exito:.0f}%)\n")

    print(f"  💾 Informe guardado: {ruta_informe}\n")

    return cv_r2_lr, cv_r2_rf, cumplidos, len(objetivos), pct_exito


# ══════════════════════════════════════════════════════════════════════════════
# MAIN – PIPELINE COMPLETO
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    t0 = time.time()

    print("=" * 65)
    print("  💧 ANÁLISIS DE AGUA POTABLE – REPÚBLICA DOMINICANA")
    print("     Proyecto CRISP-DM | Noel Reyes 21-2021 | UNPHU 2026")
    print("=" * 65)

    # 0. Cargar dataset
    df = cargar_dataset()

    # Fase 2: Comprensión
    fase2_comprension(df)

    # Fase 3: Preparación
    fase3_preparacion(df)

    # Fase 4: Modelado
    res_lr, res_rf, res_km, res_arb, X, y, feat_cols = fase4_modelado(df)

    # Fase 5: Evaluación
    cv_r2_lr, cv_r2_rf, cumplidos, total, pct_exito = \
        fase5_evaluacion(df, res_lr, res_rf, res_km, res_arb, X, y)

    # Resumen final
    print("=" * 65)
    print("  ✅ PIPELINE COMPLETO FINALIZADO")
    print(f"     ⏱  Tiempo total    : {time.time()-t0:.1f}s")
    print(f"     📊 Gráficos        : {GRAFICOS_DIR}")
    print(f"     📄 Informe         : {os.path.dirname(DATASET_PATH)}\\informe_evaluacion.txt")
    print(f"     🎯 Éxito           : {cumplidos}/{total} objetivos ({pct_exito:.0f}%)")
    print("=" * 65)
