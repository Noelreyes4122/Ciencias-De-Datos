"""
fase1_carga_exploracion.py
==========================
CRISP-DM Fase 2: Comprensión de los Datos
------------------------------------------
- Carga todos los datasets
- Imprime resumen de cada uno
- Genera gráficos de exploración inicial
- Guarda gráficos en outputs/graficos/

Ejecutar: python scripts/fase1_carga_exploracion.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

from config import *

# ══════════════════════════════════════════════════════════════════════════════
# 1. CARGA DE DATASETS
# ══════════════════════════════════════════════════════════════════════════════

def cargar_datasets():
    """Carga todos los datasets y retorna un diccionario."""
    print("=" * 65)
    print("  CARGANDO DATASETS")
    print("=" * 65)

    datasets = {}

    # ── Producción CORAASAN ──────────────────────────────────────────
    df_prod = pd.read_csv(ARCHIVO_PRODUCCION, encoding="latin-1")
    df_prod["CANTIDAD_(Millones M3)"] = (
        df_prod["CANTIDAD_(Millones M3)"].str.replace(",", ".").astype(float)
    )
    datasets["produccion"] = df_prod
    print(f"✓ Producción CORAASAN   → {df_prod.shape[0]:,} filas × {df_prod.shape[1]} columnas")

    # ── Laboratorio CORAASAN ─────────────────────────────────────────
    df_lab = pd.read_csv(ARCHIVO_LAB, encoding="latin-1")
    df_lab["INDICE_POTABILIDAD_(%)"] = (
        df_lab["INDICE_POTABILIDAD_(%)"].str.replace(",", ".").astype(float)
    )
    df_lab["CLORO_RESIDUAL_(Mg/l)"] = (
        df_lab["CLORO_RESIDUAL_(Mg/l)"].str.replace(",", ".").astype(float)
    )
    datasets["laboratorio"] = df_lab
    print(f"✓ Laboratorio CORAASAN  → {df_lab.shape[0]:,} filas × {df_lab.shape[1]} columnas")

    # ── INAPA Cobertura ──────────────────────────────────────────────
    df_inapa = pd.read_csv(ARCHIVO_INAPA, sep=";", encoding="latin-1")
    df_inapa.columns = df_inapa.columns.str.strip()
    df_inapa.rename(columns={
        "Cobertura de Potabilidad General (%)": "COBERTURA_PCT",
        "Semestre": "SEMESTRE",
        "Año": "ANO"
    }, inplace=True)
    df_inapa["COBERTURA_PCT"] = (
        df_inapa["COBERTURA_PCT"].str.replace(",", ".").astype(float)
    )
    df_inapa["ANO"] = df_inapa["ANO"].astype(int)
    datasets["inapa"] = df_inapa
    print(f"✓ INAPA Cobertura       → {df_inapa.shape[0]:,} filas × {df_inapa.shape[1]} columnas")

    # ── INDRHI Calidad ───────────────────────────────────────────────
    df_indrhi = pd.read_csv(ARCHIVO_INDRHI, sep=";", encoding="latin-1")
    df_indrhi.columns = df_indrhi.columns.str.strip()
    df_indrhi.dropna(subset=["Tipo de Servicio"], inplace=True)
    df_indrhi["FECHA_ENTRADA"] = pd.to_datetime(
        df_indrhi["Fecha de entrada"], dayfirst=True, format="mixed", errors="coerce"
    )
    df_indrhi["ANO"] = df_indrhi["FECHA_ENTRADA"].dt.year
    df_indrhi["TIEMPO_RESP"] = pd.to_numeric(
        df_indrhi["Tiempo de Respuesta"], errors="coerce"
    )
    datasets["indrhi"] = df_indrhi
    print(f"✓ INDRHI Calidad        → {df_indrhi.shape[0]:,} filas × {df_indrhi.shape[1]} columnas")

    # ── QRS Quejas ───────────────────────────────────────────────────
    df_qrs = pd.read_excel(ARCHIVO_QRS, sheet_name="DATA")
    # Filtrar quejas relacionadas con agua
    mask = (
        df_qrs["Descripción"].str.upper().str.contains("|".join(AGUA_KEYWORDS), na=False)
        | df_qrs["Institución"].str.upper().str.contains("|".join(AGUA_KEYWORDS), na=False)
    )
    df_qrs_agua = df_qrs[mask].copy()
    datasets["qrs_total"] = df_qrs
    datasets["qrs_agua"]  = df_qrs_agua
    print(f"✓ QRS 3-1-1 (total)     → {df_qrs.shape[0]:,} filas")
    print(f"  └─ Quejas agua        → {df_qrs_agua.shape[0]:,} filas filtradas")

    print()
    return datasets


# ══════════════════════════════════════════════════════════════════════════════
# 2. REPORTE DE EXPLORACIÓN
# ══════════════════════════════════════════════════════════════════════════════

def reporte_exploracion(datasets):
    """Imprime estadísticas descriptivas de cada dataset."""

    df_prod  = datasets["produccion"]
    df_lab   = datasets["laboratorio"]
    df_inapa = datasets["inapa"]
    df_indrhi= datasets["indrhi"]

    print("=" * 65)
    print("  EXPLORACIÓN INICIAL")
    print("=" * 65)

    # ── Producción ───────────────────────────────────────────────────
    print("\n📦 PRODUCCIÓN CORAASAN")
    print(f"   Años: {sorted(df_prod['ANO'].unique())}")
    print(f"   Plantas ({df_prod['PLANTA'].nunique()}):")
    for p in sorted(df_prod["PLANTA"].unique()):
        media = df_prod[df_prod["PLANTA"] == p]["CANTIDAD_(Millones M3)"].mean()
        print(f"     {p:<30} media mensual: {media:.3f} Mm³")
    print(f"\n   Valores nulos: {df_prod.isnull().sum().sum()}")
    print(f"   Registros producción=0: {(df_prod['CANTIDAD_(Millones M3)'] == 0).sum()}")
    print(f"\n   Estadísticas CANTIDAD (Millones m³):")
    print(df_prod["CANTIDAD_(Millones M3)"].describe().round(4).to_string())

    # ── Laboratorio ───────────────────────────────────────────────────
    print("\n\n🔬 LABORATORIO CORAASAN")
    print(f"   Plantas: {sorted(df_lab['PLANTA'].unique())}")
    print(f"   Valores nulos: {df_lab.isnull().sum().sum()}")
    outliers_cloro = (df_lab["CLORO_RESIDUAL_(Mg/l)"] > UMBRAL_CLORO_OUTLIER).sum()
    print(f"   Outliers Cloro > {UMBRAL_CLORO_OUTLIER} mg/L: {outliers_cloro}")
    print(f"\n   IP por planta (promedio histórico):")
    ip_planta = df_lab.groupby("PLANTA")["INDICE_POTABILIDAD_(%)"].mean().sort_values()
    for planta, val in ip_planta.items():
        estado = "⚠️ CRÍTICO" if val < UMBRAL_IP_CRITICO else ("⚡ BAJO" if val < UMBRAL_IP_BAJO else "✅ OK")
        print(f"     {planta:<30} IP={val:.1f}%  {estado}")
    print(f"\n   IP por año (promedio):")
    ip_año = df_lab.groupby("ANO")["INDICE_POTABILIDAD_(%)"].mean()
    for año, val in ip_año.items():
        tendencia = "↘️" if val < 95 else "✅"
        print(f"     {año}: {val:.2f}%  {tendencia}")

    # ── INAPA ─────────────────────────────────────────────────────────
    print("\n\n💧 INAPA – COBERTURA POTABILIDAD")
    cob_año = df_inapa.groupby("ANO")["COBERTURA_PCT"].agg(["mean", "min", "max"]).round(2)
    print(f"\n   {'AÑO':<6} {'MEDIA':>8} {'MÍNIMO':>8} {'MÁXIMO':>8}")
    for año, row in cob_año.iterrows():
        alerta = " ⚠️" if row["min"] < UMBRAL_COB_CRISIS else ""
        print(f"   {año:<6} {row['mean']:>8.2f}% {row['min']:>7.2f}% {row['max']:>7.2f}%{alerta}")
    semanas_crisis = (df_inapa["COBERTURA_PCT"] < UMBRAL_COB_CRISIS).sum()
    print(f"\n   Semanas con cobertura < {UMBRAL_COB_CRISIS}%: {semanas_crisis}")

    # ── INDRHI ────────────────────────────────────────────────────────
    print("\n\n📋 INDRHI – ANÁLISIS CALIDAD")
    print(f"   Registros válidos: {len(df_indrhi)}")
    print(f"   Condición: {df_indrhi['Condicion'].value_counts().to_dict()}")
    print(f"   Tiempo respuesta promedio: {df_indrhi['TIEMPO_RESP'].mean():.1f} días")
    print(f"   Respuestas lentas (>7 días): {(df_indrhi['TIEMPO_RESP'] > 7).sum()}")

    print()


# ══════════════════════════════════════════════════════════════════════════════
# 3. GRÁFICOS DE EXPLORACIÓN
# ══════════════════════════════════════════════════════════════════════════════

def graficos_exploracion(datasets):
    """Genera y guarda 6 gráficos de exploración inicial."""
    df_prod  = datasets["produccion"]
    df_lab   = datasets["laboratorio"]
    df_inapa = datasets["inapa"]
    df_qrs_agua = datasets["qrs_agua"]

    print("=" * 65)
    print("  GENERANDO GRÁFICOS DE EXPLORACIÓN")
    print("=" * 65)

    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor":   "#FAFAFA",
        "axes.grid":        True,
        "grid.alpha":       0.3,
        "font.family":      "sans-serif",
    })

    # ── Fig 1: IP por año ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    ip_año = df_lab.groupby("ANO")["INDICE_POTABILIDAD_(%)"].mean()
    colores = [COLOR_ROJO if v < UMBRAL_IP_BAJO else COLOR_AZUL for v in ip_año.values]
    bars = ax.bar(ip_año.index, ip_año.values, color=colores, width=0.6, zorder=3)
    ax.axhline(UMBRAL_IP_BAJO, color=COLOR_ROJO, lw=1.8, ls="--", label=f"Umbral {UMBRAL_IP_BAJO}%")
    ax.axhline(100, color=COLOR_VERDE, lw=1, ls=":", alpha=0.5)
    for b, v in zip(bars, ip_año.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.3, f"{v:.1f}%",
                ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_ylim(80, 103)
    ax.set_xlabel("Año", fontsize=12)
    ax.set_ylabel("IP Promedio (%)", fontsize=12)
    ax.set_title("Índice de Potabilidad Promedio por Año – CORAASAN (2018–2026)",
                 fontsize=13, fontweight="bold", color=COLOR_AZUL)
    ax.legend(); fig.tight_layout()
    ruta = os.path.join(GRAFICOS_DIR, "01_ip_por_ano.png")
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {ruta}")

    # ── Fig 2: Producción media por planta ────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 4))
    prod_planta = df_prod.groupby("PLANTA")["CANTIDAD_(Millones M3)"].mean().sort_values(ascending=False)
    colores_p = [COLOR_AZUL if i == 0 else COLOR_AZUL2 for i in range(len(prod_planta))]
    bars = ax.bar(range(len(prod_planta)), prod_planta.values, color=colores_p, width=0.6, zorder=3)
    ax.set_xticks(range(len(prod_planta)))
    ax.set_xticklabels([p.replace("PLANTA ", "") for p in prod_planta.index],
                       rotation=30, ha="right", fontsize=9)
    for b, v in zip(bars, prod_planta.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.05, f"{v:.2f}",
                ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    ax.set_ylabel("Producción Media Mensual (Millones m³)", fontsize=11)
    ax.set_title("Producción Media Mensual por Planta – CORAASAN (2018–2025)",
                 fontsize=13, fontweight="bold", color=COLOR_AZUL)
    fig.tight_layout()
    ruta = os.path.join(GRAFICOS_DIR, "02_produccion_por_planta.png")
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {ruta}")

    # ── Fig 3: IP promedio por planta ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    ip_planta = df_lab.groupby("PLANTA")["INDICE_POTABILIDAD_(%)"].mean().sort_values()
    colores_ip = [COLOR_ROJO if v < UMBRAL_IP_CRITICO else
                  (COLOR_NARANJA if v < UMBRAL_IP_BAJO else COLOR_AZUL2)
                  for v in ip_planta.values]
    hbars = ax.barh([p.replace("PLANTA ", "") for p in ip_planta.index],
                    ip_planta.values, color=colores_ip, height=0.6, zorder=3)
    ax.axvline(UMBRAL_IP_BAJO,     color=COLOR_ROJO,    lw=1.8, ls="--", label=f"Umbral {UMBRAL_IP_BAJO}%")
    ax.axvline(UMBRAL_IP_CRITICO,  color=COLOR_NARANJA, lw=1.4, ls=":",  label=f"Umbral {UMBRAL_IP_CRITICO}%")
    for b, v in zip(hbars, ip_planta.values):
        ax.text(v + 0.5, b.get_y() + b.get_height() / 2, f"{v:.1f}%",
                va="center", fontsize=9, fontweight="bold")
    ax.set_xlim(0, 108)
    ax.set_xlabel("IP Promedio Histórico (%)", fontsize=11)
    ax.set_title("Índice de Potabilidad Promedio por Planta – CORAASAN",
                 fontsize=13, fontweight="bold", color=COLOR_AZUL)
    ax.legend(); fig.tight_layout()
    ruta = os.path.join(GRAFICOS_DIR, "03_ip_por_planta.png")
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {ruta}")

    # ── Fig 4: Cobertura INAPA por año ────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    cob = df_inapa.groupby("ANO")["COBERTURA_PCT"].agg(["mean", "min", "max"])
    ax.fill_between(cob.index, cob["min"], cob["max"],
                    alpha=0.15, color=COLOR_AZUL2, label="Rango min–max semanal")
    ax.plot(cob.index, cob["mean"], "o-", color=COLOR_AZUL, lw=2.2,
            markersize=8, label="Media anual", zorder=3)
    ax.axhline(UMBRAL_COB_CRISIS, color=COLOR_ROJO, lw=1.8, ls="--",
               label=f"Umbral crisis {UMBRAL_COB_CRISIS}%")
    for x, v in zip(cob.index, cob["mean"]):
        ax.text(x, v + 1, f"{v:.1f}%", ha="center", va="bottom",
                fontsize=9, fontweight="bold")
    ax.set_ylim(0, 112)
    ax.set_ylabel("Cobertura de Potabilidad (%)", fontsize=11)
    ax.set_title("Cobertura de Potabilidad Semanal INAPA – Nacional (2017–2025)",
                 fontsize=13, fontweight="bold", color=COLOR_AZUL)
    ax.legend(); fig.tight_layout()
    ruta = os.path.join(GRAFICOS_DIR, "04_cobertura_inapa.png")
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {ruta}")

    # ── Fig 5: Distribución cloro residual (boxplot) ──────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    df_lab_clean = df_lab[df_lab["CLORO_RESIDUAL_(Mg/l)"] <= UMBRAL_CLORO_OUTLIER]
    plantas_ord = (df_lab_clean.groupby("PLANTA")["CLORO_RESIDUAL_(Mg/l)"]
                   .median().sort_values().index)
    data_box = [df_lab_clean[df_lab_clean["PLANTA"] == p]["CLORO_RESIDUAL_(Mg/l)"].values
                for p in plantas_ord]
    bp = ax.boxplot(data_box, patch_artist=True, notch=False)
    for patch in bp["boxes"]:
        patch.set_facecolor(COLOR_AZUL2); patch.set_alpha(0.65)
    for med in bp["medians"]:
        med.set_color(COLOR_AZUL); med.set_linewidth(2)
    ax.axhline(UMBRAL_CLORO_MIN, color=COLOR_VERDE, ls="--", lw=1.5,
               label=f"Mín OMS {UMBRAL_CLORO_MIN} mg/L")
    ax.axhline(UMBRAL_CLORO_MAX, color=COLOR_ROJO,  ls="--", lw=1.5,
               label=f"Máx OMS {UMBRAL_CLORO_MAX} mg/L")
    ax.set_xticklabels([p.replace("PLANTA ", "") for p in plantas_ord],
                       rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Cloro Residual (mg/L)", fontsize=11)
    ax.set_title("Distribución de Cloro Residual por Planta (Outlier 100 mg/L excluido)",
                 fontsize=13, fontweight="bold", color=COLOR_AZUL)
    ax.legend(); fig.tight_layout()
    ruta = os.path.join(GRAFICOS_DIR, "05_cloro_boxplot.png")
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {ruta}")

    # ── Fig 6: Quejas de agua por provincia ───────────────────────────
    fig, ax = plt.subplots(figsize=(9, 4))
    qrs_prov = df_qrs_agua["Provincia"].value_counts().head(8)
    colores_q = [COLOR_AZUL if i == 0 else COLOR_AZUL2 for i in range(len(qrs_prov))]
    hbars = ax.barh(qrs_prov.index[::-1], qrs_prov.values[::-1],
                    color=colores_q[::-1], height=0.6, zorder=3)
    for b, v in zip(hbars, qrs_prov.values[::-1]):
        ax.text(v + 0.3, b.get_y() + b.get_height() / 2,
                str(v), va="center", fontsize=10, fontweight="bold")
    ax.set_xlabel("Número de Quejas", fontsize=11)
    ax.set_title("Quejas del Sector Agua por Provincia – QRS Línea 3-1-1 (2025)",
                 fontsize=13, fontweight="bold", color=COLOR_AZUL)
    fig.tight_layout()
    ruta = os.path.join(GRAFICOS_DIR, "06_quejas_por_provincia.png")
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {ruta}")

    print()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    datasets = cargar_datasets()
    reporte_exploracion(datasets)
    graficos_exploracion(datasets)
    print("✅  Fase 1 completada.\n")
