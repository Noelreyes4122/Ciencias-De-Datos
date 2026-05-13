"""
fase3_integracion.py
====================
CRISP-DM Fase 3: Preparación de los Datos – Integración
---------------------------------------------------------
- Agrupa producción y laboratorio por PLANTA_KEY/ANO/MES
- Realiza INNER JOIN entre ambos datasets
- Construye nuevas variables derivadas
- Exporta dataset final formateado a CSV
- Genera gráficos del dataset integrado

Ejecutar: python scripts/fase3_integracion.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from config import *
from fase2_limpieza import ejecutar_limpieza


# ══════════════════════════════════════════════════════════════════════════════
# INTEGRACIÓN: JOIN PRODUCCIÓN + LABORATORIO
# ══════════════════════════════════════════════════════════════════════════════

def integrar_datasets(datasets_limpios):
    """
    Hace INNER JOIN entre Producción y Laboratorio CORAASAN.
    Retorna el dataset integrado con variables derivadas.
    """
    print("=" * 65)
    print("  INTEGRACIÓN DE DATOS")
    print("=" * 65)
    print()

    df_prod = datasets_limpios["produccion"]
    df_lab  = datasets_limpios["laboratorio"]

    # ── Paso 1: Agrupar Producción por PLANTA_KEY / ANO / MES ────────
    prod_agg = df_prod.groupby(["PLANTA_KEY", "ANO", "MES"], as_index=False).agg(
        PRODUCCION_MM3 = ("CANTIDAD_(Millones M3)", "sum"),
        LOG_PROD       = ("LOG_CANTIDAD",            "mean"),
    ).round(4)
    print(f"📦 Producción agrupada:   {prod_agg.shape[0]} grupos (PLANTA × AÑO × MES)")

    # ── Paso 2: Agrupar Laboratorio por PLANTA_KEY / ANO / MES ───────
    lab_agg = df_lab.groupby(["PLANTA_KEY", "ANO", "MES"], as_index=False).agg(
        IP_PROM    = ("INDICE_POTABILIDAD_(%)",  "mean"),
        CLORO_PROM = ("CLORO_RESIDUAL_(Mg/l)",   "mean"),
        N_MUESTRAS = ("CANT. MUESTRA",            "sum"),
        ALERTA     = ("ALERTA_IP",  lambda x: x.mode()[0] if len(x) > 0 else "NORMAL"),
    )
    lab_agg["IP_PROM"]    = lab_agg["IP_PROM"].round(2)
    lab_agg["CLORO_PROM"] = lab_agg["CLORO_PROM"].round(3)
    print(f"🔬 Laboratorio agrupado:  {lab_agg.shape[0]} grupos (PLANTA × AÑO × MES)")

    # ── Paso 3: INNER JOIN ────────────────────────────────────────────
    print(f"\n🔗 Realizando INNER JOIN por [PLANTA_KEY, ANO, MES]...")
    df_int = pd.merge(prod_agg, lab_agg, on=["PLANTA_KEY", "ANO", "MES"], how="inner")
    print(f"   Registros resultantes: {df_int.shape[0]}")
    print(f"   Plantas comunes:       {df_int['PLANTA_KEY'].nunique()} → {sorted(df_int['PLANTA_KEY'].unique())}")
    print(f"   Período:               {df_int['ANO'].min()} – {df_int['ANO'].max()}")

    # ── Paso 4: Columna FECHA ─────────────────────────────────────────
    df_int["FECHA"] = pd.to_datetime(
        dict(year=df_int["ANO"], month=df_int["MES"].map(MESES_ES), day=1),
        errors="coerce"
    )

    # ── Paso 5: Variables derivadas ───────────────────────────────────
    print("\n🔨 Construyendo variables derivadas...")

    # 5a. Índice de Eficiencia = (IP/100) × Producción
    df_int["IDX_EFICIENCIA"] = ((df_int["IP_PROM"] / 100) * df_int["PRODUCCION_MM3"]).round(4)

    # 5b. Cumple norma OMS de Cloro
    df_int["CUMPLE_OMS_CLORO"] = (
        (df_int["CLORO_PROM"] >= UMBRAL_CLORO_MIN) &
        (df_int["CLORO_PROM"] <= UMBRAL_CLORO_MAX)
    ).astype(int)

    # 5c. Variación porcentual IP respecto al mes anterior (por planta)
    df_int_sorted = df_int.sort_values(["PLANTA_KEY", "FECHA"])
    df_int["VAR_IP_MENSUAL"] = (
        df_int_sorted.groupby("PLANTA_KEY")["IP_PROM"]
        .pct_change() * 100
    ).round(2)

    # 5d. Semestre
    df_int["SEMESTRE"] = df_int["FECHA"].dt.month.apply(
        lambda m: "1er Semestre" if m <= 6 else "2do Semestre"
    )

    pct_cumple = df_int["CUMPLE_OMS_CLORO"].mean() * 100
    print(f"   IDX_EFICIENCIA creado  (rango: {df_int['IDX_EFICIENCIA'].min():.3f} – {df_int['IDX_EFICIENCIA'].max():.3f})")
    print(f"   CUMPLE_OMS_CLORO:      {pct_cumple:.1f}% de registros dentro de norma")
    print(f"   VAR_IP_MENSUAL creado  (variación mensual de IP por planta)")
    print(f"   SEMESTRE creado")

    # ── Paso 6: Renombrar columnas al español formal ──────────────────
    df_final = df_int.rename(columns={
        "PLANTA_KEY":     "PLANTA",
        "ANO":            "AÑO",
        "PRODUCCION_MM3": "PRODUCCION_MILLONES_M3",
        "LOG_PROD":       "LOG_PRODUCCION",
        "IP_PROM":        "INDICE_POTABILIDAD_PCT",
        "CLORO_PROM":     "CLORO_RESIDUAL_MGL",
        "N_MUESTRAS":     "NUM_MUESTRAS",
        "ALERTA":         "ALERTA_CALIDAD",
        "IDX_EFICIENCIA": "INDICE_EFICIENCIA",
    })

    # ── Paso 7: Orden de columnas lógico ─────────────────────────────
    columnas_ord = [
        "FECHA", "AÑO", "MES", "SEMESTRE", "PLANTA",
        "PRODUCCION_MILLONES_M3", "LOG_PRODUCCION",
        "INDICE_POTABILIDAD_PCT", "CLORO_RESIDUAL_MGL",
        "NUM_MUESTRAS", "ALERTA_CALIDAD",
        "CUMPLE_OMS_CLORO", "VAR_IP_MENSUAL", "INDICE_EFICIENCIA"
    ]
    df_final = df_final[columnas_ord]

    # ── Paso 8: Verificación final de calidad ─────────────────────────
    nulos_final = df_final.isnull().sum()
    print(f"\n✅ Dataset integrado final:")
    print(f"   Forma:  {df_final.shape[0]} filas × {df_final.shape[1]} columnas")
    print(f"   Nulos:  {nulos_final[nulos_final > 0].to_dict() if nulos_final.sum() > 0 else '0 (completitud 100%)'}")

    return df_final


# ══════════════════════════════════════════════════════════════════════════════
# EXPORTAR DATASET FINAL
# ══════════════════════════════════════════════════════════════════════════════

def exportar_dataset(df_final):
    """Guarda el dataset integrado en CSV."""
    ruta_csv = os.path.join(OUTPUT_DIR, "dataset_agua_integrado_final.csv")
    df_final["FECHA"] = df_final["FECHA"].dt.strftime("%Y-%m-%d")
    df_final.to_csv(ruta_csv, index=False, encoding="utf-8")
    print(f"\n💾 Dataset guardado: {ruta_csv}")
    print(f"   {df_final.shape[0]} registros × {df_final.shape[1]} columnas\n")
    return ruta_csv


# ══════════════════════════════════════════════════════════════════════════════
# GRÁFICOS DEL DATASET INTEGRADO
# ══════════════════════════════════════════════════════════════════════════════

def graficos_integracion(df_final):
    """Genera gráficos sobre el dataset ya integrado."""
    print("=" * 65)
    print("  GENERANDO GRÁFICOS DE INTEGRACIÓN")
    print("=" * 65)

    # ── Fig 9: Scatter IP vs Producción (coloreado por cloro) ─────────
    fig, ax = plt.subplots(figsize=(10, 5))
    sc = ax.scatter(
        df_final["PRODUCCION_MILLONES_M3"],
        df_final["INDICE_POTABILIDAD_PCT"],
        c=df_final["CLORO_RESIDUAL_MGL"],
        cmap="coolwarm_r", s=60, alpha=0.75,
        edgecolors="white", linewidth=0.4
    )
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Cloro Residual (mg/L)", fontsize=10)
    ax.axhline(UMBRAL_IP_BAJO, color=COLOR_ROJO, lw=1.5, ls="--",
               label=f"Umbral IP {UMBRAL_IP_BAJO}%", alpha=0.7)
    ax.set_xlabel("Producción Mensual (Millones m³)", fontsize=12)
    ax.set_ylabel("Índice de Potabilidad (%)", fontsize=12)
    ax.set_title("IP vs Producción – Dataset Integrado CORAASAN\n(Color = Cloro Residual mg/L)",
                 fontsize=13, fontweight="bold", color=COLOR_AZUL)
    ax.legend(fontsize=10)
    fig.tight_layout()
    ruta = os.path.join(GRAFICOS_DIR, "09_scatter_ip_vs_produccion.png")
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {ruta}")

    # ── Fig 10: Índice de Eficiencia por planta ───────────────────────
    fig, ax = plt.subplots(figsize=(9, 4))
    ef_planta = df_final.groupby("PLANTA")["INDICE_EFICIENCIA"].mean().sort_values(ascending=False)
    colores = [COLOR_AZUL if i == 0 else COLOR_AZUL2 for i in range(len(ef_planta))]
    bars = ax.bar(ef_planta.index, ef_planta.values, color=colores, width=0.6, zorder=3)
    for b, v in zip(bars, ef_planta.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.005,
                f"{v:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_ylabel("Índice Eficiencia Promedio\n(IP/100 × Producción Mm³)", fontsize=10)
    ax.set_title("Índice de Eficiencia por Planta – Dataset Integrado",
                 fontsize=13, fontweight="bold", color=COLOR_AZUL)
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    ruta = os.path.join(GRAFICOS_DIR, "10_indice_eficiencia_por_planta.png")
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {ruta}")

    # ── Fig 11: Evolución IP por planta (serie temporal) ──────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    colores_lineas = [COLOR_AZUL, COLOR_ROJO, COLOR_VERDE, COLOR_NARANJA, COLOR_AZUL2, "#9B59B6"]
    df_final["FECHA_DT"] = pd.to_datetime(df_final["FECHA"])
    for i, planta in enumerate(sorted(df_final["PLANTA"].unique())):
        sub = df_final[df_final["PLANTA"] == planta].sort_values("FECHA_DT")
        ax.plot(sub["FECHA_DT"], sub["INDICE_POTABILIDAD_PCT"],
                marker="o", markersize=4, linewidth=1.8,
                color=colores_lineas[i % len(colores_lineas)], label=planta, alpha=0.85)
    ax.axhline(UMBRAL_IP_BAJO, color=COLOR_ROJO, lw=1.5, ls="--", alpha=0.6, label=f"Umbral {UMBRAL_IP_BAJO}%")
    ax.set_ylabel("Índice de Potabilidad (%)", fontsize=11)
    ax.set_xlabel("Fecha", fontsize=11)
    ax.set_title("Evolución Mensual del Índice de Potabilidad por Planta (2018–2026)",
                 fontsize=13, fontweight="bold", color=COLOR_AZUL)
    ax.legend(fontsize=8, loc="lower left", ncol=3)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    ruta = os.path.join(GRAFICOS_DIR, "11_evolucion_ip_por_planta.png")
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {ruta}")

    # ── Fig 12: Heatmap Planta vs Año (IP promedio) ───────────────────
    fig, ax = plt.subplots(figsize=(11, 4))
    pivot = df_final.pivot_table(
        index="PLANTA", columns="AÑO",
        values="INDICE_POTABILIDAD_PCT", aggfunc="mean"
    ).round(1)
    im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto",
                   vmin=50, vmax=100)
    plt.colorbar(im, ax=ax, label="IP Promedio (%)")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, fontsize=9)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                        fontsize=8, fontweight="bold",
                        color="white" if val < 75 else "black")
    ax.set_title("Heatmap: Índice de Potabilidad (%) por Planta y Año",
                 fontsize=13, fontweight="bold", color=COLOR_AZUL)
    ax.set_xlabel("Año", fontsize=11)
    ax.set_ylabel("Planta", fontsize=11)
    fig.tight_layout()
    ruta = os.path.join(GRAFICOS_DIR, "12_heatmap_ip_planta_ano.png")
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {ruta}")

    print()


# ══════════════════════════════════════════════════════════════════════════════
# RESUMEN ESTADÍSTICO DEL DATASET FINAL
# ══════════════════════════════════════════════════════════════════════════════

def resumen_final(df_final):
    """Imprime resumen estadístico del dataset integrado."""
    print("=" * 65)
    print("  RESUMEN ESTADÍSTICO – DATASET INTEGRADO FINAL")
    print("=" * 65)

    vars_num = ["PRODUCCION_MILLONES_M3", "INDICE_POTABILIDAD_PCT",
                "CLORO_RESIDUAL_MGL", "INDICE_EFICIENCIA"]
    print(df_final[vars_num].describe().round(4).to_string())

    print(f"\n  Distribución ALERTA_CALIDAD:")
    for k, v in df_final["ALERTA_CALIDAD"].value_counts().items():
        pct = v / len(df_final) * 100
        print(f"    {k:<10} {v:>4} registros ({pct:.1f}%)")

    print(f"\n  Cumplimiento norma OMS Cloro: "
          f"{df_final['CUMPLE_OMS_CLORO'].mean()*100:.1f}% de registros")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Ejecutar limpieza primero
    datasets_limpios = ejecutar_limpieza()

    # Integrar
    df_final = integrar_datasets(datasets_limpios)

    # Resumen
    resumen_final(df_final)

    # Gráficos
    graficos_integracion(df_final)

    # Exportar CSV
    exportar_dataset(df_final)

    print("✅  Fase 3 (Integración) completada.\n")

    # Preview
    print("── Vista previa dataset integrado final ──")
    print(df_final[["FECHA", "PLANTA", "AÑO", "MES",
                     "PRODUCCION_MILLONES_M3", "INDICE_POTABILIDAD_PCT",
                     "CLORO_RESIDUAL_MGL", "ALERTA_CALIDAD",
                     "INDICE_EFICIENCIA"]].head(8).to_string(index=False))
