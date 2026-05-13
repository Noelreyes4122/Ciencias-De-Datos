"""
fase2_limpieza.py
=================
CRISP-DM Fase 3: Preparación de los Datos – Limpieza
------------------------------------------------------
- Convierte tipos de datos
- Corrige outlier de Cloro Residual = 100 mg/L
- Trata valores nulos
- Estandariza nombres de plantas (PLANTA_KEY)
- Crea columna FECHA (datetime)
- Guarda datasets limpios en memoria (retorna diccionario)

Ejecutar: python scripts/fase2_limpieza.py
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


# ══════════════════════════════════════════════════════════════════════════════
# FUNCIONES DE LIMPIEZA POR DATASET
# ══════════════════════════════════════════════════════════════════════════════

def limpiar_produccion():
    """Limpia el dataset de Producción CORAASAN."""
    print("🧹 Limpiando: Producción CORAASAN")

    df = pd.read_csv(ARCHIVO_PRODUCCION, encoding="latin-1")

    # ── Paso 1: Conversión de tipos
    df["CANTIDAD_(Millones M3)"] = (
        df["CANTIDAD_(Millones M3)"].str.replace(",", ".").astype(float)
    )

    # ── Paso 2: Eliminar columnas constantes (no aportan info)
    antes = df.shape[1]
    df.drop(columns=["PAIS", "MUNICIPIO"], inplace=True)
    print(f"   Columnas eliminadas (constantes): PAIS, MUNICIPIO  [{antes} → {df.shape[1]}]")

    # ── Paso 3: Estandarizar nombre de planta
    df["PLANTA_KEY"] = df["PLANTA"].map(PLANTA_MAP_PROD)
    sin_key = df["PLANTA_KEY"].isnull().sum()
    if sin_key > 0:
        print(f"   ⚠️  {sin_key} filas sin PLANTA_KEY (verificar mapeo)")

    # ── Paso 4: Crear columna FECHA
    df["MES_NUM"] = df["MES"].map(MESES_ES)
    df["FECHA"] = pd.to_datetime(
        dict(year=df["ANO"], month=df["MES_NUM"], day=1), errors="coerce"
    )

    # ── Paso 5: Transformación logarítmica (normalizar distribución sesgada)
    df["LOG_CANTIDAD"] = np.log1p(df["CANTIDAD_(Millones M3)"])

    # ── Reporte de nulos
    nulos = df.isnull().sum()
    nulos_totales = nulos.sum()
    print(f"   Valores nulos totales: {nulos_totales}")
    ceros = (df["CANTIDAD_(Millones M3)"] == 0).sum()
    print(f"   Registros producción = 0: {ceros}")
    print(f"   Shape final: {df.shape}\n")

    return df


def limpiar_laboratorio():
    """Limpia el dataset de Laboratorio CORAASAN."""
    print("🧹 Limpiando: Laboratorio CORAASAN")

    df = pd.read_csv(ARCHIVO_LAB, encoding="latin-1")

    # ── Paso 1: Conversión de tipos
    df["INDICE_POTABILIDAD_(%)"] = (
        df["INDICE_POTABILIDAD_(%)"].str.replace(",", ".").astype(float)
    )
    df["CLORO_RESIDUAL_(Mg/l)"] = (
        df["CLORO_RESIDUAL_(Mg/l)"].str.replace(",", ".").astype(float)
    )

    # ── Paso 2: Eliminar columnas constantes
    df.drop(columns=["PAIS", "PROVINCIA"], inplace=True)

    # ── Paso 3: Estandarizar nombre de planta
    df["PLANTA_KEY"] = df["PLANTA"].map(PLANTA_MAP_LAB)

    # ── Paso 4: Corregir OUTLIER – Cloro Residual > 10 mg/L
    outlier_mask = df["CLORO_RESIDUAL_(Mg/l)"] > UMBRAL_CLORO_OUTLIER
    n_outliers = outlier_mask.sum()
    if n_outliers > 0:
        # Calcular mediana por planta (excluyendo outliers)
        medianas = (df[~outlier_mask]
                    .groupby("PLANTA_KEY")["CLORO_RESIDUAL_(Mg/l)"]
                    .median())
        mediana_global = df["CLORO_RESIDUAL_(Mg/l)"][~outlier_mask].median()

        for idx in df[outlier_mask].index:
            pk = df.loc[idx, "PLANTA_KEY"]
            valor_original = df.loc[idx, "CLORO_RESIDUAL_(Mg/l)"]
            valor_nuevo = medianas.get(pk, mediana_global)
            df.loc[idx, "CLORO_RESIDUAL_(Mg/l)"] = valor_nuevo
            print(f"   🔧 Outlier corregido fila {idx}: {valor_original:.1f} → {valor_nuevo:.3f} mg/L (mediana planta {pk})")

    print(f"   Outliers Cloro corregidos: {n_outliers}")

    # ── Paso 5: Crear columna FECHA
    df["MES_NUM"] = df["MES"].map(MESES_ES)
    df["FECHA"] = pd.to_datetime(
        dict(year=df["ANO"], month=df["MES_NUM"], day=1), errors="coerce"
    )

    # ── Paso 6: Variable ALERTA_IP (nueva)
    df["ALERTA_IP"] = df["INDICE_POTABILIDAD_(%)"].apply(
        lambda x: "CRITICO" if x < UMBRAL_IP_CRITICO
        else ("BAJO" if x < UMBRAL_IP_BAJO else "NORMAL")
    )
    dist_alerta = df["ALERTA_IP"].value_counts().to_dict()
    print(f"   Distribución ALERTA_IP: {dist_alerta}")

    # ── Paso 7: IP media móvil 3 meses por planta
    df_sorted = df.sort_values(["PLANTA_KEY", "FECHA"])
    df["IP_ROLLING3M"] = (
        df_sorted.groupby("PLANTA_KEY")["INDICE_POTABILIDAD_(%)"]
        .transform(lambda x: x.rolling(3, min_periods=1).mean())
        .round(2)
    )

    print(f"   Valores nulos totales: {df.isnull().sum().sum()}")
    print(f"   Shape final: {df.shape}\n")

    return df


def limpiar_inapa():
    """Limpia el dataset de Cobertura INAPA."""
    print("🧹 Limpiando: Cobertura INAPA")

    df = pd.read_csv(ARCHIVO_INAPA, sep=";", encoding="latin-1")
    df.columns = df.columns.str.strip()
    df.rename(columns={
        "Cobertura de Potabilidad General (%)": "COBERTURA_PCT",
        "Semestre": "SEMESTRE",
        "Año": "ANO"
    }, inplace=True)

    # ── Conversión
    antes_nulos = df["COBERTURA_PCT"].isnull().sum()
    df["COBERTURA_PCT"] = df["COBERTURA_PCT"].str.replace(",", ".").astype(float)
    df["ANO"] = df["ANO"].astype(int)

    # ── Eliminar nulos
    df.dropna(subset=["COBERTURA_PCT"], inplace=True)
    print(f"   Nulos eliminados: {antes_nulos}")

    # ── Variables nuevas
    media_anual = df.groupby("ANO")["COBERTURA_PCT"].transform("mean")
    df["DESV_MEDIA_ANUAL"] = (df["COBERTURA_PCT"] - media_anual).round(2)
    df["SEMANA_CRITICA"] = (df["COBERTURA_PCT"] < UMBRAL_COB_CRISIS).astype(int)

    semanas_crisis = df["SEMANA_CRITICA"].sum()
    print(f"   Semanas críticas (< {UMBRAL_COB_CRISIS}%): {semanas_crisis}")
    print(f"   Shape final: {df.shape}\n")

    return df


def limpiar_indrhi():
    """Limpia el dataset de Análisis INDRHI."""
    print("🧹 Limpiando: INDRHI")

    df = pd.read_csv(ARCHIVO_INDRHI, sep=";", encoding="latin-1")
    df.columns = df.columns.str.strip()

    # ── Eliminar filas nulas (encabezados duplicados)
    antes = len(df)
    df.dropna(subset=["Tipo de Servicio"], inplace=True)
    print(f"   Filas eliminadas (nulos): {antes - len(df)}")

    # ── Parsear fechas
    df["FECHA_ENTRADA"] = pd.to_datetime(
        df["Fecha de entrada"], dayfirst=True, format="mixed", errors="coerce"
    )
    df["ANO"] = df["FECHA_ENTRADA"].dt.year
    df["TIEMPO_RESP"] = pd.to_numeric(df["Tiempo de Respuesta"], errors="coerce")

    # ── Variable nueva: respuesta lenta
    df["RESP_LENTA"] = (df["TIEMPO_RESP"] > 7).astype(int)
    print(f"   Respuestas lentas (>7 días): {df['RESP_LENTA'].sum()}")
    print(f"   Shape final: {df.shape}\n")

    return df


def limpiar_qrs():
    """Limpia y filtra el dataset de Quejas QRS."""
    print("🧹 Limpiando: QRS 3-1-1")

    df = pd.read_excel(ARCHIVO_QRS, sheet_name="DATA")
    total_original = len(df)

    # ── Filtrar quejas de agua
    mask = (
        df["Descripción"].str.upper().str.contains("|".join(AGUA_KEYWORDS), na=False)
        | df["Institución"].str.upper().str.contains("|".join(AGUA_KEYWORDS), na=False)
    )
    df_agua = df[mask].copy()
    print(f"   Total quejas: {total_original:,}  →  Quejas agua: {len(df_agua)}")

    # ── Imputar Canal nulo
    nulos_canal = df_agua["Canal"].isnull().sum()
    df_agua["Canal"] = df_agua["Canal"].fillna("No Registrado")
    print(f"   Canal nulo imputado: {nulos_canal} registros")

    # ── Variables nuevas
    df_agua["PROV"]      = df_agua["Provincia"].str.upper().str.strip()
    df_agua["ES_ROTURA"] = (
        df_agua["Clasificación"].str.upper()
        .str.contains("ROTURA|TUBERIA|FUGA", na=False).astype(int)
    )
    df_agua["DIAS_RESOL"] = pd.to_numeric(df_agua["Dias"], errors="coerce")

    print(f"   Roturas de tuberías identificadas: {df_agua['ES_ROTURA'].sum()}")
    print(f"   Shape final: {df_agua.shape}\n")

    return df_agua


# ══════════════════════════════════════════════════════════════════════════════
# GRÁFICO: ANTES vs DESPUÉS (Cloro outlier)
# ══════════════════════════════════════════════════════════════════════════════

def grafico_outlier_cloro(df_lab_limpio):
    """Muestra distribución cloro antes y después de corrección."""
    df_raw = pd.read_csv(ARCHIVO_LAB, encoding="latin-1")
    df_raw["CLORO_RESIDUAL_(Mg/l)"] = (
        df_raw["CLORO_RESIDUAL_(Mg/l)"].str.replace(",", ".").astype(float)
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, data, titulo in [
        (axes[0], df_raw["CLORO_RESIDUAL_(Mg/l)"],           "ANTES – Con outlier (100 mg/L)"),
        (axes[1], df_lab_limpio["CLORO_RESIDUAL_(Mg/l)"],    "DESPUÉS – Outlier corregido"),
    ]:
        ax.hist(data, bins=30, color=COLOR_AZUL2, edgecolor="white", alpha=0.8)
        ax.axvline(UMBRAL_CLORO_MIN, color=COLOR_VERDE, lw=1.8, ls="--", label=f"Mín OMS {UMBRAL_CLORO_MIN}")
        ax.axvline(UMBRAL_CLORO_MAX, color=COLOR_ROJO,  lw=1.8, ls="--", label=f"Máx OMS {UMBRAL_CLORO_MAX}")
        ax.set_xlabel("Cloro Residual (mg/L)", fontsize=11)
        ax.set_ylabel("Frecuencia", fontsize=11)
        ax.set_title(titulo, fontsize=12, fontweight="bold", color=COLOR_AZUL)
        ax.legend(fontsize=9)

    fig.suptitle("Corrección de Outlier – Cloro Residual CORAASAN",
                 fontsize=13, fontweight="bold", color=COLOR_AZUL)
    plt.tight_layout()
    ruta = os.path.join(GRAFICOS_DIR, "07_outlier_cloro_correccion.png")
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {ruta}")


def grafico_alertas_ip(df_lab_limpio):
    """Pie chart de distribución de ALERTA_IP."""
    conteo = df_lab_limpio["ALERTA_IP"].value_counts()
    colores_alerta = {"NORMAL": COLOR_VERDE, "BAJO": COLOR_NARANJA, "CRITICO": COLOR_ROJO}
    colores = [colores_alerta[k] for k in conteo.index]

    fig, ax = plt.subplots(figsize=(6, 5))
    wedges, texts, autotexts = ax.pie(
        conteo.values, labels=conteo.index, autopct="%1.1f%%",
        colors=colores, startangle=140, pctdistance=0.75
    )
    for t in autotexts:
        t.set_fontsize(11); t.set_fontweight("bold")
    ax.set_title("Distribución de Alertas de Calidad IP\n(CORAASAN 2018–2026)",
                 fontsize=13, fontweight="bold", color=COLOR_AZUL)
    plt.tight_layout()
    ruta = os.path.join(GRAFICOS_DIR, "08_alertas_ip.png")
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {ruta}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def ejecutar_limpieza():
    """Ejecuta toda la limpieza y retorna diccionario de datasets limpios."""
    print("=" * 65)
    print("  LIMPIEZA DE DATOS (Fase 3 CRISP-DM)")
    print("=" * 65)
    print()

    datasets_limpios = {
        "produccion":  limpiar_produccion(),
        "laboratorio": limpiar_laboratorio(),
        "inapa":       limpiar_inapa(),
        "indrhi":      limpiar_indrhi(),
        "qrs_agua":    limpiar_qrs(),
    }

    print("=" * 65)
    print("  GENERANDO GRÁFICOS DE LIMPIEZA")
    print("=" * 65)
    grafico_outlier_cloro(datasets_limpios["laboratorio"])
    grafico_alertas_ip(datasets_limpios["laboratorio"])
    print()

    return datasets_limpios


if __name__ == "__main__":
    datasets_limpios = ejecutar_limpieza()
    print("✅  Fase 2 (Limpieza) completada.\n")

    # Vista rápida del dataset de laboratorio limpio
    print("── Vista previa dataset Laboratorio limpio ──")
    print(datasets_limpios["laboratorio"][
        ["ANO", "MES", "PLANTA_KEY", "INDICE_POTABILIDAD_(%)",
         "CLORO_RESIDUAL_(Mg/l)", "ALERTA_IP"]
    ].head(10).to_string(index=False))
