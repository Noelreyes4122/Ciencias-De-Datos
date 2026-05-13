"""
config.py
=========
Rutas y constantes globales del proyecto.
Modifica DATA_DIR si tus archivos están en otra carpeta.
"""

import os

# ─── Rutas ────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(BASE_DIR, "data")
OUTPUT_DIR  = os.path.join(BASE_DIR, "outputs")
GRAFICOS_DIR= os.path.join(OUTPUT_DIR, "graficos")

# Crear carpetas de salida si no existen
os.makedirs(OUTPUT_DIR,   exist_ok=True)
os.makedirs(GRAFICOS_DIR, exist_ok=True)

# ─── Archivos de entrada ──────────────────────────────────────────────────────
ARCHIVO_PRODUCCION = os.path.join(DATA_DIR, "Datos_Produccion_Agua_Potable_CORAASAN_2018_-_2026.csv")
ARCHIVO_LAB        = os.path.join(DATA_DIR, "Datos_Laboratorio_Agua_Potable_CORAASAN_2018_-_2026.csv")
ARCHIVO_INAPA      = os.path.join(DATA_DIR, "Cobertura-de-Potabilidad-INAPA-2017-2025-2-1-1-1-1-1-1.csv")
ARCHIVO_INDRHI     = os.path.join(DATA_DIR, "Estadistica-Analisis-Calidad-del-Agua-INDRHI-2019-2026.csv")
ARCHIVO_QRS        = os.path.join(DATA_DIR, "Estadisticas-de-QRS-Linea-3-1-1-Enero-Marzo-2025.xlsx")

# ─── Colores del proyecto ─────────────────────────────────────────────────────
COLOR_AZUL    = "#1B4F8A"
COLOR_AZUL2   = "#2E75B6"
COLOR_ROJO    = "#C0392B"
COLOR_VERDE   = "#1A7A4A"
COLOR_NARANJA = "#D4730A"
COLOR_GRIS    = "#ECF0F1"

# ─── Constantes de calidad ────────────────────────────────────────────────────
UMBRAL_IP_CRITICO  = 80   # IP < 80% → CRÍTICO
UMBRAL_IP_BAJO     = 95   # IP < 95% → BAJO
UMBRAL_CLORO_MIN   = 0.2  # mg/L (norma OMS mínimo)
UMBRAL_CLORO_MAX   = 5.0  # mg/L (norma OMS máximo)
UMBRAL_COB_CRISIS  = 60   # Cobertura < 60% → semana crítica
UMBRAL_CLORO_OUTLIER = 10 # Valores > 10 mg/L son outliers

# ─── Meses en español ─────────────────────────────────────────────────────────
MESES_ES = {
    "ENERO": 1, "FEBRERO": 2, "MARZO": 3, "ABRIL": 4,
    "MAYO": 5, "JUNIO": 6, "JULIO": 7, "AGOSTO": 8,
    "SEPTIEMBRE": 9, "OCTUBRE": 10, "NOVIEMBRE": 11, "DICIEMBRE": 12
}

# ─── Mapeo de nombres de plantas (estandarización) ───────────────────────────
PLANTA_MAP_PROD = {
    "PLANTA NORIEGA":          "NORIEGA",
    "PLANTA NORIEGA II":       "NORIEGA II",
    "PLANTA 10 MGD":           "10 MGD",
    "PLANTA 25 MGD":           "25 MGD",
    "PLANTA VILLA GONZALEZ":   "VILLA GONZALEZ",
    "PLANTA VILLA GONZALEZ II":"VILLA GONZALEZ II",
    "PLANTA DE CIENFUEGOS":    "CIENFUEGOS",
    "PLANTA DE LA CANELA":     "LA CANELA",
    "PLANTA LA BARRANQUITA":   "LA BARRANQUITA",
    "SAJOMA":                  "SAJOMA",
}

PLANTA_MAP_LAB = {
    "LA NORIEGA":           "NORIEGA",
    "LA NORIEGA II":        "NORIEGA II",
    "PLANTA LA NORIEGA":    "NORIEGA",
    "PLANTA LA NORIEGA II": "NORIEGA II",
    "PLANTA 10 MGD":        "10 MGD",
    "PLANTA 25 MGD":        "25 MGD",
    "VILLA GONZALEZ":       "VILLA GONZALEZ",
    "VILLA GONZALEZ II":    "VILLA GONZALEZ II",
}

# ─── Keywords para filtrar quejas de agua ─────────────────────────────────────
AGUA_KEYWORDS = [
    "AGUA", "ACUEDUCTO", "INAPA", "CAASD", "CORAASAN",
    "TUBERIA", "POTABLE", "ALCANTARILLADO", "PLOMERIA"
]
