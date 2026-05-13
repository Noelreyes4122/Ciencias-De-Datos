# 💧 Proyecto: Análisis de Agua Potable en República Dominicana
## CRISP-DM – INF-379-02 Introducción a la Ciencia de Datos
### Noel Reyes | 21-2021 | UNPHU 2026

---

## Descripción
Análisis de producción, calidad y reclamos del agua potable en RD (2017–2026).
Fuentes: CORAASAN, INAPA, INDRHI, Línea 3-1-1 del Gobierno Dominicano.

---

## Estructura del proyecto
```
proyecto_agua/
│
├── data/                          ← Coloca aquí tus datasets
│   ├── Datos_Produccion_Agua_Potable_CORAASAN_2018_-_2026.csv
│   ├── Datos_Laboratorio_Agua_Potable_CORAASAN_2018_-_2026.csv
│   ├── Cobertura-de-Potabilidad-INAPA-2017-2025-2-1-1-1-1-1-1.csv
│   ├── Estadistica-Analisis-Calidad-del-Agua-INDRHI-2019-2026.csv
│   └── Estadisticas-de-QRS-Linea-3-1-1-Enero-Marzo-2025.xlsx
│
├── scripts/
│   ├── config.py                  ← Rutas y constantes globales
│   ├── fase1_carga_exploracion.py ← Fase 2 CRISP-DM: Comprensión datos
│   ├── fase2_limpieza.py          ← Fase 3 CRISP-DM: Preparación datos
│   ├── fase3_integracion.py       ← Integración y nuevas variables
│   └── fase4_modelado.py          ← Fase 4 CRISP-DM: Modelado
│
├── outputs/
│   └── graficos/                  ← Gráficos generados automáticamente
│
├── main.py                        ← Ejecuta todo el pipeline completo
└── requirements.txt
```

---

## Setup en VS Code

### 1. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 2. Coloca los datasets en la carpeta `data/`

### 3. Ejecutar todo el pipeline
```bash
python main.py
```

### 4. O ejecutar fase por fase
```bash
python scripts/fase1_carga_exploracion.py
python scripts/fase2_limpieza.py
python scripts/fase3_integracion.py
python scripts/fase4_modelado.py
```

---

## Datasets utilizados
| Dataset | Institución | Período | Registros |
|---------|-------------|---------|-----------|
| Producción Agua CORAASAN | CORAASAN | 2018–2026 | 870 |
| Laboratorio CORAASAN | CORAASAN | 2018–2026 | 463 |
| Cobertura Potabilidad | INAPA | 2017–2025 | 755 |
| Análisis Calidad Agua | INDRHI | 2019–2026 | 304 |
| Quejas Ciudadanas QRS | Gobierno RD | Ene–Mar 2025 | 1,948 |
