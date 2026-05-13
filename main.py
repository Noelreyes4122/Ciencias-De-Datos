"""
main.py
=======
Pipeline completo del proyecto de Ciencia de Datos.
Ejecuta todas las fases en orden.

Uso:
    python main.py              ← Ejecuta todo
    python main.py --fase 1     ← Solo exploración
    python main.py --fase 2     ← Solo limpieza
    python main.py --fase 3     ← Solo integración
    python main.py --fase 4     ← Solo modelado
"""

import sys
import os
import time

# Agregar carpeta scripts al path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))

# ── Verificar que existan los archivos de datos ────────────────────────────
from scripts.config import (DATA_DIR, OUTPUT_DIR, GRAFICOS_DIR,
                     ARCHIVO_PRODUCCION, ARCHIVO_LAB,
                     ARCHIVO_INAPA, ARCHIVO_INDRHI, ARCHIVO_QRS)

def verificar_datos():
    archivos = [ARCHIVO_PRODUCCION, ARCHIVO_LAB, ARCHIVO_INAPA,
                ARCHIVO_INDRHI, ARCHIVO_QRS]
    print("🔍 Verificando archivos de datos...")
    todos_ok = True
    for f in archivos:
        existe = os.path.exists(f)
        estado = "✅" if existe else "❌ FALTA"
        print(f"   {estado}  {os.path.basename(f)}")
        if not existe:
            todos_ok = False
    if not todos_ok:
        print(f"\n⚠️  Coloca los archivos faltantes en la carpeta: {DATA_DIR}")
        sys.exit(1)
    print()
    return True


def ejecutar_fase(num):
    t0 = time.time()

    if num == 1:
        print("\n" + "━" * 65)
        print(" FASE 1 / FASE 2 CRISP-DM: Carga y Exploración de Datos")
        print("━" * 65)
        from scripts.fase1_carga_exploracion import cargar_datasets, reporte_exploracion, graficos_exploracion
        ds = cargar_datasets()
        reporte_exploracion(ds)
        graficos_exploracion(ds)

    elif num == 2:
        print("\n" + "━" * 65)
        print(" FASE 3 CRISP-DM: Limpieza de Datos")
        print("━" * 65)
        from scripts.fase2_limpieza import ejecutar_limpieza
        ejecutar_limpieza()

    elif num == 3:
        print("\n" + "━" * 65)
        print(" FASE 3 CRISP-DM: Integración y Construcción de Datos")
        print("━" * 65)
        from scripts.fase2_limpieza   import ejecutar_limpieza
        from scripts.fase3_integracion import integrar_datasets, resumen_final, graficos_integracion, exportar_dataset
        ds_limpios = ejecutar_limpieza()
        df_final   = integrar_datasets(ds_limpios)
        resumen_final(df_final)
        graficos_integracion(df_final)
        exportar_dataset(df_final)

    elif num == 4:
        print("\n" + "━" * 65)
        print(" FASE 4 CRISP-DM: Modelado")
        print("━" * 65)
        from scripts.fase2_limpieza     import ejecutar_limpieza
        from scripts.fase3_integracion  import integrar_datasets
        from scripts.fase4_modelado     import (preparar_features, modelo_regresion_lineal,
                                         modelo_random_forest, modelo_kmeans,
                                         modelo_arbol_decision, comparar_modelos)
        from sklearn.model_selection import train_test_split

        ds_limpios = ejecutar_limpieza()
        df_final   = integrar_datasets(ds_limpios)
        X, y, X_cluster, feat_cols, df_enc = preparar_features(df_final)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42
        )
        res_lr  = modelo_regresion_lineal(X_train, X_test, y_train, y_test, feat_cols)
        res_rf  = modelo_random_forest(X_train, X_test, y_train, y_test, feat_cols)
        res_km  = modelo_kmeans(X_cluster)
        res_arb = modelo_arbol_decision(df_final)
        comparar_modelos(res_lr, res_rf)

    print(f"\n  ⏱  Tiempo: {time.time() - t0:.1f}s\n")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 65)
    print("  💧 ANÁLISIS DE AGUA POTABLE – REPÚBLICA DOMINICANA")
    print("     Proyecto CRISP-DM | Noel Reyes 21-2021 | UNPHU 2026")
    print("=" * 65)

    verificar_datos()

    # Leer argumento --fase
    fase_arg = None
    if "--fase" in sys.argv:
        idx = sys.argv.index("--fase")
        if idx + 1 < len(sys.argv):
            fase_arg = int(sys.argv[idx + 1])

    if fase_arg:
        ejecutar_fase(fase_arg)
    else:
        # Ejecutar todo el pipeline
        for f in [1, 2, 3, 4]:
            ejecutar_fase(f)
        print("=" * 65)
        print("  ✅ PIPELINE COMPLETO FINALIZADO")
        print(f"     Gráficos → {GRAFICOS_DIR}")
        print(f"     Dataset  → {OUTPUT_DIR}/dataset_agua_integrado_final.csv")
        print("=" * 65)
