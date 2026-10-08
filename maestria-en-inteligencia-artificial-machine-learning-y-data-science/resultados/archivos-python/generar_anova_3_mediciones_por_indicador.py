# ============================================================
# ANOVA DE 3 MEDICIONES POR INDICADOR
# TESIS: SISTEMA DE GESTIÓN ACADÉMICA WEB INTEGRADO CON CHATBOT
# ============================================================
#
# OBJETIVO:
#   Analizar, de forma independiente, cada uno de los 16 indicadores
#   utilizando únicamente estas tres mediciones:
#       1. Control Pre-test
#       2. Control Post-test
#       3. Experimental Pre-test
#
# IMPORTANTE:
#   SE EXCLUYE COMPLETAMENTE Experimental Post-test.
#
#   Debido a esta exclusión, ya no corresponde utilizar un ANOVA
#   factorial 2x2 ni calcular una interacción Grupo x Momento.
#   En su lugar se aplica un ANOVA de un factor (3 condiciones)
#   para cada indicador.
#
#   El análisis permite verificar si existen diferencias globales
#   entre las tres mediciones antes de incorporar Experimental Post-test.
#
# GENERA:
#   - Tabla principal por indicador.
#   - ANOVA completo por indicador.
#   - Medias, desviaciones estándar y N.
#   - Comparaciones post-hoc Tukey HSD.
#   - Tamaño del efecto eta cuadrado.
#   - Gráfico de medias de las 3 mediciones por indicador.
#   - Excel y resumen TXT.
# ============================================================

from pathlib import Path
import warnings
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd

warnings.filterwarnings("ignore")

# ============================================================
# 1. CONFIGURACIÓN
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

ARCHIVOS = {
    "Control Pre-test": BASE_DIR / "grupo-control-pre-test-codificado.xlsx",
    "Control Post-test": BASE_DIR / "grupo-control-post-test-codificado.xlsx",
    "Experimental Pre-test": BASE_DIR / "grupo-experimental-pre-test-codificado.xlsx",
}

SALIDA = BASE_DIR / "resultados_anova_3_mediciones_por_indicador"
GRAFICOS = SALIDA / "graficos"
SALIDA.mkdir(parents=True, exist_ok=True)
GRAFICOS.mkdir(parents=True, exist_ok=True)

ALPHA = 0.05

INDICADORES = [
    "Tiempo Inscripción",
    "Facilidad Inscripción",
    "Acceso Inscripción",
    "Satisfacción Inscripciones",
    "Tiempo información",
    "Facilidad información",
    "Numero Consultas",
    "Calidad Atención",
    "Tiempo Calificación",
    "Facilidad Calificación",
    "Acceso Calificaciones",
    "Satisfacción Calificaciones",
    "Tiempo Atención Cliente",
    "Facilidad Atención Cliente",
    "Acceso Atención Cliente",
    "Satisfacción Atención Cliente",
]

DIMENSIONES = {
    "Sistema de gestión académica web": INDICADORES[0:4],
    "Chatbot": INDICADORES[4:8],
    "Gestión académica": INDICADORES[8:12],
    "Atención al cliente": INDICADORES[12:16],
}

INDICADOR_DIMENSION = {
    indicador: dimension
    for dimension, indicadores in DIMENSIONES.items()
    for indicador in indicadores
}

# ============================================================
# 2. FUNCIONES
# ============================================================

def nombre_archivo(texto):
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", texto.lower()).strip("_")


def leer_archivo(ruta):
    if not ruta.exists():
        raise FileNotFoundError(
            f"\nNo se encontró el archivo:\n{ruta}\n"
            "Colócalo en la misma carpeta del programa."
        )

    df = pd.read_excel(ruta)

    faltantes = [c for c in INDICADORES if c not in df.columns]
    if faltantes:
        raise ValueError(
            f"\nEl archivo {ruta.name} no contiene los indicadores requeridos:\n"
            + "\n".join(f"- {c}" for c in faltantes)
        )

    for columna in INDICADORES:
        df[columna] = pd.to_numeric(df[columna], errors="coerce")
        valores = df[columna].dropna()
        invalidos = valores[~valores.isin([1, 2, 3, 4, 5])]

        if not invalidos.empty:
            raise ValueError(
                f"\nLa columna '{columna}' del archivo {ruta.name} "
                f"contiene valores fuera de la escala 1-5: "
                f"{sorted(invalidos.unique().tolist())}"
            )

    return df


def interpretar_eta(eta):
    if pd.isna(eta):
        return "No calculable"
    if eta < 0.01:
        return "Efecto muy pequeño"
    if eta < 0.06:
        return "Efecto pequeño"
    if eta < 0.14:
        return "Efecto mediano"
    return "Efecto grande"


def decision(p):
    if pd.isna(p):
        return "No calculable"
    return "Rechazar H0" if p < ALPHA else "No rechazar H0"


def significancia(p):
    if pd.isna(p):
        return "No calculable"
    if p < 0.001:
        return "p < 0.001"
    if p < 0.01:
        return "p < 0.01"
    if p < 0.05:
        return "p < 0.05"
    return "p ≥ 0.05"


# ============================================================
# 3. CARGAR DATOS
# ============================================================

print("=" * 90)
print("ANOVA DE 3 MEDICIONES POR INDICADOR")
print("=" * 90)
print("Se excluye completamente: Experimental Post-test")
print()

datos = {}

for nombre, ruta in ARCHIVOS.items():
    print(f"Leyendo: {ruta.name}")
    datos[nombre] = leer_archivo(ruta)
    print(f"  Participantes: {len(datos[nombre])}")

# ============================================================
# 4. ANOVA DE UN FACTOR POR INDICADOR
# ============================================================

resultados = []
tablas_completas = []
resumen_medias = []
posthoc_resultados = []

condiciones = list(ARCHIVOS.keys())

for indicador in INDICADORES:

    grupos = []
    nombres_grupos = []

    for condicion in condiciones:
        serie = datos[condicion][indicador].dropna().astype(float)
        grupos.append(serie.values)
        nombres_grupos.append(condicion)

        resumen_medias.append({
            "Dimensión": INDICADOR_DIMENSION[indicador],
            "Indicador": indicador,
            "Condición": condicion,
            "N": len(serie),
            "Media": serie.mean() if len(serie) else np.nan,
            "Desviación estándar": serie.std(ddof=1) if len(serie) > 1 else np.nan,
            "Mediana": serie.median() if len(serie) else np.nan,
        })

    # ANOVA de un factor
    if all(len(g) >= 2 for g in grupos):
        f_stat, p_valor = f_oneway(*grupos)

        todos = np.concatenate(grupos)
        media_general = np.mean(todos)
        ss_total = np.sum((todos - media_general) ** 2)
        ss_entre = sum(len(g) * (np.mean(g) - media_general) ** 2 for g in grupos)
        eta2 = ss_entre / ss_total if ss_total > 0 else np.nan
    else:
        f_stat = np.nan
        p_valor = np.nan
        eta2 = np.nan

    n_total = sum(len(g) for g in grupos)
    gl_entre = len(grupos) - 1
    gl_dentro = n_total - len(grupos)

    medias = [np.mean(g) if len(g) else np.nan for g in grupos]
    sds = [np.std(g, ddof=1) if len(g) > 1 else np.nan for g in grupos]

    resultados.append({
        "Dimensión": INDICADOR_DIMENSION[indicador],
        "Indicador": indicador,
        "N total": n_total,
        "Control Pre-test - Media": medias[0],
        "Control Post-test - Media": medias[1],
        "Experimental Pre-test - Media": medias[2],
        "Control Pre-test - DE": sds[0],
        "Control Post-test - DE": sds[1],
        "Experimental Pre-test - DE": sds[2],
        "F": f_stat,
        "gl entre": gl_entre,
        "gl dentro": gl_dentro,
        "p-valor": p_valor,
        "Significancia": significancia(p_valor),
        "Eta²": eta2,
        "Magnitud Eta²": interpretar_eta(eta2),
        "Decisión": decision(p_valor),
    })

    # Tabla ANOVA completa por indicador
    ss_total = np.nan
    ss_entre = np.nan
    ss_dentro = np.nan

    if all(len(g) >= 2 for g in grupos):
        todos = np.concatenate(grupos)
        media_general = np.mean(todos)
        ss_total = np.sum((todos - media_general) ** 2)
        ss_entre = sum(len(g) * (np.mean(g) - media_general) ** 2 for g in grupos)
        ss_dentro = ss_total - ss_entre

    ms_entre = ss_entre / gl_entre if pd.notna(ss_entre) and gl_entre > 0 else np.nan
    ms_dentro = ss_dentro / gl_dentro if pd.notna(ss_dentro) and gl_dentro > 0 else np.nan

    tablas_completas.extend([
        {
            "Dimensión": INDICADOR_DIMENSION[indicador],
            "Indicador": indicador,
            "Fuente de variación": "Entre condiciones",
            "Suma de cuadrados": ss_entre,
            "gl": gl_entre,
            "Cuadrado medio": ms_entre,
            "F": f_stat,
            "p-valor": p_valor,
        },
        {
            "Dimensión": INDICADOR_DIMENSION[indicador],
            "Indicador": indicador,
            "Fuente de variación": "Dentro de condiciones (Error)",
            "Suma de cuadrados": ss_dentro,
            "gl": gl_dentro,
            "Cuadrado medio": ms_dentro,
            "F": np.nan,
            "p-valor": np.nan,
        },
        {
            "Dimensión": INDICADOR_DIMENSION[indicador],
            "Indicador": indicador,
            "Fuente de variación": "Total",
            "Suma de cuadrados": ss_total,
            "gl": n_total - 1,
            "Cuadrado medio": np.nan,
            "F": np.nan,
            "p-valor": np.nan,
        },
    ])

    # --------------------------------------------------------
    # Tukey HSD: comparación por pares
    # --------------------------------------------------------
    valores = []
    etiquetas = []

    for condicion in condiciones:
        serie = datos[condicion][indicador].dropna().astype(float)
        valores.extend(serie.tolist())
        etiquetas.extend([condicion] * len(serie))

    if len(set(etiquetas)) == 3 and len(valores) > 3:
        tukey = pairwise_tukeyhsd(
            endog=np.array(valores),
            groups=np.array(etiquetas),
            alpha=ALPHA,
        )

        tabla_tukey = pd.DataFrame(
            data=tukey._results_table.data[1:],
            columns=tukey._results_table.data[0],
        )

        for _, fila in tabla_tukey.iterrows():
            posthoc_resultados.append({
                "Dimensión": INDICADOR_DIMENSION[indicador],
                "Indicador": indicador,
                "Grupo 1": fila["group1"],
                "Grupo 2": fila["group2"],
                "Diferencia media": float(fila["meandiff"]),
                "p-ajustado": float(fila["p-adj"]),
                "Límite inferior": float(fila["lower"]),
                "Límite superior": float(fila["upper"]),
                "Significativo": "Sí" if str(fila["reject"]).lower() == "true" else "No",
            })

    # --------------------------------------------------------
    # Gráfico por indicador
    # --------------------------------------------------------
    medias_grafico = [np.mean(g) if len(g) else np.nan for g in grupos]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(condiciones))
    barras = ax.bar(x, medias_grafico)

    ax.set_xticks(x)
    ax.set_xticklabels(
        ["Control Pre-test", "Control Post-test", "Experimental Pre-test"],
        rotation=15,
        ha="right",
    )
    ax.set_ylabel("Media")
    ax.set_xlabel("Condición de medición")
    ax.set_ylim(0, 5)
    ax.set_title(
        f"Comparación de tres mediciones - {indicador}",
        fontsize=13,
        fontweight="bold",
    )
    ax.grid(axis="y", alpha=0.25)

    for barra, valor in zip(barras, medias_grafico):
        if pd.notna(valor):
            ax.text(
                barra.get_x() + barra.get_width() / 2,
                valor + 0.08,
                f"{valor:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.tight_layout()
    plt.savefig(
        GRAFICOS / f"anova_3_{nombre_archivo(indicador)}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

# ============================================================
# 5. DATAFRAMES DE SALIDA
# ============================================================

tabla_principal = pd.DataFrame(resultados)
tabla_anova_completa = pd.DataFrame(tablas_completas)
tabla_medias = pd.DataFrame(resumen_medias)
tabla_tukey = pd.DataFrame(posthoc_resultados)

# ============================================================
# 6. TABLA PRINCIPAL PARA TESIS
# ============================================================

tabla_tesis = tabla_principal[[
    "Dimensión",
    "Indicador",
    "Control Pre-test - Media",
    "Control Post-test - Media",
    "Experimental Pre-test - Media",
    "F",
    "gl entre",
    "gl dentro",
    "p-valor",
    "Eta²",
    "Magnitud Eta²",
    "Decisión",
]].copy()

# ============================================================
# 7. EXPORTAR EXCEL
# ============================================================

archivo_excel = SALIDA / "anova_3_mediciones_por_indicador.xlsx"

with pd.ExcelWriter(archivo_excel, engine="openpyxl") as writer:
    tabla_tesis.to_excel(
        writer,
        sheet_name="Tabla para tesis",
        index=False,
    )
    tabla_principal.to_excel(
        writer,
        sheet_name="Resultados ANOVA",
        index=False,
    )
    tabla_anova_completa.to_excel(
        writer,
        sheet_name="ANOVA Completo",
        index=False,
    )
    tabla_medias.to_excel(
        writer,
        sheet_name="Medias y DE",
        index=False,
    )
    tabla_tukey.to_excel(
        writer,
        sheet_name="Tukey Post-hoc",
        index=False,
    )

# ============================================================
# 8. RESUMEN TXT
# ============================================================

archivo_txt = SALIDA / "resumen_anova_3_mediciones.txt"

with open(archivo_txt, "w", encoding="utf-8") as f:
    f.write("ANOVA DE 3 MEDICIONES POR INDICADOR\n")
    f.write("=" * 90 + "\n\n")
    f.write("Condiciones analizadas:\n")
    f.write("1. Control Pre-test\n")
    f.write("2. Control Post-test\n")
    f.write("3. Experimental Pre-test\n")
    f.write("\nExperimental Post-test: EXCLUIDO\n")
    f.write("Nivel de significancia: α = 0.05\n\n")
    f.write(
        "El análisis corresponde a un ANOVA de un factor aplicado "
        "independientemente a cada uno de los 16 indicadores.\n"
    )
    f.write(
        "No se calcula interacción Grupo × Momento porque Experimental "
        "Post-test fue excluido del análisis.\n\n"
    )

    for _, fila in tabla_principal.iterrows():
        f.write("-" * 90 + "\n")
        f.write(f"Dimensión: {fila['Dimensión']}\n")
        f.write(f"Indicador: {fila['Indicador']}\n")
        f.write(f"Control Pre-test: {fila['Control Pre-test - Media']:.4f}\n")
        f.write(f"Control Post-test: {fila['Control Post-test - Media']:.4f}\n")
        f.write(f"Experimental Pre-test: {fila['Experimental Pre-test - Media']:.4f}\n")
        f.write(f"F = {fila['F']:.4f}\n")
        f.write(f"gl = ({fila['gl entre']}, {fila['gl dentro']})\n")
        f.write(f"p = {fila['p-valor']:.6f}\n")
        f.write(f"Eta² = {fila['Eta²']:.4f}\n")
        f.write(f"Magnitud = {fila['Magnitud Eta²']}\n")
        f.write(f"Decisión = {fila['Decisión']}\n")

# ============================================================
# 9. INFORMACIÓN FINAL
# ============================================================

print("\n" + "=" * 90)
print("PROCESO TERMINADO")
print("=" * 90)
print(f"\nExcel: {archivo_excel}")
print(f"TXT:   {archivo_txt}")
print(f"Gráficos: {GRAFICOS}")
print("\nSe analizaron los 16 indicadores utilizando únicamente:")
print("  1. Control Pre-test")
print("  2. Control Post-test")
print("  3. Experimental Pre-test")
print("\nExperimental Post-test fue excluido completamente.")
print("No se calcula interacción Grupo × Momento en este archivo.")
print("\nListo.")
