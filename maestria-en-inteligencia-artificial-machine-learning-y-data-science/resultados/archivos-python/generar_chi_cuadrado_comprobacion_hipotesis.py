# ============================================================
# COMPROBACIÓN DE HIPÓTESIS - PRUEBA CHI-CUADRADO
# TESIS: SISTEMA DE GESTIÓN ACADÉMICA WEB INTEGRADO CON CHATBOT
# ============================================================
#
# ESTRUCTURA REAL:
#
# 1. Sistema de gestión académica web -> indicadores 1-4
# 2. Chatbot                         -> indicadores 5-8
# 3. Gestión académica              -> indicadores 9-12
# 4. Atención al cliente            -> indicadores 13-16
#
# OBJETIVO:
# Complementar la comprobación de hipótesis mediante pruebas
# de asociación entre GRUPO y las respuestas categóricas.
#
# ANÁLISIS PRINCIPAL:
# - Chi-cuadrado de Pearson entre:
#       Grupo (Control / Experimental)
#       y
#       Nivel de respuesta en el POST-TEST
#
# Se realizan dos niveles de análisis:
#
# A) POR INDICADOR:
#    Grupo x categorías 1,2,3,4,5
#
# B) POR VARIABLE:
#    Grupo x nivel:
#       Desfavorable = 1-2
#       Moderado     = 3
#       Favorable    = 4-5
#
# También se realiza:
#
# C) ANÁLISIS PRE-TEST:
#    Permite comprobar si ambos grupos presentan una distribución
#    comparable antes de la intervención.
#
# D) COMPARACIÓN PRE-TEST / POST-TEST:
#    Se presenta descriptivamente mediante tablas de frecuencias
#    y porcentajes favorables. El chi-cuadrado se mantiene como
#    prueba de asociación entre grupos, no como prueba pareada.
#
# Para cada prueba se calcula:
# - Chi-cuadrado de Pearson
# - grados de libertad
# - p-valor
# - V de Cramer (tamaño de asociación)
# - frecuencias esperadas
# - decisión estadística con alfa = 0.05
#
# GENERA:
# - Excel con tablas
# - Tablas de frecuencias
# - Tablas de porcentajes
# - Gráficos comparativos
# - Gráficos de asociación
# - Resumen TXT
#
# Requisitos:
# pandas==2.2.3
# openpyxl==3.1.5
# matplotlib==3.9.2
# numpy==2.0.2
# scipy==1.13.1
# ============================================================

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import chi2_contingency

warnings.filterwarnings("ignore")


# ============================================================
# 1. CONFIGURACIÓN
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

ARCHIVOS = {
    "Control Pre-test": BASE_DIR / "grupo-control-pre-test-codificado.xlsx",
    "Control Post-test": BASE_DIR / "grupo-control-post-test-codificado.xlsx",
    "Experimental Pre-test": BASE_DIR / "grupo-experimental-pre-test-codificado.xlsx",
    "Experimental Post-test": BASE_DIR / "grupo-experimental-post-test-codificado.xlsx",
}

SALIDA = BASE_DIR / "resultados_chi_cuadrado"
GRAFICOS = SALIDA / "graficos"

SALIDA.mkdir(parents=True, exist_ok=True)
GRAFICOS.mkdir(parents=True, exist_ok=True)

ALPHA = 0.05


# ============================================================
# 2. INDICADORES Y VARIABLES
# ============================================================

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

VARIABLES = {
    "Sistema de gestión académica web": INDICADORES[0:4],
    "Chatbot": INDICADORES[4:8],
    "Gestión académica": INDICADORES[8:12],
    "Atención al cliente": INDICADORES[12:16],
}


# ============================================================
# 3. FUNCIONES DE LECTURA
# ============================================================

def leer_archivo(ruta):
    if not ruta.exists():
        raise FileNotFoundError(
            f"\nNo se encontró el archivo:\n{ruta}\n"
            "Colócalo en la misma carpeta del programa."
        )

    df = pd.read_excel(ruta)

    faltantes = [
        c for c in INDICADORES
        if c not in df.columns
    ]

    if faltantes:
        raise ValueError(
            f"\nEl archivo {ruta.name} no contiene:\n"
            + "\n".join(f"- {c}" for c in faltantes)
        )

    for columna in INDICADORES:

        df[columna] = pd.to_numeric(
            df[columna],
            errors="coerce"
        )

        valores = df[columna].dropna()

        invalidos = valores[
            ~valores.isin([1, 2, 3, 4, 5])
        ]

        if not invalidos.empty:
            raise ValueError(
                f"\nLa columna '{columna}' contiene valores "
                f"fuera de la escala 1-5: "
                f"{sorted(invalidos.unique().tolist())}"
            )

    return df


# ============================================================
# 4. FUNCIONES ESTADÍSTICAS
# ============================================================

def cramers_v(tabla, chi2, n):
    """
    V de Cramer:
    mide la intensidad de asociación entre dos variables
    categóricas.
    """
    filas, columnas = tabla.shape

    minimo = min(
        filas - 1,
        columnas - 1
    )

    if n == 0 or minimo <= 0:
        return np.nan

    return np.sqrt(
        chi2 / (n * minimo)
    )


def interpretar_v(v):
    if pd.isna(v):
        return "No calculable"

    if v < 0.10:
        return "asociación trivial"
    elif v < 0.30:
        return "asociación débil"
    elif v < 0.50:
        return "asociación moderada"
    else:
        return "asociación fuerte"


def decision_p(p):
    if pd.isna(p):
        return "No calculable"

    if p < ALPHA:
        return "Rechazar H0"

    return "No rechazar H0"


def significancia(p):
    if pd.isna(p):
        return "No calculable"

    if p < 0.001:
        return "p < 0.001"
    elif p < 0.01:
        return "p < 0.01"
    elif p < 0.05:
        return "p < 0.05"

    return "p ≥ 0.05"


def chi_cuadrado(tabla):
    """
    Ejecuta chi-cuadrado de Pearson.
    """
    chi2, p, gl, esperadas = chi2_contingency(
        tabla,
        correction=False
    )

    n = tabla.values.sum()

    v = cramers_v(
        tabla,
        chi2,
        n
    )

    return (
        chi2,
        p,
        gl,
        esperadas,
        v
    )


# ============================================================
# 5. CARGAR LOS ARCHIVOS
# ============================================================

print("=" * 80)
print("COMPROBACIÓN DE HIPÓTESIS - CHI-CUADRADO")
print("=" * 80)

datos = {}

for nombre, ruta in ARCHIVOS.items():

    print(f"\nLeyendo: {ruta.name}")

    datos[nombre] = leer_archivo(ruta)

    print(
        f"  Participantes: {len(datos[nombre])}"
    )


# ============================================================
# 6. CHI-CUADRADO POR INDICADOR
# ============================================================
#
# Se utiliza la distribución original de 1 a 5.
#
# POST-TEST:
# Control vs Experimental
#
# Esto permite comprobar si la distribución de respuestas
# es independiente del grupo.
# ============================================================

resultados_indicadores = []

frecuencias_indicadores = []

porcentajes_indicadores = []

for indicador in INDICADORES:

    control = datos[
        "Control Post-test"
    ][indicador].dropna().astype(int)

    experimental = datos[
        "Experimental Post-test"
    ][indicador].dropna().astype(int)

    tabla = pd.crosstab(
        pd.Series(
            ["Control"] * len(control)
            + ["Experimental"] * len(experimental),
            name="Grupo"
        ),
        pd.Series(
            control.tolist()
            + experimental.tolist(),
            name="Respuesta"
        )
    )

    tabla = tabla.reindex(
        index=["Control", "Experimental"],
        columns=[1, 2, 3, 4, 5],
        fill_value=0
    )

    chi2, p, gl, esperadas, v = chi_cuadrado(
        tabla
    )

    resultados_indicadores.append({
        "Indicador": indicador,
        "Chi-cuadrado": chi2,
        "gl": gl,
        "p-valor": p,
        "V de Cramer": v,
        "Interpretación V": interpretar_v(v),
        "Significancia": significancia(p),
        "Decisión": decision_p(p),
        "N": int(tabla.values.sum()),
    })

    # Frecuencias observadas
    fila = {
        "Indicador": indicador,
        "Control 1": tabla.loc["Control", 1],
        "Control 2": tabla.loc["Control", 2],
        "Control 3": tabla.loc["Control", 3],
        "Control 4": tabla.loc["Control", 4],
        "Control 5": tabla.loc["Control", 5],
        "Experimental 1": tabla.loc["Experimental", 1],
        "Experimental 2": tabla.loc["Experimental", 2],
        "Experimental 3": tabla.loc["Experimental", 3],
        "Experimental 4": tabla.loc["Experimental", 4],
        "Experimental 5": tabla.loc["Experimental", 5],
    }

    frecuencias_indicadores.append(fila)

    # Porcentajes por fila
    porcentajes = (
        tabla.div(
            tabla.sum(axis=1),
            axis=0
        ) * 100
    )

    fila_porcentaje = {
        "Indicador": indicador,
        "Control 1 (%)": porcentajes.loc["Control", 1],
        "Control 2 (%)": porcentajes.loc["Control", 2],
        "Control 3 (%)": porcentajes.loc["Control", 3],
        "Control 4 (%)": porcentajes.loc["Control", 4],
        "Control 5 (%)": porcentajes.loc["Control", 5],
        "Experimental 1 (%)": porcentajes.loc["Experimental", 1],
        "Experimental 2 (%)": porcentajes.loc["Experimental", 2],
        "Experimental 3 (%)": porcentajes.loc["Experimental", 3],
        "Experimental 4 (%)": porcentajes.loc["Experimental", 4],
        "Experimental 5 (%)": porcentajes.loc["Experimental", 5],
    }

    porcentajes_indicadores.append(
        fila_porcentaje
    )


tabla_chi_indicadores = pd.DataFrame(
    resultados_indicadores
)

tabla_frecuencias_indicadores = pd.DataFrame(
    frecuencias_indicadores
)

tabla_porcentajes_indicadores = pd.DataFrame(
    porcentajes_indicadores
)


# ============================================================
# 7. CHI-CUADRADO POR VARIABLE
# ============================================================
#
# Para cada variable se agrupan sus cuatro indicadores.
#
# Cada respuesta se clasifica:
#
# 1-2 = Desfavorable
# 3   = Moderado
# 4-5 = Favorable
#
# Se compara:
# Control Post-test vs Experimental Post-test.
# ============================================================

resultados_variables = []

frecuencias_variables = []

porcentajes_variables = []

for variable, indicadores in VARIABLES.items():

    control_valores = (
        datos["Control Post-test"][indicadores]
        .values
        .flatten()
    )

    experimental_valores = (
        datos["Experimental Post-test"][indicadores]
        .values
        .flatten()
    )

    control_valores = (
        pd.Series(control_valores)
        .dropna()
        .astype(int)
    )

    experimental_valores = (
        pd.Series(experimental_valores)
        .dropna()
        .astype(int)
    )

    def clasificar(valor):
        if valor in [1, 2]:
            return "Desfavorable"
        elif valor == 3:
            return "Moderado"
        elif valor in [4, 5]:
            return "Favorable"

        return np.nan

    control_cat = control_valores.map(
        clasificar
    )

    experimental_cat = experimental_valores.map(
        clasificar
    )

    tabla = pd.crosstab(
        pd.Series(
            ["Control"] * len(control_cat)
            + ["Experimental"] * len(experimental_cat),
            name="Grupo"
        ),
        pd.Series(
            control_cat.tolist()
            + experimental_cat.tolist(),
            name="Nivel"
        )
    )

    categorias = [
        "Desfavorable",
        "Moderado",
        "Favorable"
    ]

    tabla = tabla.reindex(
        index=["Control", "Experimental"],
        columns=categorias,
        fill_value=0
    )

    chi2, p, gl, esperadas, v = chi_cuadrado(
        tabla
    )

    resultados_variables.append({
        "Variable": variable,
        "Chi-cuadrado": chi2,
        "gl": gl,
        "p-valor": p,
        "V de Cramer": v,
        "Interpretación V": interpretar_v(v),
        "Significancia": significancia(p),
        "Decisión": decision_p(p),
        "N respuestas": int(tabla.values.sum()),
    })

    fila = {
        "Variable": variable,
        "Control Desfavorable": tabla.loc[
            "Control", "Desfavorable"
        ],
        "Control Moderado": tabla.loc[
            "Control", "Moderado"
        ],
        "Control Favorable": tabla.loc[
            "Control", "Favorable"
        ],
        "Experimental Desfavorable": tabla.loc[
            "Experimental", "Desfavorable"
        ],
        "Experimental Moderado": tabla.loc[
            "Experimental", "Moderado"
        ],
        "Experimental Favorable": tabla.loc[
            "Experimental", "Favorable"
        ],
    }

    frecuencias_variables.append(
        fila
    )

    porcentajes = (
        tabla.div(
            tabla.sum(axis=1),
            axis=0
        ) * 100
    )

    fila_porcentaje = {
        "Variable": variable,
        "Control Desfavorable (%)":
            porcentajes.loc["Control", "Desfavorable"],
        "Control Moderado (%)":
            porcentajes.loc["Control", "Moderado"],
        "Control Favorable (%)":
            porcentajes.loc["Control", "Favorable"],
        "Experimental Desfavorable (%)":
            porcentajes.loc[
                "Experimental", "Desfavorable"
            ],
        "Experimental Moderado (%)":
            porcentajes.loc[
                "Experimental", "Moderado"
            ],
        "Experimental Favorable (%)":
            porcentajes.loc[
                "Experimental", "Favorable"
            ],
    }

    porcentajes_variables.append(
        fila_porcentaje
    )


tabla_chi_variables = pd.DataFrame(
    resultados_variables
)

tabla_frecuencias_variables = pd.DataFrame(
    frecuencias_variables
)

tabla_porcentajes_variables = pd.DataFrame(
    porcentajes_variables
)


# ============================================================
# 8. ANÁLISIS PRE-TEST
# ============================================================
#
# Se repite el análisis por variable usando PRE-TEST.
#
# Esto permite comprobar que las distribuciones de ambos grupos
# no presentaban diferencias estadísticamente significativas
# antes de la intervención.
# ============================================================

resultados_pre = []

for variable, indicadores in VARIABLES.items():

    control_valores = (
        datos["Control Pre-test"][indicadores]
        .values
        .flatten()
    )

    experimental_valores = (
        datos["Experimental Pre-test"][indicadores]
        .values
        .flatten()
    )

    control_valores = (
        pd.Series(control_valores)
        .dropna()
        .astype(int)
    )

    experimental_valores = (
        pd.Series(experimental_valores)
        .dropna()
        .astype(int)
    )

    def clasificar_pre(valor):
        if valor in [1, 2]:
            return "Desfavorable"
        elif valor == 3:
            return "Moderado"
        elif valor in [4, 5]:
            return "Favorable"

        return np.nan

    control_cat = control_valores.map(
        clasificar_pre
    )

    experimental_cat = experimental_valores.map(
        clasificar_pre
    )

    tabla = pd.crosstab(
        pd.Series(
            ["Control"] * len(control_cat)
            + ["Experimental"] * len(experimental_cat),
            name="Grupo"
        ),
        pd.Series(
            control_cat.tolist()
            + experimental_cat.tolist(),
            name="Nivel"
        )
    )

    categorias = [
        "Desfavorable",
        "Moderado",
        "Favorable"
    ]

    tabla = tabla.reindex(
        index=["Control", "Experimental"],
        columns=categorias,
        fill_value=0
    )

    chi2, p, gl, esperadas, v = chi_cuadrado(
        tabla
    )

    resultados_pre.append({
        "Variable": variable,
        "Chi-cuadrado": chi2,
        "gl": gl,
        "p-valor": p,
        "V de Cramer": v,
        "Interpretación V": interpretar_v(v),
        "Significancia": significancia(p),
        "Decisión": decision_p(p),
    })


tabla_chi_pre = pd.DataFrame(
    resultados_pre
)


# ============================================================
# 9. TABLA RESUMEN PARA COMPROBACIÓN DE HIPÓTESIS
# ============================================================

resumen_hipotesis = tabla_chi_variables[
    [
        "Variable",
        "Chi-cuadrado",
        "gl",
        "p-valor",
        "V de Cramer",
        "Interpretación V",
        "Decisión",
    ]
].copy()

resumen_hipotesis.insert(
    0,
    "Comparación",
    "Control Post-test vs Experimental Post-test"
)


# ============================================================
# 10. GRÁFICO: FAVORABLE POR VARIABLE
# ============================================================

variables = (
    tabla_porcentajes_variables["Variable"]
    .tolist()
)

control_favorable = (
    tabla_porcentajes_variables[
        "Control Favorable (%)"
    ].tolist()
)

experimental_favorable = (
    tabla_porcentajes_variables[
        "Experimental Favorable (%)"
    ].tolist()
)

x = np.arange(len(variables))
ancho = 0.36

fig, ax = plt.subplots(
    figsize=(12, 7)
)

ax.bar(
    x - ancho / 2,
    control_favorable,
    ancho,
    label="Control Post-test"
)

ax.bar(
    x + ancho / 2,
    experimental_favorable,
    ancho,
    label="Experimental Post-test"
)

ax.set_xticks(x)
ax.set_xticklabels(
    variables,
    rotation=25,
    ha="right"
)

ax.set_ylabel(
    "Respuestas favorables (%)"
)

ax.set_title(
    "Comparación de respuestas favorables por variable",
    fontsize=13,
    fontweight="bold"
)

ax.set_ylim(
    0,
    100
)

ax.legend()

ax.grid(
    axis="y",
    alpha=0.25
)

plt.tight_layout()

plt.savefig(
    GRAFICOS
    / "favorable_control_vs_experimental.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 11. GRÁFICO: P-VALORES POR VARIABLE
# ============================================================

fig, ax = plt.subplots(
    figsize=(12, 7)
)

p = tabla_chi_variables[
    "p-valor"
].tolist()

ax.bar(
    variables,
    p
)

ax.axhline(
    ALPHA,
    linestyle="--",
    linewidth=2,
    label="α = 0.05"
)

ax.set_ylabel(
    "p-valor"
)

ax.set_title(
    "Prueba Chi-cuadrado por variable",
    fontsize=13,
    fontweight="bold"
)

ax.set_yscale(
    "log"
)

ax.tick_params(
    axis="x",
    rotation=25
)

ax.legend()

ax.grid(
    axis="y",
    alpha=0.25
)

plt.tight_layout()

plt.savefig(
    GRAFICOS
    / "p_valores_chi_cuadrado_variables.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 12. GRÁFICO: V DE CRAMER
# ============================================================

v_cramer = tabla_chi_variables[
    "V de Cramer"
].tolist()

fig, ax = plt.subplots(
    figsize=(12, 7)
)

ax.bar(
    variables,
    v_cramer
)

ax.set_ylabel(
    "V de Cramer"
)

ax.set_title(
    "Magnitud de asociación entre grupo y variable",
    fontsize=13,
    fontweight="bold"
)

ax.set_ylim(
    0,
    max(
        0.5,
        max(v_cramer) * 1.20
    )
)

ax.tick_params(
    axis="x",
    rotation=25
)

ax.grid(
    axis="y",
    alpha=0.25
)

plt.tight_layout()

plt.savefig(
    GRAFICOS
    / "v_cramer_variables.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 13. GRÁFICOS DE DISTRIBUCIÓN FAVORABLE/MODERADO/DESFAVORABLE
# ============================================================

categorias = [
    "Desfavorable",
    "Moderado",
    "Favorable"
]

for grupo in [
    "Control",
    "Experimental"
]:

    columnas = [
        f"{grupo} {cat} (%)"
        for cat in categorias
    ]

    valores = [
        tabla_porcentajes_variables[col].tolist()
        for col in columnas
    ]

    fig, ax = plt.subplots(
        figsize=(12, 7)
    )

    bottom = np.zeros(
        len(variables)
    )

    for categoria, serie in zip(
        categorias,
        valores
    ):

        ax.bar(
            variables,
            serie,
            bottom=bottom,
            label=categoria
        )

        bottom += np.array(
            serie
        )

    ax.set_ylabel(
        "Porcentaje (%)"
    )

    ax.set_title(
        f"Distribución de respuestas por nivel - {grupo} Post-test",
        fontsize=13,
        fontweight="bold"
    )

    ax.set_ylim(
        0,
        100
    )

    ax.tick_params(
        axis="x",
        rotation=25
    )

    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        GRAFICOS
        / f"distribucion_niveles_{grupo.lower()}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# 14. EXPORTAR A EXCEL
# ============================================================

archivo_excel = (
    SALIDA
    / "tablas_chi_cuadrado.xlsx"
)

with pd.ExcelWriter(
    archivo_excel,
    engine="openpyxl"
) as writer:

    tabla_chi_indicadores.to_excel(
        writer,
        sheet_name="Chi Indicadores",
        index=False
    )

    tabla_frecuencias_indicadores.to_excel(
        writer,
        sheet_name="Frecuencias Indicadores",
        index=False
    )

    tabla_porcentajes_indicadores.to_excel(
        writer,
        sheet_name="Porcentajes Indicadores",
        index=False
    )

    tabla_chi_variables.to_excel(
        writer,
        sheet_name="Chi Variables",
        index=False
    )

    tabla_frecuencias_variables.to_excel(
        writer,
        sheet_name="Frecuencias Variables",
        index=False
    )

    tabla_porcentajes_variables.to_excel(
        writer,
        sheet_name="Porcentajes Variables",
        index=False
    )

    tabla_chi_pre.to_excel(
        writer,
        sheet_name="Chi Pre-test",
        index=False
    )

    resumen_hipotesis.to_excel(
        writer,
        sheet_name="Resumen Hipotesis",
        index=False
    )


# ============================================================
# 15. RESUMEN TXT
# ============================================================

archivo_txt = (
    SALIDA
    / "resumen_chi_cuadrado.txt"
)

with open(
    archivo_txt,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "COMPROBACIÓN DE HIPÓTESIS - CHI-CUADRADO DE PEARSON\n"
    )

    f.write(
        "=" * 80 + "\n\n"
    )

    f.write(
        "Nivel de significancia: α = 0.05\n\n"
    )

    f.write(
        "PRUEBA PRINCIPAL:\n"
        "Chi-cuadrado de Pearson entre grupo y nivel de respuesta "
        "en el post-test.\n\n"
    )

    f.write(
        "Clasificación por variable:\n"
        "1-2 = Desfavorable\n"
        "3   = Moderado\n"
        "4-5 = Favorable\n\n"
    )

    f.write(
        "RESULTADOS POR VARIABLE\n"
    )

    f.write(
        "-" * 80 + "\n\n"
    )

    for _, fila in tabla_chi_variables.iterrows():

        f.write(
            f"Variable: {fila['Variable']}\n"
            f"  Chi-cuadrado = "
            f"{fila['Chi-cuadrado']:.4f}\n"
            f"  gl = {fila['gl']}\n"
            f"  p = {fila['p-valor']:.6f}\n"
            f"  V de Cramer = "
            f"{fila['V de Cramer']:.4f}\n"
            f"  Asociación: "
            f"{fila['Interpretación V']}\n"
            f"  Significancia: "
            f"{fila['Significancia']}\n"
            f"  Decisión: "
            f"{fila['Decisión']}\n\n"
        )

    f.write(
        "=" * 80 + "\n"
    )

    f.write(
        "ANÁLISIS PRE-TEST\n"
    )

    f.write(
        "=" * 80 + "\n\n"
    )

    for _, fila in tabla_chi_pre.iterrows():

        f.write(
            f"{fila['Variable']} | "
            f"χ²={fila['Chi-cuadrado']:.4f} | "
            f"gl={fila['gl']} | "
            f"p={fila['p-valor']:.6f} | "
            f"V={fila['V de Cramer']:.4f} | "
            f"{fila['Decisión']}\n"
        )


# ============================================================
# 16. INFORMACIÓN FINAL
# ============================================================

print("\n" + "=" * 80)
print("PROCESO TERMINADO")
print("=" * 80)

print("\nExcel:")
print(archivo_excel)

print("\nGráficos:")
print(GRAFICOS)

print("\nAnálisis generado:")
print("1. Chi-cuadrado por indicador")
print("2. Chi-cuadrado por variable")
print("3. Chi-cuadrado de pre-test")
print("4. V de Cramer")
print("5. Distribución favorable/moderado/desfavorable")
print("6. Comparación de p-valores")

print("\n" + "=" * 80)
print("CRITERIO DE DECISIÓN")
print("=" * 80)

print(
    "\nSi p < 0.05: se rechaza H0."
    "\nSi p >= 0.05: no se rechaza H0."
)

print(
    "\nNota: el chi-cuadrado analiza asociación entre variables "
    "categóricas. No debe interpretarse como una prueba causal."
)

print("\nListo.")
