# ============================================================
# RESULTADOS COMPARATIVOS
# ============================================================
# Compara el porcentaje de respuestas favorables (4-5) entre:
# - Control Pre-test / Post-test
# - Experimental Pre-test / Post-test
#
# Variables:
# 1. Sistema de gestión académica web -> indicadores 1-4
# 2. Chatbot                         -> indicadores 5-8
# 3. Gestión académica              -> indicadores 9-12
# 4. Atención al cliente            -> indicadores 13-16
#
# Genera tablas y gráficos para el análisis comparativo.
#
# Requisitos:
# pip install pandas openpyxl matplotlib numpy
# ============================================================

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

SALIDA = BASE_DIR / "resultados_comparativos"
GRAFICOS = SALIDA / "graficos"

SALIDA.mkdir(exist_ok=True)
GRAFICOS.mkdir(exist_ok=True)

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
    "Satisfacción Atención Cliente"
]

VARIABLES = {
    "Sistema de gestión académica web": INDICADORES[0:4],
    "Chatbot": INDICADORES[4:8],
    "Gestión académica": INDICADORES[8:12],
    "Atención al cliente": INDICADORES[12:16],
}

# ============================================================
# 3. CONFIGURACIÓN DEL NIVEL FAVORABLE
# ============================================================
# En la escala codificada:
# 1-2 = desfavorable
# 3   = moderado
# 4-5 = favorable

VALORES_FAVORABLES = [4, 5]

# ============================================================
# 4. FUNCIONES
# ============================================================

def leer_excel(ruta):
    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo:\n{ruta}"
        )

    df = pd.read_excel(ruta)

    for columna in INDICADORES:

        if columna not in df.columns:
            raise ValueError(
                f"El archivo {ruta.name} no contiene la columna "
                f"'{columna}'."
            )

        df[columna] = pd.to_numeric(
            df[columna],
            errors="coerce"
        )

        valores = df[columna].dropna()

        if not valores.between(1, 5).all():
            raise ValueError(
                f"La columna '{columna}' en {ruta.name} "
                f"contiene valores fuera de la escala 1-5."
            )

    return df


def porcentaje_favorable(serie):
    serie = pd.to_numeric(
        serie,
        errors="coerce"
    ).dropna()

    if len(serie) == 0:
        return np.nan

    return (
        serie.isin(VALORES_FAVORABLES).mean()
        * 100
    )


def porcentaje_variable(df, indicadores):
    valores = df[indicadores].values.flatten()
    valores = pd.to_numeric(
        pd.Series(valores),
        errors="coerce"
    ).dropna()

    if len(valores) == 0:
        return np.nan

    return (
        valores.isin(VALORES_FAVORABLES).mean()
        * 100
    )


def diferencia_puntos_porcentuales(pre, post):
    return post - pre


def diferencia_de_diferencias(
    cambio_control,
    cambio_experimental
):
    return cambio_experimental - cambio_control


def crear_grafico(
    datos,
    titulo,
    archivo,
    ylabel="Porcentaje favorable (%)"
):

    ax = datos.plot(
        kind="bar",
        figsize=(11, 6)
    )

    ax.set_title(
        titulo,
        fontsize=14,
        fontweight="bold"
    )

    ax.set_ylabel(ylabel)
    ax.set_xlabel("")

    ax.set_ylim(0, 100)

    ax.grid(
        axis="y",
        alpha=0.25
    )

    plt.xticks(rotation=0)

    plt.tight_layout()

    plt.savefig(
        archivo,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# 5. LECTURA DE LOS CUATRO ARCHIVOS
# ============================================================

print("=" * 75)
print("LECTURA DE DATOS")
print("=" * 75)

datos = {}

for nombre, archivo in ARCHIVOS.items():

    print(f"Leyendo: {archivo.name}")

    datos[nombre] = leer_excel(archivo)

    print(
        f"  Filas: {len(datos[nombre])} | "
        f"Columnas: {len(datos[nombre].columns)}"
    )

# ============================================================
# 6. TABLA POR INDICADOR
# ============================================================

filas_indicadores = []

for numero, indicador in enumerate(
    INDICADORES,
    start=1
):

    fila = {
        "N.º": numero,
        "Indicador": indicador
    }

    for nombre in ARCHIVOS.keys():

        fila[nombre] = porcentaje_favorable(
            datos[nombre][indicador]
        )

    fila["Cambio Control (pp)"] = (
        fila["Control Post-test"]
        - fila["Control Pre-test"]
    )

    fila["Cambio Experimental (pp)"] = (
        fila["Experimental Post-test"]
        - fila["Experimental Pre-test"]
    )

    fila["Diferencia de diferencias (pp)"] = (
        fila["Cambio Experimental (pp)"]
        - fila["Cambio Control (pp)"]
    )

    filas_indicadores.append(fila)


tabla_indicadores = pd.DataFrame(
    filas_indicadores
)

# ============================================================
# 7. TABLA POR VARIABLE
# ============================================================

filas_variables = []

for variable, indicadores in VARIABLES.items():

    control_pre = porcentaje_variable(
        datos["Control Pre-test"],
        indicadores
    )

    control_post = porcentaje_variable(
        datos["Control Post-test"],
        indicadores
    )

    experimental_pre = porcentaje_variable(
        datos["Experimental Pre-test"],
        indicadores
    )

    experimental_post = porcentaje_variable(
        datos["Experimental Post-test"],
        indicadores
    )

    cambio_control = (
        control_post - control_pre
    )

    cambio_experimental = (
        experimental_post - experimental_pre
    )

    did = diferencia_de_diferencias(
        cambio_control,
        cambio_experimental
    )

    filas_variables.append({

        "Variable": variable,

        "Control Pre-test (%)":
            control_pre,

        "Control Post-test (%)":
            control_post,

        "Incremento Control (pp)":
            cambio_control,

        "Experimental Pre-test (%)":
            experimental_pre,

        "Experimental Post-test (%)":
            experimental_post,

        "Incremento Experimental (pp)":
            cambio_experimental,

        "Diferencia de diferencias (pp)":
            did,

        "N.º Indicadores":
            len(indicadores)
    })


tabla_variables = pd.DataFrame(
    filas_variables
)

# ============================================================
# 8. TABLA RESUMEN
# ============================================================

tabla_resumen = tabla_variables[
    [
        "Variable",
        "Control Pre-test (%)",
        "Control Post-test (%)",
        "Incremento Control (pp)",
        "Experimental Pre-test (%)",
        "Experimental Post-test (%)",
        "Incremento Experimental (pp)",
        "Diferencia de diferencias (pp)"
    ]
].copy()

# ============================================================
# 9. RESULTADO GLOBAL
# ============================================================

control_pre_global = tabla_variables[
    "Control Pre-test (%)"
].mean()

control_post_global = tabla_variables[
    "Control Post-test (%)"
].mean()

experimental_pre_global = tabla_variables[
    "Experimental Pre-test (%)"
].mean()

experimental_post_global = tabla_variables[
    "Experimental Post-test (%)"
].mean()

incremento_control_global = (
    control_post_global
    - control_pre_global
)

incremento_experimental_global = (
    experimental_post_global
    - experimental_pre_global
)

did_global = (
    incremento_experimental_global
    - incremento_control_global
)

tabla_global = pd.DataFrame([{

    "Control Pre-test (%)":
        control_pre_global,

    "Control Post-test (%)":
        control_post_global,

    "Incremento Control (pp)":
        incremento_control_global,

    "Experimental Pre-test (%)":
        experimental_pre_global,

    "Experimental Post-test (%)":
        experimental_post_global,

    "Incremento Experimental (pp)":
        incremento_experimental_global,

    "Diferencia de diferencias (pp)":
        did_global
}])

# ============================================================
# 10. TABLA DE DISTRIBUCIÓN FAVORABLE / NO FAVORABLE
# ============================================================

filas_distribucion = []

for variable, indicadores in VARIABLES.items():

    for grupo in [
        "Control Pre-test",
        "Control Post-test",
        "Experimental Pre-test",
        "Experimental Post-test"
    ]:

        valores = datos[grupo][indicadores].values.flatten()

        valores = pd.to_numeric(
            pd.Series(valores),
            errors="coerce"
        ).dropna()

        favorable = (
            valores.isin(VALORES_FAVORABLES).mean()
            * 100
        )

        no_favorable = 100 - favorable

        filas_distribucion.append({

            "Variable": variable,
            "Grupo": grupo,
            "Favorable (%)": favorable,
            "No favorable (%)": no_favorable
        })


tabla_distribucion = pd.DataFrame(
    filas_distribucion
)

# ============================================================
# 11. EXPORTAR EXCEL
# ============================================================

archivo_excel = (
    SALIDA
    / "tablas_resultados_comparativos.xlsx"
)

with pd.ExcelWriter(
    archivo_excel,
    engine="openpyxl"
) as writer:

    tabla_indicadores.round(2).to_excel(
        writer,
        sheet_name="Por indicador",
        index=False
    )

    tabla_variables.round(2).to_excel(
        writer,
        sheet_name="Por variable",
        index=False
    )

    tabla_resumen.round(2).to_excel(
        writer,
        sheet_name="Resumen variables",
        index=False
    )

    tabla_distribucion.round(2).to_excel(
        writer,
        sheet_name="Favorable no favorable",
        index=False
    )

    tabla_global.round(2).to_excel(
        writer,
        sheet_name="Resultado global",
        index=False
    )

# ============================================================
# 12. GRÁFICOS POR VARIABLE
# ============================================================

for numero, variable in enumerate(
    VARIABLES.keys(),
    start=1
):

    fila = tabla_variables[
        tabla_variables["Variable"] == variable
    ].iloc[0]

    grafico = pd.DataFrame({

        "Control": [
            fila["Control Pre-test (%)"],
            fila["Control Post-test (%)"]
        ],

        "Experimental": [
            fila["Experimental Pre-test (%)"],
            fila["Experimental Post-test (%)"]
        ]

    }, index=[
        "Pre-test",
        "Post-test"
    ])

    nombre = (
        f"{numero:02d}_"
        + variable.replace(" ", "_")
        + ".png"
    )

    crear_grafico(
        grafico,
        f"{variable}: porcentaje favorable",
        GRAFICOS / nombre
    )

# ============================================================
# 13. GRÁFICO DE INCREMENTOS
# ============================================================

grafico_incrementos = tabla_variables.set_index(
    "Variable"
)[
    [
        "Incremento Control (pp)",
        "Incremento Experimental (pp)"
    ]
]

crear_grafico(
    grafico_incrementos,
    "Incremento del porcentaje favorable",
    GRAFICOS / "05_incremento_control_vs_experimental.png",
    ylabel="Incremento (puntos porcentuales)"
)

# ============================================================
# 14. GRÁFICO DE DIFERENCIA DE DIFERENCIAS
# ============================================================

grafico_did = tabla_variables.set_index(
    "Variable"
)[
    ["Diferencia de diferencias (pp)"]
]

crear_grafico(
    grafico_did,
    "Diferencia de diferencias por variable",
    GRAFICOS / "06_diferencia_de_diferencias.png",
    ylabel="Diferencia (puntos porcentuales)"
)

# ============================================================
# 15. GRÁFICO GLOBAL
# ============================================================

grafico_global = pd.DataFrame({

    "Control": [
        control_pre_global,
        control_post_global
    ],

    "Experimental": [
        experimental_pre_global,
        experimental_post_global
    ]

}, index=[
    "Pre-test",
    "Post-test"
])

crear_grafico(
    grafico_global,
    "Porcentaje favorable: Control vs Experimental",
    GRAFICOS / "07_resultado_global.png"
)

# ============================================================
# 16. RESUMEN TXT
# ============================================================

archivo_txt = (
    SALIDA
    / "resumen_resultados.txt"
)

with open(
    archivo_txt,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "RESULTADOS COMPARATIVOS\n"
    )

    f.write("=" * 75 + "\n\n")

    f.write(
        "Criterio de resultado favorable: "
        "respuestas codificadas con valores 4 y 5.\n\n"
    )

    f.write(
        "RESULTADOS POR VARIABLE\n"
    )

    f.write("-" * 75 + "\n\n")

    f.write(
        tabla_variables.round(2).to_string(
            index=False
        )
    )

    f.write("\n\nRESULTADO GLOBAL\n")
    f.write("-" * 75 + "\n\n")

    f.write(
        tabla_global.round(2).to_string(
            index=False
        )
    )

# ============================================================
# 17. RESULTADOS EN CONSOLA
# ============================================================

print()
print("=" * 75)
print("RESULTADOS COMPARATIVOS GENERADOS")
print("=" * 75)

print()
print(
    "RESULTADOS POR VARIABLE:"
)

print()

print(
    tabla_variables.round(2).to_string(
        index=False
    )
)

print()
print("=" * 75)
print("RESULTADO GLOBAL")
print("=" * 75)

print(
    f"Control Pre-test:       "
    f"{control_pre_global:.2f}%"
)

print(
    f"Control Post-test:      "
    f"{control_post_global:.2f}%"
)

print(
    f"Incremento Control:     "
    f"{incremento_control_global:.2f} pp"
)

print(
    f"Experimental Pre-test:  "
    f"{experimental_pre_global:.2f}%"
)

print(
    f"Experimental Post-test: "
    f"{experimental_post_global:.2f}%"
)

print(
    f"Incremento Experimental:"
    f" {incremento_experimental_global:.2f} pp"
)

print(
    f"Diferencia diferencias: "
    f"{did_global:.2f} pp"
)

print()
print("=" * 75)
print("ARCHIVOS GENERADOS")
print("=" * 75)

print(f"Excel: {archivo_excel}")
print(f"Gráficos: {GRAFICOS}")
print(f"Resumen: {archivo_txt}")
print("=" * 75)
