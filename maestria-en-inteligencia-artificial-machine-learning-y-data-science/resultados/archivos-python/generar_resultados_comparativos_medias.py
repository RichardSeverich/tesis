# ============================================================
# RESULTADOS COMPARATIVOS
# TESIS: SISTEMA DE GESTIÓN ACADÉMICA WEB INTEGRADO CON CHATBOT
# ============================================================
#
# Estructura REAL de las variables:
#
# 1. Sistema de gestión académica web -> Indicadores 1-4
# 2. Chatbot                         -> Indicadores 5-8
# 3. Gestión académica              -> Indicadores 9-12
# 4. Atención al cliente            -> Indicadores 13-16
#
# El programa compara:
# - Control: Pre-test vs Post-test
# - Experimental: Pre-test vs Post-test
# - Cambio de cada grupo
# - Diferencia de diferencias
#
# Genera:
# - Excel con tablas
# - Gráficos por variable
# - Gráfico comparativo de las 4 variables
# - Resumen TXT
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
# 2. ESCALAS DE RESPUESTA
# ============================================================

ESCALA_TIEMPO = {
    5: "Menos de 2",
    4: "2 - 4",
    3: "5 - 7",
    2: "8 - 10",
    1: "Más de 10"
}

ESCALA_FACILIDAD = {
    5: "Muy fácil",
    4: "Fácil",
    3: "Moderado",
    2: "Difícil",
    1: "Muy difícil"
}

ESCALA_ACCESO = {
    5: "1 o menos",
    4: "2 - 3",
    3: "4 - 5",
    2: "6 - 7",
    1: "8 o más"
}

ESCALA_SATISFACCION = {
    5: "Muy bueno",
    4: "Bueno",
    3: "Moderado",
    2: "Malo",
    1: "Muy malo"
}

# ============================================================
# 3. INDICADORES Y VARIABLES
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
# 4. FUNCIONES
# ============================================================

def leer_excel(ruta):
    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo:\n{ruta}"
        )

    df = pd.read_excel(ruta)

    # Convertir indicadores a valores numéricos
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


def media_variable(df, indicadores):
    return df[indicadores].mean().mean()


def variacion_porcentual(pre, post):
    if pre == 0 or pd.isna(pre):
        return np.nan

    return ((post - pre) / abs(pre)) * 100


def diferencia_de_diferencias(cambio_control, cambio_experimental):
    return cambio_experimental - cambio_control


def crear_grafico(datos, titulo, archivo, ylabel="Media"):
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
# 5. LEER LOS CUATRO ARCHIVOS
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

for numero, indicador in enumerate(INDICADORES, start=1):

    fila = {
        "N.º": numero,
        "Indicador": indicador
    }

    for nombre in ARCHIVOS.keys():

        fila[nombre] = datos[nombre][indicador].mean()

    fila["Cambio Control"] = (
        fila["Control Post-test"]
        - fila["Control Pre-test"]
    )

    fila["Cambio Experimental"] = (
        fila["Experimental Post-test"]
        - fila["Experimental Pre-test"]
    )

    fila["Diferencia de diferencias"] = (
        fila["Cambio Experimental"]
        - fila["Cambio Control"]
    )

    filas_indicadores.append(fila)


tabla_indicadores = pd.DataFrame(
    filas_indicadores
)

# ============================================================
# 7. RESULTADOS POR VARIABLE
# ============================================================

filas_variables = []

for variable, indicadores in VARIABLES.items():

    control_pre = media_variable(
        datos["Control Pre-test"],
        indicadores
    )

    control_post = media_variable(
        datos["Control Post-test"],
        indicadores
    )

    experimental_pre = media_variable(
        datos["Experimental Pre-test"],
        indicadores
    )

    experimental_post = media_variable(
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

        "Control Pre-test": control_pre,
        "Control Post-test": control_post,
        "Cambio Control": cambio_control,
        "Variación Control %":
            variacion_porcentual(
                control_pre,
                control_post
            ),

        "Experimental Pre-test":
            experimental_pre,

        "Experimental Post-test":
            experimental_post,

        "Cambio Experimental":
            cambio_experimental,

        "Variación Experimental %":
            variacion_porcentual(
                experimental_pre,
                experimental_post
            ),

        "Diferencia de diferencias":
            did,

        "N.º Indicadores":
            len(indicadores)
    })


tabla_variables = pd.DataFrame(
    filas_variables
)

# ============================================================
# 8. TABLA RESUMEN DE LAS CUATRO VARIABLES
# ============================================================

tabla_resumen = tabla_variables[
    [
        "Variable",
        "Control Pre-test",
        "Control Post-test",
        "Cambio Control",
        "Experimental Pre-test",
        "Experimental Post-test",
        "Cambio Experimental",
        "Diferencia de diferencias"
    ]
].copy()

# ============================================================
# 9. RESULTADO GLOBAL DE LAS CUATRO VARIABLES
# ============================================================

control_pre_global = tabla_variables[
    "Control Pre-test"
].mean()

control_post_global = tabla_variables[
    "Control Post-test"
].mean()

experimental_pre_global = tabla_variables[
    "Experimental Pre-test"
].mean()

experimental_post_global = tabla_variables[
    "Experimental Post-test"
].mean()

cambio_control_global = (
    control_post_global
    - control_pre_global
)

cambio_experimental_global = (
    experimental_post_global
    - experimental_pre_global
)

did_global = (
    cambio_experimental_global
    - cambio_control_global
)

tabla_global = pd.DataFrame([{

    "Control Pre-test":
        control_pre_global,

    "Control Post-test":
        control_post_global,

    "Cambio Control":
        cambio_control_global,

    "Experimental Pre-test":
        experimental_pre_global,

    "Experimental Post-test":
        experimental_post_global,

    "Cambio Experimental":
        cambio_experimental_global,

    "Diferencia de diferencias":
        did_global
}])

# ============================================================
# 10. EXPORTAR TABLAS A EXCEL
# ============================================================

archivo_excel = (
    SALIDA
    / "tablas_resultados_comparativos.xlsx"
)

with pd.ExcelWriter(
    archivo_excel,
    engine="openpyxl"
) as writer:

    tabla_indicadores.round(4).to_excel(
        writer,
        sheet_name="Por indicador",
        index=False
    )

    tabla_variables.round(4).to_excel(
        writer,
        sheet_name="Por variable",
        index=False
    )

    tabla_resumen.round(4).to_excel(
        writer,
        sheet_name="Resumen 4 variables",
        index=False
    )

    tabla_global.round(4).to_excel(
        writer,
        sheet_name="Resultado global",
        index=False
    )

# ============================================================
# 11. GRÁFICOS INDIVIDUALES DE CADA VARIABLE
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
            fila["Control Pre-test"],
            fila["Control Post-test"]
        ],

        "Experimental": [
            fila["Experimental Pre-test"],
            fila["Experimental Post-test"]
        ]

    }, index=[
        "Pre-test",
        "Post-test"
    ])

    nombre_archivo = (
        f"{numero:02d}_"
        + variable.replace(" ", "_")
        .replace("/", "-")
        + ".png"
    )

    crear_grafico(
        grafico,
        f"{variable}: Pre-test vs Post-test",
        GRAFICOS / nombre_archivo
    )

# ============================================================
# 12. GRÁFICO DE CAMBIOS
# ============================================================

grafico_cambios = tabla_variables.set_index(
    "Variable"
)[
    [
        "Cambio Control",
        "Cambio Experimental"
    ]
]

crear_grafico(
    grafico_cambios,
    "Comparación de cambios entre grupos",
    GRAFICOS / "05_cambios_control_vs_experimental.png",
    ylabel="Cambio de la media"
)

# ============================================================
# 13. GRÁFICO DE DIFERENCIA DE DIFERENCIAS
# ============================================================

grafico_did = tabla_variables.set_index(
    "Variable"
)[
    ["Diferencia de diferencias"]
]

crear_grafico(
    grafico_did,
    "Diferencia de diferencias por variable",
    GRAFICOS / "06_diferencia_de_diferencias.png",
    ylabel="Diferencia"
)

# ============================================================
# 14. GRÁFICO GLOBAL DE LAS CUATRO VARIABLES
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
    "Resultado global: Control vs Experimental",
    GRAFICOS / "07_resultado_global.png",
    ylabel="Media global"
)

# ============================================================
# 15. RESUMEN TXT
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

    f.write("VARIABLES ANALIZADAS\n")
    f.write("-" * 75 + "\n")

    for variable, indicadores in VARIABLES.items():

        f.write(
            f"\n{variable} ({len(indicadores)} indicadores):\n"
        )

        for indicador in indicadores:
            f.write(f"  - {indicador}\n")

    f.write("\n\nRESULTADOS POR VARIABLE\n")
    f.write("-" * 75 + "\n\n")

    f.write(
        tabla_variables.round(4).to_string(
            index=False
        )
    )

    f.write("\n\nRESULTADO GLOBAL\n")
    f.write("-" * 75 + "\n")

    f.write(
        tabla_global.round(4).to_string(
            index=False
        )
    )

# ============================================================
# 16. MOSTRAR RESULTADOS EN CONSOLA
# ============================================================

print()
print("=" * 75)
print("RESULTADOS COMPARATIVOS GENERADOS")
print("=" * 75)

print()
print("RESULTADOS POR VARIABLE:")
print()

print(
    tabla_variables.round(4).to_string(
        index=False
    )
)

print()
print("=" * 75)
print("RESULTADO GLOBAL")
print("=" * 75)

print(
    f"Control Pre-test:       "
    f"{control_pre_global:.4f}"
)

print(
    f"Control Post-test:      "
    f"{control_post_global:.4f}"
)

print(
    f"Cambio Control:         "
    f"{cambio_control_global:.4f}"
)

print(
    f"Experimental Pre-test:  "
    f"{experimental_pre_global:.4f}"
)

print(
    f"Experimental Post-test: "
    f"{experimental_post_global:.4f}"
)

print(
    f"Cambio Experimental:    "
    f"{cambio_experimental_global:.4f}"
)

print(
    f"Diferencia diferencias: "
    f"{did_global:.4f}"
)

print()
print("=" * 75)
print("ARCHIVOS GENERADOS")
print("=" * 75)

print(f"Excel: {archivo_excel}")
print(f"Gráficos: {GRAFICOS}")
print(f"Resumen: {archivo_txt}")
print("=" * 75)
