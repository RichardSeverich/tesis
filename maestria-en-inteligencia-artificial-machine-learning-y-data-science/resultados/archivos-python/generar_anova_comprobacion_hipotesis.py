# ============================================================
# COMPROBACIÓN DE HIPÓTESIS - ANOVA FACTORIAL 2x2
# TESIS: SISTEMA DE GESTIÓN ACADÉMICA WEB INTEGRADO CON CHATBOT
# ============================================================
#
# DISEÑO:
#   Factor 1: GRUPO
#       Control
#       Experimental
#
#   Factor 2: MOMENTO
#       Pre-test
#       Post-test
#
#   Interacción:
#       Grupo × Momento
#
# El efecto de mayor interés es la INTERACCIÓN Grupo × Momento,
# porque permite determinar si el cambio entre Pre-test y Post-test
# difiere entre el grupo experimental y el grupo control.
#
# Variables:
#   1. Sistema de gestión académica web
#   2. Chatbot
#   3. Gestión académica
#   4. Atención al cliente
#
# Para cada variable se calcula:
#   - Suma de cuadrados
#   - gl
#   - F
#   - p-valor
#   - eta cuadrado parcial
#
# También genera:
#   - Medias por grupo y momento
#   - Gráfico de interacción
#   - Gráfico de medias
#   - Excel
#   - Resumen TXT
#
# NOTA:
# Este ANOVA factorial se presenta como análisis complementario.
# Debido a que Pre-test y Post-test corresponden a las mismas
# personas dentro de cada grupo, la estructura es longitudinal.
# Por ello, además del ANOVA, se mantienen las pruebas t del
# archivo anterior como parte del análisis principal del cambio.
#
# Requisitos:
# pandas==2.2.3
# openpyxl==3.1.5
# matplotlib==3.9.2
# numpy==2.0.2
# scipy==1.13.1
# statsmodels==0.14.4
# ============================================================

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

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

SALIDA = BASE_DIR / "resultados_anova"
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
# 3. LECTURA Y VALIDACIÓN
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
                f"\nLa columna '{columna}' del archivo "
                f"{ruta.name} contiene valores fuera de 1-5: "
                f"{sorted(invalidos.unique().tolist())}"
            )

    return df


def crear_variables(df):

    resultado = pd.DataFrame(index=df.index)

    for variable, indicadores in VARIABLES.items():

        resultado[variable] = df[
            indicadores
        ].mean(axis=1)

    return resultado


# ============================================================
# 4. CARGA DE LOS DATOS
# ============================================================

print("=" * 80)
print("COMPROBACIÓN DE HIPÓTESIS - ANOVA FACTORIAL 2x2")
print("=" * 80)

datos_variables = {}

for nombre, ruta in ARCHIVOS.items():

    print(f"\nLeyendo: {ruta.name}")

    df = leer_archivo(ruta)

    datos_variables[nombre] = crear_variables(df)

    print(
        f"  Participantes: {len(df)}"
    )


# ============================================================
# 5. CONSTRUIR DATAFRAME LONGITUDINAL
# ============================================================
#
# Se conserva la correspondencia por fila.
#
# Cada participante tiene:
#   Control Pre
#   Control Post
# o
#   Experimental Pre
#   Experimental Post
#
# El identificador es local al grupo.
# ============================================================

longitudinal = []

for clave, grupo, momento in [
    ("Control Pre-test", "Control", "Pre-test"),
    ("Control Post-test", "Control", "Post-test"),
    ("Experimental Pre-test", "Experimental", "Pre-test"),
    ("Experimental Post-test", "Experimental", "Post-test"),
]:

    df = datos_variables[clave].copy()

    df["ID"] = np.arange(
        1,
        len(df) + 1
    )

    df["Grupo"] = grupo
    df["Momento"] = momento

    longitudinal.append(df)

longitudinal = pd.concat(
    longitudinal,
    ignore_index=True
)


# ============================================================
# 6. ANOVA 2x2 POR VARIABLE
# ============================================================
#
# Modelo:
#
# Y = Grupo + Momento + Grupo:Momento
#
# Se utiliza Type II ANOVA.
#
# La interacción Grupo:Momento es el efecto central.
# ============================================================

resultados_anova = []
tablas_anova = []

for variable in VARIABLES:

    datos_modelo = longitudinal[
        [
            "ID",
            "Grupo",
            "Momento",
            variable
        ]
    ].dropna().copy()

    datos_modelo = datos_modelo.rename(
        columns={
            variable: "Y"
        }
    )

    # Modelo factorial 2x2
    modelo = ols(
        "Y ~ C(Grupo) * C(Momento)",
        data=datos_modelo
    ).fit()

    tabla = anova_lm(
        modelo,
        typ=2
    )

    # Error residual
    ss_error = tabla.loc[
        "Residual",
        "sum_sq"
    ]

    # eta parcial para cada efecto
    for efecto in [
        "C(Grupo)",
        "C(Momento)",
        "C(Grupo):C(Momento)"
    ]:

        ss_efecto = tabla.loc[
            efecto,
            "sum_sq"
        ]

        eta_parcial = (
            ss_efecto
            / (ss_efecto + ss_error)
        )

        nombre_efecto = {
            "C(Grupo)": "Efecto Grupo",
            "C(Momento)": "Efecto Momento",
            "C(Grupo):C(Momento)": "Interacción Grupo × Momento"
        }[efecto]

        resultados_anova.append({
            "Variable": variable,
            "Efecto": nombre_efecto,
            "Suma de cuadrados": ss_efecto,
            "gl": tabla.loc[efecto, "df"],
            "F": tabla.loc[efecto, "F"],
            "p-valor": tabla.loc[efecto, "PR(>F)"],
            "Eta² parcial": eta_parcial,
            "Significancia":
                "Significativo"
                if tabla.loc[efecto, "PR(>F)"] < ALPHA
                else "No significativo",
            "Decisión":
                "Rechazar H0"
                if tabla.loc[efecto, "PR(>F)"] < ALPHA
                else "No rechazar H0",
        })

    # Guardar tabla ANOVA completa
    tabla_salida = tabla.reset_index()

    tabla_salida.insert(
        0,
        "Variable",
        variable
    )

    tabla_salida.rename(
        columns={
            "index": "Efecto",
            "sum_sq": "Suma de cuadrados",
            "df": "gl",
            "PR(>F)": "p-valor"
        },
        inplace=True
    )

    tablas_anova.append(
        tabla_salida
    )


tabla_resultados = pd.DataFrame(
    resultados_anova
)

tabla_anova_completa = pd.concat(
    tablas_anova,
    ignore_index=True
)


# ============================================================
# 7. TABLA DE MEDIAS POR GRUPO Y MOMENTO
# ============================================================

medias = (
    longitudinal
    .groupby(
        ["Grupo", "Momento"]
    )[list(VARIABLES.keys())]
    .agg(
        ["mean", "std", "count"]
    )
    .reset_index()
)

# Aplanar columnas
nuevas_columnas = []

for columna in medias.columns:

    if isinstance(columna, tuple):

        if columna[1] == "":
            nuevas_columnas.append(
                columna[0]
            )
        else:
            nuevas_columnas.append(
                f"{columna[0]} - {columna[1]}"
            )
    else:
        nuevas_columnas.append(
            columna
        )

medias.columns = nuevas_columnas


# ============================================================
# 8. TABLA ESPECÍFICA DE LA INTERACCIÓN
# ============================================================

interaccion = tabla_resultados[
    tabla_resultados["Efecto"]
    == "Interacción Grupo × Momento"
].copy()

interaccion = interaccion[
    [
        "Variable",
        "F",
        "gl",
        "p-valor",
        "Eta² parcial",
        "Significancia",
        "Decisión"
    ]
]


# ============================================================
# 9. INTERPRETACIÓN DE ETA² PARCIAL
# ============================================================

def interpretar_eta(eta):

    if pd.isna(eta):
        return "No calculable"

    if eta < 0.01:
        return "efecto muy pequeño"
    elif eta < 0.06:
        return "efecto pequeño"
    elif eta < 0.14:
        return "efecto mediano"
    else:
        return "efecto grande"


tabla_resultados[
    "Magnitud Eta² parcial"
] = tabla_resultados[
    "Eta² parcial"
].apply(
    interpretar_eta
)


interaccion[
    "Magnitud Eta² parcial"
] = interaccion[
    "Eta² parcial"
].apply(
    interpretar_eta
)


# ============================================================
# 10. GRÁFICOS DE INTERACCIÓN
# ============================================================

for variable in VARIABLES:

    resumen = (
        longitudinal
        .groupby(
            ["Grupo", "Momento"]
        )[variable]
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    for grupo in [
        "Control",
        "Experimental"
    ]:

        datos_grupo = resumen[
            resumen["Grupo"] == grupo
        ]

        datos_grupo = datos_grupo.set_index(
            "Momento"
        ).reindex(
            ["Pre-test", "Post-test"]
        )

        ax.plot(
            ["Pre-test", "Post-test"],
            datos_grupo[variable].values,
            marker="o",
            linewidth=2,
            label=grupo
        )

    ax.set_ylabel(
        "Media"
    )

    ax.set_xlabel(
        "Momento"
    )

    ax.set_title(
        f"Interacción Grupo × Momento\n{variable}",
        fontsize=13,
        fontweight="bold"
    )

    ax.legend()

    ax.grid(
        True,
        alpha=0.25
    )

    plt.tight_layout()

    nombre = (
        variable
        .lower()
        .replace(" ", "_")
        .replace("×", "x")
    )

    plt.savefig(
        GRAFICOS
        / f"interaccion_{nombre}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# 11. GRÁFICO COMPARATIVO DE LOS EFECTOS F
# ============================================================

efectos = [
    "Efecto Grupo",
    "Efecto Momento",
    "Interacción Grupo × Momento"
]

for efecto in efectos:

    datos_efecto = tabla_resultados[
        tabla_resultados["Efecto"] == efecto
    ]

    fig, ax = plt.subplots(
        figsize=(11, 6)
    )

    ax.bar(
        datos_efecto["Variable"],
        datos_efecto["F"]
    )

    ax.set_ylabel(
        "Estadístico F"
    )

    ax.set_title(
        f"Estadístico F por variable - {efecto}",
        fontsize=13,
        fontweight="bold"
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

    nombre = (
        efecto
        .lower()
        .replace(" ", "_")
        .replace("×", "x")
    )

    plt.savefig(
        GRAFICOS
        / f"F_{nombre}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# 12. GRÁFICO DE P-VALORES DE LA INTERACCIÓN
# ============================================================

fig, ax = plt.subplots(
    figsize=(10, 6)
)

ax.bar(
    interaccion["Variable"],
    interaccion["p-valor"]
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
    "Significancia de la interacción Grupo × Momento",
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
    / "p_valores_interaccion.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 13. GRÁFICO DE ETA² PARCIAL
# ============================================================

fig, ax = plt.subplots(
    figsize=(10, 6)
)

ax.bar(
    interaccion["Variable"],
    interaccion["Eta² parcial"]
)

ax.set_ylabel(
    "Eta² parcial"
)

ax.set_title(
    "Tamaño del efecto de la interacción Grupo × Momento",
    fontsize=13,
    fontweight="bold"
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
    / "eta_cuadrado_parcial_interaccion.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 14. EXPORTACIÓN A EXCEL
# ============================================================

archivo_excel = (
    SALIDA
    / "tablas_anova.xlsx"
)

with pd.ExcelWriter(
    archivo_excel,
    engine="openpyxl"
) as writer:

    tabla_resultados.to_excel(
        writer,
        sheet_name="ANOVA Resultados",
        index=False
    )

    tabla_anova_completa.to_excel(
        writer,
        sheet_name="ANOVA Completo",
        index=False
    )

    interaccion.to_excel(
        writer,
        sheet_name="Interaccion",
        index=False
    )

    medias.to_excel(
        writer,
        sheet_name="Medias",
        index=False
    )

    longitudinal.to_excel(
        writer,
        sheet_name="Datos Longitudinales",
        index=False
    )


# ============================================================
# 15. RESUMEN TXT
# ============================================================

archivo_txt = (
    SALIDA
    / "resumen_anova.txt"
)

with open(
    archivo_txt,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "COMPROBACIÓN DE HIPÓTESIS - ANOVA FACTORIAL 2x2\n"
    )

    f.write(
        "=" * 80 + "\n\n"
    )

    f.write(
        "Factor 1: Grupo (Control / Experimental)\n"
        "Factor 2: Momento (Pre-test / Post-test)\n"
        "Efecto principal de interés: Grupo × Momento\n"
        "Nivel de significancia: α = 0.05\n\n"
    )

    f.write(
        "INTERPRETACIÓN:\n"
        "La interacción Grupo × Momento permite evaluar si la "
        "evolución entre el pre-test y el post-test difiere "
        "entre el grupo control y el experimental.\n\n"
    )

    for _, fila in interaccion.iterrows():

        f.write(
            f"Variable: {fila['Variable']}\n"
            f"  F = {fila['F']:.4f}\n"
            f"  gl = {fila['gl']}\n"
            f"  p = {fila['p-valor']:.6f}\n"
            f"  Eta² parcial = "
            f"{fila['Eta² parcial']:.4f}\n"
            f"  Magnitud: "
            f"{fila['Magnitud Eta² parcial']}\n"
            f"  Significancia: "
            f"{fila['Significancia']}\n"
            f"  Decisión: "
            f"{fila['Decisión']}\n\n"
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
print("1. ANOVA factorial 2x2 por variable")
print("2. Efecto Grupo")
print("3. Efecto Momento")
print("4. Interacción Grupo × Momento")
print("5. Eta² parcial")
print("6. Gráficos de interacción")
print("7. Comparación de estadísticos F")
print("8. P-valores de interacción")

print("\n" + "=" * 80)
print("CRITERIO DE DECISIÓN")
print("=" * 80)

print(
    "\nSi p < 0.05: se rechaza H0."
    "\nSi p >= 0.05: no se rechaza H0."
)

print(
    "\nLa interacción Grupo × Momento es el resultado de mayor "
    "interés para determinar si la evolución temporal difiere "
    "entre el grupo experimental y el grupo control."
)

print("\nListo.")
