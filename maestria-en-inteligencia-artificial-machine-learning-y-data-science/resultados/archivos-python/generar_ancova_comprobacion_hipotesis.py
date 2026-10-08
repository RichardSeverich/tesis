# ============================================================
# COMPROBACIÓN DE HIPÓTESIS - ANCOVA
# TESIS: SISTEMA DE GESTIÓN ACADÉMICA WEB INTEGRADO CON CHATBOT
# ============================================================
#
# PRUEBA EXTRA RECOMENDADA: ANCOVA
#
# Diseño:
#   Variable independiente: Grupo
#       Control / Experimental
#
#   Covariable: Pre-test
#
#   Variable dependiente: Post-test
#
# Modelo principal:
#   Post-test = Pre-test + Grupo
#
# ¿POR QUÉ ANCOVA?
# En un diseño cuasi-experimental con pre-test y post-test,
# ANCOVA permite comparar los resultados finales entre el grupo
# control y experimental, ajustando estadísticamente las diferencias
# existentes en el nivel inicial (pre-test).
#
# Se aplica a las cuatro variables:
#   1. Sistema de gestión académica web
#   2. Chatbot
#   3. Gestión académica
#   4. Atención al cliente
#
# El efecto de mayor interés es:
#   GRUPO
#
# Si p < 0.05 para Grupo, existe evidencia de una diferencia
# estadísticamente significativa en el post-test entre Control
# y Experimental después de controlar el pre-test.
#
# Además se comprueba el supuesto de homogeneidad de pendientes:
#
#   Post-test = Pre-test + Grupo + Pre-test:Grupo
#
# Si la interacción Pre-test × Grupo no es significativa,
# la aplicación del ANCOVA convencional es compatible con este
# supuesto.
#
# CALCULA:
#   - Medias pre-test
#   - Medias post-test
#   - Medias post-test ajustadas
#   - F
#   - gl
#   - p-valor
#   - Eta² parcial
#   - Cohen's d ajustado
#   - Prueba de homogeneidad de pendientes
#
# GENERA:
#   - Excel
#   - Resumen TXT
#   - Gráficos de medias observadas
#   - Gráficos de medias ajustadas
#   - Gráficos pre-test vs post-test
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

from scipy.stats import t
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

SALIDA = BASE_DIR / "resultados_ancova"
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
# 3. FUNCIONES
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
                f"fuera de 1-5: "
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


def decision(p):

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


# ============================================================
# 4. CARGAR DATOS
# ============================================================

print("=" * 80)
print("COMPROBACIÓN DE HIPÓTESIS - ANCOVA")
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
# 5. CONSTRUIR DATOS PAREADOS PRE/POST
# ============================================================
#
# Cada fila representa al mismo participante dentro de su grupo.
# ============================================================

datos_ancova = []

for grupo in ["Control", "Experimental"]:

    pre = datos_variables[
        f"{grupo} Pre-test"
    ].copy()

    post = datos_variables[
        f"{grupo} Post-test"
    ].copy()

    for i in range(len(pre)):

        fila = {
            "ID": i + 1,
            "Grupo": grupo
        }

        for variable in VARIABLES:

            fila[
                f"{variable} Pre-test"
            ] = pre.iloc[i][variable]

            fila[
                f"{variable} Post-test"
            ] = post.iloc[i][variable]

        datos_ancova.append(fila)


datos_ancova = pd.DataFrame(
    datos_ancova
)


# ============================================================
# 6. ANCOVA POR VARIABLE
# ============================================================
#
# Modelo principal:
#
#   Post ~ Pre + Grupo
#
# Se utiliza Type II ANOVA sobre el modelo.
#
# El efecto "Grupo" es el efecto ajustado por el pre-test.
# ============================================================

resultados_ancova = []
tablas_ancova = []
pendientes = []
medias_ajustadas = []


for variable in VARIABLES:

    pre_col = f"{variable} Pre-test"
    post_col = f"{variable} Post-test"

    df = datos_ancova[
        [
            "ID",
            "Grupo",
            pre_col,
            post_col
        ]
    ].copy()

    df = df.rename(
        columns={
            pre_col: "Pre",
            post_col: "Post"
        }
    )

    df = df.dropna()

    # --------------------------------------------------------
    # Modelo ANCOVA principal
    # --------------------------------------------------------

    modelo = ols(
        "Post ~ Pre + C(Grupo)",
        data=df
    ).fit()

    tabla = anova_lm(
        modelo,
        typ=2
    )

    ss_error = tabla.loc[
        "Residual",
        "sum_sq"
    ]

    # Efecto de Grupo
    ss_grupo = tabla.loc[
        "C(Grupo)",
        "sum_sq"
    ]

    f_grupo = tabla.loc[
        "C(Grupo)",
        "F"
    ]

    p_grupo = tabla.loc[
        "C(Grupo)",
        "PR(>F)"
    ]

    eta_grupo = (
        ss_grupo
        / (ss_grupo + ss_error)
    )

    # Efecto de Pre-test
    ss_pre = tabla.loc[
        "Pre",
        "sum_sq"
    ]

    f_pre = tabla.loc[
        "Pre",
        "F"
    ]

    p_pre = tabla.loc[
        "Pre",
        "PR(>F)"
    ]

    eta_pre = (
        ss_pre
        / (ss_pre + ss_error)
    )

    resultados_ancova.append({
        "Variable": variable,
        "N": len(df),
        "F Grupo": f_grupo,
        "gl Grupo": tabla.loc[
            "C(Grupo)", "df"
        ],
        "p Grupo": p_grupo,
        "Eta² parcial Grupo": eta_grupo,
        "Magnitud efecto Grupo":
            interpretar_eta(eta_grupo),
        "Decisión Grupo":
            decision(p_grupo),

        "F Pre-test": f_pre,
        "gl Pre-test": tabla.loc[
            "Pre", "df"
        ],
        "p Pre-test": p_pre,
        "Eta² parcial Pre-test": eta_pre,

        "R² modelo": modelo.rsquared,
        "R² ajustado": modelo.rsquared_adj,
    })

    tabla_completa = tabla.reset_index()

    tabla_completa.insert(
        0,
        "Variable",
        variable
    )

    tabla_completa.rename(
        columns={
            "index": "Efecto",
            "sum_sq": "Suma de cuadrados",
            "df": "gl",
            "PR(>F)": "p-valor"
        },
        inplace=True
    )

    tablas_ancova.append(
        tabla_completa
    )

    # --------------------------------------------------------
    # Prueba de homogeneidad de pendientes
    # Post ~ Pre * Grupo
    # --------------------------------------------------------

    modelo_interaccion = ols(
        "Post ~ Pre * C(Grupo)",
        data=df
    ).fit()

    tabla_interaccion = anova_lm(
        modelo_interaccion,
        typ=2
    )

    nombre_interaccion = "Pre:C(Grupo)"

    if nombre_interaccion in tabla_interaccion.index:

        f_interaccion = tabla_interaccion.loc[
            nombre_interaccion,
            "F"
        ]

        p_interaccion = tabla_interaccion.loc[
            nombre_interaccion,
            "PR(>F)"
        ]

    else:
        f_interaccion = np.nan
        p_interaccion = np.nan

    pendientes.append({
        "Variable": variable,
        "F Pre-test × Grupo": f_interaccion,
        "p Pre-test × Grupo": p_interaccion,
        "Significancia": significancia(
            p_interaccion
        ),
        "Supuesto de pendientes":
            "Compatible"
            if p_interaccion >= ALPHA
            else "No compatible"
    })

    # --------------------------------------------------------
    # Medias observadas
    # --------------------------------------------------------

    medias = (
        df.groupby("Grupo")
        .agg(
            N=("Post", "count"),
            Media_Pre=("Pre", "mean"),
            Media_Post=("Post", "mean"),
            SD_Pre=("Pre", "std"),
            SD_Post=("Post", "std")
        )
        .reset_index()
    )

    # --------------------------------------------------------
    # Medias ajustadas por ANCOVA
    #
    # Se ajustan ambos grupos a la media general del pre-test.
    # --------------------------------------------------------

    media_pre_general = df["Pre"].mean()

    for grupo in ["Control", "Experimental"]:

        fila = medias[
            medias["Grupo"] == grupo
        ]

        if fila.empty:
            continue

        media_observada = (
            fila["Media_Post"].iloc[0]
        )

        # Predicción del modelo para el grupo,
        # manteniendo Pre en la media general.
        nuevo = pd.DataFrame({
            "Pre": [media_pre_general],
            "Grupo": [grupo]
        })

        media_ajustada = modelo.predict(
            nuevo
        ).iloc[0]

        medias_ajustadas.append({
            "Variable": variable,
            "Grupo": grupo,
            "N": int(
                fila["N"].iloc[0]
            ),
            "Media Pre-test":
                fila["Media_Pre"].iloc[0],
            "Media Post-test observada":
                media_observada,
            "Media Post-test ajustada":
                media_ajustada,
            "Media Pre-test general":
                media_pre_general
        })


tabla_resultados = pd.DataFrame(
    resultados_ancova
)

tabla_ancova_completa = pd.concat(
    tablas_ancova,
    ignore_index=True
)

tabla_pendientes = pd.DataFrame(
    pendientes
)

tabla_medias_ajustadas = pd.DataFrame(
    medias_ajustadas
)


# ============================================================
# 7. TABLA RESUMEN PRINCIPAL
# ============================================================

resumen = tabla_resultados[
    [
        "Variable",
        "N",
        "F Grupo",
        "gl Grupo",
        "p Grupo",
        "Eta² parcial Grupo",
        "Magnitud efecto Grupo",
        "Decisión Grupo",
        "R² modelo",
        "R² ajustado"
    ]
].copy()


# ============================================================
# 8. GRÁFICO DE MEDIAS OBSERVADAS
# ============================================================

variables = list(VARIABLES.keys())

control_pre = [
    datos_variables[
        "Control Pre-test"
    ][v].mean()
    for v in variables
]

control_post = [
    datos_variables[
        "Control Post-test"
    ][v].mean()
    for v in variables
]

experimental_pre = [
    datos_variables[
        "Experimental Pre-test"
    ][v].mean()
    for v in variables
]

experimental_post = [
    datos_variables[
        "Experimental Post-test"
    ][v].mean()
    for v in variables
]

x = np.arange(
    len(variables)
)

ancho = 0.20

fig, ax = plt.subplots(
    figsize=(13, 7)
)

ax.bar(
    x - 1.5 * ancho,
    control_pre,
    ancho,
    label="Control Pre-test"
)

ax.bar(
    x - 0.5 * ancho,
    control_post,
    ancho,
    label="Control Post-test"
)

ax.bar(
    x + 0.5 * ancho,
    experimental_pre,
    ancho,
    label="Experimental Pre-test"
)

ax.bar(
    x + 1.5 * ancho,
    experimental_post,
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
    "Media"
)

ax.set_title(
    "Comparación de medias Pre-test y Post-test",
    fontsize=13,
    fontweight="bold"
)

ax.legend()

ax.grid(
    axis="y",
    alpha=0.25
)

plt.tight_layout()

plt.savefig(
    GRAFICOS
    / "medias_observadas_pre_post.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 9. GRÁFICO DE MEDIAS POST-TEST AJUSTADAS
# ============================================================

fig, ax = plt.subplots(
    figsize=(12, 7)
)

control_ajustado = []
experimental_ajustado = []

for variable in variables:

    fila_control = tabla_medias_ajustadas[
        (tabla_medias_ajustadas["Variable"] == variable)
        & (tabla_medias_ajustadas["Grupo"] == "Control")
    ]

    fila_exp = tabla_medias_ajustadas[
        (tabla_medias_ajustadas["Variable"] == variable)
        & (tabla_medias_ajustadas["Grupo"] == "Experimental")
    ]

    control_ajustado.append(
        fila_control[
            "Media Post-test ajustada"
        ].iloc[0]
    )

    experimental_ajustado.append(
        fila_exp[
            "Media Post-test ajustada"
        ].iloc[0]
    )

x = np.arange(
    len(variables)
)

ax.bar(
    x - ancho / 2,
    control_ajustado,
    ancho,
    label="Control ajustado"
)

ax.bar(
    x + ancho / 2,
    experimental_ajustado,
    ancho,
    label="Experimental ajustado"
)

ax.set_xticks(x)

ax.set_xticklabels(
    variables,
    rotation=25,
    ha="right"
)

ax.set_ylabel(
    "Media Post-test ajustada"
)

ax.set_title(
    "Comparación de medias Post-test ajustadas por ANCOVA",
    fontsize=13,
    fontweight="bold"
)

ax.legend()

ax.grid(
    axis="y",
    alpha=0.25
)

plt.tight_layout()

plt.savefig(
    GRAFICOS
    / "medias_post_test_ajustadas.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 10. GRÁFICO DE P-VALORES DEL EFECTO GRUPO
# ============================================================

fig, ax = plt.subplots(
    figsize=(11, 6)
)

ax.bar(
    resumen["Variable"],
    resumen["p Grupo"]
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
    "Significancia del efecto Grupo en ANCOVA",
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
    / "p_valores_ancova_grupo.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 11. GRÁFICO DE ETA² PARCIAL
# ============================================================

fig, ax = plt.subplots(
    figsize=(11, 6)
)

ax.bar(
    resumen["Variable"],
    resumen["Eta² parcial Grupo"]
)

ax.set_ylabel(
    "Eta² parcial"
)

ax.set_title(
    "Tamaño del efecto del Grupo - ANCOVA",
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
    / "eta_cuadrado_ancova.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 12. EXPORTAR EXCEL
# ============================================================

archivo_excel = (
    SALIDA
    / "tablas_ancova.xlsx"
)

with pd.ExcelWriter(
    archivo_excel,
    engine="openpyxl"
) as writer:

    resumen.to_excel(
        writer,
        sheet_name="Resumen ANCOVA",
        index=False
    )

    tabla_resultados.to_excel(
        writer,
        sheet_name="Resultados ANCOVA",
        index=False
    )

    tabla_ancova_completa.to_excel(
        writer,
        sheet_name="ANCOVA Completo",
        index=False
    )

    tabla_medias_ajustadas.to_excel(
        writer,
        sheet_name="Medias Ajustadas",
        index=False
    )

    tabla_pendientes.to_excel(
        writer,
        sheet_name="Supuesto Pendientes",
        index=False
    )

    datos_ancova.to_excel(
        writer,
        sheet_name="Datos ANCOVA",
        index=False
    )


# ============================================================
# 13. RESUMEN TXT
# ============================================================

archivo_txt = (
    SALIDA
    / "resumen_ancova.txt"
)

with open(
    archivo_txt,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "COMPROBACIÓN DE HIPÓTESIS - ANCOVA\n"
    )

    f.write(
        "=" * 80 + "\n\n"
    )

    f.write(
        "Modelo: Post-test ~ Pre-test + Grupo\n"
        "Nivel de significancia: α = 0.05\n\n"
    )

    f.write(
        "El efecto de mayor interés es GRUPO, "
        "porque determina si existen diferencias "
        "entre Control y Experimental en el post-test "
        "después de controlar estadísticamente el pre-test.\n\n"
    )

    f.write(
        "RESULTADOS\n"
    )

    f.write(
        "-" * 80 + "\n\n"
    )

    for _, fila in resumen.iterrows():

        f.write(
            f"Variable: {fila['Variable']}\n"
            f"  N = {fila['N']}\n"
            f"  F Grupo = {fila['F Grupo']:.4f}\n"
            f"  gl Grupo = {fila['gl Grupo']}\n"
            f"  p Grupo = {fila['p Grupo']:.6f}\n"
            f"  Eta² parcial = "
            f"{fila['Eta² parcial Grupo']:.4f}\n"
            f"  Magnitud = "
            f"{fila['Magnitud efecto Grupo']}\n"
            f"  Decisión = "
            f"{fila['Decisión Grupo']}\n"
            f"  R² = {fila['R² modelo']:.4f}\n"
            f"  R² ajustado = "
            f"{fila['R² ajustado']:.4f}\n\n"
        )

    f.write(
        "=" * 80 + "\n"
    )

    f.write(
        "SUPUESTO DE HOMOGENEIDAD DE PENDIENTES\n"
    )

    f.write(
        "=" * 80 + "\n\n"
    )

    for _, fila in tabla_pendientes.iterrows():

        f.write(
            f"{fila['Variable']} | "
            f"F={fila['F Pre-test × Grupo']:.4f} | "
            f"p={fila['p Pre-test × Grupo']:.6f} | "
            f"{fila['Supuesto de pendientes']}\n"
        )


# ============================================================
# 14. SALIDA
# ============================================================

print("\n" + "=" * 80)
print("PROCESO TERMINADO")
print("=" * 80)

print("\nExcel:")
print(archivo_excel)

print("\nGráficos:")
print(GRAFICOS)

print("\nAnálisis generado:")
print("1. ANCOVA por variable")
print("2. Efecto Grupo ajustado por Pre-test")
print("3. Efecto Pre-test")
print("4. Eta² parcial")
print("5. Medias Post-test ajustadas")
print("6. Homogeneidad de pendientes")
print("7. Gráficos comparativos")

print("\n" + "=" * 80)
print("CRITERIO DE DECISIÓN")
print("=" * 80)

print(
    "\nSi p Grupo < 0.05: se rechaza H0."
    "\nSi p Grupo >= 0.05: no se rechaza H0."
)

print(
    "\nLa interacción Pre-test × Grupo debe ser no significativa "
    "para considerar compatible el supuesto de homogeneidad "
    "de pendientes del ANCOVA convencional."
)

print("\nListo.")
