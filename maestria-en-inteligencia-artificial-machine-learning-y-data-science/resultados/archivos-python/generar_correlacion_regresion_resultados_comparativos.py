# ============================================================
# ANÁLISIS DE CORRELACIÓN Y REGRESIÓN LINEAL
# TESIS: SISTEMA DE GESTIÓN ACADÉMICA WEB INTEGRADO CON CHATBOT
# ============================================================
#
# ESTRUCTURA REAL DE LAS VARIABLES
#
# 1. Sistema de gestión académica web -> indicadores 1-4
# 2. Chatbot                         -> indicadores 5-8
# 3. Gestión académica              -> indicadores 9-12
# 4. Atención al cliente            -> indicadores 13-16
#
# El programa genera:
#
# A) CORRELACIÓN DE PEARSON
#    - Matrices por grupo y momento
#    - Correlaciones entre variables independientes y dependientes
#
# B) REGRESIÓN LINEAL MÚLTIPLE
#    Modelo 1: Gestión académica ~ Sistema web + Chatbot
#    Modelo 2: Atención al cliente ~ Sistema web + Chatbot
#
# C) REGRESIÓN DE CAMBIO EN EL GRUPO EXPERIMENTAL
#    Δ Gestión académica ~ Δ Sistema web + Δ Chatbot
#    Δ Atención al cliente ~ Δ Sistema web + Δ Chatbot
#
# D) TABLAS COMPARATIVAS
#    - Pearson por grupo/momento
#    - R, R², R² ajustado, F y p
#    - Coeficientes B, Beta y p de cada predictor
#    - Comparación de R² entre grupos/momentos
#
# E) GRÁFICOS
#    - Heatmaps de correlaciones
#    - Dispersión + línea de regresión para las relaciones
#      principales del grupo experimental post-test
#    - Comparación de R² de los modelos
#
# IMPORTANTE:
# Este análisis estudia asociación y capacidad predictiva lineal.
# Una regresión por sí sola NO demuestra causalidad.
#
# Requisitos:
# pip install pandas openpyxl numpy scipy matplotlib statsmodels
# ============================================================

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import pearsonr
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

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

SALIDA = BASE_DIR / "resultados_correlacion_regresion"
GRAFICOS = SALIDA / "graficos"

SALIDA.mkdir(parents=True, exist_ok=True)
GRAFICOS.mkdir(parents=True, exist_ok=True)


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

INDEPENDIENTES = [
    "Sistema de gestión académica web",
    "Chatbot",
]

DEPENDIENTES = [
    "Gestión académica",
    "Atención al cliente",
]


# ============================================================
# 3. FUNCIONES
# ============================================================

def leer_archivo(ruta):
    """Lee y valida uno de los cuatro archivos."""
    if not ruta.exists():
        raise FileNotFoundError(
            f"\nNo se encontró el archivo:\n{ruta}\n"
            f"Colócalo en la misma carpeta del programa."
        )

    df = pd.read_excel(ruta)

    faltantes = [c for c in INDICADORES if c not in df.columns]

    if faltantes:
        raise ValueError(
            f"\nEl archivo {ruta.name} no contiene estas columnas:\n"
            + "\n".join(f"- {c}" for c in faltantes)
        )

    for columna in INDICADORES:
        df[columna] = pd.to_numeric(df[columna], errors="coerce")

        valores = df[columna].dropna()

        if valores.empty:
            raise ValueError(
                f"La columna '{columna}' del archivo {ruta.name} "
                f"no contiene datos numéricos."
            )

        invalidos = valores[~valores.isin([1, 2, 3, 4, 5])]

        if not invalidos.empty:
            raise ValueError(
                f"La columna '{columna}' del archivo {ruta.name} "
                f"contiene valores fuera de 1-5: "
                f"{sorted(invalidos.unique().tolist())}"
            )

    return df


def crear_variables(df):
    """
    Calcula un puntaje por variable para cada participante.

    Cada variable está formada por 4 indicadores.
    Se utiliza la media de esos 4 indicadores por participante.
    """
    resultado = pd.DataFrame(index=df.index)

    for variable, indicadores in VARIABLES.items():
        resultado[variable] = df[indicadores].mean(axis=1)

    return resultado


def correlacion_pearson(x, y):
    """Devuelve r y p de Pearson."""
    datos = pd.concat([x, y], axis=1).dropna()

    if len(datos) < 3:
        return np.nan, np.nan, len(datos)

    r, p = pearsonr(datos.iloc[:, 0], datos.iloc[:, 1])

    return r, p, len(datos)


def interpretar_r(r):
    """Interpretación convencional de la magnitud de r."""
    if pd.isna(r):
        return "No calculable"

    a = abs(r)

    if a < 0.10:
        fuerza = "muy débil"
    elif a < 0.30:
        fuerza = "débil"
    elif a < 0.50:
        fuerza = "moderada"
    elif a < 0.70:
        fuerza = "fuerte"
    elif a < 0.90:
        fuerza = "muy fuerte"
    else:
        fuerza = "extremadamente fuerte"

    direccion = "positiva" if r > 0 else "negativa" if r < 0 else "nula"

    return f"{fuerza} {direccion}"


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


def regresion_multiple(datos, dependiente):
    """
    Regresión lineal múltiple:
        Y = b0 + b1 X1 + b2 X2

    X1 = Sistema de gestión académica web
    X2 = Chatbot
    """
    columnas = INDEPENDIENTES + [dependiente]

    datos = datos[columnas].dropna().copy()

    X = datos[INDEPENDIENTES]
    y = datos[dependiente]

    X_const = sm.add_constant(X)

    modelo = sm.OLS(y, X_const).fit()

    # Betas estandarizados
    beta_estandarizados = {}

    for predictor in INDEPENDIENTES:
        sd_x = X[predictor].std(ddof=1)
        sd_y = y.std(ddof=1)

        if sd_x == 0 or sd_y == 0:
            beta_estandarizados[predictor] = np.nan
        else:
            beta_estandarizados[predictor] = (
                modelo.params[predictor] * sd_x / sd_y
            )

    # VIF
    vif_resultados = {}

    X_vif = sm.add_constant(X)

    for i, predictor in enumerate(INDEPENDIENTES, start=1):
        try:
            vif_resultados[predictor] = variance_inflation_factor(
                X_vif.values,
                i
            )
        except Exception:
            vif_resultados[predictor] = np.nan

    return modelo, datos, beta_estandarizados, vif_resultados


def guardar_heatmap(matriz, titulo, nombre_archivo):
    """Genera un heatmap de correlaciones."""
    fig, ax = plt.subplots(figsize=(9, 7))

    imagen = ax.imshow(
        matriz.values,
        vmin=-1,
        vmax=1,
        aspect="auto"
    )

    ax.set_xticks(range(len(matriz.columns)))
    ax.set_xticklabels(
        matriz.columns,
        rotation=45,
        ha="right"
    )

    ax.set_yticks(range(len(matriz.index)))
    ax.set_yticklabels(matriz.index)

    for i in range(len(matriz.index)):
        for j in range(len(matriz.columns)):
            valor = matriz.iloc[i, j]

            if pd.notna(valor):
                ax.text(
                    j,
                    i,
                    f"{valor:.2f}",
                    ha="center",
                    va="center",
                    fontsize=9
                )

    ax.set_title(titulo, fontsize=13, fontweight="bold")
    fig.colorbar(imagen, ax=ax, label="r de Pearson")

    plt.tight_layout()
    plt.savefig(
        GRAFICOS / nombre_archivo,
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()


def guardar_dispersion(datos, x, y, titulo, nombre_archivo):
    """Gráfico de dispersión con recta de regresión."""
    pares = datos[[x, y]].dropna()

    if len(pares) < 3:
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(
        pares[x],
        pares[y],
        alpha=0.65
    )

    coef = np.polyfit(
        pares[x],
        pares[y],
        1
    )

    linea_x = np.linspace(
        pares[x].min(),
        pares[x].max(),
        100
    )

    linea_y = coef[0] * linea_x + coef[1]

    ax.plot(
        linea_x,
        linea_y,
        linewidth=2
    )

    r, p, n = correlacion_pearson(
        pares[x],
        pares[y]
    )

    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(
        f"{titulo}\nr = {r:.3f} | p = {p:.4f} | n = {n}",
        fontsize=12,
        fontweight="bold"
    )

    ax.grid(
        True,
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        GRAFICOS / nombre_archivo,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


def guardar_comparacion_r2(regresiones):
    """Compara R² de los modelos mediante gráfico de barras."""
    datos = regresiones[
        ["Grupo/Momento", "Variable dependiente", "R²"]
    ].copy()

    datos["Modelo"] = (
        datos["Grupo/Momento"]
        + " - "
        + datos["Variable dependiente"]
    )

    fig, ax = plt.subplots(figsize=(12, 7))

    ax.bar(
        datos["Modelo"],
        datos["R²"]
    )

    ax.set_ylabel("R²")
    ax.set_xlabel("Grupo y momento")
    ax.set_title(
        "Comparación del coeficiente de determinación de los modelos",
        fontsize=13,
        fontweight="bold"
    )

    ax.set_ylim(0, 1)

    ax.tick_params(
        axis="x",
        rotation=55
    )

    ax.grid(
        axis="y",
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        GRAFICOS / "comparacion_R2_modelos.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# 4. LECTURA DE LOS CUATRO GRUPOS
# ============================================================

print("=" * 75)
print("ANÁLISIS DE CORRELACIÓN DE PEARSON Y REGRESIÓN LINEAL")
print("=" * 75)

datos_originales = {}
datos_variables = {}

for nombre, ruta in ARCHIVOS.items():

    print(f"\nLeyendo: {ruta.name}")

    df = leer_archivo(ruta)

    variables = crear_variables(df)

    datos_originales[nombre] = df
    datos_variables[nombre] = variables

    print(f"  Filas: {len(df)}")
    print(f"  Variables calculadas: {len(variables.columns)}")


# ============================================================
# 5. CORRELACIONES DE PEARSON
# ============================================================

correlaciones = []

for grupo_momento, datos in datos_variables.items():

    for independiente in INDEPENDIENTES:

        for dependiente in DEPENDIENTES:

            r, p, n = correlacion_pearson(
                datos[independiente],
                datos[dependiente]
            )

            correlaciones.append({
                "Grupo/Momento": grupo_momento,
                "Variable independiente": independiente,
                "Variable dependiente": dependiente,
                "r de Pearson": r,
                "p-valor": p,
                "n": n,
                "Interpretación": interpretar_r(r),
                "Significancia": significancia(p)
            })

tabla_correlaciones = pd.DataFrame(correlaciones)


# ============================================================
# 6. MATRICES DE CORRELACIÓN COMPLETAS
# ============================================================

matrices_correlacion = {}

for grupo_momento, datos in datos_variables.items():

    matriz = datos.corr(method="pearson")

    matrices_correlacion[grupo_momento] = matriz


# ============================================================
# 7. REGRESIÓN LINEAL MÚLTIPLE
# ============================================================

resultados_regresion = []
resultados_coeficientes = []

for grupo_momento, datos in datos_variables.items():

    for dependiente in DEPENDIENTES:

        modelo, datos_modelo, betas, vifs = regresion_multiple(
            datos,
            dependiente
        )

        resultados_regresion.append({
            "Grupo/Momento": grupo_momento,
            "Variable dependiente": dependiente,
            "N": int(modelo.nobs),
            "R": np.sqrt(max(modelo.rsquared, 0)),
            "R²": modelo.rsquared,
            "R² ajustado": modelo.rsquared_adj,
            "F": modelo.fvalue,
            "p-valor modelo": modelo.f_pvalue,
            "Significancia": significancia(modelo.f_pvalue),
        })

        for predictor in INDEPENDIENTES:

            resultados_coeficientes.append({
                "Grupo/Momento": grupo_momento,
                "Variable dependiente": dependiente,
                "Predictor": predictor,
                "B no estandarizado": modelo.params[predictor],
                "Beta estandarizado": betas[predictor],
                "Error estándar": modelo.bse[predictor],
                "t": modelo.tvalues[predictor],
                "p-valor": modelo.pvalues[predictor],
                "VIF": vifs[predictor],
                "Significancia": significancia(modelo.pvalues[predictor]),
            })


tabla_regresiones = pd.DataFrame(resultados_regresion)
tabla_coeficientes = pd.DataFrame(resultados_coeficientes)


# ============================================================
# 8. REGRESIÓN SOBRE CAMBIOS: GRUPO EXPERIMENTAL
# ============================================================
#
# Esta parte es especialmente útil para complementar el análisis
# comparativo del pre-test/post-test.
#
# Se calcula:
# Δ variable = Post-test - Pre-test
#
# Luego:
# Δ Y = b0 + b1 Δ Sistema + b2 Δ Chatbot
#
# No se interpreta como prueba causal automática, sino como
# análisis de asociación entre las mejoras observadas.
# ============================================================

exp_pre = datos_variables["Experimental Pre-test"]
exp_post = datos_variables["Experimental Post-test"]

cambios = pd.DataFrame(index=exp_pre.index)

for variable in VARIABLES:
    cambios[f"Δ {variable}"] = (
        exp_post[variable] - exp_pre[variable]
    )

regresiones_cambio = []
coeficientes_cambio = []

for dependiente in DEPENDIENTES:

    y_col = f"Δ {dependiente}"

    x1 = f"Δ {INDEPENDIENTES[0]}"
    x2 = f"Δ {INDEPENDIENTES[1]}"

    datos = cambios[[x1, x2, y_col]].dropna()

    X = sm.add_constant(
        datos[[x1, x2]]
    )

    y = datos[y_col]

    modelo = sm.OLS(y, X).fit()

    sd_y = y.std(ddof=1)

    betas = {}

    for predictor in [x1, x2]:

        sd_x = datos[predictor].std(ddof=1)

        if sd_x == 0 or sd_y == 0:
            betas[predictor] = np.nan
        else:
            betas[predictor] = (
                modelo.params[predictor]
                * sd_x
                / sd_y
            )

    regresiones_cambio.append({
        "Modelo": "Cambio Experimental",
        "Variable dependiente": dependiente,
        "N": int(modelo.nobs),
        "R": np.sqrt(max(modelo.rsquared, 0)),
        "R²": modelo.rsquared,
        "R² ajustado": modelo.rsquared_adj,
        "F": modelo.fvalue,
        "p-valor modelo": modelo.f_pvalue,
        "Significancia": significancia(modelo.f_pvalue),
    })

    for predictor in [x1, x2]:

        coeficientes_cambio.append({
            "Modelo": "Cambio Experimental",
            "Variable dependiente": dependiente,
            "Predictor": predictor.replace("Δ ", ""),
            "B no estandarizado": modelo.params[predictor],
            "Beta estandarizado": betas[predictor],
            "Error estándar": modelo.bse[predictor],
            "t": modelo.tvalues[predictor],
            "p-valor": modelo.pvalues[predictor],
            "Significancia": significancia(modelo.pvalues[predictor]),
        })


tabla_regresiones_cambio = pd.DataFrame(regresiones_cambio)
tabla_coeficientes_cambio = pd.DataFrame(coeficientes_cambio)


# ============================================================
# 9. TABLA COMPARATIVA RESUMIDA
# ============================================================

tabla_comparativa = tabla_correlaciones.copy()

tabla_comparativa["r²"] = (
    tabla_comparativa["r de Pearson"] ** 2
)

tabla_comparativa["Porcentaje de varianza compartida (%)"] = (
    tabla_comparativa["r²"] * 100
)

tabla_comparativa = tabla_comparativa[
    [
        "Grupo/Momento",
        "Variable independiente",
        "Variable dependiente",
        "r de Pearson",
        "r²",
        "Porcentaje de varianza compartida (%)",
        "p-valor",
        "n",
        "Interpretación",
        "Significancia",
    ]
]


# ============================================================
# 10. GRÁFICOS DE CORRELACIÓN
# ============================================================

for grupo_momento, matriz in matrices_correlacion.items():

    nombre = (
        grupo_momento
        .lower()
        .replace(" ", "_")
        .replace("-", "")
    )

    guardar_heatmap(
        matriz,
        f"Matriz de correlaciones de Pearson\n{grupo_momento}",
        f"heatmap_{nombre}.png"
    )


# ============================================================
# 11. DISPERSIÓN + REGRESIÓN
# ============================================================
#
# Se generan los cuatro cruces:
#
# Sistema web -> Gestión académica
# Sistema web -> Atención al cliente
# Chatbot -> Gestión académica
# Chatbot -> Atención al cliente
#
# Para los cuatro momentos/grupos.
# ============================================================

for grupo_momento, datos in datos_variables.items():

    prefijo = (
        grupo_momento
        .lower()
        .replace(" ", "_")
        .replace("-", "")
    )

    for independiente in INDEPENDIENTES:

        for dependiente in DEPENDIENTES:

            x_nombre = independiente[:18]
            y_nombre = dependiente[:18]

            nombre_archivo = (
                f"dispersion_{prefijo}_"
                f"{x_nombre}_{y_nombre}.png"
            )

            nombre_archivo = (
                nombre_archivo
                .replace(" ", "_")
                .replace("/", "_")
            )

            guardar_dispersion(
                datos,
                independiente,
                dependiente,
                f"{grupo_momento}: {independiente} vs {dependiente}",
                nombre_archivo
            )


# ============================================================
# 12. COMPARACIÓN DE R²
# ============================================================

guardar_comparacion_r2(tabla_regresiones)


# ============================================================
# 13. EXPORTACIÓN A EXCEL
# ============================================================

archivo_excel = SALIDA / "analisis_correlacion_regresion.xlsx"

with pd.ExcelWriter(
    archivo_excel,
    engine="openpyxl"
) as writer:

    tabla_comparativa.to_excel(
        writer,
        sheet_name="Comparacion Pearson",
        index=False
    )

    tabla_correlaciones.to_excel(
        writer,
        sheet_name="Pearson Detallado",
        index=False
    )

    tabla_regresiones.to_excel(
        writer,
        sheet_name="Regresiones",
        index=False
    )

    tabla_coeficientes.to_excel(
        writer,
        sheet_name="Coeficientes",
        index=False
    )

    tabla_regresiones_cambio.to_excel(
        writer,
        sheet_name="Regresion Cambio",
        index=False
    )

    tabla_coeficientes_cambio.to_excel(
        writer,
        sheet_name="Coef Cambio",
        index=False
    )

    for grupo_momento, matriz in matrices_correlacion.items():

        nombre = (
            "Matriz "
            + grupo_momento[:24]
        )

        matriz.to_excel(
            writer,
            sheet_name=nombre[:31]
        )


# ============================================================
# 14. RESUMEN TXT
# ============================================================

archivo_txt = SALIDA / "resumen_analisis.txt"

with open(
    archivo_txt,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "ANÁLISIS DE CORRELACIÓN DE PEARSON Y REGRESIÓN LINEAL\n"
    )
    f.write("=" * 75 + "\n\n")

    f.write(
        "VARIABLES INDEPENDIENTES:\n"
        "- Sistema de gestión académica web\n"
        "- Chatbot\n\n"
    )

    f.write(
        "VARIABLES DEPENDIENTES:\n"
        "- Gestión académica\n"
        "- Atención al cliente\n\n"
    )

    f.write(
        "INTERPRETACIÓN DE PEARSON:\n"
        "El coeficiente r indica la dirección y fuerza de la relación "
        "lineal entre dos variables.\n"
        "El p-valor permite evaluar su significancia estadística.\n\n"
    )

    f.write(
        "INTERPRETACIÓN DE REGRESIÓN:\n"
        "R² indica la proporción de variabilidad de la variable "
        "dependiente explicada conjuntamente por los dos predictores.\n"
        "Los coeficientes Beta permiten comparar la contribución relativa "
        "de los predictores dentro de cada modelo.\n\n"
    )

    f.write("=" * 75 + "\n")
    f.write("CORRELACIONES PRINCIPALES\n")
    f.write("=" * 75 + "\n\n")

    for _, fila in tabla_correlaciones.iterrows():

        f.write(
            f"{fila['Grupo/Momento']} | "
            f"{fila['Variable independiente']} -> "
            f"{fila['Variable dependiente']} | "
            f"r={fila['r de Pearson']:.4f} | "
            f"p={fila['p-valor']:.4f} | "
            f"{fila['Interpretación']}\n"
        )

    f.write("\n")
    f.write("=" * 75 + "\n")
    f.write("REGRESIONES\n")
    f.write("=" * 75 + "\n\n")

    for _, fila in tabla_regresiones.iterrows():

        f.write(
            f"{fila['Grupo/Momento']} | "
            f"{fila['Variable dependiente']} | "
            f"R={fila['R']:.4f} | "
            f"R²={fila['R²']:.4f} | "
            f"R² ajustado={fila['R² ajustado']:.4f} | "
            f"F={fila['F']:.4f} | "
            f"p={fila['p-valor modelo']:.4f}\n"
        )

    f.write("\n")
    f.write("=" * 75 + "\n")
    f.write("REGRESIÓN SOBRE CAMBIOS - GRUPO EXPERIMENTAL\n")
    f.write("=" * 75 + "\n\n")

    for _, fila in tabla_regresiones_cambio.iterrows():

        f.write(
            f"{fila['Variable dependiente']} | "
            f"R={fila['R']:.4f} | "
            f"R²={fila['R²']:.4f} | "
            f"R² ajustado={fila['R² ajustado']:.4f} | "
            f"F={fila['F']:.4f} | "
            f"p={fila['p-valor modelo']:.4f}\n"
        )


# ============================================================
# 15. CONSOLA
# ============================================================

print("\n" + "=" * 75)
print("PROCESO TERMINADO")
print("=" * 75)

print(f"\nExcel generado:")
print(archivo_excel)

print("\nCarpeta de gráficos:")
print(GRAFICOS)

print("\nArchivos principales:")

print("1. Comparacion Pearson")
print("2. Pearson Detallado")
print("3. Regresiones")
print("4. Coeficientes")
print("5. Regresion Cambio")
print("6. Coef Cambio")

print("\n" + "=" * 75)
print("NOTA METODOLÓGICA")
print("=" * 75)

print(
    "\nLa correlación de Pearson analiza la relación lineal entre "
    "variables."
)

print(
    "\nLa regresión lineal múltiple estima la capacidad predictiva "
    "conjunta de Sistema web y Chatbot sobre Gestión académica y "
    "Atención al cliente."
)

print(
    "\nLa regresión sobre cambios utiliza Post-test - Pre-test "
    "del grupo experimental para complementar el análisis."
)

print(
    "\nLos resultados de correlación y regresión deben interpretarse "
    "como evidencia de asociación/predicción lineal y no como "
    "prueba causal por sí solos."
)

print("\nListo.")
