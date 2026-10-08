# ============================================================
# COMPROBACIÓN DE HIPÓTESIS - PRUEBA T DE STUDENT
# TESIS: SISTEMA DE GESTIÓN ACADÉMICA WEB INTEGRADO CON CHATBOT
# ============================================================
#
# Estructura REAL:
#
# 1. Sistema de gestión académica web -> indicadores 1-4
# 2. Chatbot                         -> indicadores 5-8
# 3. Gestión académica              -> indicadores 9-12
# 4. Atención al cliente            -> indicadores 13-16
#
# Este programa realiza cuatro análisis:
#
# 1) T de Student para muestras relacionadas:
#    Pre-test vs Post-test del grupo control.
#
# 2) T de Student para muestras relacionadas:
#    Pre-test vs Post-test del grupo experimental.
#
# 3) T de Student para muestras independientes:
#    Post-test Experimental vs Post-test Control.
#
# 4) T de Student para muestras independientes sobre el cambio:
#    (Experimental Post - Experimental Pre)
#    vs
#    (Control Post - Control Pre)
#
# El análisis 4 es especialmente útil para un diseño
# cuasi-experimental porque compara directamente la magnitud
# del cambio entre ambos grupos.
#
# Además calcula:
# - Media pre-test
# - Media post-test
# - Diferencia de medias
# - t
# - grados de libertad
# - p-valor
# - intervalo de confianza del 95 %
# - tamaño del efecto Cohen's d
# - decisión estadística con alfa = 0.05
#
# Genera:
# - Excel con tablas
# - Gráficos comparativos
# - Gráficos de diferencias
# - Resumen TXT
#
# Requisitos:
# pip install pandas==2.2.3 openpyxl==3.1.5 matplotlib==3.9.2 numpy==2.0.2 scipy==1.13.1
# ============================================================

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import ttest_rel, ttest_ind

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

SALIDA = BASE_DIR / "resultados_t_student"
GRAFICOS = SALIDA / "graficos"

SALIDA.mkdir(parents=True, exist_ok=True)
GRAFICOS.mkdir(parents=True, exist_ok=True)

ALPHA = 0.05
Z_95 = 1.96


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
# 3. FUNCIONES DE LECTURA Y VALIDACIÓN
# ============================================================

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
            f"\nEl archivo {ruta.name} no contiene:\n"
            + "\n".join(f"- {c}" for c in faltantes)
        )

    for columna in INDICADORES:
        df[columna] = pd.to_numeric(
            df[columna],
            errors="coerce"
        )

        valores = df[columna].dropna()

        invalidos = valores[~valores.isin([1, 2, 3, 4, 5])]

        if not invalidos.empty:
            raise ValueError(
                f"\nLa columna '{columna}' contiene valores "
                f"fuera de la escala 1-5: "
                f"{sorted(invalidos.unique().tolist())}"
            )

    return df


def crear_variables(df):
    """
    Calcula el puntaje de cada variable como la media de sus
    cuatro indicadores para cada participante.
    """
    resultado = pd.DataFrame(index=df.index)

    for variable, indicadores in VARIABLES.items():
        resultado[variable] = df[indicadores].mean(axis=1)

    return resultado


# ============================================================
# 4. FUNCIONES ESTADÍSTICAS
# ============================================================

def cohen_d_paired(pre, post):
    """
    Cohen's d para muestras relacionadas.
    Se calcula sobre las diferencias Post - Pre.
    """
    diferencias = post - pre
    sd = diferencias.std(ddof=1)

    if sd == 0 or pd.isna(sd):
        return np.nan

    return diferencias.mean() / sd


def cohen_d_independent(x, y):
    """
    Cohen's d para muestras independientes usando
    desviación estándar agrupada.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]

    if len(x) < 2 or len(y) < 2:
        return np.nan

    nx = len(x)
    ny = len(y)

    sx = np.std(x, ddof=1)
    sy = np.std(y, ddof=1)

    sp = np.sqrt(
        (
            (nx - 1) * sx ** 2
            + (ny - 1) * sy ** 2
        )
        / (nx + ny - 2)
    )

    if sp == 0:
        return np.nan

    return (np.mean(x) - np.mean(y)) / sp


def interpretar_d(d):
    if pd.isna(d):
        return "No calculable"

    valor = abs(d)

    if valor < 0.20:
        return "efecto trivial"
    elif valor < 0.50:
        return "efecto pequeño"
    elif valor < 0.80:
        return "efecto mediano"
    else:
        return "efecto grande"


def decision_p(p):
    if pd.isna(p):
        return "No calculable"

    if p < ALPHA:
        return "Rechazar H0"
    return "No rechazar H0"


def interpretar_significancia(p):
    if pd.isna(p):
        return "No calculable"

    if p < 0.001:
        return "p < 0.001"
    elif p < 0.01:
        return "p < 0.01"
    elif p < 0.05:
        return "p < 0.05"
    else:
        return "p ≥ 0.05"


def ic_diferencia_pareada(pre, post):
    """
    IC95% para la diferencia Post - Pre.
    """
    diferencias = (post - pre).dropna()

    n = len(diferencias)

    if n < 2:
        return np.nan, np.nan

    media = diferencias.mean()
    error = diferencias.std(ddof=1) / np.sqrt(n)

    # Para muestras pequeñas se aproxima mediante t crítico.
    # scipy se utiliza para obtener el valor crítico.
    from scipy.stats import t

    critico = t.ppf(0.975, df=n - 1)

    return (
        media - critico * error,
        media + critico * error
    )


def ic_diferencia_independiente(x, y):
    """
    IC95% para diferencia de medias:
    x - y
    """
    x = pd.Series(x).dropna().astype(float)
    y = pd.Series(y).dropna().astype(float)

    nx = len(x)
    ny = len(y)

    if nx < 2 or ny < 2:
        return np.nan, np.nan

    mx = x.mean()
    my = y.mean()

    vx = x.var(ddof=1)
    vy = y.var(ddof=1)

    diferencia = mx - my

    error = np.sqrt(
        vx / nx
        + vy / ny
    )

    # Welch-Satterthwaite
    numerador = (
        vx / nx
        + vy / ny
    ) ** 2

    denominador = (
        ((vx / nx) ** 2) / (nx - 1)
        + ((vy / ny) ** 2) / (ny - 1)
    )

    if denominador == 0:
        return np.nan, np.nan

    df = numerador / denominador

    from scipy.stats import t

    critico = t.ppf(0.975, df=df)

    return (
        diferencia - critico * error,
        diferencia + critico * error
    )


# ============================================================
# 5. CARGA DE LOS CUATRO ARCHIVOS
# ============================================================

print("=" * 80)
print("COMPROBACIÓN DE HIPÓTESIS - PRUEBA T DE STUDENT")
print("=" * 80)

datos_variables = {}

for nombre, ruta in ARCHIVOS.items():

    print(f"\nLeyendo: {ruta.name}")

    df = leer_archivo(ruta)

    variables = crear_variables(df)

    datos_variables[nombre] = variables

    print(f"  Participantes: {len(df)}")
    print("  Variables: 4")


# ============================================================
# 6. T DE STUDENT PAREADA
# ============================================================
#
# Control:
# Pre-test vs Post-test
#
# Experimental:
# Pre-test vs Post-test
# ============================================================

resultados_pareada = []

pares = [
    (
        "Control",
        datos_variables["Control Pre-test"],
        datos_variables["Control Post-test"]
    ),
    (
        "Experimental",
        datos_variables["Experimental Pre-test"],
        datos_variables["Experimental Post-test"]
    ),
]

for grupo, pre, post in pares:

    for variable in VARIABLES:

        x = pre[variable]
        y = post[variable]

        datos = pd.concat(
            [x, y],
            axis=1
        ).dropna()

        x = datos.iloc[:, 0]
        y = datos.iloc[:, 1]

        if len(x) >= 2:

            resultado = ttest_rel(
                x,
                y
            )

            t = resultado.statistic
            p = resultado.pvalue

            ic_inf, ic_sup = ic_diferencia_pareada(
                x,
                y
            )

            d = cohen_d_paired(
                x,
                y
            )

            resultados_pareada.append({
                "Grupo": grupo,
                "Variable": variable,
                "N": len(x),
                "Media Pre-test": x.mean(),
                "Media Post-test": y.mean(),
                "Diferencia Post - Pre": y.mean() - x.mean(),
                "t de Student": t,
                "gl": len(x) - 1,
                "p-valor": p,
                "IC95% inferior": ic_inf,
                "IC95% superior": ic_sup,
                "Cohen's d": d,
                "Magnitud del efecto": interpretar_d(d),
                "Significancia": interpretar_significancia(p),
                "Decisión": decision_p(p),
            })


tabla_pareada = pd.DataFrame(
    resultados_pareada
)


# ============================================================
# 7. T DE STUDENT INDEPENDIENTE - POST-TEST
# ============================================================
#
# Experimental Post-test vs Control Post-test
#
# Se utiliza Welch's t-test porque no exige asumir igualdad
# de varianzas.
# ============================================================

resultados_post = []

control_post = datos_variables["Control Post-test"]
experimental_post = datos_variables["Experimental Post-test"]

for variable in VARIABLES:

    x = experimental_post[variable].dropna()
    y = control_post[variable].dropna()

    resultado = ttest_ind(
        x,
        y,
        equal_var=False
    )

    t = resultado.statistic
    p = resultado.pvalue

    nx = len(x)
    ny = len(y)

    vx = x.var(ddof=1)
    vy = y.var(ddof=1)

    numerador = (
        vx / nx
        + vy / ny
    ) ** 2

    denominador = (
        ((vx / nx) ** 2) / (nx - 1)
        + ((vy / ny) ** 2) / (ny - 1)
    )

    gl = numerador / denominador

    ic_inf, ic_sup = ic_diferencia_independiente(
        x,
        y
    )

    d = cohen_d_independent(
        x,
        y
    )

    resultados_post.append({
        "Variable": variable,
        "N Experimental": len(x),
        "N Control": len(y),
        "Media Experimental Post-test": x.mean(),
        "Media Control Post-test": y.mean(),
        "Diferencia Experimental - Control": x.mean() - y.mean(),
        "t de Student": t,
        "gl": gl,
        "p-valor": p,
        "IC95% inferior": ic_inf,
        "IC95% superior": ic_sup,
        "Cohen's d": d,
        "Magnitud del efecto": interpretar_d(d),
        "Significancia": interpretar_significancia(p),
        "Decisión": decision_p(p),
    })


tabla_post = pd.DataFrame(
    resultados_post
)


# ============================================================
# 8. T DE STUDENT INDEPENDIENTE SOBRE EL CAMBIO
# ============================================================
#
# Cambio = Post-test - Pre-test
#
# Experimental vs Control
#
# Esta comparación es la más importante como complemento
# del análisis pre-test/post-test porque determina si la
# magnitud de la mejora difiere entre ambos grupos.
# ============================================================

control_cambio = (
    datos_variables["Control Post-test"]
    - datos_variables["Control Pre-test"]
)

experimental_cambio = (
    datos_variables["Experimental Post-test"]
    - datos_variables["Experimental Pre-test"]
)

resultados_cambio = []

for variable in VARIABLES:

    x = experimental_cambio[variable].dropna()
    y = control_cambio[variable].dropna()

    resultado = ttest_ind(
        x,
        y,
        equal_var=False
    )

    t = resultado.statistic
    p = resultado.pvalue

    nx = len(x)
    ny = len(y)

    vx = x.var(ddof=1)
    vy = y.var(ddof=1)

    numerador = (
        vx / nx
        + vy / ny
    ) ** 2

    denominador = (
        ((vx / nx) ** 2) / (nx - 1)
        + ((vy / ny) ** 2) / (ny - 1)
    )

    gl = numerador / denominador

    ic_inf, ic_sup = ic_diferencia_independiente(
        x,
        y
    )

    d = cohen_d_independent(
        x,
        y
    )

    resultados_cambio.append({
        "Variable": variable,
        "N Experimental": len(x),
        "N Control": len(y),
        "Media Cambio Experimental": x.mean(),
        "Media Cambio Control": y.mean(),
        "Diferencia de cambios": x.mean() - y.mean(),
        "t de Student": t,
        "gl": gl,
        "p-valor": p,
        "IC95% inferior": ic_inf,
        "IC95% superior": ic_sup,
        "Cohen's d": d,
        "Magnitud del efecto": interpretar_d(d),
        "Significancia": interpretar_significancia(p),
        "Decisión": decision_p(p),
    })


tabla_cambio = pd.DataFrame(
    resultados_cambio
)


# ============================================================
# 9. TABLA RESUMEN PARA LA COMPROBACIÓN DE HIPÓTESIS
# ============================================================

resumen = tabla_cambio[
    [
        "Variable",
        "Media Cambio Experimental",
        "Media Cambio Control",
        "Diferencia de cambios",
        "t de Student",
        "gl",
        "p-valor",
        "Cohen's d",
        "Magnitud del efecto",
        "Decisión",
    ]
].copy()

resumen.insert(
    0,
    "Prueba",
    "t independiente sobre cambio"
)


# ============================================================
# 10. GRÁFICO PRE-TEST VS POST-TEST
# ============================================================

for grupo in ["Control", "Experimental"]:

    if grupo == "Control":
        pre = datos_variables["Control Pre-test"]
        post = datos_variables["Control Post-test"]
    else:
        pre = datos_variables["Experimental Pre-test"]
        post = datos_variables["Experimental Post-test"]

    medias_pre = [
        pre[v].mean()
        for v in VARIABLES
    ]

    medias_post = [
        post[v].mean()
        for v in VARIABLES
    ]

    nombres = list(VARIABLES.keys())

    x_pos = np.arange(len(nombres))
    ancho = 0.36

    fig, ax = plt.subplots(figsize=(12, 7))

    ax.bar(
        x_pos - ancho / 2,
        medias_pre,
        ancho,
        label="Pre-test"
    )

    ax.bar(
        x_pos + ancho / 2,
        medias_post,
        ancho,
        label="Post-test"
    )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(
        nombres,
        rotation=25,
        ha="right"
    )

    ax.set_ylabel("Media")
    ax.set_title(
        f"Comparación de medias Pre-test y Post-test - {grupo}",
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
        / f"medias_pre_post_{grupo.lower()}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# 11. GRÁFICO DE DIFERENCIA DE CAMBIOS
# ============================================================

nombres = resumen["Variable"].tolist()
valores = resumen["Diferencia de cambios"].tolist()

fig, ax = plt.subplots(figsize=(12, 7))

barras = ax.bar(
    nombres,
    valores
)

ax.axhline(
    0,
    linewidth=1
)

ax.set_ylabel(
    "Diferencia de cambios (Experimental - Control)"
)

ax.set_title(
    "Diferencia de cambios entre grupos",
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

for barra, valor in zip(barras, valores):

    ax.text(
        barra.get_x()
        + barra.get_width() / 2,
        valor,
        f"{valor:.2f}",
        ha="center",
        va="bottom" if valor >= 0 else "top"
    )

plt.tight_layout()

plt.savefig(
    GRAFICOS
    / "diferencia_de_cambios.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 12. GRÁFICO DE P-VALORES
# ============================================================

variables = tabla_cambio["Variable"].tolist()
p_valores = tabla_cambio["p-valor"].tolist()

fig, ax = plt.subplots(figsize=(12, 7))

ax.bar(
    variables,
    p_valores
)

ax.axhline(
    ALPHA,
    linestyle="--",
    linewidth=2,
    label="α = 0.05"
)

ax.set_ylabel("p-valor")
ax.set_title(
    "Significancia estadística de la diferencia de cambios",
    fontsize=13,
    fontweight="bold"
)

ax.set_yscale("log")

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
    / "p_valores_diferencia_cambios.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 13. EXPORTAR A EXCEL
# ============================================================

archivo_excel = (
    SALIDA
    / "tablas_t_student.xlsx"
)

with pd.ExcelWriter(
    archivo_excel,
    engine="openpyxl"
) as writer:

    tabla_pareada.to_excel(
        writer,
        sheet_name="T pareada",
        index=False
    )

    tabla_post.to_excel(
        writer,
        sheet_name="T independiente Post",
        index=False
    )

    tabla_cambio.to_excel(
        writer,
        sheet_name="T independiente Cambio",
        index=False
    )

    resumen.to_excel(
        writer,
        sheet_name="Resumen Hipotesis",
        index=False
    )


# ============================================================
# 14. RESUMEN TXT
# ============================================================

archivo_txt = (
    SALIDA
    / "resumen_t_student.txt"
)

with open(
    archivo_txt,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "COMPROBACIÓN DE HIPÓTESIS - PRUEBA T DE STUDENT\n"
    )
    f.write("=" * 80 + "\n\n")

    f.write(
        "Nivel de significancia: α = 0.05\n\n"
    )

    f.write(
        "PRUEBA PRINCIPAL: T DE STUDENT INDEPENDIENTE SOBRE "
        "EL CAMBIO PRE-TEST/POST-TEST\n\n"
    )

    f.write(
        "Cambio = Post-test - Pre-test\n"
        "Comparación = Cambio Experimental - Cambio Control\n\n"
    )

    for _, fila in tabla_cambio.iterrows():

        cohens_d = fila["Cohen's d"]

        f.write(
            f"Variable: {fila['Variable']}\n"
            f"  Media cambio experimental: "
            f"{fila['Media Cambio Experimental']:.4f}\n"
            f"  Media cambio control: "
            f"{fila['Media Cambio Control']:.4f}\n"
            f"  Diferencia de cambios: "
            f"{fila['Diferencia de cambios']:.4f}\n"
            f"  t = {fila['t de Student']:.4f}\n"
            f"  gl = {fila['gl']:.2f}\n"
            f"  p = {fila['p-valor']:.6f}\n"
            f"  Cohen's d = {cohens_d:.4f}\\n"
            f"  Efecto: {fila['Magnitud del efecto']}\n"
            f"  Decisión: {fila['Decisión']}\n\n"
        )


# ============================================================
# 15. INFORMACIÓN FINAL
# ============================================================

print("\n" + "=" * 80)
print("PROCESO TERMINADO")
print("=" * 80)

print("\nExcel:")
print(archivo_excel)

print("\nGráficos:")
print(GRAFICOS)

print("\nPruebas generadas:")
print("1. T pareada Control")
print("2. T pareada Experimental")
print("3. T independiente Post-test")
print("4. T independiente sobre Cambio (principal)")

print("\n" + "=" * 80)
print("CRITERIO DE DECISIÓN")
print("=" * 80)

print(
    "\nSi p < 0.05: se rechaza H0."
    "\nSi p >= 0.05: no se rechaza H0."
)

print(
    "\nLa prueba principal para comparar la mejora entre grupos "
    "es la T independiente aplicada a los cambios "
    "(Post-test - Pre-test)."
)

print("\nListo.")
