# ============================================================
# COMPROBACIÓN DE HIPÓTESIS - ANOVA FACTORIAL 2x2 POR INDICADOR
# TESIS: SISTEMA DE GESTIÓN ACADÉMICA WEB INTEGRADO CON CHATBOT
# ============================================================
#
# Diseño factorial:
#   Factor 1: Grupo (Control / Experimental)
#   Factor 2: Momento (Pre-test / Post-test)
#
# IMPORTANTE:
#   El análisis se realiza POR CADA UNO DE LOS 16 INDICADORES.
#   Las cuatro variables/dimensiones se conservan únicamente
#   como clasificación temática de los indicadores.
#
# El efecto de mayor interés es Grupo × Momento, porque permite
# determinar si el cambio entre Pre-test y Post-test difiere
# entre el grupo experimental y el grupo control.
#
# NOTA METODOLÓGICA:
#   Este ANOVA factorial se utiliza como análisis complementario.
#   Como Pre-test y Post-test corresponden a las mismas personas,
#   la estructura es longitudinal. Por ello, las pruebas t y el
#   ANCOVA por indicador pueden mantenerse como análisis principales
#   según el diseño de la tesis.
# ============================================================

from pathlib import Path
import warnings
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

SALIDA = BASE_DIR / "resultados_anova_por_indicador"
GRAFICOS = SALIDA / "graficos"
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

MAPA_DIMENSION = {
    indicador: dimension
    for dimension, indicadores in DIMENSIONES.items()
    for indicador in indicadores
}

# ============================================================
# 2. FUNCIONES
# ============================================================

def nombre_archivo(texto):
    texto = str(texto).lower()
    texto = re.sub(r"[^a-z0-9áéíóúñü]+", "_", texto)
    return texto.strip("_")


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
            f"\nEl archivo {ruta.name} no contiene estos indicadores:\n"
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
        return "Muy pequeño"
    if eta < 0.06:
        return "Pequeño"
    if eta < 0.14:
        return "Mediano"
    return "Grande"


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


def crear_dataframe_longitudinal(datos):
    filas = []

    configuracion = [
        ("Control Pre-test", "Control", "Pre-test"),
        ("Control Post-test", "Control", "Post-test"),
        ("Experimental Pre-test", "Experimental", "Pre-test"),
        ("Experimental Post-test", "Experimental", "Post-test"),
    ]

    for clave, grupo, momento in configuracion:
        df = datos[clave].copy()
        df["ID"] = np.arange(1, len(df) + 1)
        df["Grupo"] = grupo
        df["Momento"] = momento
        filas.append(df)

    return pd.concat(filas, ignore_index=True)


# ============================================================
# 3. CARGA DE DATOS
# ============================================================

print("=" * 90)
print("COMPROBACIÓN DE HIPÓTESIS - ANOVA FACTORIAL 2x2 POR INDICADOR")
print("=" * 90)

datos = {}
for nombre, ruta in ARCHIVOS.items():
    print(f"\nLeyendo: {ruta.name}")
    datos[nombre] = leer_archivo(ruta)
    print(f"  Participantes: {len(datos[nombre])}")

longitudinal = crear_dataframe_longitudinal(datos)

# ============================================================
# 4. ANOVA 2x2 POR INDICADOR
# ============================================================

resultados = []
tablas_completas = []
medias = []

for indicador in INDICADORES:
    df = longitudinal[["ID", "Grupo", "Momento", indicador]].dropna().copy()
    df = df.rename(columns={indicador: "Y"})

    modelo = ols("Y ~ C(Grupo) * C(Momento)", data=df).fit()
    tabla = anova_lm(modelo, typ=2)

    ss_error = tabla.loc["Residual", "sum_sq"]

    for efecto, nombre_efecto in [
        ("C(Grupo)", "Efecto Grupo"),
        ("C(Momento)", "Efecto Momento"),
        ("C(Grupo):C(Momento)", "Interacción Grupo × Momento"),
    ]:
        ss = tabla.loc[efecto, "sum_sq"]
        p = tabla.loc[efecto, "PR(>F)"]
        eta = ss / (ss + ss_error)

        resultados.append({
            "Dimensión": MAPA_DIMENSION[indicador],
            "Indicador": indicador,
            "Efecto": nombre_efecto,
            "Suma de cuadrados": ss,
            "gl": int(tabla.loc[efecto, "df"]),
            "F": tabla.loc[efecto, "F"],
            "p-valor": p,
            "Significancia": significancia(p),
            "Eta² parcial": eta,
            "Magnitud Eta² parcial": interpretar_eta(eta),
            "Decisión": decision(p),
            "N": len(df),
        })

    completa = tabla.reset_index().rename(columns={
        "index": "Efecto",
        "sum_sq": "Suma de cuadrados",
        "df": "gl",
        "PR(>F)": "p-valor",
    })
    completa.insert(0, "Indicador", indicador)
    completa.insert(0, "Dimensión", MAPA_DIMENSION[indicador])
    tablas_completas.append(completa)

    # Estadísticos descriptivos por grupo y momento
    desc = (
        df.groupby(["Grupo", "Momento"])["Y"]
        .agg(Media="mean", DE="std", N="count")
        .reset_index()
    )
    desc.insert(0, "Indicador", indicador)
    desc.insert(0, "Dimensión", MAPA_DIMENSION[indicador])
    medias.append(desc)

tabla_resultados = pd.DataFrame(resultados)
tabla_anova_completa = pd.concat(tablas_completas, ignore_index=True)
tabla_medias = pd.concat(medias, ignore_index=True)

# ============================================================
# 5. TABLA PRINCIPAL: INTERACCIÓN POR INDICADOR
# ============================================================

interaccion = tabla_resultados[
    tabla_resultados["Efecto"] == "Interacción Grupo × Momento"
].copy()

interaccion = interaccion[
    [
        "Dimensión", "Indicador", "N", "F", "gl", "p-valor",
        "Significancia", "Eta² parcial", "Magnitud Eta² parcial",
        "Decisión"
    ]
].sort_values(["Dimensión", "Indicador"])

# ============================================================
# 6. RESUMEN DE INTERACCIÓN POR DIMENSIÓN
# ============================================================

resumen_dimension = (
    interaccion.groupby("Dimensión")
    .agg(
        Indicadores=("Indicador", "count"),
        Indicadores_significativos=("p-valor", lambda x: int((x < ALPHA).sum())),
        F_promedio=("F", "mean"),
        Eta2_parcial_promedio=("Eta² parcial", "mean"),
    )
    .reset_index()
)

# ============================================================
# 7. GRÁFICOS DE INTERACCIÓN POR INDICADOR
# ============================================================

for indicador in INDICADORES:
    resumen = (
        longitudinal.groupby(["Grupo", "Momento"])[indicador]
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    for grupo in ["Control", "Experimental"]:
        datos_grupo = (
            resumen[resumen["Grupo"] == grupo]
            .set_index("Momento")
            .reindex(["Pre-test", "Post-test"])
        )

        ax.plot(
            ["Pre-test", "Post-test"],
            datos_grupo[indicador].values,
            marker="o",
            linewidth=2,
            label=grupo,
        )

    fila = interaccion[interaccion["Indicador"] == indicador].iloc[0]
    p = fila["p-valor"]

    ax.set_ylim(1, 5)
    ax.set_ylabel("Media")
    ax.set_xlabel("Momento")
    ax.set_title(
        f"Interacción Grupo × Momento\n{indicador}",
        fontsize=12,
        fontweight="bold",
    )
    ax.text(
        0.5, 0.02,
        f"F = {fila['F']:.3f} | p = {p:.4g} | η²p = {fila['Eta² parcial']:.3f}",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=9,
    )
    ax.legend()
    ax.grid(True, alpha=0.25)
    plt.tight_layout()

    plt.savefig(
        GRAFICOS / f"interaccion_{nombre_archivo(indicador)}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

# ============================================================
# 8. GRÁFICOS RESUMEN
# ============================================================

for efecto in ["Efecto Grupo", "Efecto Momento", "Interacción Grupo × Momento"]:
    datos_efecto = tabla_resultados[tabla_resultados["Efecto"] == efecto]

    fig, ax = plt.subplots(figsize=(13, 6.5))
    ax.bar(datos_efecto["Indicador"], datos_efecto["F"])
    ax.set_ylabel("Estadístico F")
    ax.set_title(f"Estadístico F por indicador - {efecto}",
                 fontsize=13, fontweight="bold")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()

    plt.savefig(
        GRAFICOS / f"F_{nombre_archivo(efecto)}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

fig, ax = plt.subplots(figsize=(13, 6.5))
ax.bar(interaccion["Indicador"], interaccion["p-valor"])
ax.axhline(ALPHA, linestyle="--", linewidth=2, label="α = 0.05")
ax.set_ylabel("p-valor")
ax.set_title("Significancia de la interacción Grupo × Momento por indicador",
             fontsize=13, fontweight="bold")
ax.tick_params(axis="x", rotation=45)
ax.legend()
ax.grid(axis="y", alpha=0.25)
plt.tight_layout()
plt.savefig(GRAFICOS / "p_valores_interaccion_por_indicador.png",
            dpi=300, bbox_inches="tight")
plt.close()

fig, ax = plt.subplots(figsize=(13, 6.5))
ax.bar(interaccion["Indicador"], interaccion["Eta² parcial"])
ax.set_ylabel("Eta² parcial")
ax.set_title("Tamaño del efecto de la interacción Grupo × Momento por indicador",
             fontsize=13, fontweight="bold")
ax.tick_params(axis="x", rotation=45)
ax.grid(axis="y", alpha=0.25)
plt.tight_layout()
plt.savefig(GRAFICOS / "eta_cuadrado_parcial_por_indicador.png",
            dpi=300, bbox_inches="tight")
plt.close()

# ============================================================
# 9. TABLAS DE COMPARACIÓN PARA LA TESIS
# ============================================================
# Estas tablas permiten distinguir:
# A) Situación inicial: Control Pre, Control Post y Experimental Pre.
# B) Comparación completa: se incorpora Experimental Post.
#
# IMPORTANTE: no se fuerza ningún resultado de significancia.
# Los p-valores se calculan con los datos reales.
# La comparación completa se interpreta mediante el ANOVA factorial
# y, especialmente, la interacción Grupo × Momento.
# ============================================================

from scipy.stats import ttest_rel, ttest_ind

comparacion_inicial = []
comparacion_completa = []

for indicador in INDICADORES:
    c_pre = datos["Control Pre-test"][indicador].dropna().to_numpy(dtype=float)
    c_post = datos["Control Post-test"][indicador].dropna().to_numpy(dtype=float)
    e_pre = datos["Experimental Pre-test"][indicador].dropna().to_numpy(dtype=float)
    e_post = datos["Experimental Post-test"][indicador].dropna().to_numpy(dtype=float)

    # Comparación pre/post dentro del Control: prueba t pareada.
    n_pareado = min(len(c_pre), len(c_post))
    t_control, p_control = (np.nan, np.nan)
    if n_pareado >= 2:
        t_control, p_control = ttest_rel(c_pre[:n_pareado], c_post[:n_pareado])

    # Comparación inicial entre Control Pre y Experimental Pre: t independiente.
    t_inicial, p_inicial = ttest_ind(e_pre, c_pre, equal_var=False, nan_policy="omit")

    # ANOVA factorial completo: extraemos directamente la interacción.
    fila_inter = interaccion[interaccion["Indicador"] == indicador].iloc[0]

    comparacion_inicial.append({
        "Dimensión": MAPA_DIMENSION[indicador],
        "Indicador": indicador,
        "Control Pre - Media": np.mean(c_pre),
        "Control Post - Media": np.mean(c_post),
        "Experimental Pre - Media": np.mean(e_pre),
        "Control Pre vs Control Post - p": p_control,
        "Control Pre vs Experimental Pre - p": p_inicial,
        "Resultado Control Pre/Post": (
            "Significativo" if p_control < ALPHA else "No significativo"
        ) if not pd.isna(p_control) else "No calculable",
        "Resultado equivalencia inicial": (
            "Significativo" if p_inicial < ALPHA else "No significativo"
        ) if not pd.isna(p_inicial) else "No calculable",
    })

    comparacion_completa.append({
        "Dimensión": MAPA_DIMENSION[indicador],
        "Indicador": indicador,
        "Control Pre - Media": np.mean(c_pre),
        "Control Post - Media": np.mean(c_post),
        "Experimental Pre - Media": np.mean(e_pre),
        "Experimental Post - Media": np.mean(e_post),
        "Cambio Control": np.mean(c_post) - np.mean(c_pre),
        "Cambio Experimental": np.mean(e_post) - np.mean(e_pre),
        "Diferencia de cambios (Exp - Control)":
            (np.mean(e_post) - np.mean(e_pre)) -
            (np.mean(c_post) - np.mean(c_pre)),
        "F Interacción": fila_inter["F"],
        "p Interacción": fila_inter["p-valor"],
        "Eta² parcial": fila_inter["Eta² parcial"],
        "Resultado": (
            "Significativo" if fila_inter["p-valor"] < ALPHA
            else "No significativo"
        ),
        "Decisión": fila_inter["Decisión"],
    })

tabla_comparacion_inicial = pd.DataFrame(comparacion_inicial)
tabla_comparacion_completa = pd.DataFrame(comparacion_completa)

# Tabla compacta para insertar directamente en la tesis.
tabla_tesis_comparativa = tabla_comparacion_completa[[
    "Dimensión", "Indicador",
    "Control Pre - Media", "Control Post - Media",
    "Experimental Pre - Media", "Experimental Post - Media",
    "Cambio Control", "Cambio Experimental",
    "Diferencia de cambios (Exp - Control)",
    "F Interacción", "p Interacción", "Eta² parcial", "Resultado"
]].copy()

# ============================================================
# 10. EXPORTACIÓN A EXCEL
# ============================================================

archivo_excel = SALIDA / "tablas_anova_por_indicador.xlsx"

with pd.ExcelWriter(archivo_excel, engine="openpyxl") as writer:
    interaccion.to_excel(writer, sheet_name="Interacción por indicador", index=False)
    tabla_resultados.to_excel(writer, sheet_name="Todos los efectos", index=False)
    tabla_anova_completa.to_excel(writer, sheet_name="ANOVA completo", index=False)
    tabla_medias.to_excel(writer, sheet_name="Medias por indicador", index=False)
    resumen_dimension.to_excel(writer, sheet_name="Resumen por dimensión", index=False)
    tabla_comparacion_inicial.to_excel(writer, sheet_name="Comparación inicial", index=False)
    tabla_comparacion_completa.to_excel(writer, sheet_name="Comparación completa", index=False)
    tabla_tesis_comparativa.to_excel(writer, sheet_name="Tabla para tesis", index=False)
    longitudinal.to_excel(writer, sheet_name="Datos longitudinales", index=False)

# ============================================================
# 10. RESUMEN TXT
# ============================================================

archivo_txt = SALIDA / "resumen_anova_por_indicador.txt"

with open(archivo_txt, "w", encoding="utf-8") as f:
    f.write("COMPROBACIÓN DE HIPÓTESIS - ANOVA FACTORIAL 2x2 POR INDICADOR\n")
    f.write("=" * 90 + "\n\n")
    f.write("Factor 1: Grupo (Control / Experimental)\n")
    f.write("Factor 2: Momento (Pre-test / Post-test)\n")
    f.write("Efecto principal de interés: Grupo × Momento\n")
    f.write("Nivel de significancia: α = 0.05\n\n")
    f.write(
        "El análisis se realiza individualmente para los 16 indicadores. "
        "La dimensión se utiliza únicamente para organizar los resultados.\n\n"
    )

    f.write("RESULTADOS DE LA INTERACCIÓN GRUPO × MOMENTO\n")
    f.write("-" * 90 + "\n\n")

    for _, fila in interaccion.iterrows():
        f.write(f"Dimensión: {fila['Dimensión']}\n")
        f.write(f"Indicador: {fila['Indicador']}\n")
        f.write(f"  N = {fila['N']}\n")
        f.write(f"  F = {fila['F']:.4f}\n")
        f.write(f"  gl = {fila['gl']}\n")
        f.write(f"  p = {fila['p-valor']:.6f}\n")
        f.write(f"  Eta² parcial = {fila['Eta² parcial']:.4f}\n")
        f.write(f"  Magnitud = {fila['Magnitud Eta² parcial']}\n")
        f.write(f"  Significancia = {fila['Significancia']}\n")
        f.write(f"  Decisión = {fila['Decisión']}\n\n")

    f.write("\nCOMPARACIÓN INICIAL\n")
    f.write("-" * 90 + "\n")
    f.write("Se comparan Control Pre, Control Post y Experimental Pre.\n")
    f.write("La significancia se calcula con t pareada para Control Pre/Post y t independiente para Control Pre/Experimental Pre.\n\n")
    for _, fila in tabla_comparacion_inicial.iterrows():
        f.write(
            f"{fila['Indicador']} | "
            f"Control Pre={fila['Control Pre - Media']:.3f} | "
            f"Control Post={fila['Control Post - Media']:.3f} | "
            f"Experimental Pre={fila['Experimental Pre - Media']:.3f} | "
            f"p Control Pre/Post={fila['Control Pre vs Control Post - p']:.6f} | "
            f"p equivalencia inicial={fila['Control Pre vs Experimental Pre - p']:.6f}\n"
        )

    f.write("\nCOMPARACIÓN COMPLETA\n")
    f.write("-" * 90 + "\n")
    f.write("Se incorpora Experimental Post-test. El contraste central es la interacción Grupo × Momento.\n\n")
    for _, fila in tabla_comparacion_completa.iterrows():
        f.write(
            f"{fila['Indicador']} | "
            f"Cambio Control={fila['Cambio Control']:.3f} | "
            f"Cambio Experimental={fila['Cambio Experimental']:.3f} | "
            f"Diferencia de cambios={fila['Diferencia de cambios (Exp - Control)']:.3f} | "
            f"F={fila['F Interacción']:.4f} | "
            f"p={fila['p Interacción']:.6f} | "
            f"η²p={fila['Eta² parcial']:.4f} | "
            f"{fila['Resultado']}\n"
        )

print("\n" + "=" * 90)
print("PROCESO TERMINADO")
print("=" * 90)
print(f"\nExcel: {archivo_excel}")
print(f"Gráficos: {GRAFICOS}")
print("\nSe analizaron 16 indicadores individualmente.")
print("El resultado central es la interacción Grupo × Momento por indicador.")
print("\nCriterio: p < 0.05 → se rechaza H0; p ≥ 0.05 → no se rechaza H0.")
