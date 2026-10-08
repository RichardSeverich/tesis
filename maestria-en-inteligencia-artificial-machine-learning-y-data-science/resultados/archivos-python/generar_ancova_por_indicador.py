# ============================================================
# COMPROBACIÓN DE HIPÓTESIS - ANCOVA POR INDICADOR
# TESIS: SISTEMA DE GESTIÓN ACADÉMICA WEB INTEGRADO CON CHATBOT
# ============================================================
#
# Para cada indicador:
#   Variable dependiente: Post-test
#   Covariable: Pre-test
#   Factor: Grupo (Control / Experimental)
#
# Modelo principal:
#   Post-test = Pre-test + Grupo
#
# El efecto de mayor interés es GRUPO, porque permite determinar
# si existen diferencias en el Post-test entre Control y Experimental
# después de controlar estadísticamente el nivel inicial.
#
# IMPORTANTE:
#   El ANCOVA se ejecuta POR CADA UNO DE LOS 16 INDICADORES.
#   Las cuatro variables/dimensiones se conservan como clasificación.
#
# También se comprueba la homogeneidad de pendientes:
#   Post = Pre * Grupo
#
# Si p de Pre × Grupo >= 0.05, el supuesto es compatible.
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

SALIDA = BASE_DIR / "resultados_ancova_por_indicador"
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


def cohens_d(x1, x2):
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)

    x1 = x1[~np.isnan(x1)]
    x2 = x2[~np.isnan(x2)]

    n1, n2 = len(x1), len(x2)
    if n1 < 2 or n2 < 2:
        return np.nan

    s1, s2 = np.std(x1, ddof=1), np.std(x2, ddof=1)
    sp_num = (n1 - 1) * s1**2 + (n2 - 1) * s2**2
    sp_den = n1 + n2 - 2

    if sp_den <= 0:
        return np.nan

    sp = np.sqrt(sp_num / sp_den)
    if sp == 0:
        return np.nan

    return (np.mean(x1) - np.mean(x2)) / sp


# ============================================================
# 3. CARGA DE DATOS
# ============================================================

print("=" * 90)
print("COMPROBACIÓN DE HIPÓTESIS - ANCOVA POR INDICADOR")
print("=" * 90)

datos = {}
for nombre, ruta in ARCHIVOS.items():
    print(f"\nLeyendo: {ruta.name}")
    datos[nombre] = leer_archivo(ruta)
    print(f"  Participantes: {len(datos[nombre])}")

# ============================================================
# 4. CONSTRUIR DATOS PRE/POST POR INDICADOR
# ============================================================

datos_ancova = []

for grupo in ["Control", "Experimental"]:
    pre = datos[f"{grupo} Pre-test"]
    post = datos[f"{grupo} Post-test"]

    if len(pre) != len(post):
        raise ValueError(
            f"El grupo {grupo} tiene diferente cantidad de participantes "
            f"entre Pre-test ({len(pre)}) y Post-test ({len(post)})."
        )

    for i in range(len(pre)):
        fila = {
            "ID": i + 1,
            "Grupo": grupo,
        }

        for indicador in INDICADORES:
            fila[f"{indicador} Pre-test"] = pre.iloc[i][indicador]
            fila[f"{indicador} Post-test"] = post.iloc[i][indicador]

        datos_ancova.append(fila)

datos_ancova = pd.DataFrame(datos_ancova)

# ============================================================
# 5. ANCOVA POR INDICADOR
# ============================================================

resultados = []
tablas_completas = []
pendientes = []
medias_ajustadas = []
medias_observadas = []

for indicador in INDICADORES:
    pre_col = f"{indicador} Pre-test"
    post_col = f"{indicador} Post-test"

    df = datos_ancova[["ID", "Grupo", pre_col, post_col]].copy()
    df = df.rename(columns={pre_col: "Pre", post_col: "Post"})
    df = df.dropna()

    if df["Grupo"].nunique() < 2:
        raise ValueError(
            f"El indicador '{indicador}' no tiene datos de ambos grupos."
        )

    # --------------------------------------------------------
    # Modelo ANCOVA principal: Post ~ Pre + Grupo
    # --------------------------------------------------------

    modelo = ols("Post ~ Pre + C(Grupo)", data=df).fit()
    tabla = anova_lm(modelo, typ=2)

    ss_error = tabla.loc["Residual", "sum_sq"]

    ss_grupo = tabla.loc["C(Grupo)", "sum_sq"]
    f_grupo = tabla.loc["C(Grupo)", "F"]
    p_grupo = tabla.loc["C(Grupo)", "PR(>F)"]
    gl_grupo = int(tabla.loc["C(Grupo)", "df"])
    eta_grupo = ss_grupo / (ss_grupo + ss_error)

    ss_pre = tabla.loc["Pre", "sum_sq"]
    f_pre = tabla.loc["Pre", "F"]
    p_pre = tabla.loc["Pre", "PR(>F)"]
    gl_pre = int(tabla.loc["Pre", "df"])
    eta_pre = ss_pre / (ss_pre + ss_error)

    # --------------------------------------------------------
    # Homogeneidad de pendientes: Post ~ Pre * Grupo
    # --------------------------------------------------------

    modelo_interaccion = ols("Post ~ Pre * C(Grupo)", data=df).fit()
    tabla_interaccion = anova_lm(modelo_interaccion, typ=2)

    nombre_interaccion = "Pre:C(Grupo)"

    if nombre_interaccion in tabla_interaccion.index:
        f_interaccion = tabla_interaccion.loc[nombre_interaccion, "F"]
        p_interaccion = tabla_interaccion.loc[nombre_interaccion, "PR(>F)"]
    else:
        f_interaccion = np.nan
        p_interaccion = np.nan

    supuesto = (
        "Compatible"
        if not pd.isna(p_interaccion) and p_interaccion >= ALPHA
        else "No compatible"
        if not pd.isna(p_interaccion)
        else "No calculable"
    )

    # --------------------------------------------------------
    # Medias observadas
    # --------------------------------------------------------

    medias = (
        df.groupby("Grupo")
        .agg(
            N=("Post", "count"),
            Media_Pre=("Pre", "mean"),
            DE_Pre=("Pre", "std"),
            Media_Post=("Post", "mean"),
            DE_Post=("Post", "std"),
        )
        .reset_index()
    )

    for _, fila in medias.iterrows():
        medias_observadas.append({
            "Dimensión": MAPA_DIMENSION[indicador],
            "Indicador": indicador,
            "Grupo": fila["Grupo"],
            "N": int(fila["N"]),
            "Media Pre-test": fila["Media_Pre"],
            "DE Pre-test": fila["DE_Pre"],
            "Media Post-test": fila["Media_Post"],
            "DE Post-test": fila["DE_Post"],
        })

    # --------------------------------------------------------
    # Medias Post-test ajustadas
    # --------------------------------------------------------

    media_pre_general = df["Pre"].mean()
    ajustados = {}

    for grupo in ["Control", "Experimental"]:
        fila = medias[medias["Grupo"] == grupo]

        if fila.empty:
            continue

        nuevo = pd.DataFrame({
            "Pre": [media_pre_general],
            "Grupo": [grupo],
        })

        media_ajustada = float(modelo.predict(nuevo).iloc[0])
        ajustados[grupo] = media_ajustada

        medias_ajustadas.append({
            "Dimensión": MAPA_DIMENSION[indicador],
            "Indicador": indicador,
            "Grupo": grupo,
            "N": int(fila["N"].iloc[0]),
            "Media Pre-test": fila["Media_Pre"].iloc[0],
            "Media Post-test observada": fila["Media_Post"].iloc[0],
            "Media Post-test ajustada": media_ajustada,
            "Media Pre-test general": media_pre_general,
        })

    # Cohen's d sobre medias post-test observadas
    post_control = df.loc[df["Grupo"] == "Control", "Post"]
    post_exp = df.loc[df["Grupo"] == "Experimental", "Post"]
    d_post = cohens_d(post_exp, post_control)

    resultados.append({
        "Dimensión": MAPA_DIMENSION[indicador],
        "Indicador": indicador,
        "N": len(df),
        "F Grupo": f_grupo,
        "gl Grupo": gl_grupo,
        "p Grupo": p_grupo,
        "Significancia Grupo": significancia(p_grupo),
        "Eta² parcial Grupo": eta_grupo,
        "Magnitud Eta² Grupo": interpretar_eta(eta_grupo),
        "Decisión Grupo": decision(p_grupo),
        "F Pre-test": f_pre,
        "gl Pre-test": gl_pre,
        "p Pre-test": p_pre,
        "Eta² parcial Pre-test": eta_pre,
        "R² modelo": modelo.rsquared,
        "R² ajustado": modelo.rsquared_adj,
        "Cohen's d Post-test": d_post,
        "F Pre-test × Grupo": f_interaccion,
        "p Pre-test × Grupo": p_interaccion,
        "Supuesto pendientes": supuesto,
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

tabla_resultados = pd.DataFrame(resultados)
tabla_ancova_completa = pd.concat(tablas_completas, ignore_index=True)
tabla_medias_observadas = pd.DataFrame(medias_observadas)
tabla_medias_ajustadas = pd.DataFrame(medias_ajustadas)
tabla_pendientes = tabla_resultados[
    [
        "Dimensión", "Indicador", "F Pre-test × Grupo",
        "p Pre-test × Grupo", "Supuesto pendientes"
    ]
].copy()

# ============================================================
# 6. TABLA PRINCIPAL PARA LA TESIS
# ============================================================

resumen = tabla_resultados[
    [
        "Dimensión", "Indicador", "N",
        "F Grupo", "gl Grupo", "p Grupo",
        "Significancia Grupo", "Eta² parcial Grupo",
        "Magnitud Eta² Grupo", "Decisión Grupo",
        "R² modelo", "R² ajustado",
    ]
].copy()

# ============================================================
# 7. RESUMEN POR DIMENSIÓN
# ============================================================

resumen_dimension = (
    tabla_resultados.groupby("Dimensión")
    .agg(
        Indicadores=("Indicador", "count"),
        Indicadores_significativos=("p Grupo", lambda x: int((x < ALPHA).sum())),
        F_promedio=("F Grupo", "mean"),
        Eta2_parcial_promedio=("Eta² parcial Grupo", "mean"),
    )
    .reset_index()
)

# ============================================================
# 8. GRÁFICOS POR INDICADOR
# ============================================================

for indicador in INDICADORES:
    fila = tabla_resultados[tabla_resultados["Indicador"] == indicador].iloc[0]

    obs = tabla_medias_observadas[
        tabla_medias_observadas["Indicador"] == indicador
    ]

    adj = tabla_medias_ajustadas[
        tabla_medias_ajustadas["Indicador"] == indicador
    ]

    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    grupos = ["Control", "Experimental"]
    x = np.arange(len(grupos))
    ancho = 0.35

    obs_vals = [
        obs.loc[obs["Grupo"] == g, "Media Post-test"].iloc[0]
        for g in grupos
    ]
    adj_vals = [
        adj.loc[adj["Grupo"] == g, "Media Post-test ajustada"].iloc[0]
        for g in grupos
    ]

    ax.bar(x - ancho / 2, obs_vals, ancho, label="Post-test observado")
    ax.bar(x + ancho / 2, adj_vals, ancho, label="Post-test ajustado")

    ax.set_xticks(x)
    ax.set_xticklabels(grupos)
    ax.set_ylim(1, 5)
    ax.set_ylabel("Media")
    ax.set_title(
        f"ANCOVA por indicador\n{indicador}",
        fontsize=12,
        fontweight="bold",
    )
    ax.text(
        0.5, 0.02,
        f"F Grupo = {fila['F Grupo']:.3f} | "
        f"p = {fila['p Grupo']:.4g} | "
        f"η²p = {fila['Eta² parcial Grupo']:.3f}",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=9,
    )
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()

    plt.savefig(
        GRAFICOS / f"ancova_{nombre_archivo(indicador)}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

# ============================================================
# 9. GRÁFICOS RESUMEN
# ============================================================

fig, ax = plt.subplots(figsize=(13, 6.5))
ax.bar(resumen["Indicador"], resumen["p Grupo"])
ax.axhline(ALPHA, linestyle="--", linewidth=2, label="α = 0.05")
ax.set_ylabel("p-valor")
ax.set_title("Significancia del efecto Grupo en ANCOVA por indicador",
             fontsize=13, fontweight="bold")
ax.tick_params(axis="x", rotation=45)
ax.legend()
ax.grid(axis="y", alpha=0.25)
plt.tight_layout()
plt.savefig(GRAFICOS / "p_valores_ancova_por_indicador.png",
            dpi=300, bbox_inches="tight")
plt.close()

fig, ax = plt.subplots(figsize=(13, 6.5))
ax.bar(resumen["Indicador"], resumen["Eta² parcial Grupo"])
ax.set_ylabel("Eta² parcial")
ax.set_title("Tamaño del efecto del Grupo en ANCOVA por indicador",
             fontsize=13, fontweight="bold")
ax.tick_params(axis="x", rotation=45)
ax.grid(axis="y", alpha=0.25)
plt.tight_layout()
plt.savefig(GRAFICOS / "eta_cuadrado_ancova_por_indicador.png",
            dpi=300, bbox_inches="tight")
plt.close()

# ============================================================
# 10. TABLAS DE COMPARACIÓN PARA LA TESIS
# ============================================================
# Se incorporan dos vistas complementarias:
# 1) Situación inicial: Control Pre, Control Post y Experimental Pre.
# 2) Comparación completa: incluye Experimental Post.
#
# El ANCOVA sigue siendo la prueba ajustada principal; estas tablas
# sirven para visualizar los cambios y relacionarlos con sus pruebas.
# ============================================================

from scipy.stats import ttest_rel, ttest_ind

comparacion_inicial = []
comparacion_completa = []

for indicador in INDICADORES:
    c_pre = datos["Control Pre-test"][indicador].dropna().to_numpy(dtype=float)
    c_post = datos["Control Post-test"][indicador].dropna().to_numpy(dtype=float)
    e_pre = datos["Experimental Pre-test"][indicador].dropna().to_numpy(dtype=float)
    e_post = datos["Experimental Post-test"][indicador].dropna().to_numpy(dtype=float)

    n_pareado = min(len(c_pre), len(c_post))
    t_control, p_control = (np.nan, np.nan)
    if n_pareado >= 2:
        t_control, p_control = ttest_rel(c_pre[:n_pareado], c_post[:n_pareado])

    t_inicial, p_inicial = ttest_ind(
        e_pre, c_pre, equal_var=False, nan_policy="omit"
    )

    fila_ancova = tabla_resultados[
        tabla_resultados["Indicador"] == indicador
    ].iloc[0]

    # El cambio bruto se muestra de forma descriptiva.
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
        "Diferencia de cambios (Exp - Control)": (
            np.mean(e_post) - np.mean(e_pre)
        ) - (
            np.mean(c_post) - np.mean(c_pre)
        ),
        "F Grupo ANCOVA": fila_ancova["F Grupo"],
        "p Grupo ANCOVA": fila_ancova["p Grupo"],
        "Eta² parcial Grupo": fila_ancova["Eta² parcial Grupo"],
        "Media Post Control ajustada": (
            tabla_medias_ajustadas[
                (tabla_medias_ajustadas["Indicador"] == indicador) &
                (tabla_medias_ajustadas["Grupo"] == "Control")
            ]["Media Post-test ajustada"].iloc[0]
        ),
        "Media Post Experimental ajustada": (
            tabla_medias_ajustadas[
                (tabla_medias_ajustadas["Indicador"] == indicador) &
                (tabla_medias_ajustadas["Grupo"] == "Experimental")
            ]["Media Post-test ajustada"].iloc[0]
        ),
        "Resultado ANCOVA": (
            "Significativo" if fila_ancova["p Grupo"] < ALPHA
            else "No significativo"
        ),
    })

tabla_comparacion_inicial = pd.DataFrame(comparacion_inicial)
tabla_comparacion_completa = pd.DataFrame(comparacion_completa)

tabla_tesis_comparativa = tabla_comparacion_completa[[
    "Dimensión", "Indicador",
    "Control Pre - Media", "Control Post - Media",
    "Experimental Pre - Media", "Experimental Post - Media",
    "Cambio Control", "Cambio Experimental",
    "Diferencia de cambios (Exp - Control)",
    "Media Post Control ajustada", "Media Post Experimental ajustada",
    "F Grupo ANCOVA", "p Grupo ANCOVA", "Eta² parcial Grupo",
    "Resultado ANCOVA"
]].copy()

# ============================================================
# 11. EXPORTACIÓN A EXCEL
# ============================================================

archivo_excel = SALIDA / "tablas_ancova_por_indicador.xlsx"

with pd.ExcelWriter(archivo_excel, engine="openpyxl") as writer:
    resumen.to_excel(writer, sheet_name="Resumen ANCOVA", index=False)
    tabla_resultados.to_excel(writer, sheet_name="Resultados ANCOVA", index=False)
    tabla_ancova_completa.to_excel(writer, sheet_name="ANCOVA completo", index=False)
    tabla_medias_observadas.to_excel(writer, sheet_name="Medias observadas", index=False)
    tabla_medias_ajustadas.to_excel(writer, sheet_name="Medias ajustadas", index=False)
    tabla_pendientes.to_excel(writer, sheet_name="Homogeneidad pendientes", index=False)
    resumen_dimension.to_excel(writer, sheet_name="Resumen por dimensión", index=False)
    tabla_comparacion_inicial.to_excel(writer, sheet_name="Comparación inicial", index=False)
    tabla_comparacion_completa.to_excel(writer, sheet_name="Comparación completa", index=False)
    tabla_tesis_comparativa.to_excel(writer, sheet_name="Tabla para tesis", index=False)
    datos_ancova.to_excel(writer, sheet_name="Datos ANCOVA", index=False)

# ============================================================
# 11. RESUMEN TXT
# ============================================================

archivo_txt = SALIDA / "resumen_ancova_por_indicador.txt"

with open(archivo_txt, "w", encoding="utf-8") as f:
    f.write("COMPROBACIÓN DE HIPÓTESIS - ANCOVA POR INDICADOR\n")
    f.write("=" * 90 + "\n\n")
    f.write("Modelo: Post-test ~ Pre-test + Grupo\n")
    f.write("Nivel de significancia: α = 0.05\n")
    f.write(
        "El efecto de mayor interés es GRUPO, ajustado por el Pre-test.\n"
    )
    f.write(
        "El análisis se realiza individualmente para los 16 indicadores.\n\n"
    )

    f.write("RESULTADOS PRINCIPALES\n")
    f.write("-" * 90 + "\n\n")

    for _, fila in resumen.iterrows():
        f.write(f"Dimensión: {fila['Dimensión']}\n")
        f.write(f"Indicador: {fila['Indicador']}\n")
        f.write(f"  N = {fila['N']}\n")
        f.write(f"  F Grupo = {fila['F Grupo']:.4f}\n")
        f.write(f"  gl Grupo = {fila['gl Grupo']}\n")
        f.write(f"  p Grupo = {fila['p Grupo']:.6f}\n")
        f.write(f"  Eta² parcial = {fila['Eta² parcial Grupo']:.4f}\n")
        f.write(f"  Magnitud = {fila['Magnitud Eta² Grupo']}\n")
        f.write(f"  Decisión = {fila['Decisión Grupo']}\n")
        f.write(f"  R² = {fila['R² modelo']:.4f}\n")
        f.write(f"  R² ajustado = {fila['R² ajustado']:.4f}\n\n")

    f.write("=" * 90 + "\n")
    f.write("SUPUESTO DE HOMOGENEIDAD DE PENDIENTES\n")
    f.write("=" * 90 + "\n\n")

    for _, fila in tabla_pendientes.iterrows():
        f.write(
            f"{fila['Indicador']} | "
            f"F={fila['F Pre-test × Grupo']:.4f} | "
            f"p={fila['p Pre-test × Grupo']:.6f} | "
            f"{fila['Supuesto pendientes']}\n"
        )

    f.write("\nCOMPARACIÓN INICIAL\n")
    f.write("-" * 90 + "\n")
    f.write("Se comparan Control Pre, Control Post y Experimental Pre.\n")
    f.write("Se utilizan t pareada para Control Pre/Post y t independiente para Control Pre/Experimental Pre.\n\n")
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
    f.write("Se incorpora Experimental Post-test y se muestran los cambios brutos y el resultado ANCOVA ajustado.\n\n")
    for _, fila in tabla_comparacion_completa.iterrows():
        f.write(
            f"{fila['Indicador']} | "
            f"Cambio Control={fila['Cambio Control']:.3f} | "
            f"Cambio Experimental={fila['Cambio Experimental']:.3f} | "
            f"Diferencia de cambios={fila['Diferencia de cambios (Exp - Control)']:.3f} | "
            f"F ANCOVA={fila['F Grupo ANCOVA']:.4f} | "
            f"p ANCOVA={fila['p Grupo ANCOVA']:.6f} | "
            f"η²p={fila['Eta² parcial Grupo']:.4f} | "
            f"{fila['Resultado ANCOVA']}\n"
        )

print("\n" + "=" * 90)
print("PROCESO TERMINADO")
print("=" * 90)
print(f"\nExcel: {archivo_excel}")
print(f"Gráficos: {GRAFICOS}")
print("\nSe analizaron 16 indicadores individualmente.")
print("El efecto central del ANCOVA es Grupo, ajustado por Pre-test.")
print("\nCriterio: p < 0.05 → se rechaza H0; p ≥ 0.05 → no se rechaza H0.")
