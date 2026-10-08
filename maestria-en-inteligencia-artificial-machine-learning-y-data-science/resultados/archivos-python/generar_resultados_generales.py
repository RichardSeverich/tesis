import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURACIÓN
# ============================================================

ARCHIVO_CONTROL_PRE = "grupo-control-pre-test-codificado.xlsx"
ARCHIVO_CONTROL_POST = "grupo-control-post-test-codificado.xlsx"
ARCHIVO_EXPERIMENTAL_PRE = "grupo-experimental-pre-test-codificado.xlsx"
ARCHIVO_EXPERIMENTAL_POST = "grupo-experimental-post-test-codificado.xlsx"

CARPETA_SALIDA = "resultados_generales"
CARPETA_GRAFICOS = os.path.join(CARPETA_SALIDA, "graficos")

os.makedirs(CARPETA_SALIDA, exist_ok=True)
os.makedirs(CARPETA_GRAFICOS, exist_ok=True)

ARCHIVO_SALIDA = os.path.join(
    CARPETA_SALIDA,
    "resultados_generales.xlsx"
)

# ============================================================
# FUNCIONES
# ============================================================

def verificar_archivo(nombre):
    """
    Verifica que el archivo exista.
    """
    if not os.path.exists(nombre):
        raise FileNotFoundError(
            f"\nERROR: No se encontró el archivo:\n{nombre}\n"
            f"Colócalo en la misma carpeta que este script."
        )


def cargar_excel(nombre):
    """
    Lee el archivo Excel.
    """
    verificar_archivo(nombre)

    df = pd.read_excel(nombre)

    print(f"\nArchivo leído: {nombre}")
    print(f"Filas: {df.shape[0]}")
    print(f"Columnas: {df.shape[1]}")

    return df


def convertir_numericas(df):
    """
    Convierte columnas que contienen respuestas numéricas
    a formato numérico cuando sea posible.
    """

    resultado = df.copy()

    for columna in resultado.columns:
        resultado[columna] = pd.to_numeric(
            resultado[columna],
            errors="ignore"
        )

    return resultado


def obtener_columnas_numericas(df):
    """
    Obtiene únicamente las columnas numéricas.
    """

    return df.select_dtypes(
        include=np.number
    ).columns.tolist()


def calcular_estadisticos(df, grupo, momento):
    """
    Calcula estadísticos para todas las columnas numéricas.
    """

    columnas = obtener_columnas_numericas(df)

    resultados = []

    for columna in columnas:

        datos = pd.to_numeric(
            df[columna],
            errors="coerce"
        ).dropna()

        if len(datos) == 0:
            continue

        resultados.append({
            "Grupo": grupo,
            "Momento": momento,
            "Variable": columna,
            "N": len(datos),
            "Media": datos.mean(),
            "Mediana": datos.median(),
            "Desviacion_Estandar": datos.std(),
            "Minimo": datos.min(),
            "Maximo": datos.max()
        })

    return pd.DataFrame(resultados)


# ============================================================
# 1. LEER LOS CUATRO ARCHIVOS REALES
# ============================================================

print("=" * 70)
print("LECTURA DE DATOS REALES")
print("=" * 70)

control_pre = cargar_excel(
    ARCHIVO_CONTROL_PRE
)

control_post = cargar_excel(
    ARCHIVO_CONTROL_POST
)

experimental_pre = cargar_excel(
    ARCHIVO_EXPERIMENTAL_PRE
)

experimental_post = cargar_excel(
    ARCHIVO_EXPERIMENTAL_POST
)

# ============================================================
# 2. CONVERTIR COLUMNAS NUMÉRICAS
# ============================================================

control_pre = convertir_numericas(control_pre)
control_post = convertir_numericas(control_post)

experimental_pre = convertir_numericas(experimental_pre)
experimental_post = convertir_numericas(experimental_post)

# ============================================================
# 3. INFORMACIÓN GENERAL DE LA MUESTRA
# ============================================================

cantidad_control_pre = len(control_pre)
cantidad_control_post = len(control_post)

cantidad_experimental_pre = len(experimental_pre)
cantidad_experimental_post = len(experimental_post)

# Utilizamos el pretest para determinar el tamaño inicial
# de cada grupo.

n_control = cantidad_control_pre
n_experimental = cantidad_experimental_pre
n_total = n_control + n_experimental

tabla_muestra = pd.DataFrame({
    "Grupo": [
        "Control",
        "Experimental",
        "Total"
    ],
    "Participantes": [
        n_control,
        n_experimental,
        n_total
    ],
    "Porcentaje": [
        n_control / n_total * 100,
        n_experimental / n_total * 100,
        100
    ]
})

# ============================================================
# 4. ESTADÍSTICOS DEL CONTROL PRETEST
# ============================================================

estad_control_pre = calcular_estadisticos(
    control_pre,
    "Control",
    "Pretest"
)

# ============================================================
# 5. ESTADÍSTICOS DEL CONTROL POSTEST
# ============================================================

estad_control_post = calcular_estadisticos(
    control_post,
    "Control",
    "Postest"
)

# ============================================================
# 6. ESTADÍSTICOS DEL EXPERIMENTAL PRETEST
# ============================================================

estad_experimental_pre = calcular_estadisticos(
    experimental_pre,
    "Experimental",
    "Pretest"
)

# ============================================================
# 7. ESTADÍSTICOS DEL EXPERIMENTAL POSTEST
# ============================================================

estad_experimental_post = calcular_estadisticos(
    experimental_post,
    "Experimental",
    "Postest"
)

# ============================================================
# 8. UNIFICAR ESTADÍSTICOS
# ============================================================

estadisticos_generales = pd.concat([
    estad_control_pre,
    estad_control_post,
    estad_experimental_pre,
    estad_experimental_post
], ignore_index=True)

# Redondear resultados
columnas_redondear = [
    "Media",
    "Mediana",
    "Desviacion_Estandar",
    "Minimo",
    "Maximo"
]

estadisticos_generales[columnas_redondear] = (
    estadisticos_generales[columnas_redondear]
    .round(2)
)

# ============================================================
# 9. CALCULAR MEDIA GENERAL POR GRUPO Y MOMENTO
# ============================================================
#
# IMPORTANTE:
# Se calcula la media de todas las respuestas numéricas
# disponibles en cada archivo.
#
# Esto sirve para una visión GENERAL.
# Los indicadores específicos se analizarán posteriormente
# en el capítulo 4.2.

def media_general(df):

    columnas_numericas = obtener_columnas_numericas(df)

    if not columnas_numericas:
        return np.nan

    datos = df[columnas_numericas]

    return datos.stack().mean()


resumen_general = pd.DataFrame({

    "Grupo": [
        "Control",
        "Control",
        "Experimental",
        "Experimental"
    ],

    "Momento": [
        "Pretest",
        "Postest",
        "Pretest",
        "Postest"
    ],

    "Participantes": [
        n_control,
        cantidad_control_post,
        n_experimental,
        cantidad_experimental_post
    ],

    "Media_General": [
        media_general(control_pre),
        media_general(control_post),
        media_general(experimental_pre),
        media_general(experimental_post)
    ]
})

resumen_general["Media_General"] = (
    resumen_general["Media_General"]
    .round(2)
)

# ============================================================
# 10. COMPARACIÓN PRETEST / POSTEST
# ============================================================

comparacion = pd.DataFrame({

    "Grupo": [
        "Control",
        "Experimental"
    ],

    "Pretest": [
        media_general(control_pre),
        media_general(experimental_pre)
    ],

    "Postest": [
        media_general(control_post),
        media_general(experimental_post)
    ]
})

comparacion["Diferencia"] = (
    comparacion["Postest"]
    - comparacion["Pretest"]
)

comparacion["Variacion_Porcentual"] = (
    (
        comparacion["Postest"]
        - comparacion["Pretest"]
    )
    / comparacion["Pretest"]
) * 100

comparacion = comparacion.round(2)

# ============================================================
# 11. GENERAR ARCHIVO EXCEL
# ============================================================

print("\nGenerando archivo Excel...")

with pd.ExcelWriter(
    ARCHIVO_SALIDA,
    engine="openpyxl"
) as writer:

    tabla_muestra.to_excel(
        writer,
        sheet_name="Caracterizacion",
        index=False
    )

    resumen_general.to_excel(
        writer,
        sheet_name="Resultados_Generales",
        index=False
    )

    estadisticos_generales.to_excel(
        writer,
        sheet_name="Estadisticos",
        index=False
    )

    comparacion.to_excel(
        writer,
        sheet_name="Comparacion",
        index=False
    )

    # También guardar los datos originales
    control_pre.to_excel(
        writer,
        sheet_name="Control_Pretest",
        index=False
    )

    control_post.to_excel(
        writer,
        sheet_name="Control_Postest",
        index=False
    )

    experimental_pre.to_excel(
        writer,
        sheet_name="Experimental_Pretest",
        index=False
    )

    experimental_post.to_excel(
        writer,
        sheet_name="Experimental_Postest",
        index=False
    )

# ============================================================
# 12. GRÁFICO 1
# DISTRIBUCIÓN DE PARTICIPANTES
# ============================================================

plt.figure(figsize=(8, 5))

plt.bar(
    tabla_muestra["Grupo"][:2],
    tabla_muestra["Participantes"][:2]
)

plt.title(
    "Distribución de participantes por grupo"
)

plt.xlabel("Grupo")
plt.ylabel("Número de participantes")

plt.tight_layout()

plt.savefig(
    os.path.join(
        CARPETA_GRAFICOS,
        "01_distribucion_participantes.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ============================================================
# 13. GRÁFICO 2
# RESULTADOS GENERALES DEL PRETEST
# ============================================================

datos_pretest = comparacion[
    ["Grupo", "Pretest"]
]

plt.figure(figsize=(8, 5))

plt.bar(
    datos_pretest["Grupo"],
    datos_pretest["Pretest"]
)

plt.title(
    "Resultados generales del pretest"
)

plt.xlabel("Grupo")
plt.ylabel("Media general")

plt.tight_layout()

plt.savefig(
    os.path.join(
        CARPETA_GRAFICOS,
        "02_resultados_pretest.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ============================================================
# 14. GRÁFICO 3
# RESULTADOS GENERALES DEL POSTEST
# ============================================================

datos_postest = comparacion[
    ["Grupo", "Postest"]
]

plt.figure(figsize=(8, 5))

plt.bar(
    datos_postest["Grupo"],
    datos_postest["Postest"]
)

plt.title(
    "Resultados generales del postest"
)

plt.xlabel("Grupo")
plt.ylabel("Media general")

plt.tight_layout()

plt.savefig(
    os.path.join(
        CARPETA_GRAFICOS,
        "03_resultados_postest.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ============================================================
# 15. GRÁFICO 4
# COMPARACIÓN PRETEST VS POSTEST
# ============================================================

x = np.arange(
    len(comparacion["Grupo"])
)

ancho = 0.35

plt.figure(figsize=(9, 5))

plt.bar(
    x - ancho / 2,
    comparacion["Pretest"],
    ancho,
    label="Pretest"
)

plt.bar(
    x + ancho / 2,
    comparacion["Postest"],
    ancho,
    label="Postest"
)

plt.xticks(
    x,
    comparacion["Grupo"]
)

plt.xlabel("Grupo")
plt.ylabel("Media general")

plt.title(
    "Comparación de resultados generales: pretest y postest"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        CARPETA_GRAFICOS,
        "04_comparacion_pretest_postest.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ============================================================
# 16. GRÁFICO 5
# DIFERENCIA ENTRE PRETEST Y POSTEST
# ============================================================

plt.figure(figsize=(8, 5))

plt.bar(
    comparacion["Grupo"],
    comparacion["Diferencia"]
)

plt.axhline(
    y=0,
    linewidth=1
)

plt.title(
    "Variación de los resultados entre pretest y postest"
)

plt.xlabel("Grupo")
plt.ylabel("Diferencia de medias")

plt.tight_layout()

plt.savefig(
    os.path.join(
        CARPETA_GRAFICOS,
        "05_diferencia_pretest_postest.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ============================================================
# 17. MOSTRAR RESULTADOS EN CONSOLA
# ============================================================

print("\n")
print("=" * 70)
print("RESULTADOS GENERALES")
print("=" * 70)

print("\nParticipantes:")
print(tabla_muestra.to_string(index=False))

print("\n")
print("Comparación general:")
print(comparacion.to_string(index=False))

print("\n")
print("=" * 70)
print("PROCESO COMPLETADO")
print("=" * 70)

print(f"\nArchivo generado:")
print(os.path.abspath(ARCHIVO_SALIDA))

print("\nGráficos generados en:")
print(os.path.abspath(CARPETA_GRAFICOS))

print("\nArchivos:")
print("01_distribucion_participantes.png")
print("02_resultados_pretest.png")
print("03_resultados_postest.png")
print("04_comparacion_pretest_postest.png")
print("05_diferencia_pretest_postest.png")
