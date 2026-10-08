import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# CONFIGURACIÓN
# ============================================================

ARCHIVOS = {
    "Control Pre-test": "grupo-control-pre-test-codificado.xlsx",
    "Control Post-test": "grupo-control-post-test-codificado.xlsx",
    "Experimental Pre-test": "grupo-experimental-pre-test-codificado.xlsx",
    "Experimental Post-test": "grupo-experimental-post-test-codificado.xlsx"
}

# Columnas que NO deben considerarse como ítems
# Agrega aquí nombres de columnas de identificación si existen.
COLUMNAS_EXCLUIR = [
    "ID",
    "Id",
    "id",
    "Participante",
    "Participante_ID",
    "Código",
    "Codigo",
    "N°",
    "Nº",
    "Numero",
    "Número"
]

CARPETA_SALIDA = Path("resultados_estadistica_general")
CARPETA_SALIDA.mkdir(exist_ok=True)


# ============================================================
# FUNCIÓN PARA LEER Y OBTENER LOS ÍTEMS
# ============================================================

def obtener_datos_numericos(archivo):
    """
    Lee el Excel y obtiene únicamente las columnas numéricas
    correspondientes a los ítems.
    """

    df = pd.read_excel(archivo)

    # Eliminar columnas de identificación
    columnas_validas = [
        col for col in df.columns
        if col not in COLUMNAS_EXCLUIR
    ]

    df = df[columnas_validas]

    # Convertir a numérico cuando sea posible
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Mantener únicamente columnas que tengan datos numéricos
    columnas_numericas = df.select_dtypes(include=[np.number]).columns

    df = df[columnas_numericas]

    return df


# ============================================================
# FUNCIÓN ESTADÍSTICA
# ============================================================

def calcular_estadisticas(df):
    """
    Une todos los ítems y calcula estadística descriptiva general.
    """

    # Convertir todas las respuestas de todos los ítems
    # en una sola serie.
    valores = df.values.flatten()

    # Eliminar valores vacíos
    valores = valores[~pd.isna(valores)]

    # Convertir a array numérico
    valores = pd.to_numeric(valores, errors="coerce")
    valores = valores[~pd.isna(valores)]

    # Estadísticas
    media = np.mean(valores)

    mediana = np.median(valores)

    moda = pd.Series(valores).mode()

    # Si existen varias modas, tomar la primera
    moda = moda.iloc[0]

    desviacion = np.std(valores, ddof=1)

    minimo = np.min(valores)

    maximo = np.max(valores)

    return {
        "Media": media,
        "Moda": moda,
        "Mediana": mediana,
        "Desv. estándar": desviacion,
        "Mínimo": minimo,
        "Máximo": maximo
    }


# ============================================================
# PROCESAMIENTO
# ============================================================

resultados = []

print("=" * 70)
print("ESTADÍSTICA DESCRIPTIVA GENERAL")
print("=" * 70)

for grupo, archivo in ARCHIVOS.items():

    print(f"\nProcesando: {archivo}")

    if not Path(archivo).exists():
        print(f"ERROR: No se encontró el archivo: {archivo}")
        continue

    df = obtener_datos_numericos(archivo)

    print(f"Filas: {len(df)}")
    print(f"Ítems considerados: {len(df.columns)}")

    estadisticas = calcular_estadisticas(df)

    resultados.append({
        "Grupo": grupo,
        **estadisticas
    })


# ============================================================
# CREAR TABLA
# ============================================================

tabla = pd.DataFrame(resultados)

# Redondear resultados
tabla["Media"] = tabla["Media"].round(4)
tabla["Moda"] = tabla["Moda"].round(0).astype(int)
tabla["Mediana"] = tabla["Mediana"].round(0).astype(int)
tabla["Desv. estándar"] = tabla["Desv. estándar"].round(4)
tabla["Mínimo"] = tabla["Mínimo"].round(0).astype(int)
tabla["Máximo"] = tabla["Máximo"].round(0).astype(int)


# ============================================================
# MOSTRAR TABLA EN CONSOLA
# ============================================================

print("\n")
print("=" * 90)
print("RESULTADOS GENERALES - ESTADÍSTICA DESCRIPTIVA")
print("=" * 90)

print(tabla.to_string(index=False))


# ============================================================
# GUARDAR TABLA EN EXCEL
# ============================================================

archivo_excel = CARPETA_SALIDA / "estadistica_descriptiva_general.xlsx"

tabla.to_excel(
    archivo_excel,
    index=False
)

print(f"\nExcel generado: {archivo_excel}")


# ============================================================
# GRÁFICO
# ============================================================

# Estadísticas que aparecerán en el gráfico
estadisticas_grafico = [
    "Media",
    "Moda",
    "Mediana",
    "Desv. estándar",
    "Mínimo",
    "Máximo"
]

# Crear figura
fig, ax = plt.subplots(figsize=(11, 7))

x = np.arange(len(estadisticas_grafico))

# Ancho de cada barra
ancho = 0.20

# Dibujar cada grupo
for i, fila in tabla.iterrows():

    valores = [
        fila["Media"],
        fila["Moda"],
        fila["Mediana"],
        fila["Desv. estándar"],
        fila["Mínimo"],
        fila["Máximo"]
    ]

    posiciones = x + (i - 1.5) * ancho

    ax.bar(
        posiciones,
        valores,
        width=ancho,
        label=fila["Grupo"]
    )


# ============================================================
# CONFIGURACIÓN DEL GRÁFICO
# ============================================================

ax.set_title(
    "Estadística Descriptiva",
    fontsize=15,
    pad=15
)

ax.set_xticks(x)

ax.set_xticklabels(
    estadisticas_grafico,
    fontsize=10
)

ax.set_ylabel("Valor", fontsize=11)

ax.set_ylim(0, 6)

ax.set_yticks(np.arange(0, 7, 1))

ax.grid(
    axis="y",
    linestyle="-",
    alpha=0.25
)

ax.set_axisbelow(True)

ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.12),
    ncol=2,
    frameon=False
)

plt.tight_layout()


# ============================================================
# GUARDAR GRÁFICO
# ============================================================

archivo_grafico = CARPETA_SALIDA / "figura_estadistica_descriptiva_general.png"

plt.savefig(
    archivo_grafico,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(f"Gráfico generado: {archivo_grafico}")

print("\n")
print("=" * 70)
print("PROCESO FINALIZADO")
print("=" * 70)