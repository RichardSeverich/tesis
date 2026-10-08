import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import re


# ============================================================
# CONFIGURACIÓN
# ============================================================

ARCHIVOS = {
    "Grupo Control Pre-test": "grupo-control-pre-test-codificado.xlsx",
    "Grupo Control Post-test": "grupo-control-post-test-codificado.xlsx",
    "Grupo Experimental Pre-test": "grupo-experimental-pre-test-codificado.xlsx",
    "Grupo Experimental Post-test": "grupo-experimental-post-test-codificado.xlsx"
}


# ============================================================
# ESCALAS DE RESPUESTA
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
# CONFIGURACIÓN DE CADA INDICADOR
# ============================================================

INDICADORES = {

    # --------------------------------------------------------
    # INSCRIPCIONES
    # --------------------------------------------------------

    "Tiempo Inscripción": ESCALA_TIEMPO,

    "Facilidad Inscripción": ESCALA_FACILIDAD,

    "Acceso Inscripción": ESCALA_ACCESO,

    "Satisfacción Inscripciones": ESCALA_SATISFACCION,


    # --------------------------------------------------------
    # INFORMACIÓN
    # --------------------------------------------------------

    "Tiempo información": ESCALA_TIEMPO,

    "Facilidad información": ESCALA_FACILIDAD,

    "Numero Consultas": ESCALA_ACCESO,

    "Calidad Atención": ESCALA_SATISFACCION,


    # --------------------------------------------------------
    # CALIFICACIONES
    # --------------------------------------------------------

    "Tiempo Calificación": ESCALA_TIEMPO,

    "Facilidad Calificación": ESCALA_FACILIDAD,

    "Acceso Calificaciones": ESCALA_ACCESO,

    "Satisfacción Calificaciones": ESCALA_SATISFACCION,


    # --------------------------------------------------------
    # ATENCIÓN AL CLIENTE
    # --------------------------------------------------------

    "Tiempo Atención Cliente": ESCALA_TIEMPO,

    "Facilidad Atención Cliente": ESCALA_FACILIDAD,

    "Acceso Atención Cliente": ESCALA_ACCESO,

    "Satisfacción Atención Cliente": ESCALA_SATISFACCION
}


# ============================================================
# CARPETAS DE SALIDA
# ============================================================

CARPETA_SALIDA = Path("resultados_indicadores")

CARPETA_TABLAS = CARPETA_SALIDA / "tablas"

CARPETA_GRAFICOS = CARPETA_SALIDA / "graficos"

CARPETA_TABLAS.mkdir(parents=True, exist_ok=True)

CARPETA_GRAFICOS.mkdir(parents=True, exist_ok=True)


# ============================================================
# LIMPIAR NOMBRE DE ARCHIVO
# ============================================================

def limpiar_nombre(nombre):

    nombre = str(nombre)

    nombre = re.sub(
        r'[\\/*?:"<>|]',
        "",
        nombre
    )

    nombre = nombre.replace(" ", "_")

    return nombre[:100]


# ============================================================
# LEER LOS ARCHIVOS
# ============================================================

datos = {}

print("=" * 80)
print("LECTURA DE DATOS")
print("=" * 80)


for grupo, archivo in ARCHIVOS.items():

    print(f"\nLeyendo: {archivo}")

    if not Path(archivo).exists():

        print(
            f"ERROR: No se encontró el archivo {archivo}"
        )

        continue

    df = pd.read_excel(archivo)

    print(f"Filas: {len(df)}")
    print(f"Columnas: {len(df.columns)}")

    # Convertir todos los indicadores a numérico
    for columna in df.columns:

        df[columna] = pd.to_numeric(
            df[columna],
            errors="coerce"
        )

    datos[grupo] = df


# ============================================================
# VERIFICACIÓN
# ============================================================

if len(datos) != 4:

    print("\nERROR: Deben existir los cuatro archivos.")

    exit()


# ============================================================
# VERIFICAR INDICADORES
# ============================================================

print("\n")
print("=" * 80)
print("VERIFICACIÓN DE INDICADORES")
print("=" * 80)


for indicador in INDICADORES:

    encontrado = False

    for df in datos.values():

        if indicador in df.columns:

            encontrado = True
            break

    if encontrado:

        print(f"OK: {indicador}")

    else:

        print(
            f"ADVERTENCIA: No se encontró: {indicador}"
        )


# ============================================================
# PROCESAR INDICADORES
# ============================================================

for numero, (indicador, escala) in enumerate(
    INDICADORES.items(),
    start=1
):

    print("\n")
    print("=" * 80)
    print(f"INDICADOR {numero}: {indicador}")
    print("=" * 80)


    # ========================================================
    # TABLA DE FRECUENCIAS
    # ========================================================

    tabla = pd.DataFrame(
        index=list(escala.values())
    )


    # --------------------------------------------------------
    # Contar respuestas
    # --------------------------------------------------------

    for grupo, df in datos.items():

        if indicador not in df.columns:

            print(
                f"ADVERTENCIA: {indicador} "
                f"no existe en {grupo}"
            )

            tabla[grupo] = 0

            continue


        frecuencias = (
            df[indicador]
            .value_counts()
        # IMPORTANTE: las etiquetas están ordenadas 5, 4, 3, 2, 1;
        # las frecuencias deben recuperarse en ese mismo orden.
            .reindex(
                [5, 4, 3, 2, 1],
                fill_value=0
            )
        )


        tabla[grupo] = frecuencias.values


    # ========================================================
    # TOTAL
    # ========================================================

    tabla["Total"] = tabla.sum(axis=1)


    # ========================================================
    # CREAR TABLA DE PRESENTACIÓN
    # FRECUENCIA + PORCENTAJE
    # ========================================================

    # Se conserva "tabla" con valores numéricos para que
    # los gráficos continúen funcionando correctamente.
    #
    # "tabla_presentacion" será la tabla que se guardará
    # en Excel y mostrará:
    #
    #       Frecuencia (Porcentaje)
    #
    # Ejemplo:
    #       25 (32.1%)

    tabla_presentacion = pd.DataFrame(
        index=tabla.index
    )

    # --------------------------------------------------------
    # Calcular porcentaje dentro de cada grupo
    # --------------------------------------------------------

    for grupo in datos.keys():

        # Total de respuestas válidas del grupo
        total_grupo = tabla[grupo].sum()

        valores_presentacion = []

        for frecuencia in tabla[grupo]:

            if total_grupo > 0:

                porcentaje = (
                    frecuencia / total_grupo
                ) * 100

            else:

                porcentaje = 0

            valores_presentacion.append(
                f"{int(frecuencia)} "
                f"({porcentaje:.1f}%)"
            )

        tabla_presentacion[grupo] = (
            valores_presentacion
        )

    # --------------------------------------------------------
    # Columna TOTAL
    # --------------------------------------------------------

    # El total se mantiene como frecuencia numérica.
    # Esto permite identificar fácilmente el total de
    # respuestas de cada categoría entre los cuatro grupos.

    tabla_presentacion["Total"] = (
        tabla["Total"].astype(int)
    )


    # ========================================================
    # MOSTRAR TABLA
    # ========================================================

    print("\nTabla de frecuencias y porcentajes:")

    print(
        tabla_presentacion.to_string()
    )


    # ========================================================
    # GUARDAR TABLA INDIVIDUAL
    # ========================================================

    nombre_tabla = (
        f"{numero:02d}_"
        f"{limpiar_nombre(indicador)}.xlsx"
    )


    tabla_presentacion.to_excel(
        CARPETA_TABLAS / nombre_tabla,
        index_label="Respuesta"
    )


    # ========================================================
    # CREAR GRÁFICO
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(11, 7)
    )


    categorias = list(
        escala.values()
    )


    posiciones_base = range(
        len(categorias)
    )


    ancho = 0.18


    # --------------------------------------------------------
    # Dibujar los cuatro grupos
    # --------------------------------------------------------

    for i, grupo in enumerate(datos.keys()):

        posiciones = [
            x + (i - 1.5) * ancho
            for x in posiciones_base
        ]


        valores = tabla[grupo].values


        ax.bar(
            posiciones,
            valores,
            width=ancho,
            label=grupo
        )


    # ========================================================
    # CONFIGURACIÓN DEL GRÁFICO
    # ========================================================

    ax.set_title(
        indicador,
        fontsize=15,
        pad=15
    )


    ax.set_xlabel(
        "Categoría de respuesta",
        fontsize=11
    )


    ax.set_ylabel(
        "Frecuencia",
        fontsize=11
    )


    ax.set_xticks(
        list(posiciones_base)
    )


    ax.set_xticklabels(
        categorias,
        rotation=0
    )


    ax.grid(
        axis="y",
        linestyle="-",
        alpha=0.25
    )


    ax.set_axisbelow(True)


    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        ncol=2,
        frameon=False
    )


    plt.tight_layout()


    # ========================================================
    # GUARDAR GRÁFICO
    # ========================================================

    nombre_grafico = (
        f"{numero:02d}_"
        f"{limpiar_nombre(indicador)}.png"
    )


    plt.savefig(
        CARPETA_GRAFICOS / nombre_grafico,
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()


    print(
        f"Generado: {indicador}"
    )


# ============================================================
# FINAL
# ============================================================

print("\n")
print("=" * 80)
print("PROCESO FINALIZADO")
print("=" * 80)

print(
    "\nTablas guardadas en:"
)

print(
    CARPETA_TABLAS
)

print(
    "\nGráficos guardados en:"
)

print(
    CARPETA_GRAFICOS
)
