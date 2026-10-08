import pandas as pd
from pathlib import Path


# ============================================================
# CODIFICACIÓN DEL CUESTIONARIO
# 1 = peor resultado
# 5 = mejor resultado
# ============================================================

CODIFICACION = {

    # TIEMPO
    "Tiempo Inscripción": {
        "Más de 10": 1,
        "8 - 10": 2,
        "5 - 7": 3,
        "2 - 4": 4,
        "Menos de 2": 5
    },

    "Tiempo información": {
        "Más de 10": 1,
        "8 - 10": 2,
        "5 - 7": 3,
        "2 - 4": 4,
        "Menos de 2": 5
    },

    "Tiempo Calificación": {
        "Más de 10": 1,
        "8 - 10": 2,
        "5 - 7": 3,
        "2 - 4": 4,
        "Menos de 2": 5
    },

    "Tiempo Atención Cliente": {
        "Más de 10": 1,
        "8 - 10": 2,
        "5 - 7": 3,
        "2 - 4": 4,
        "Menos de 2": 5
    },


    # FACILIDAD
    "Facilidad Inscripción": {
        "Muy difícil": 1,
        "Difícil": 2,
        "Moderado": 3,
        "Fácil": 4,
        "Muy fácil": 5
    },

    "Facilidad información": {
        "Muy difícil": 1,
        "Difícil": 2,
        "Moderado": 3,
        "Fácil": 4,
        "Muy fácil": 5
    },

    "Facilidad Calificación": {
        "Muy difícil": 1,
        "Difícil": 2,
        "Moderado": 3,
        "Fácil": 4,
        "Muy fácil": 5
    },

    "Facilidad Atención Cliente": {
        "Muy difícil": 1,
        "Difícil": 2,
        "Moderado": 3,
        "Fácil": 4,
        "Muy fácil": 5
    },


    # ACCESO / CONSULTAS
    "Acceso Inscripción": {
        "8 o más": 1,
        "6 - 7": 2,
        "4 - 5": 3,
        "2 - 3": 4,
        "1 o menos": 5
    },

    "Numero Consultas": {
        "8 o más": 1,
        "6 - 7": 2,
        "4 - 5": 3,
        "2 - 3": 4,
        "1 o menos": 5
    },

    "Acceso Calificaciones": {
        "8 o más": 1,
        "6 - 7": 2,
        "4 - 5": 3,
        "2 - 3": 4,
        "1 o menos": 5
    },

    "Acceso Atención Cliente": {
        "8 o más": 1,
        "6 - 7": 2,
        "4 - 5": 3,
        "2 - 3": 4,
        "1 o menos": 5
    },


    # SATISFACCIÓN / CALIDAD
    "Satisfacción Inscripciones": {
        "Muy malo": 1,
        "Malo": 2,
        "Moderado": 3,
        "Bueno": 4,
        "Muy bueno": 5
    },

    "Calidad Atención": {
        "Muy malo": 1,
        "Malo": 2,
        "Moderado": 3,
        "Bueno": 4,
        "Muy bueno": 5
    },

    "Satisfacción Calificaciones": {
        "Muy malo": 1,
        "Malo": 2,
        "Moderado": 3,
        "Bueno": 4,
        "Muy bueno": 5
    },

    "Satisfacción Atención Cliente": {
        "Muy malo": 1,
        "Malo": 2,
        "Moderado": 3,
        "Bueno": 4,
        "Muy bueno": 5
    }
}


def convertir_a_numerico(nombre_archivo):
    """
    Lee un archivo Excel con respuestas textuales,
    convierte las respuestas a valores numéricos
    y genera un nuevo archivo con '-codificado'.
    """

    # --------------------------------------------------------
    # 1. Cargar archivo
    # --------------------------------------------------------
    archivo = Path(nombre_archivo)

    if not archivo.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo: {nombre_archivo}"
        )

    df = pd.read_excel(archivo)

    # --------------------------------------------------------
    # 2. Verificar columnas
    # --------------------------------------------------------
    columnas_faltantes = [
        columna
        for columna in CODIFICACION
        if columna not in df.columns
    ]

    if columnas_faltantes:
        raise ValueError(
            "Faltan las siguientes columnas en el Excel:\n"
            + "\n".join(columnas_faltantes)
        )

    # --------------------------------------------------------
    # 3. Convertir respuestas a números
    # --------------------------------------------------------
    for columna, mapa in CODIFICACION.items():

        valores_originales = set(
            df[columna].dropna().astype(str).str.strip()
        )

        valores_permitidos = set(mapa.keys())

        valores_desconocidos = (
            valores_originales - valores_permitidos
        )

        if valores_desconocidos:
            raise ValueError(
                f"Se encontraron respuestas no reconocidas "
                f"en '{columna}': {valores_desconocidos}"
            )

        df[columna] = (
            df[columna]
            .astype(str)
            .str.strip()
            .map(mapa)
        )

    # --------------------------------------------------------
    # 4. Crear nombre del nuevo archivo
    # --------------------------------------------------------
    nuevo_nombre = archivo.with_name(
        f"{archivo.stem}-codificado{archivo.suffix}"
    )

    # --------------------------------------------------------
    # 5. Guardar Excel
    # --------------------------------------------------------
    df.to_excel(nuevo_nombre, index=False)

    print("Conversión completada correctamente.")
    print(f"Archivo original : {archivo}")
    print(f"Archivo generado : {nuevo_nombre}")
    print(f"Participantes    : {len(df)}")
    print(f"Indicadores      : {len(df.columns)}")

    return nuevo_nombre


# ============================================================
# EJEMPLO DE USO
# ============================================================

convertir_a_numerico("grupo-control-pre-test.xlsx")

convertir_a_numerico("grupo-control-post-test.xlsx")

convertir_a_numerico("grupo-experimental-pre-test.xlsx")

convertir_a_numerico("grupo-experimental-post-test.xlsx")
