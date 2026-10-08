# Conversión de respuestas a datos numéricos

Este proyecto permite convertir las respuestas textuales de los cuestionarios almacenados en archivos Excel a valores numéricos del **1 al 5**, utilizando la codificación definida para el análisis estadístico de la investigación.

## 1. Estructura del proyecto

Todos los archivos deben encontrarse en la misma carpeta:

```text
datos-codificacion/
│
├── convertir_a_numerico.py
│
├── grupo-control-pre-test.xlsx
├── grupo-control-post-test.xlsx
├── grupo-experimental-pre-test.xlsx
└── grupo-experimental-post-test.xlsx
```

## 2. Requisitos

Se requiere tener instalado:

- Python Python 3.9.13
- pandas
- openpyxl

## 3. Instalar las dependencias

Abrir una terminal en la carpeta del proyecto y ejecutar:

```bash
py -m pip install pandas openpyxl
```

En caso de que `py` no esté disponible, utilizar:

```bash
python -m pip install pandas openpyxl
```

## 4. Ejecutar el programa

Ubicarse mediante la terminal en la carpeta donde se encuentra el archivo:

```text
convertir_a_numerico.py
```

Por ejemplo:

```bash
cd "E:\ACADEMIC MAESTRIA - CHATBOT\datos-codificacion"
```

Luego ejecutar:

```bash
py convertir_a_numerico.py
```

También puede ejecutarse con:

```bash
python convertir_a_numerico.py
```

## 5. Archivos de entrada

El programa procesa los siguientes archivos:

```text
grupo-control-pre-test.xlsx
grupo-control-post-test.xlsx
grupo-experimental-pre-test.xlsx
grupo-experimental-post-test.xlsx
```

Cada archivo contiene las respuestas textuales de los participantes.

## 6. Codificación utilizada

La conversión utiliza una escala de **1 a 5**, donde:

### Tiempo

| Respuesta | Valor |
|---|---:|
| Más de 10 | 1 |
| 8 - 10 | 2 |
| 5 - 7 | 3 |
| 2 - 4 | 4 |
| Menos de 2 | 5 |

### Facilidad

| Respuesta | Valor |
|---|---:|
| Muy difícil | 1 |
| Difícil | 2 |
| Moderado | 3 |
| Fácil | 4 |
| Muy fácil | 5 |

### Acceso / Consultas

| Respuesta | Valor |
|---|---:|
| 8 o más | 1 |
| 6 - 7 | 2 |
| 4 - 5 | 3 |
| 2 - 3 | 4 |
| 1 o menos | 5 |

### Satisfacción / Calidad

| Respuesta | Valor |
|---|---:|
| Muy malo | 1 |
| Malo | 2 |
| Moderado | 3 |
| Bueno | 4 |
| Muy bueno | 5 |

De esta manera:

**1 = peor resultado**

**5 = mejor resultado**

## 7. Archivos generados

Al ejecutar el programa, se crean automáticamente cuatro nuevos archivos:

```text
grupo-control-pre-test-codificado.xlsx
grupo-control-post-test-codificado.xlsx
grupo-experimental-pre-test-codificado.xlsx
grupo-experimental-post-test-codificado.xlsx
```

Los archivos originales **no son modificados**.

## 8. Ejemplo

Si el archivo contiene:

| Tiempo Inscripción | Facilidad Inscripción |
|---|---|
| Más de 10 | Muy difícil |
| 2 - 4 | Fácil |
| Menos de 2 | Muy fácil |

Después de ejecutar el programa:

| Tiempo Inscripción | Facilidad Inscripción |
|---:|---:|
| 1 | 1 |
| 4 | 4 |
| 5 | 5 |

## 9. Ejecución completa

Una vez instaladas las dependencias, el proceso completo es:

```bash
cd "E:\ACADEMIC MAESTRIA - CHATBOT\datos-codificacion"

py -m pip install pandas openpyxl

py convertir_a_numerico.py
```

El programa mostrará en la terminal información similar a:

```text
Conversión completada correctamente.
Archivo original : grupo-control-pre-test.xlsx
Archivo generado : grupo-control-pre-test-codificado.xlsx
Participantes    : 78
Indicadores      : 16
```

Y los cuatro archivos codificados quedarán en la misma carpeta.
