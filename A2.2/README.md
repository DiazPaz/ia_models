# A2.2 LDA y Árboles de Decisión

Este proyecto consiste en la implementación y evaluación de modelos de aprendizaje automático para la clasificación, utilizando las técnicas de **Análisis de Discriminante Lineal (LDA)** y **Árboles de Decisión**.

## Autor

* **Nombre:** David Díaz Paz Y Puente

## Estructura de la Carpeta

* `main.ipynb`: Jupyter Notebook que contiene el código fuente, el preprocesamiento de datos, la implementación de los modelos y el análisis de resultados.
* `main.pdf` / `main.html`: Versiones exportadas del notebook para facilitar su visualización.
* `features_lasso.csv`: Conjunto de datos con las características seleccionadas previamente (posiblemente mediante LASSO) para alimentar los modelos de esta actividad.

## Descripción del Proyecto

El objetivo principal es comparar el desempeño de un modelo estadístico tradicional como LDA frente a un modelo basado en reglas como los Árboles de Decisión.

### Contenido del Notebook:

1. **Carga de Datos:** Importación del archivo `features_lasso.csv`.
2. **Análisis de Discriminante Lineal (LDA):**
* Reducción de dimensionalidad o clasificación directa.
* Evaluación mediante métricas de precisión.


3. **Árboles de Decisión:**
* Entrenamiento del clasificador.
* Visualización de la estructura del árbol.


4. **Comparativa:** Análisis de las matrices de confusión y métricas de desempeño (Accuracy, Precision, Recall) para determinar qué modelo se adapta mejor a los datos.

## Requisitos

Para ejecutar el notebook, se requiere un entorno de Python con las siguientes librerías:

* `pandas`
* `numpy`
* `scikit-learn`
* `matplotlib` / `seaborn` (para visualización)
