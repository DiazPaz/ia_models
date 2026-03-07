# Actividad 2.2: Clasificación con LDA y Árboles de Decisión

## Descripción General

Esta actividad consiste en el desarrollo y aplicación de modelos de aprendizaje supervisado para tareas de clasificación. En particular, se implementan y comparan dos enfoques distintos:

1. **Análisis de Discriminante Lineal (LDA):** Utilizado para encontrar una combinación lineal de características que caractericen o separen dos o más clases de objetos.
2. **Árboles de Decisión:** Un modelo predictivo que mapea observaciones sobre un artículo a conclusiones sobre el valor objetivo del mismo.

El objetivo principal es evaluar el desempeño de estos algoritmos en la clasificación de datos tras un proceso de selección de características.

## Base de Datos

* **Origen:** Los datos utilizados en esta actividad provienen de un conjunto de datos previamente procesado, donde se han identificado las características más relevantes mediante métodos de regularización como Lasso.
* **Características:**
** **Entradas:** El modelo utiliza un conjunto de variables predictoras seleccionadas (almacenadas en `features_lasso.csv`) que representan las dimensiones con mayor poder explicativo para el fenómeno en estudio.
** **Variable Objetivo:** Una variable categórica que define la clase o etiqueta a la que pertenece cada instancia.
** **Procesamiento:** Se asume un preprocesamiento previo que incluye limpieza de datos y escalamiento de variables antes de la ejecución de los modelos.

## Índice de Archivos

A continuación, se describen los archivos contenidos en este folder:

1. **[main.ipynb](main.ipynb):** Jupyter Notebook que contiene el código fuente, el análisis de datos, la implementación de los modelos LDA y Árboles de Decisión, y las métricas de evaluación.
2. **[main.html](main.html):** Versión en formato HTML del notebook para una visualización rápida en navegadores.
3. **[main.pdf](main.pdf):** Documento PDF con el reporte final de la actividad.
4. **[features_lasso.csv](features_lasso.csv):** Archivo CSV que contiene el conjunto de características seleccionadas utilizadas para entrenar los modelos.
