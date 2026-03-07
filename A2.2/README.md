# A2.2 LDA y Árboles de Decisión

## Descripción General

Esta actividad consiste en resolver un problema de clasificación mediante el desarrollo y comparación de dos modelos distintos: **Linear Discriminant Analysis (LDA)** y **Árboles de Decisión**. El objetivo es analizar y determinar cuál de ambos enfoques resulta más adecuado para caracterizar familias en estratos socioeconómicos basándose en las condiciones de su vivienda.

## Base de Datos

* **Origen:** Los datos provienen de la **Encuesta Nacional de Ingresos y Gastos de los Hogares (ENIGH) 2024**, específicamente de la muestra correspondiente al estado de **Nuevo León**.
* **Características:** * **Variable de salida:** Se clasifica el estrato socioeconómico en tres categorías: **Bajo**, **Medio** y **Alto** (estas dos últimas resultantes de la fusión de las clases "Medio alto" y "Alto" para simplificar el análisis).
* **Variables de entrada:** El conjunto de datos utiliza variables predictoras previamente seleccionadas mediante el método **LASSO**, las cuales se encuentran almacenadas en el archivo `features_lasso.csv`.

## Índice de Archivos

* [main.ipynb](main.ipynb): Notebook principal con el desarrollo del código, entrenamiento de modelos y visualización de resultados.
* [main.pdf](main.pdf): Reporte detallado de la actividad en formato PDF que incluye la introducción, metodología y conclusiones.
* [main.html](main.html): Versión exportada en HTML para visualización rápida del análisis.
* [features_lasso.csv](features_lasso.csv): Dataset que contiene las características seleccionadas utilizadas para el entrenamiento de los modelos LDA y de árboles de decisión.
