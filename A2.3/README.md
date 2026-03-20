# A2.3 Modelos de ensamble, SVM y Redes Neuronales

## Descripción General

Esta actividad consiste en analizar el comportamiento y resolver un problema de clasificación mediante el entrenamiento y comparación de distintos modelos avanzados: **Random Forest**, **Boosting (AdaBoost)**, **Support Vector Machine (SVM)** y **Redes Neuronales (MLP)**. El objetivo es evaluar el desempeño de estos enfoques para diagnosticar el estrato socioeconómico de una familia basándose en las condiciones de su vivienda.

## Base de Datos

  * **Origen:** Los datos provienen de la **Encuesta Nacional de Ingresos y Gastos de los Hogares (ENIGH) 2024**, específicamente de la muestra correspondiente al estado de **Nuevo León**.
  * **Variable de salida:** Se clasifica el estrato socioeconómico en cuatro categorías: **Bajo (1)**, **Medio bajo (2)**, **Medio alto (3)** y **Alto (4)**.
  * **Variables de entrada:** El conjunto de datos consta de 3,767 observaciones y utiliza 44 variables predictoras previamente seleccionadas mediante el método **LASSO**, las cuales se encuentran almacenadas en el archivo `features_lasso.csv`.

## Índice de Archivos

  * [main.ipynb](https://www.google.com/search?q=main.ipynb): Notebook principal con el desarrollo del código, optimización de hiperparámetros (mediante `RandomizedSearchCV` y `GridSearchCV`) y evaluación de los modelos implementados.
  * [main.pdf](https://www.google.com/search?q=main.pdf): Reporte detallado de la actividad en formato PDF que incluye la introducción, metodología, resultados comparativos y conclusiones.
  * [main.html](https://www.google.com/search?q=main.html): Versión exportada en HTML para visualización rápida del análisis.
  * [features\_lasso.csv](https://www.google.com/search?q=features_lasso.csv): Dataset que contiene las características seleccionadas utilizadas para el entrenamiento (70%) y prueba (30%) de los modelos.
