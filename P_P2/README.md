# P_P2 Proyecto Final - Unidad 2

## Descripción General

Este proyecto consiste en analizar el comportamiento y resolver un problema de clasificación mediante el entrenamiento y comparación de distintos modelos predictivos: Regresión Logística, Análisis Discriminante Lineal (LDA), Random Forest, Gradient Boosting, Support Vector Machine (SVM) y Redes Neuronales (MLP). El objetivo principal es evaluar el desempeño de estos enfoques para diagnosticar el estrato socioeconómico de una familia en México basándose en las características financieras y la infraestructura de su hogar.

## Base de Datos

* **Origen:** Los datos provienen de la Encuesta Nacional de Ingresos y Gastos de los Hogares (ENIGH) 2024.
* **Variable de salida:** Se clasifica la variable objetivo (`est_socio`) en cuatro categorías de estrato socioeconómico: Bajo (1), Medio bajo (2), Medio alto (3) y Alto (4).
* **Variables de entrada:** El conjunto de datos consta de 3,767 observaciones y utiliza 44 variables predictoras que fueron previamente seleccionadas mediante el método LASSO, las cuales se encuentran listas para usarse en el archivo `base_ENIGH_lasso.csv`.

## Índice de Archivos

* [main.ipynb](main.ipynb): Notebook principal con la exploración de datos, desarrollo del código, optimización de hiperparámetros (mediante `RandomizedSearchCV` y `GridSearchCV`) y evaluación de los modelos implementados.
* [main.pdf](main.pdf): Reporte detallado del proyecto en formato PDF que incluye el contexto de los datos, metodología, resultados comparativos y conclusiones.
* [main.html](main.html): Versión exportada en HTML para visualización rápida de las gráficas y el análisis general.
* [base_ENIGH_lasso.csv](base_ENIGH_lasso.csv): Dataset que contiene las características procesadas y utilizadas para el entrenamiento y prueba de los modelos.
* [base_ENIGH.csv](base_ENIGH.csv): Dataset original y completo de la ENIGH previo al proceso de selección de atributos.
* [diccionario_datos.csv](diccionario_datos.csv) y carpeta [catalogos](catalogos/): Archivos de referencia y tablas que contienen la descripción y codificación detallada de las variables de la encuesta.
