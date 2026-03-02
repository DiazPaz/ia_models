# A2.1 Regresión logística y validación cruzada

**Autor:** David Díaz Paz y Puente | **Matrícula:** 650794

## Descripción del Proyecto

Este directorio contiene la resolución de la actividad **A2.1**, en la cual se aborda un problema de **clasificación supervisada** para predecir el estrato socioeconómico de los hogares (`est_socio`). 

A diferencia de etapas anteriores donde se estimaba el ingreso per cápita de forma continua, en este proyecto el problema se reformuló para predecir a qué categoría pertenece un hogar de acuerdo con sus características estructurales y condiciones de vivienda. Las clases a predecir son cuatro categorías ordinales:
1. **Bajo**
2. **Medio bajo**
3. **Medio alto**
4. **Alto**

El objetivo principal es entrenar y evaluar un modelo de **Regresión Logística** robusto, asegurando su capacidad de generalización mediante validación cruzada y realizando una interpretación profunda de sus métricas, umbrales y coeficientes.

---

## Estructura del Directorio

* `main.ipynb`: Jupyter Notebook principal que contiene todo el código fuente. Incluye desde la carga y separación de los datos, hasta el entrenamiento mediante *Pipelines*, validación cruzada, evaluación en el conjunto de prueba y el análisis visual/estadístico del modelo.
* `features_lasso.csv`: Conjunto de datos utilizado para el entrenamiento y prueba del modelo. Contiene las características de las viviendas previamente seleccionadas.

---

## Metodología y Técnicas Aplicadas

El desarrollo del modelo en [`main.ipynb`](main.ipynb) sigue un flujo de trabajo estructurado de Machine Learning:

1. **Preparación de Datos:** * División del dataset en conjuntos de entrenamiento (80%) y prueba (20%) mediante muestreo estratificado (`stratify=y`) para mantener el balance original de las clases.
2. **Entrenamiento y Validación Cruzada:**
   * Construcción de un `Pipeline` integrando estandarización (`StandardScaler`) y el modelo clasificador (`LogisticRegression`).
   * Evaluación de la estabilidad del modelo mediante **Validación Cruzada Estratificada (5-folds)** utilizando métricas como *Accuracy*, *F1-macro* y *F1-weighted*.
3. **Evaluación Final en Prueba:**
   * Generación del reporte de clasificación y Matriz de Confusión en datos no vistos.
   * **Análisis de Umbrales:** Estudio del balance entre Precisión y Sensibilidad (*Recall*) al modificar el punto de corte (threshold) específicamente para aislar a la clase vulnerable ("Bajo").
   * **Curvas ROC y AUC:** Evaluación del rendimiento global del modelo utilizando una estrategia *One-vs-Rest (OvR)*.
4. **Interpretación del Modelo (Caja Blanca):**
   * Extracción y análisis de los **coeficientes** de la regresión logística para identificar qué características arquitectónicas, económicas o de servicios (ej. `estim_pago`, `drenaje_2`, `eli_basura_4`) actúan como "drivers" de pobreza o riqueza en la toma de decisión del algoritmo.
