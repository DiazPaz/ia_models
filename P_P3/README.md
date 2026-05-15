Este es un archivo README estructurado para tu proyecto de la Unidad 3, siguiendo el estilo y formato del documento que compartiste anteriormente.

---

# Proyecto Final - Unidad 3: Aprendizaje No Supervisado en Datos de Expresión Génica

## Descripción General

Este proyecto tiene como objetivo identificar subgrupos moleculares de pacientes con **Sarcoma Uterino (UCS)** mediante técnicas de aprendizaje no supervisado. Dado que el UCS es un grupo heterogéneo de tumores malignos, el análisis de expresión génica permite descubrir perfiles moleculares compartidos que no siempre son visibles en la clasificación clínica convencional.

Para lograrlo, se implementa una metodología que incluye la reducción de dimensionalidad mediante **PCA** y la aplicación de algoritmos de clustering (**K-means** y **Clustering Jerárquico**). Finalmente, los grupos encontrados se contrastan con variables clínicas y datos de supervivencia para evaluar su relevancia biológica.

## Base de Datos

* **Origen:** Los datos provienen de **The Cancer Genome Atlas (TCGA)**, obtenidos a través de la plataforma UCSC Xena.
* **Archivos utilizados:**
* `TCGA-UCS.star_tpm.tsv.gz`: Datos de expresión génica (60,660 genes × 57 muestras) en escala TPM.
* `TCGA-UCS.clinical.tsv.gz`: Variables clínicas y fenotípicas de los pacientes.
* `TCGA-UCS.survival.tsv.gz`: Datos de supervivencia global (OS y OS.time).


* **Variables de entrada:** Se utiliza la matriz de expresión génica completa, procesando la varianza para seleccionar los genes más informativos antes del clustering.

## Índice de Archivos

* [main.html](main.html) / [main.ipynb](main.ipynb): Notebook principal que contiene el preprocesamiento de datos, análisis de componentes principales (PCA), implementación de modelos de agrupamiento y visualización de resultados (dendrogramas, diagramas de dispersión y curvas de supervivencia).
* **[data_bases](data_bases/)**: Carpeta (local) que contiene los archivos comprimidos de expresión, clínica y supervivencia del TCGA.
* **[results](results/)**: Carpeta donde se almacenan las gráficas generadas y los archivos de salida del análisis.

## Metodología Aplicada

1. **Limpieza y Normalización:** Carga de datos y alineación de muestras entre los archivos de expresión y clínica.
2. **Reducción de Dimensionalidad:** Aplicación de PCA para manejar la alta dimensionalidad de los datos genómicos y capturar la mayor varianza posible.
3. **Clustering:** * **K-means:** Evaluación mediante el método del codo y el coeficiente de silueta.
* **Clustering Jerárquico:** Uso de vinculación de Ward para identificar la estructura del dendrograma.


4. **Caracterización Clínica:** Análisis estadístico de los clusters resultantes frente a estadios del tumor, diagnóstico histológico y supervivencia de los pacientes.

## Requisitos

Para ejecutar el código de este proyecto, se necesitan las siguientes librerías de Python:

```python
pandas
numpy
matplotlib
seaborn
scikit-learn
scipy

```

---

**Autor:** David Díaz Paz y Puente

**Institución:** Universidad de Monterrey (UDEM)

**Curso:** SC3314 Inteligencia Artificial

**Instructor:** Dr. Antonio Martínez Torteya
