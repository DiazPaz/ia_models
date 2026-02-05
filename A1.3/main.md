Nombre: David Díaz Paz y Puente | Matrícula: 650794

# A1.3 Solución de problemas y selección de características

## 1. Introducción

Para esta actividad partimos de la idea de que en ciencia de datos casi nunca recibimos conjuntos de datos "listos para usar": es normal encontrar variables categóricas mal codificadas, escalas inconsistentes, valores atípicos y hasta información redundante entre variables, lo cual puede afectar directamente la calidad y la interpretabilidad de cualquier modelo predictivo. 

Trabajaremos con una base de datos que incluye información demográfica y académica de estudiantes, además de sus calificaciones parciales y final. El objetivo del reporte es construir un modelo de regresión lineal múltiple para predecir la calificación final, enfrentando explícitamente los retos de usar datos reales y justificando la selección de variables explicativas más relevantes. Para lograrlo, se describirá el proceso de exploración, limpieza y transformación de datos, se analizarán relaciones entre variables para detectar colinealidad o redundancia, y finalmente se aplicará un criterio de selección de características que permita quedarse con un subconjunto informativo. Con esas variables seleccionadas, se entrenará y evaluará el modelo comparando su desempeño en entrenamiento y prueba (evitando fuga de datos), y se cerrará con una reflexión crítica sobre limitaciones y posibles mejoras.

```python
import pandas as pd

ruta = f'calificaciones.csv'
df = pd.read_csv(ruta)
```

## 2. Metodología

### 2.1 Exploración y comprensión del conjunto de datos

De acuerdo a la fuente, la base de datos utilizada reúne información real de estudiantes de nivel secundaria en dos escuelas de Portugal, y está diseñada para relacionar el desempeño académico con características del estudiante y de su contexto. Se compone de atributos (variables) que incluyen: (1) calificaciones por periodo (G1, G2) y la calificación final del año (G3), (2) datos demográficos (por ejemplo, edad y sexo), (3) factores sociales y familiares (como apoyo en casa o condiciones del entorno), y (4) variables escolares (relacionadas con hábitos o dinámica académica).

```python
print(df.head())
print(df.info())
```

```
  Escuela Sexo  Edad  HorasDeEstudio  Reprobadas Internet  Faltas  G1  G2  G3
0      GP    F    18               2           0       no       6   5   6   6
1      GP    F    17               2           0      yes       4   5   5   6
2      GP    F    15               2           3      yes      10   7   8  10
3      GP    F    15               3           0      yes       2  15  14  15
4      GP    F    16               2           0       no       4   6  10  10
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 395 entries, 0 to 394
Data columns (total 10 columns):
 #   Column          Non-Null Count  Dtype 
---  ------          --------------  ----- 
 0   Escuela         395 non-null    object
 1   Sexo            395 non-null    object
 2   Edad            395 non-null    int64 
 3   HorasDeEstudio  395 non-null    int64 
 4   Reprobadas      395 non-null    int64 
 5   Internet        395 non-null    object
 6   Faltas          395 non-null    int64 
 7   G1              395 non-null    int64 
 8   G2              395 non-null    int64 
 9   G3              395 non-null    int64 
dtypes: int64(7), object(3)
memory usage: 31.0+ KB
None
```

A partir del resumen del dataframe, se identifica que el dataset está compuesto por 395 observaciones y 10 variables, donde cada registro representa a un estudiante con características académicas, personales y de contexto escolar. Esta primera revisión permite clasificar las variables según su naturaleza, lo cual es clave para decidir cómo tratarlas dentro de un modelo de regresión lineal. A continuación se presenta la descripción para cada variable de acuerdo a la fuente de la base de datos ([archive.ics.uci.edu](https://archive.ics.uci.edu/dataset/320/student+performance)):

| Variable Name | Role | Type | Descripction |
|-----------|-----------|-----------|-----------
| school | Feature | Categorical | student's school (binary: 'GP' - Gabriel Pereira or 'MS' - Mousinho da Silveira) | 
| sex | Feature | Binary | student's sex (binary: 'F' - female or 'M' - male) |
| age | Feature | Integer |  student's age (numeric: from 15 to 22)  |
| studytime | Feature | Integer   | weekly study time (numeric: 1 - <2 hours, 2 - 2 to 5 hours, 3 - 5 to 10 hours, or 4 - >10 hours) |
| failures | Feature | Integer   | number of past class failures (numeric: n if 1<=n<3, else 4) |
| internet | Feature | Binary   | Internet access at home (binary: yes or no)
| absences | Feature | Integer   | number of school absences (numeric: from 0 to 93) |
| G1 | Target | Categorical |   first period grade (numeric: from 0 to 20) |
| G2 | Target | Categorical |   second period grade (numeric: from 0 to 20) |
| G3 | Target | Categorical |   final grade (numeric: from 0 to 20, output target) |

### 2.2 Procesamiento de los datos

Después de haber analizado y entendido la naturaleza y estructura de los datos, el siguiente paso consiste en transformarlos de tal manera que puedan ser empleados directamente en un modelo de regresión lineal múltiple. Este procesamiento incluye preparar tanto la variable objetivo (G3) como las variables predictoras, asegurando que todas las variables categóricas se conviertan en formatos numéricos, ya que muchos algoritmos de ML requieren que las entradas sean numéricas. Existen distintas técnicas para transformar variables categóricas en numéricas, como ordinal encoding, one-hot encoding y label encoding. Cada una es apropiada para diferentes tipos de variables categóricas. 

En este caso, se usará **One-Hot Encoding** para aquellas variables categóricas sin orden intrínseco entre categorías (por ejemplo, "Sexo" o "Escuela"); este método crea una nueva columna binaria (0 o 1) por cada categoría de la variable categórica, permitiendo que el modelo trate cada categoría de manera independiente sin imponer un orden artificial. Por otro lado, se considerará **Label Encoding** para variables binarias como "Internet", donde solo hay dos posibles valores y el orden no es relevante.

```python
# Crear una copia del dataframe original
df_procesado = df.copy()

# Obtener las columnas categóricas
categorical_cols = df_procesado.select_dtypes(include='object').columns
print("Columnas categóricas:", categorical_cols.tolist())

# Codificación de variables categóricas utilizando One-Hot Encoding
df_procesado = pd.get_dummies(df_procesado, columns=['Escuela', 'Sexo'], drop_first=True, dtype=int)

# Codificación de la variable binaria 'Internet' usando Label Encoding
df_procesado['Internet'] = df_procesado['Internet'].map({'no': 0, 'yes': 1})

# Mostrar las primeras filas después de la codificación
print("\nPrimeras filas después de la codificación:")
print(df_procesado.head())
```

```
Columnas categóricas: ['Escuela', 'Sexo', 'Internet']

Primeras filas después de la codificación:
   Edad  HorasDeEstudio  Reprobadas  Internet  Faltas  G1  G2  G3  Escuela_MS  \
0    18               2           0         0       6   5   6   6           0   
1    17               2           0         1       4   5   5   6           0   
2    15               2           3         1      10   7   8  10           0   
3    15               3           0         1       2  15  14  15           0   
4    16               2           0         0       4   6  10  10           0   

   Sexo_M  
0       0  
1       0  
2       0  
3       0  
4       0  
```

Con la codificación completada, las variables categóricas ahora se encuentran en formato numérico. A continuación, es necesario separar las variables predictoras de la variable objetivo. En este análisis, el objetivo es predecir la calificación final del estudiante (G3) a partir de las demás características disponibles. Por lo tanto, todas las variables restantes (Edad, HorasDeEstudio, Reprobadas, Internet, Faltas, G1, G2, Escuela_MS, Sexo_M) serán consideradas como variables predictoras (X), mientras que G3 será nuestra variable objetivo (y). 

```python
# Separar la variable objetivo (G3) de las características (X)
X = df_procesado.drop('G3', axis=1)
y = df_procesado['G3']

# Verificar las dimensiones de las nuevas matrices
print("Dimensiones de X:", X.shape)
print("Dimensiones de y:", y.shape)

# Mostrar los nombres de las columnas en X
print("\nVariables predictoras (X):")
print(X.columns.tolist())
```

```
Dimensiones de X: (395, 9)
Dimensiones de y: (395,)

Variables predictoras (X):
['Edad', 'HorasDeEstudio', 'Reprobadas', 'Internet', 'Faltas', 'G1', 'G2', 'Escuela_MS', 'Sexo_M']
```

### 2.3 División de datos y normalización

En la construcción de modelos de ML, es esencial dividir el conjunto de datos en un conjunto de entrenamiento y un conjunto de prueba. Esta división permite entrenar el modelo sobre un subconjunto de los datos y posteriormente evaluar su desempeño sobre datos nuevos y no vistos, con el fin de evitar el sobreajuste y asegurar que el modelo generalice adecuadamente. Una convención común es asignar aproximadamente el 70-80% de los datos para entrenamiento y el 20-30% restante para prueba. En este caso, se empleará una división 80-20.

Luego de dividir los datos, es importante realizar una normalización de las características. Los datos tienen diferentes escalas: por ejemplo, la edad puede variar entre 15 y 22, mientras que el número de faltas puede ir de 0 a 93. Utilizar variables con diferentes escalas puede llevar a que el modelo asigne de manera implícita mayor peso a aquellas con rango más amplio. Normalizar las características asegura que todas estén en un rango o escala común, facilitando la interpretación de los coeficientes del modelo de regresión lineal y permitiendo que la regularización aplicada (Lasso) funcione de forma más efectiva.

Un método comúnmente utilizado es la estandarización, que transforma cada variable para que tenga media 0 y desviación estándar 1. Es crucial aplicar la normalización después de la separación de los datos. De lo contrario, estaríamos introduciendo información del conjunto de prueba en el proceso de escalado del conjunto de entrenamiento (lo cual se denomina "fuga de datos" o data leakage), generando estimaciones de desempeño demasiado optimistas.

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Dividir el conjunto de datos en entrenamiento y prueba (80-20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Mostrar las dimensiones de cada conjunto
print("Datos de entrenamiento (X_train):", X_train.shape)
print("Datos de prueba (X_test):", X_test.shape)
print("Etiquetas de entrenamiento (y_train):", y_train.shape)
print("Etiquetas de prueba (y_test):", y_test.shape)

# Normalización de las características
scaler = StandardScaler()

# Ajustar el escalador únicamente con los datos de entrenamiento y transformar ambos conjuntos
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nPrimeras 5 filas de X_train después de la normalización:")
print(X_train_scaled[:5])
```

```
Datos de entrenamiento (X_train): (316, 9)
Datos de prueba (X_test): (79, 9)
Etiquetas de entrenamiento (y_train): (316,)
Etiquetas de prueba (y_test): (79,)

Primeras 5 filas de X_train después de la normalización:
[[-0.72335518 -0.05066952  0.03333062 -1.19958042 -0.07398234 -0.17667479
   0.06896027  0.28373835  0.82967869]
 [ 1.52854923 -1.06788906 -0.42073827 -1.19958042  0.48479825  1.78659652
   1.27374164  0.28373835  0.82967869]
 [-1.29605619  0.47044385 -0.42073827  0.83392417  0.58673355  0.47691815
   0.45012063  0.28373835  0.82967869]
 [-0.15065417  0.47044385 -0.42073827  0.83392417 -0.37785293 -0.59734029
   0.06896027 -3.52486989 -1.20524398]
 [ 1.52854923 -0.04116473 -0.42073827  0.83392417  1.60415201  1.78659652
   1.27374164  0.28373835 -1.20524398]]
```

### 2.4 Análisis exploratorio de datos (EDA)

Antes de proceder con la construcción del modelo, es fundamental realizar un análisis exploratorio de los datos. El objetivo de esta etapa es entender las relaciones y distribuciones de las variables, identificar posibles patrones, correlaciones y detectar características que podrían ser redundantes o poco informativas. Mediante técnicas de visualización e índices numéricos, podemos validar nuestras intuiciones sobre el dataset y tomar decisiones informadas para el modelado.

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Mostrar correlaciones entre variables (antes de dividir los datos)
plt.figure(figsize=(10, 8))
sns.heatmap(df_procesado.corr(), annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("Matriz de correlación entre las variables")
plt.show()

# Distribución de la variable objetivo
plt.figure(figsize=(6, 4))
sns.histplot(df_procesado['G3'], kde=True, bins=20, color='steelblue')
plt.title("Distribución de la calificación final (G3)")
plt.xlabel("Calificación final (G3)")
plt.ylabel("Frecuencia")
plt.show()
```

*[Gráficas generadas por el código]*

A partir de la matriz de correlación, podemos observar que las variables G1 y G2 tienen una correlación muy alta con la variable objetivo G3. Esto es esperado porque G1 y G2 son calificaciones de períodos previos, por lo que guardan una relación directa con la calificación final. Sin embargo, también se puede notar que G1 y G2 tienen una alta correlación entre sí, indicando cierto grado de multicolinealidad. Por su parte, variables como Edad, HorasDeEstudio, Reprobadas, Internet, Faltas, Escuela_MS y Sexo_M presentan correlaciones más débiles con G3. De estas, la variable Reprobadas muestra una correlación negativa moderada con G3, lo que sugiere que un mayor número de materias reprobadas en el pasado tiende a asociarse con calificaciones finales más bajas.

La distribución de la variable objetivo muestra que la mayoría de los estudiantes obtienen calificaciones finales entre 8 y 14 puntos (en una escala de 0 a 20). Existen relativamente pocos estudiantes con calificaciones muy bajas (cercanas a 0) o extremadamente altas (cercanas a 20). Esta distribución no es simétrica, y presenta una ligera tendencia hacia valores intermedios-altos, siendo esto positivo puesto a que no hay valores atípicos extremos ni distribuciones sesgadas que puedan afectar el rendimiento del modelo de regresión lineal.

### 2.5 Selección de características con Lasso

Una vez que hemos preparado y explorado los datos, el siguiente paso es seleccionar las características más relevantes para el modelo. En este análisis se utilizará **Regresión Lasso (Least Absolute Shrinkage and Selection Operator)**, un método de regularización que penaliza la magnitud de los coeficientes, permitiendo que algunos se reduzcan a cero. De esta forma, Lasso no solo ayuda a prevenir el sobreajuste, sino que además realiza una selección automática de características, eliminando aquellas que no aportan información relevante al modelo.

El parámetro clave en Lasso es lambda (α), que controla la intensidad de la regularización. Un valor bajo de α implica poca regularización, mientras que un valor alto puede llevar a que muchos coeficientes se reduzcan a cero. Para determinar el valor óptimo de α, se utiliza **validación cruzada (cross-validation)**, la cual prueba diferentes valores y selecciona el que minimiza el error de predicción. En este análisis, se usará validación cruzada de 5 pliegues (5-fold).

```python
from sklearn.linear_model import LassoCV

# Definir un rango de valores para el parámetro alpha (lambda)
alphas = [0.001, 0.01, 0.1, 1, 10, 100]

# Crear un modelo Lasso con validación cruzada de 5 pliegues
lasso_cv = LassoCV(alphas=alphas, cv=5, random_state=42, max_iter=10000)

# Ajustar el modelo con los datos de entrenamiento normalizados
lasso_cv.fit(X_train_scaled, y_train)

# Mostrar el valor óptimo de alpha seleccionado por validación cruzada
print("Valor óptimo de alpha (λ):", lasso_cv.alpha_)

# Obtener los coeficientes del modelo
coeficientes = pd.DataFrame({
    'Variable': X.columns,
    'Coeficiente': lasso_cv.coef_
})

# Filtrar las variables con coeficiente distinto de cero
coeficientes_seleccionados = coeficientes[coeficientes['Coeficiente'] != 0]

print("\nCoeficientes seleccionados por Lasso:")
print(coeficientes_seleccionados)

# Visualizar los coeficientes seleccionados
plt.figure(figsize=(10, 6))
plt.barh(coeficientes_seleccionados['Variable'], coeficientes_seleccionados['Coeficiente'], color='teal')
plt.xlabel("Valor del coeficiente")
plt.ylabel("Variable")
plt.title("Coeficientes seleccionados por Lasso")
plt.axvline(x=0, color='red', linestyle='--')
plt.show()
```

```
Valor óptimo de alpha (λ): 0.001

Coeficientes seleccionados por Lasso:
         Variable  Coeficiente
0            Edad    -0.246926
2      Reprobadas    -0.197614
3        Internet     0.147625
4          Faltas     0.042863
5              G1     0.205669
6              G2     0.808098
7      Escuela_MS    -0.001988
8          Sexo_M     0.179324
```

*[Gráfica de coeficientes generada por el código]*

El modelo Lasso seleccionó un valor óptimo de λ = 0.001, lo que indica que se aplicó una regularización leve, manteniendo la mayoría de las características del modelo. De las 9 variables predictoras iniciales, 8 fueron retenidas con coeficientes distintos de cero, mientras que HorasDeEstudio fue eliminada (coeficiente = 0), lo que sugiere que esta variable no aporta información relevante para predecir G3 cuando se controla por las demás variables.

Los coeficientes obtenidos permiten interpretar la contribución de cada característica a la predicción de la calificación final:

- **G2 (0.808)**: La calificación del segundo periodo es el predictor más importante. Un aumento en G2 se asocia con un aumento significativo en G3.
- **Edad (-0.247)**: Los estudiantes de mayor edad tienden a obtener calificaciones finales ligeramente más bajas.
- **G1 (0.206)**: La calificación del primer periodo también contribuye positivamente a G3, aunque en menor medida que G2.
- **Reprobadas (-0.198)**: Un mayor número de materias reprobadas en el pasado se asocia con calificaciones finales más bajas.
- **Sexo_M (0.179)**: Ser hombre se asocia con una calificación final ligeramente más alta en comparación con mujeres.
- **Internet (0.148)**: Tener acceso a internet en casa se asocia con una mejora en la calificación final.
- **Faltas (0.043)**: Sorprendentemente, un mayor número de faltas muestra una relación positiva con G3, aunque esta contribución es muy pequeña.
- **Escuela_MS (-0.002)**: Pertenecer a la escuela Mousinho da Silveira en lugar de Gabriel Pereira tiene un efecto prácticamente nulo en la calificación final.

### 2.6 Entrenamiento y evaluación del modelo

Con las características seleccionadas, se ha entrenado un modelo de regresión lineal con regularización Lasso. El modelo fue ajustado utilizando los datos de entrenamiento normalizados y posteriormente se evaluó su desempeño tanto en el conjunto de entrenamiento como en el conjunto de prueba. Esta evaluación es crucial para determinar si el modelo es capaz de generalizar bien a datos nuevos o si presenta sobreajuste.

Para evaluar el desempeño del modelo se utilizaron dos métricas principales:
- **Error Cuadrático Medio (MSE)**: Mide el promedio de los errores al cuadrado entre las predicciones y los valores reales. Un MSE bajo indica que el modelo tiene buena precisión.
- **Coeficiente de determinación (R²)**: Indica qué porcentaje de la variabilidad en la variable objetivo es explicado por el modelo. Un R² cercano a 1 indica un muy buen ajuste.

```python
from sklearn.metrics import mean_squared_error, r2_score

# Realizar predicciones en el conjunto de prueba
y_test_pred = lasso_cv.predict(X_test_scaled)
y_train_pred = lasso_cv.predict(X_train_scaled)

# Calcular métricas de desempeño: MSE y R²
mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)

# Mostrar las métricas de desempeño
print("\nMétricas de evaluación:")
print(f"MSE en entrenamiento: {mse_train:.2f}")
print(f"MSE en prueba: {mse_test:.2f}")
print(f"R² en entrenamiento: {r2_train:.2f}")
print(f"R² en prueba: {r2_test:.2f}")

# Visualización: comparación entre valores reales y predicciones
plt.figure(figsize=(10, 5))

# Graficar los valores reales frente a las predicciones para los datos de prueba
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_test_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', label='Línea ideal')
plt.title("Valores reales vs. Predicciones (Prueba)")
plt.xlabel("Valores reales")
plt.ylabel("Predicciones")
plt.legend()

# Graficar los residuos (diferencia entre predicción y valor real)
residuals = y_test - y_test_pred
plt.subplot(1, 2, 2)
plt.scatter(y_test_pred, residuals)
plt.hlines(y=0, xmin=y_test_pred.min(), xmax=y_test_pred.max(), color='r', linestyle='--')
plt.title("Residuos vs. Predicciones (Prueba)")
plt.xlabel("Predicciones")
plt.ylabel("Residuos")

plt.tight_layout()
plt.show()
```

*[Gráficas generadas por el código]*

El Error Cuadrático Medio (MSE) tanto para el entrenamiento como para la prueba muestran ser bajos; la diferencia promedio entre los valores reales y las predicciones del modelo es relativamente poca, lo que significa que las predicciones están cerca de los valores reales. El MSE del conjunto de prueba indica que el modelo tiene un ligeramente mayor error en la predicción de los datos de prueba. Esto es normal y esperado, ya que el modelo generalmente tiene un rendimiento ligeramente peor en los datos no vistos.

El coeficiente de determinación R² mide qué tan bien el modelo explica la variabilidad de la variable dependiente (G3) con base en las variables independientes. Un R² de 0.84 en entrenamiento significa que el modelo explica el 84% de la variabilidad de la calificación final en los datos de entrenamiento. Esto es un buen indicador de que el modelo se ajusta bien a los datos de entrenamiento. En prueba indica que el modelo explica el 78% de la variabilidad en los datos de prueba. Aunque es ligeramente más bajo que el R² en entrenamiento (0.84), sigue siendo un buen desempeño para un modelo predictivo.

## 3. Conclusiones y reflexiones

### 3.1 Reflexión

El modelo generado en base a la regularización Lasso muestra relaciones que considieraría contraintuitivas, como por ejemplo: incrementar el número de faltas en promedio aumentaría (poco) tus probabilidades de obtener una mayor calificación final. Otro ejemplo es que dentro de los coeficientes seleccionados no se encuentra la variable explicativa de HorasDeEstudio, y una razón a esto la podemos ver incluso dentro de la matriz de correlación donde la correlacionalidad entre la variable HorasDeEstudio es muy baja en relación a otras. 

Finalmente, cada una de las variables tiene un impacto diferente en la calificación final (G3). El modelo Lasso ha identificado que algunas características positivas (como las calificaciones de los periodos anteriores) contribuyen significativamente a una mayor calificación final, mientras que otras variables negativas (como la edad y las materias reprobadas) están asociadas con un rendimiento más bajo. El impacto de características como sexo o acceso a Internet sugiere que el contexto educativo y social de cada estudiante juega un papel importante en el desempeño académico.

### 3.2 Conclusión

El proceso de análisis realizado ha permitido construir un modelo de regresión lineal múltiple con regularización Lasso para predecir la calificación final (G3) de los estudiantes. A través de la preparación de datos y la selección de características realizadas, hemos logrado mejorar la precisión del modelo al eliminar variables irrelevantes y reducir la complejidad del modelo. La regularización Lasso permitió seleccionar de manera eficiente las variables más relevantes, evitando problemas de multicolinealidad y sobreajuste, lo que resultó en un modelo más robusto y con mejor capacidad de generalización.

La preparación de los datos juega un papel fundamental en el éxito del análisis. La normalización de las características fue esencial para garantizar que todas las variables estuvieran en la misma escala, lo que es particularmente importante en modelos de regularización como Lasso. Además, la división adecuada entre conjuntos de entrenamiento y prueba, junto con la validación cruzada, contribuyó a evitar la fuga de datos y a evaluar el modelo de manera más confiable.

En conclusión, este proceso ha proporcionado valiosos aprendizajes sobre cómo construir y evaluar modelos predictivos, y la importancia de realizar un análisis detallado y riguroso durante todas las etapas, desde la preparación de los datos hasta la evaluación del modelo. Las limitaciones actuales brindan una excelente oportunidad para seguir mejorando y explorando nuevas técnicas que fortalezcan el modelo en el futuro.
