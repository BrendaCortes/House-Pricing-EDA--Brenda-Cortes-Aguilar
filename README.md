# House-Pricing-EDA--Brenda-Cortes-Aguilar
Proyecto U1_  EDA Completo 

El siguiente proyecto tomará como set de datos un conjunto de información sobre los precios de las casas.
El set de datos tiene datos nulos, datos tipo objeto, anomalías, inconsistencias, por lo que es necesario conocimientos previos de EDA para poder prepararlo.
El objetivo es preparar un set de datos para poder usarlo posteriormente en un modelo de ML, no basta con solo volverlo numérico, necesitamos tener conocimientos sobre los datos mismos.
El resultado de esta exploración debe ser:
1. Gráficas y conclusiones acerca de los datos.
2. Eliminación de datos nulos, anomalías e inconsistencias.
3. Generación de un dataframe listo para usarse sobre un modelo de ML (numérico), con las variables, cuya correlación es más fuerte con la variable objetivo, identificadas.
Usando como referencia el set de datos “casas_dataset.csv”

Análisis y exploración.

Paso 1: Importar las librerías que se van a utilizar y mostrar las primeras y últimas 5 filas del dataset.

Es importante importar las librerías que nos pueden ayudar a realizar el análisis exploratorio y cargar el set de datos correspondiente. 

Paso 2: Mostrar información sobre el dataset, qué tipos son, cuántos nulos hay, datos estadísticos.

Realizar una exploración previa de los datos nos ayuda a conocer nuestro data set, de este modo poder ir detectando que tanto trabajo tendremos que realizar para limpiarlo, por ejemplo:
Dentro del listado de la información general de la data set, podemos observar que nos enfrentamos a un data set que cuenta con 81 columnas y 1460 registros. Podemos observar que el data set contiene columnas de tipo "Object", "Int" y "float" y que más del 50% de nuestro data set, está representado por columnas tipo objeto.  Al listar los datos nulos que existen por columna, podemos darnos cuenta de que, si excluimos las columnas que están nulas casi en su totalidad, en general, no se presenta número elevado en datos nulos y las columnas con un mayor dato, pueden ser consideradas para eliminarse totalmente. Por otro lado, al observar los datos estadísticos, podemos encontrar información esencial para entender las características de los datos, misma que nos apoyara a tomar decisiones acerca del cómo vamos a preparar los datos.  De este modo si observamos los números que nos muestra los cuartiles, podemos predecir que nos encontraremos con demasiados atípicos en alguna columna del data set. 

Paso 3: Mostrar información sobre las variables "objeto" y revisar si se pueden categorizar

Observamos que en la gran mayoría de las columnas, presentan valores únicos que van desde 1-6 y algunas cuantas más con un número más grande, pero razonable, es decir que podríamos considerar que la gran mayoría de ella es candidata para la categorización. 
Sin embargo, es muy claro que las columnas representan características de un "casa" u "vivienda", pero dentro de los registros se expresan demasiadas abreviaturas, que, posiblemente más adelante, podrían llegar a representar un reto enorme.

Paso 4: Mostrar información sobre las correlaciones (variables numéricas), la variable/columna objetivo es "SalePrice".

Al imprimir la matriz de correlación podemos darnos cuenta de que es poco un complicado visualizar con claridad por la cantidad de valores que posee, sin embargo, ya que conocemos cuál es nuestra variable objetivo, podemos filtrarla para visualizar mejor la correlación que existe entre la columna objetivo y todas las demás. 
Existen varias columnas que presentan una correlación positiva fuerte, una de ella es 'OverallQual', la cual nos podría indicar que a medida que la calidad general de la casa aumenta, el precio de venta también tiende a aumentar.


Paso 5: Muestra de gráficas de las variables numéricas y categóricas.

Gracias a la visualización de las gráficas podemos observar que nos enfrentaremos con datos atípicos en la gran mayoría de las columnas. Incluso en las gráficas de "BsmtFinSF2", "LowQualFinSF", "BsmtHalfBath", "KitchebvGr", "EnclosedPorch", "3SsnPorch", "ScreenPorch", "PoolArea" y "MiscVal" no se esta graficando la caja que representa el IQR, solo se ven puntos dispersos, lo que podria significar que la mayoría de los datos están muy concentrados en un rango estrecho, y los datos atípicos están muy separados de esta concentración. 
Así que debería de analizar más a fondo para determinar si son errores que deben corregirse, o si son datos legítimos, pero inusuales, que proporcionan información valiosa para la variable que deseamos predecir. 
Por otro lado, al desplegar los histogramas de las columnas categóricas, podemos observar que existen columnas que presentan una carga de datos masiva en una de muchas categorías, dejando prácticamente en 0 los demás valores que presentan, por ejemplo "Street", "Utilities", "LandSlope", "Condition 1", "Condition 2", "RoofMatl", "Heating".
Habría que valorar si esta carga masiva a un valor en especial es normal, o si existen inconsistencias y que por esa razón los demás valores no contengan datos. 

Tratamiento de los datos.

Paso 1: Crear un límite para eliminar los datos nulos, mostrar las variables que harán eliminación a sus nulos y mostrar los conteos antes, eliminar los datos nulos y mostrar los conteos de nuevo.

Como ya se había comentado anteriormente, dentro de nuestro data set contamos con pocas columnas con datos nulos, y la mayoría de estas no superan el 5% recomendado para eliminarlos. Así que se realizara la eliminación de dichos registros en todas las columnas que no excedan el límite.
Por otro, contamos con las columnas "PoolQC", "MiscFeature", "Alley" y "Fence" columnas categóricas que tienen más del 50% de datos nulos, por lo cual es recomendable checar la correlación que tienen con la variable objetivo y si esta no es fuerte, podríamos considerar eliminarlas por completo.     
Podemos notar que no existe una correlación fuerte con nuestra variable objetivo, por lo cual, vamos a eliminar la columna completa. 

Paso 2: Si existen nulos aún, dependiendo de la gráficas anteriores, determinar si hay que imputar por medio de la moda, la mediana o la media; realizar la imputación.

Para la columna categórica no es posible utilizar otro dato estadístico que no sea la moda, por lo cual, "FireplaceQu", "GarageFinish", "GarageQual" serán imputados con la moda.    
Por otro lado, "GarageYrBlt", "LotFrontage", parece ser simetrica, por cuál, imputaremos con la media. 

 Paso 3: Analizar los datos numéricos, determinar si hay anomalías y utilizar el rango intercuartílico para tratarlos, mostrar gráficos antes y después del tratamiento, debe verse si la distribución se vio afectada.

Aplicaremos el método del rango intercuartílico a todas las columnas numéricas que reflejan datos atípicos en se gráfica, exceptuando las columnas "LowQualFinSF", "BsmtHalfBath", "KitchebvGr", "EnclosedPorch", "3SsnPorch", "ScreenPorch", "PoolArea" y "MiscVal", las cuales ni siquiera presentan los cuartiles y de aplicar este método en estas columnas, perderíamos la totalidad de los registros de nuestro data set.  
Es importante que conservemos estas columnas, ya que necesitamos profundizar en el análisis si estos datos son producto de un error o son importantes para la predicción de la variable o se pueden corregir aplicando algún otro método.

Paso 4: En base a la exploración previa, determinar si alguna columna puede ser categorizada y realizar la categorización.

Ya que la gran mayoría de columnas presentan valores únicos bajos, si es viable la categorización en la gran mayoría, en aquellas columnas cuyos valores únicos rondan de entre 10 a los 30 es complicado considerar si es posible categorizar para disminuir el número, porque la columna cuenta con abreviaturas las cuales complican el proceso.

Paso 5: De las columnas categorizadas, buscar si hay inconsistencias, en caso de que las haya, hay que mostrarlas y tratarlas.

Esta parte es una de las más complicadas del análisis, el data set está muy abreviado, y con él no se proporcionó diccionarios para saber el significado. 
Así que a simple vista es muy complicado determinar si alguno de los valores presentado es inconsistente, retomando un poco los histogramas que observamos anteriormente, existen columnas que presentan pocos datos en los valores categóricos que presentan, las cargas de datos recen en una sola categoría, por lo cual se debería de analizar si a fondo si estos comportamientos son normales o nos encontramos ante un data set con datos que no son de calidad. 
Por ejemplo, en las columnas donde esperaríamos únicamente un (Y o N) como un "YES" o "NO", columna "PavedDrive" se cuela una tercera categoria denominada "P", pero a ciencia cierta no podemos determinar que estamos ante una inconsistencia porque este dato podría tal vez representar un "Parcialmente".
Decidí conservar las categorías que se presentan esperando que el proceso de transformación a numérico no se complique más adelante. 

Manipulación y preparación de los datos.

Paso 1: Mostrar la matriz de correlación de nuevo, identificar las columnas que más esté correlacionadas con "SalePrice", mostrar numéricamente las 10 variables que estén correlacionadas más fuertemente a la variable objetivo.

En este paso podemos identificar algunas de las variables más relacionada que nos pueden ser útiles para predecir nuestra variable objetivo.

Paso 2: Responder las preguntas.
1. ¿Con las variables numéricas que se tienen es suficiente para predecir la variable objetivo?

Al listar las 10 variables más correlacionadas podemos deducir que entre los factores que más influyen en el precio de la casa/ vivienda son:
"OverallQual", la calidad general, tiene una correlación muy fuerte con el precio de venta, lo cual es esperado, ya que la calidad general de una casa suele estar relacionada con su precio. "YearBuilt", el año en que se construyó la casa. Las casas más nuevas suelen tener un precio más alto, "FullBath" el número de baños completos, "GarageCars" la Capacidad de la cochera. Un garaje con mayor capacidad suele asociarse con un precio de venta más alto. Imagino que "YearRemodAdd”, es el año en el que ocurrió la última remodelación, por lo general, las remodelaciones recientes pueden aumentar el valor de una casa.
Entre otras como "GarageArea", "GarageYrBlt", "TotalBsmtSF" "TotRmsAbvGrd", que en general que son variables que representan la capacidad, tamaño y antiguedad de la casa. 
Factores que sin duda son de mucha ayuda, así que sí, considero que podrían ser suficientes para predecir la variable objetivo.

2. ¿Alguna de las variables categóricas servirá realmente para determinar la variable objetivo?

Sí, yo creo que aunque con las numéricas sería suficiente para realizar una buena predicción, tener información acerca de la calle, el vecindario y la zona (Columnas categóricas) en la que se ubica la casa/ vivienda podrían ser factores que influyan fuertemente en el costo de la misma.
Sin embargo, no podemos afirmar esto hasta que se realizase el proceso pertinente y checar la correlación que surja entre estas y la variable objetivo. 

Paso 3: Conversión de categórico a numérico. Hay que seleccionar las columnas que ya fueron categorizadas y hay que sacar su valor con un "one-hot encoder", luego hay que agregarlas al dataset y eliminar su columna categórica. Hay que mostrar de nuevo las correlaciones para ver si cambiaron las variables más correlacionadas con la variable objetivo.

Como era de esperarse, el one-hot encoder agrego muchísimas columnas nuevas, pasamos de tener un data set con alrededor de 81 columnas, a tener un data set con 157 columnas, tanto que ahora es más difícil tratar de interpretar la matriz de correlación.
Una vez que mostramos las correlaciones más fuertes, vemos que cambiaron con respecto a las que se mostraron más fuertes, cuando solo comparamos las numéricas, se puede decir que "ExterQual_Gd" y "Foundation_PConc" que por la abreviación que tiene no logro descifrar a que sígnica influyen más en el precio que "TotalBsmtSF " y "TotRmsAbvGrd"

Paso 4: Conversión de las demás columnas objeto a numérico. Para ello se va a requerir un encoder más avanzado, usar la clase "MultiColumnLabelEncoder" vista en clase, el dataframe resultante va a ser la versión consolidada y completamente numérica.

Podemos observar la diferencia de cómo trabaja el Label encoder a comparación del one-hot encoder, ya que este no agrego más columnas, permaneció con el mismo número porque categorizo en una sola columna. Ocupando el lugar de las columnas "object", es por esta razón que el número de columnas no aumento.

Paso 5: Mostrar la información del nuevo dataframe (numérico), mostrar que no contenga nulos, que todos los datos sean de tipo int/float/uint. Mostrar de nuevo las correlaciones, filtrar para que solo muestre las 10 más correlacionadas a la variable objetivo

Al final del análisis conseguimos un data frame sin datos nulos, totalmente numérico y conservando las mismas correlaciones positivas, indicando que estos factores podrían ser útiles para predecir el precio de una casa/ vivienda.

Conclusión

En conclusión, realizar un Análisis Exploratorio de Datos (EDA) es una etapa fundamental en cualquier proyecto de análisis de datos. Ya que nos proporciona una comprensión completa de la estructura, distribución y relaciones dentro de los datos, permitiendo identificar errores, patrones clave. Además, facilita la preparación y limpieza de datos, la selección de variables relevantes y la validación de suposiciones iniciales. El EDA es esencial para tomar decisiones informadas y formular hipótesis, estableciendo una base sólida para análisis y modelado subsiguientes.
Por ejemplo, durante la realización de este EDA pude estar realizando predicciones y observar por medio del análisis si mis predicciones eran correctas.
