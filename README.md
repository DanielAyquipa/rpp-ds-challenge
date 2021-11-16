# rpp-ds-challenge

La carpeta data contiene un fichero .csv con los datos del modelo. <\n>
La carpeta src contiene un fichero .py con la solución al ds-challenge. <\n>
El fichero requirements.txt contiene las versiones de los módulos de python que se usan en el .py <\n>

### Conclusiones

En el exploratorio de los datos, se encontró que las clientes más fraudulentos han realizado un número de transacciones mayor durante el mes. Además, suelen hacerlo de diversos establecimientos, en diferentes fechas (diferentes días) y tienden a usar distintos tipos de dispositivos. Por otro lado, el histograma indica que el monto de transacción de los clientes fraudulentos es mayor que los no fraudulentos (ver el histograma) y hay una cantidad muy pequeña de clientes prime que son fraudulentos.

Para la segmentación de clientes, se hace usom de un KPrototypes (Unsupervised Learning) que es un modelo que considera tanto las features númericas (Kmeans) como las features categóricas (Kmodes) y con esto se intentó segmentar a los clientes. Los resultados indicaban que un óptimo número de clusters sería 4 o 3, y con estos podríamos hacer alguna interpretación de los datos. Podemos ver que si elegimos 4 clusters y observamos los histogramas de los montos de transacciones de los clientes que conforman cada uno de ellos, se aprecia que posiblemente si existe un patrón similar dado que guardaban similaridad en su forma. Una posible mejora es probar reducir el número de clusters y ver qué tan bien se forman estos histogramas.

Antes de la implementacion del modelo, se tomó en cuenta la presencia de variables categóricas para el entrenamiento de los datos (haciendo uso de get_dummies) y posteriormente, se buscó entrenar un modelo baseline que es el Logistic Regression (Supervised Learning) teniendo en cuenta el problema de data desbalanceada, para solucionar este problema se uso un método de oversampling llamado SMOTE que permitió tener el mismo número de valores de la clase desbalanceada en el entrenamiento de los datos. Los resultados de esta implementación, no fueron los mejores y se buscó implementar un modelo no lineal Random Forest, para ello se uso un cross validation y se incluyeron hiperparámetros para su optimización. En la parte final se obtienen los feature importance del random forest y se puede notar que  las features más importante son el medio por el cuál se realiza la transacción y el device_score y entre las features menos importantes tenemos si el cliente es prime o no, el día y la hora.

