#! /usr/bin/python
"""
Created on Sun Nov 14 12:06:11 2021

@author: danie
"""

# Import modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from functools import reduce
from kmodes.kprototypes import KPrototypes
from sklearn.cluster import KMeans
import matplotlib.cm as cm
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split,RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score,classification_report

# Import data
dirname = os.path.abspath(os.path.dirname(__file__))
filename = os.path.join(dirname, '../data/ds_challenge_2021.csv')
data = pd.read_csv(filename)

# Features
print("Columns: ", data.columns)

# Modificación del feature 'dispositivo'
data_dispositivo = data["dispositivo"].apply(lambda x : dict(eval(x))).apply(pd.Series)
data.drop(columns = ['dispositivo'], axis = 1, inplace = True)

# Data Frame con todas las features
df_features = pd.concat([data,data_dispositivo],axis = 1)

# Exploracion del data frame
print("Features: ",df_features.shape[1])
print("Registros: ",df_features.shape[0])

# Veamos las features
print(df_features.columns)

# Veamos a las features que contienen missing values
print((df_features.isna().sum())[df_features.isna().sum() != 0])
features_with_na = (df_features.isna().sum())[df_features.isna().sum() != 0].keys()

# Veamos los valores features que contienen missing values
print(df_features.apply(lambda col: col.unique()))

# Feature 'genero'
df_features['genero'].loc[df_features['genero'] == '--'] = np.nan

# Drop feature 'model'
df_features.drop(columns = ['model'], axis = 1, inplace = True)

# Revisamos los missing values por feature
df_features.isna().sum()

# Feature 'establecimiento', 'ciudad' y 'genero'
# A los missing values se le asigna el valor 'VACIO'
df_features['establecimiento'].fillna('VACIO',inplace = True)
df_features['ciudad'].fillna('VACIO',inplace = True)
df_features['genero'].fillna('VACIO',inplace = True)

# Veamos el 'target'
df_features['fraude'].value_counts().plot.bar()

# Subset fraude
df_fraude = df_features.loc[df_features['fraude'] == True,:]

# --------------------------------
# Análisis exploratorio - Parte I
# --------------------------------

# Función personalizada para generar histogramas
def plot_histograma(df,color,leyenda,titulo,xlabel,ylabel,bins):

    # Tamaño del gráfico
    plt.figure(figsize=(10, 7))
    # Label del eje "y"
    plt.ylabel(ylabel)
    # Label del eje "x"
    plt.xlabel(xlabel)
    # Titulo
    plt.title(titulo)
    # Creacion del histograma
    plt.hist(\
             # Filtramos una lista con las edades de los hombres y otra con las edades de las mujeres
             [df],\
             # Elegimos un color para cada género
             color = [color],\
             # Le damos un label a cada género, en este caso "Hombre" y "Mujer"
             label = [leyenda],\
             # Definimos la orientación del histograma, en este caso "vertical"
             orientation = 'vertical',\
             # Definimos la transparencia "alpha"
             alpha = 0.4,\
             # Definimos los bins
             bins = bins \
            )
    # Definimos un grid en el eje "y" con lineas "cortadas"
    plt.grid(color='gray', linestyle='dotted', linewidth=1, axis='y')
    # Creamos los ticks que irán en el eje "x"
    plt.xticks(bins)
    # Creamos la leyenda
    plt.legend()
    # Creamos el plot
    plt.show()

### 'monto'

# Descripción del feature
print(df_features.describe()['monto'])

# Histograma del feature 'monto' en el data frame total
plot_histograma(df = df_features.monto,
                color = 'red',
                leyenda = 'Montos Total',
                titulo = 'Histograma del feature monto',
                xlabel = 'Monto',
                ylabel = 'Frecuencia',
                bins = [i*100 for i in range(1,11)])

# Histograma del feature 'monto' en el data frame que contiene fraudes
plot_histograma(df = df_fraude.monto,
                color = 'blue',
                leyenda = 'Montos Fraude',
                titulo = 'Histograma del feature monto en el la data con fraude',
                xlabel = 'Monto',
                ylabel = 'Frecuencia',
                bins = [i*100 for i in range(1,11)])

### 'cash back'

# Descripción del feature
print(df_features.describe()['cashback'])

# Histograma del feature 'cashback' en el data frame total
plot_histograma(df = df_features.cashback,
                color = 'red',
                leyenda = 'Cashback Total',
                titulo = 'Histograma del feature cashback',
                xlabel = 'Cashback',
                ylabel = 'Frecuencia',
                bins = [i*2 for i in range(1,11)])

# Histograma del feature 'cashback' en el data frame total
plot_histograma(df = df_fraude.cashback,
                color = 'blue',
                leyenda = 'Cashback Fraude',
                titulo = 'Histograma del feature cashback en el la data con fraude',
                xlabel = 'Cashback',
                ylabel = 'Frecuencia',
                bins = [i*2 for i in range(1,11)])

### 'dcto'

# Descripción del feature
print(df_features.describe()['dcto'])

# Histograma del feature 'cashback' en el data frame total
plot_histograma(df = df_features.dcto,
                color = 'red',
                leyenda = 'dcto Total',
                titulo = 'Histograma del feature dcto',
                xlabel = 'dcto',
                ylabel = 'Frecuencia',
                bins = [i*20 for i in range(0,11)])

# Histograma del feature 'cashback' en el data frame total
plot_histograma(df = df_fraude.dcto,
                color = 'blue',
                leyenda = 'dcto Fraude',
                titulo = 'Histograma del feature dcto en el la data con fraude',
                xlabel = 'dcto',
                ylabel = 'Frecuencia',
                bins = [i*20 for i in range(0,11)])
    
### 'ciudad'

# Veamos un gráfico de barras para el feature ciudad
df_features.ciudad.value_counts().plot.bar()

# Veamos un gráfico de barras para el feature ciudad
df_fraude.ciudad.value_counts().plot.bar()

### 'establecimiento'

# Veamos un gráfico de barras para el feature ciudad
df_features.establecimiento.value_counts().plot.bar()

# Veamos un gráfico de barras para el feature ciudad
df_fraude.establecimiento.value_counts().plot.bar()

### 'tipo_tc'

# Veamos un gráfico de barras para el feature ciudad
df_features.tipo_tc.value_counts().plot.bar()

# Veamos un gráfico de barras para el feature ciudad
df_fraude.tipo_tc.value_counts().plot.bar()

### 'tipo_tc'

# Veamos un gráfico de barras para el feature ciudad
df_features.tipo_tc.value_counts().plot.bar()

# Veamos un gráfico de barras para el feature ciudad
df_fraude.tipo_tc.value_counts().plot.bar()

### 'is_prime'

# Veamos un gráfico de barras para el feature ciudad
df_features.is_prime.value_counts().plot.bar()

# Veamos un gráfico de barras para el feature ciudad
df_fraude.is_prime.value_counts().plot.bar()

### 'device_score'

# Veamos un gráfico de barras para el feature ciudad
df_features.device_score.value_counts().plot.bar()

# Veamos un gráfico de barras para el feature ciudad
df_fraude.device_score.value_counts().plot.bar()

### 'os'

# Veamos un gráfico de barras para el feature ciudad
df_features.os.value_counts().plot.bar()

# Veamos un gráfico de barras para el feature ciudad
df_fraude.os.value_counts().plot.bar()

### 'genero'

# Veamos un gráfico de barras para el feature ciudad
df_features.genero.value_counts().plot.bar()

# Veamos un gráfico de barras para el feature ciudad
df_fraude.genero.value_counts().plot.bar()

### 'status_txn'

# Veamos un gráfico de barras para el feature ciudad
df_features.status_txn.value_counts().plot.bar()

# Veamos un gráfico de barras para el feature ciudad
df_fraude.status_txn.value_counts().plot.bar()

### 'fecha'
print(data.fecha.min())
print(data.fecha.max())

print(pd.to_datetime(df_features.fecha).dt.year.value_counts())
print(pd.to_datetime(df_features.fecha).dt.month.value_counts())
print(pd.to_datetime(df_features.fecha).dt.day.value_counts())

df_fraude.hora.value_counts().plot.bar()

# --------------------------------
# Análisis exploratorio - Parte II
# --------------------------------

# Import data
dirname = os.path.abspath(os.path.dirname(__file__))
filename = os.path.join(dirname, '../data/ds_challenge_2021.csv')
data = pd.read_csv(filename)

data.columns

# Transformación de la columna dispositivo
data_dispositivo = data["dispositivo"].apply(lambda x : dict(eval(x))).apply(pd.Series)
data.drop(columns = ['dispositivo'], axis = 1, inplace = True)
# Data Frame con todas las features
data = pd.concat([data,data_dispositivo],axis = 1)
data.drop(['model'],axis = 1,inplace=True)

# A los missing values se le asigna el valor 'VACIO'
data['establecimiento'].fillna('VACIO',inplace = True)
data['ciudad'].fillna('VACIO',inplace = True)
data['genero'].fillna('VACIO',inplace = True)

# Creacion de data frames
# 1. Suma de features numéricos
df_feat_numerica = data.groupby(['ID_USER'])[['monto','cashback','dcto','fraude']].sum().reset_index().rename(columns={"fraude":"n_fraude"})
# 2. Conteo de features por ID_USER
df_feat_count = data.groupby(['ID_USER'])[['ciudad','establecimiento','os','fecha','status_txn','tipo_tc']].nunique().reset_index()
# 3. Nº por ID_USER
df_n_transacciones = data.groupby(['ID_USER'])[['ID_USER']].count().rename(columns={"ID_USER":"N_TRANS"}).reset_index()
# 4. ID_USER que ha cometido al menos un fraude
df_ID_fraude = data.loc[data['fraude']][['ID_USER','fraude']].drop_duplicates()
# 5. ID_USER, is_prime
df_ID_isprime = data[['ID_USER','is_prime']].drop_duplicates()
# 6. ID_USER, linea_tc
df_ID_lineatc = data.groupby(['ID_USER'])[['linea_tc']].mean().reset_index()

# Union de data frames
data_frames = [df_n_transacciones, df_feat_count, df_feat_numerica, df_ID_isprime, df_ID_lineatc]
df_total = reduce(lambda left,right: pd.merge(left,right,on=['ID_USER'],how='outer'), data_frames)
df_total = pd.merge(df_total,df_ID_fraude,on='ID_USER',how='left')
df_total['fraude'].fillna(False,inplace=True)

# PATRON 1
pd.concat({
            '#Clientes No Fraudulentos': df_total[~df_total.fraude].establecimiento.value_counts().sort_index(),
            '#Clientes Fraudulentos': df_total[df_total.fraude].establecimiento.value_counts().sort_index()
            }, axis=1).plot.bar(title='# de clientes por # de establecimientos')
plt.xlabel("# Establecimientos")
plt.ylabel("# Clientes")

# PATRON 2
pd.concat({
            '#Clientes No Fraudulentos': df_total[~df_total.fraude].fecha.value_counts().sort_index(),
            '#Clientes Fraudulentos': df_total[df_total.fraude].fecha.value_counts().sort_index()
            }, axis=1).plot.bar(title='# de clientes por fecha')
plt.xlabel("# Fechas")
plt.ylabel("# Clientes")

# PATRON 3
pd.concat({
            '#Clientes No Fraudulentos': df_total[~df_total.fraude].os.value_counts().sort_index(),
            '#Clientes Fraudulentos': df_total[df_total.fraude].os.value_counts().sort_index()
            }, axis=1).plot.bar(title='# de clientes por # de dispositivos')
plt.xlabel("# Dispositivos")
plt.ylabel("# Clientes")

# PATRON 4
fig, (ax1, ax2) = plt.subplots(1, 2)
fig.set_size_inches(18, 7)
ax1.set_xlim([0, max(df_total[~df_total.fraude].monto)])
ax1.hist(df_total[~df_total.fraude].monto)
ax1.set_title('Histograma del feature monto - Clientes No Fraudulentos')
ax2.set_xlim([0, max(df_total[df_total.fraude].monto)])
ax2.hist(df_total[df_total.fraude].monto)
ax2.set_title('Histograma del feature monto - Clientes Fraudulentos')

# PATRON 5
fig, (ax1, ax2) = plt.subplots(1, 2)
fig.set_size_inches(18, 7)
ax1.set_xlim([0, max(df_total[~df_total.fraude].N_TRANS)])
ax1.hist(df_total[~df_total.fraude].N_TRANS)
ax1.set_title('Histograma del feature N_TRANS - Clientes No Fraudulentos')
ax2.set_xlim([0, max(df_total[df_total.fraude].N_TRANS)])
ax2.hist(df_total[df_total.fraude].N_TRANS)
ax2.set_title('Histograma del feature N_TRANS - Clientes Fraudulentos')

# PATRON 6
fig, (ax1, ax2) = plt.subplots(1, 2)
fig.set_size_inches(18, 7)
ax1.set_xlim([0, max(df_total[~df_total.fraude].dcto)])
ax1.hist(df_total[~df_total.fraude].dcto)
ax1.set_title('Histograma del feature dcto - Clientes No Fraudulentos')
ax2.set_xlim([0, max(df_total[df_total.fraude].dcto)])
ax2.hist(df_total[df_total.fraude].dcto)
ax2.set_title('Histograma del feature dcto - Clientes Fraudulentos')

# PATRON 7
fig, (ax1, ax2) = plt.subplots(1, 2)
fig.set_size_inches(18, 7)
ax1.set_xlim([0, max(df_total[~df_total.fraude].cashback)])
ax1.hist(df_total[~df_total.fraude].cashback)
ax1.set_title('Histograma del feature cashback - Clientes No Fraudulentos')
ax2.set_xlim([0, max(df_total[df_total.fraude].cashback)])
ax2.hist(df_total[df_total.fraude].cashback)
ax2.set_title('Histograma del feature cashback - Clientes Fraudulentos')

# PATRON 8
pd.concat({
            '#Clientes No Fraudulentos': df_total[~df_total.fraude].is_prime.value_counts().sort_index(),
            '#Clientes Fraudulentos': df_total[df_total.fraude].is_prime.value_counts().sort_index()
            }, axis=1).plot.bar(title='# de clientes por prime/no prime')
plt.xlabel("Prime/ No Prime")
plt.ylabel("# Clientes")

# ---------------------------------------
# Segmentación de Clientes por Clustering
# ---------------------------------------

# Definición de X, y
X = df_total[[i for i in df_total.columns if i not in ["fraude","ID_USER"] ]]
y = df_total["fraude"]

# Categorical Features
categorical_features = ['is_prime']
cat_idx = [i for i in range(len(X.columns)) if X.columns[i] in categorical_features]

# Optimo k, usando K-Prototypes (Considerando categorical features)
cost = []
for num_clusters in list(range(2,8)):
    print(num_clusters)
    kproto = KPrototypes(n_clusters=num_clusters)
    kproto.fit_predict(X, categorical=cat_idx)
    cost.append(kproto.cost_)
print("Listo")

# Elbow plot
plt.xticks(ticks=range(len(cost)), labels=[i+1for i in range(1,len(cost)+1)])
plt.plot(cost) # Your data
plt.title('Elbow plot') # Your data
plt.show()

# Clusters = 4
kproto = KPrototypes(n_clusters=4)
kproto.fit_predict(X, categorical=cat_idx)

# Data frame con clusters (Kprototype)
clusters = kproto.predict(X, categorical=cat_idx)
df_cluster = X.copy()
df_cluster['cluster'] = list(clusters)

# Each cluster
df_cluster_0 = df_cluster[df_cluster['cluster']==0]
df_cluster_1 = df_cluster[df_cluster['cluster']==1]
df_cluster_2 = df_cluster[df_cluster['cluster']==2]
df_cluster_3 = df_cluster[df_cluster['cluster']==3]

# Optimo k, usando K-Means
X_kmeans = X.copy()
X_kmeans.drop(['is_prime'],axis=1,inplace=True)

# kmeans para diferente numero de clusters
silh_list = []
for num_clusters in list(range(2,8)):
    print(num_clusters)
    kmeans_ = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans_.fit_predict(X_kmeans)
    silh_list += [silhouette_score(X_kmeans, cluster_labels)]
print("Listo")

# Kmeans con 3 clusters
kmeans_ = KMeans(n_clusters=3, random_state=42)
cluster_labels = kmeans_.fit_predict(X_kmeans)

# Kmeans scatter plot
fig, (ax1) = plt.subplots(1)
fig.set_size_inches(12, 7)
colors = cm.nipy_spectral(clusters.astype(float) / 3)
ax1.scatter(np.array(X_kmeans)[:, 4],
            np.array(X_kmeans)[:, 2],
            marker='.', s=30, lw=0, alpha=0.7,c=colors, edgecolor='k')
ax1.scatter(kmeans_.cluster_centers_[:, 4],
            kmeans_.cluster_centers_[:, 2],
            marker='o', c="white", alpha=1, s=200, edgecolor='k')

# ---------------------------------------
# Monto por Cluster
# ---------------------------------------

# Monto
fig, ax = plt.subplots(2, 2)
fig.set_size_inches(15, 7)
ax[0][0].hist(df_cluster_0.monto)
ax[0][0].legend(['Cluster 0'])
ax[0][1].hist(df_cluster_1.monto)
ax[0][1].legend(['Cluster 1'])
ax[1][0].hist(df_cluster_2.monto)
ax[1][0].legend(['Cluster 2'])
ax[1][1].hist(df_cluster_3.monto)
ax[1][1].legend(['Cluster 3'])
plt.show()

# Cashback
fig, ax = plt.subplots(2, 2)
fig.set_size_inches(15, 7)
ax[0][0].hist(df_cluster_0.cashback)
ax[0][0].legend(['Cluster 0'])
ax[0][1].hist(df_cluster_1.cashback)
ax[0][1].legend(['Cluster 1'])
ax[1][0].hist(df_cluster_2.cashback)
ax[1][0].legend(['Cluster 2'])
ax[1][1].hist(df_cluster_3.cashback)
ax[1][1].legend(['Cluster 3'])
plt.show()

# linea_tc
fig, ax = plt.subplots(2, 2)
fig.set_size_inches(15, 7)
ax[0][0].hist(df_cluster_0.linea_tc)
ax[0][0].legend(['Cluster 0'])
ax[0][1].hist(df_cluster_1.linea_tc)
ax[0][1].legend(['Cluster 1'])
ax[1][0].hist(df_cluster_2.linea_tc)
ax[1][0].legend(['Cluster 2'])
ax[1][1].hist(df_cluster_3.linea_tc)
ax[1][1].legend(['Cluster 3'])
plt.show()

# N_TRANS
fig, ax = plt.subplots(2, 2)
fig.set_size_inches(15, 7)
ax[0][0].hist(df_cluster_0.N_TRANS)
ax[0][0].legend(['Cluster 0'])
ax[0][1].hist(df_cluster_1.N_TRANS)
ax[0][1].legend(['Cluster 1'])
ax[1][0].hist(df_cluster_2.N_TRANS)
ax[1][0].legend(['Cluster 2'])
ax[1][1].hist(df_cluster_3.N_TRANS)
ax[1][1].legend(['Cluster 3'])
plt.show()

# ----------------------------------------
# Machine Learning Model - Fraud Detection
# ----------------------------------------

# Definición de categorical features en el dataset
categorical_features = ['establecimiento','ciudad','tipo_tc','status_txn','is_prime','device_score',
                        'os','genero']

for i in categorical_features:
    print(i)
    df_features = pd.concat([df_features,pd.get_dummies(df_features[i], prefix=i)],axis=1)
    df_features.drop([i],axis=1,inplace=True)
print("Listo")

# Train/Test split 
X = df_features[[ i for i in df_features.columns if i != "fraude"]]
y = df_features['fraude']
X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=0.33, random_state=42)
print("Antes de hacer OverSampling, # de registros con fraude: {}".format(sum(y_train == 1)))
print("Antes de hacer OverSampling, # de registros sin fraude: {}".format(sum(y_train == 0)))

# SMOTE - TECNICA DE OVERSAMPLING
sm = SMOTE(random_state = 2)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train.ravel())

print('Después de hacer OverSampling, las dimnesiones del train_X: {}'.format(X_train_res.shape))
print('Después de hacer OverSampling, las dimnesiones del train_y: {}'.format(y_train_res.shape)) 
print("Después de hacer OverSampling, # de registros con fraude: {}".format(sum(y_train_res == 1)))
print("Después de hacer OverSampling, # de registros sin fraude: {}".format(sum(y_train_res == 0)))

# Usando un modelo base line - Regresión Logística
scaler = MinMaxScaler()
model_lr = LogisticRegression(random_state=42)
pipeline_lr = Pipeline(steps=[('scaler', scaler),
                              ('model', model_lr)])
pipeline_lr.fit(X_train_res, y_train_res)
predictions_lr_train = pipeline_lr.predict(X_train_res)
predictions_lr_test = pipeline_lr.predict(X_test)

print(classification_report(y_train_res,predictions_lr_train))
print(classification_report(y_test,predictions_lr_test))

# Usando un modelo - Random Forest
random_grid = {
                "n_estimators" : [500],
                "max_depth": [3, 5, 10, 15],
                "min_samples_split": [2, 4, 8],
                "min_samples_leaf": [4, 6, 8],
                "criterion": ["gini"]
                }
model_rf = RandomForestClassifier(random_state=42)
rf_random = RandomizedSearchCV(estimator = model_rf,
                               param_distributions = random_grid,
                               n_iter = 100, cv = 3, verbose=2, n_jobs = 4)
rf_random.fit(X_train_res, y_train_res)

predictions_rf_train = rf_random.predict(X_train_res)
predictions_rf_test = rf_random.predict(X_test)

print(classification_report(y_train_res,predictions_lr_train))
print(classification_report(y_test,predictions_lr_test))

# Feature Importance
importance = rf_random.best_estimator_.feature_importances_

df_importances = pd.DataFrame({'features':rf_random.best_estimator_.feature_names_in_,
                               'importance':importance}).set_index('features').sort_values(by='importance')

fig, ax = plt.subplots(1, 1)
fig.set_size_inches(15, 12)
ax.barh(df_importances.index,
        df_importances.importance)
plt.show()

# ----------------------------------------
# Conclusiones
# ----------------------------------------

# En el exploratorio de los datos, se encontró que las clientes más fraudulentos han realizado un número de transacciones mayor
# durante el mes. Además, suelen hacerlo de diversos establecimientos, en diferentes fechas (diferentes días) y tienden a usar
# distintos tipos de dispositivos. Por otro lado, el histograma indica que el monto de transacción de los clientes fraudulentos
# es mayor que los no fraudulentos (ver el histograma) y hay una cantidad muy pequeña de clientes prime que son fraudulentos.

# Para la segmentación de clientes, se hace usom de un KPrototypes (Unsupervised Learning) que es un modelo que considera tanto las features númericas (Kmeans)
# como las features categóricas (Kmodes) y con esto se intentó segmentar a los clientes. Los resultados indicaban que un óptimo número de
# clusters sería 4 o 3, y con estos podríamos hacer alguna interpretación de los datos.
# Podemos ver que si elegimos 4 clusters y observamos los histogramas de los montos de transacciones de los clientes que conforman
# cada uno de ellos, se aprecia que posiblemente si existe un patrón similar dado que guardaban similaridad en su forma.
# Una posible mejora es probar reducir el número de clusters y ver qué tan bien se forman estos histogramas.

# Antes de la implementacion del modelo, se tomó en cuenta la presencia de variables categóricas para el entrenamiento de los datos
# (haciendo uso de get_dummies) y posteriormente, se buscó entrenar un modelo baseline que es el Logistic Regression (Supervised Learning)
# teniendo en cuenta el problema de data desbalanceada, para solucionar este problema se uso un método de oversampling llamado SMOTE que permitió
# tener el mismo número de valores de la clase desbalanceada en el entrenamiento de los datos. Los resultados de esta implementación,
# no fueron los mejores y se buscó implementar un modelo no lineal Random Forest, para ello se uso un cross validation y se incluyeron
# hiperparámetros para su optimización. En la parte final se obtienen los feature importance del random forest y se puede notar que 
# las features más importante son el medio por el cuál se realiza la transacción y el device_score y entre las features menos importantes
# tenemos si el cliente es prime o no, el día y la hora.








