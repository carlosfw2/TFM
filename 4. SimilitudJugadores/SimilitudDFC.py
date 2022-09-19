import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_excel("../Wyscout/Jugadores2+2BFinal.xlsx")
df.head()

df2 = df.copy()
#Filtramos por posicion
df2 = df2[(df2.Posicion == "DFC") & (df2.MinutosJugados >= 900)]
df2.shape

#Columnas que vamos a ponderar
subdf2 = pd.DataFrame(df2,columns=['AccionesDefensivasRealizadas/90','%DuelosDefensivosGanados','%DuelosAereosGanados',
          'Interceptaciones/90','%PrecisionPasesLargos',
          'PasesProgresivos/90','TirosInterceptados/90','GolesCabeza/90']) #Varias columnas
subdf2.head()

#Añadimos las columnas calculadas que queramos
#subdf2['xG/RemateContra'] = df2.apply(lambda row: row['xGEnContra']/row['RematesContra'], axis=1).round(2)

#Normalizamos cada columna por el mayor y damos el valor de la ponderizacion
subdf2['AccionesDefensivasRealizadas/90'] = ((subdf2['AccionesDefensivasRealizadas/90']/subdf2['AccionesDefensivasRealizadas/90'].max())*2.5).round(2)
subdf2['%DuelosDefensivosGanados'] = ((subdf2['%DuelosDefensivosGanados']/subdf2['%DuelosDefensivosGanados'].max())*3).round(2)
subdf2['%DuelosAereosGanados'] = ((subdf2['%DuelosAereosGanados']/subdf2['%DuelosAereosGanados'].max())*3).round(2)
subdf2['Interceptaciones/90'] = ((subdf2['Interceptaciones/90']/subdf2['Interceptaciones/90'].max())*2).round(2)
subdf2['%PrecisionPasesLargos'] = ((subdf2['%PrecisionPasesLargos']/subdf2['%PrecisionPasesLargos'].max())*1.2).round(2)
subdf2['PasesProgresivos/90'] = ((subdf2['PasesProgresivos/90']/subdf2['PasesProgresivos/90'].max())*1.2).round(2)

subdf2['TirosInterceptados/90'] = ((subdf2['TirosInterceptados/90']/subdf2['TirosInterceptados/90'].max())*1.5).round(2)
subdf2['GolesCabeza/90'] = ((subdf2['GolesCabeza/90']/subdf2['GolesCabeza/90'].max())*1).round(2)

#Añadimos la columna con el nombre del equipo y la colocamos en primer lugar
subdf2['Jugador'] = df2['Jugador']
first_cols = ['Jugador']
last_cols = [col for col in subdf2.columns if col not in first_cols]

subdf2 = subdf2[first_cols+last_cols]
subdf2.head()
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

subdf2.shape

#Hacemos el PCA con el numero de componentes
def filtradoDataframe(df):
    X, y = df.iloc[:,1:8].values, df.iloc[:,0].values
    sc = StandardScaler()
    X_std = sc.fit_transform(X)
    
    pca = PCA(n_components = 4)
    pca.fit(X_std)
    X_pca = pca.transform(X_std)
    
    exp1 = pca.explained_variance_ratio_
    print('Varianza 4 componentes',sum(exp1[0:4]))
    print(df.shape)
    
    df_pca_resultado = pd.DataFrame(data = X_pca, columns = ["PC1","PC2","PC3","PC4"], index=y)
    #df_pca_resultado = df_pca_resultado.rename_axis('Equipo')
    return df_pca_resultado.sort_values(by=['PC1'])
df_pca_resultado = filtradoDataframe(subdf2)
df_pca_resultado.head()

#Matriz de correlacion METODO PEARSON
df_corr = df_pca_resultado.T.corr(method='pearson')
df_corr.head()

#Metodo1: Sin Dendogram
df_corr2 = df_corr.filter(like='B. Espinosa', axis=0)
df_corr2 =df_corr2.T
df_corr2 = df_corr2*100

df_corr2.sort_values(by=['B. Espinosa'],ascending=False).head(11).iloc[1:]

#Metodo2: Con Dendogram WARD
unique_index = pd.Index(list(df_corr))
unique_index.get_loc('B. Espinosa')

from scipy.cluster.hierarchy import dendrogram, linkage
Z = linkage(df_corr,"ward")

plt.figure(figsize=(40, 15))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('sample index')
plt.ylabel('distance')
dendrogram(
    Z,
    leaf_rotation=90.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
plt.show()

df_corr.iloc[[191]]


