
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_excel("../Instat/equiposFinal.xlsx")
df.head()

print(df.columns.to_list())

#AÑADIMOS VARIABLES Y NORMALIZAMOS
df2 = df.copy()
df2['TirosContraPorteria'] = df2['TirosContraPorteria']/df2['TirosContra']
df2['TirosContraFueraArea'] = df2['TirosContraFueraArea']/df2['TirosContra']

#Creamos el dataframe de ataque con las columnas que queremos
dfAtaque = pd.DataFrame(df2, columns = ['Faltas','Entradas'
            ,'Interceptaciones','BalonesRecuperados','RecuperacionesCampoRival','Presion'
            ,'Despejes','PPDA','TirosContra'
            ,'TirosContraFueraArea','ABPContra'])
dfAtaque.head()

dfAtaque.shape

#Normalizamos todas las columnas a valores entre 0 y 1
dfAtNorm = (dfAtaque - dfAtaque.min())/(dfAtaque.max()-dfAtaque.min())
dfAtNorm = dfAtNorm.round(2)
dfAtNorm.head()

#MATRIZ DE POSICION(COLORES)

#Añadimos la columna con el nombre del equipo y la colocamos en primer lugar
dfAtNorm['Equipo'] = df['Equipos']
first_cols = ['Equipo']
last_cols = [col for col in dfAtNorm.columns if col not in first_cols]
dfAtNorm = dfAtNorm[first_cols+last_cols]
dfAtNorm.head()

dfAtNorm.shape

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

#Hacemos el PCA con el numero de componentes
def filtradoDataframe(df):
    X, y = df.iloc[:,1:11].values, df.iloc[:,0].values
    sc = StandardScaler()
    X_std = sc.fit_transform(X)
    
    pca = PCA(n_components = 8)
    pca.fit(X_std)
    X_pca = pca.transform(X_std)
    
    exp1 = pca.explained_variance_ratio_
    print('Varianza 4 componentes',sum(exp1[0:4]))
    print('Varianza 8 componentes',sum(exp1[0:8]))
    print(df.shape)
    
    df_pca_resultado = pd.DataFrame(data = X_pca, columns = ["PC1","PC2","PC3","PC4","PC5","PC6","PC7","PC8"], index=y)
    #df_pca_resultado = df_pca_resultado.rename_axis('Equipo')
    return df_pca_resultado.sort_values(by=['PC1'])


#Varianza del 0.99 con 12 componentes
df_pca_resultado = filtradoDataframe(dfAtNorm)

#Matriz de correlacion METODO PEARSON
df_corr = df_pca_resultado.T.corr(method='pearson')
df_corr = df_corr.round(2)
df_corr.head()

#VISUALIZACION DEL MAPA DE CALOR
fig = plt.figure(figsize=(20,20))
ax = plt.axes()

plt.yticks(fontsize=14, weight='bold') 
plt.xticks(fontsize=14,weight='bold') 

ax.set_facecolor('#0077c8')

ax.spines['left'].set_color('white')
ax.spines['bottom'].set_color('white')
ax.spines['right'].set_color('white')
ax.spines['top'].set_color('white')
ax.xaxis.label.set_color('white')
ax.yaxis.label.set_color('white')
ax.tick_params(axis='x', colors='white')
ax.tick_params(axis='y', colors='white')

sns.set(font_scale=1.2, rc={'axes.facecolor':'#0077c8', 'figure.facecolor':'#0077c8'})
g1 = sns.heatmap(df_corr, linewidths=0.2,vmax=1.0, square=True, linecolor="white", annot=True, cmap="YlGnBu",
           cbar_kws={"shrink": 0.5})
plt.show()

#DENDOGRAMA A PARTIR MATRIZ CORRELACION

#dfAtNorm = dfAtNorm.drop(['Equipo'], axis=1)
#dfAtNorm.head()
df_corrInd = df_corr.rename(columns={'Lugo': '0','Real Sporting': '3','CE Sabadell': '9','Cartagena': '1',
    'Ponferradina': '13','UD Logrones': '8','AD Alcorcon': '18','Las Palmas': '15','Malaga': '19',
    'Albacete': '6','Real Zaragoza': '4','Girona FC': '11','CF Fuenlabrada': '7','Real Oviedo': '14',
    'Castellon': '2','Tenerife': '17','Almeria': '5','Leganes': '16','Espanyol': '10','Mirandes': '20',
    'Mallorca': '12','Rayo Vallecano': '21'})
df_corrInd.head()

from scipy.cluster.hierarchy import dendrogram, linkage

Z = linkage(df_corrInd,"ward")

# Personalizacion de colores
dflt_col = "#FFFFFF"   # Unclustered gray
D_leaf_colors = {"0": dflt_col,
                 "1": "#2630FE", 
                 "2": "#2630FE",
                 "3": "#FFFFFF",
                 "4": "#2630FE",
                 "5": "#0404b4",
                 "6": "#0404b4",
                 "7": "#5583fa", 
                 "8": "#5583fa",
                 "9": "#5583fa",
                 "10": "#7cf4d9", 
                 "11": "#0404b4",
                 "12": "#7cf4d9",
                 "13": "#fbfc21",
                 "14": "#5583fa",
                 "15": "#0404b4",
                 "16": "#fbfc21", 
                 "17": "#7cf4d9",
                 "18": "#fbfc21",
                 "19": "#fbfc21",
                 "20": "#fbfc21",
                 "21": "#7cf4d9",
                 }
link_cols = {}
for i, i12 in enumerate(Z[:,:2].astype(int)):
    c1, c2 = (link_cols[x] if x > len(Z) else D_leaf_colors["%d"%x]
    for x in i12)
    link_cols[i+1+len(Z)] = c1 if c1 == c2 else dflt_col

#Representacion del dendograma
fig= plt.figure(figsize=(25,10))
ax = plt.axes()
ax.set_facecolor('#0077c8')
plt.xlabel('Equipos', fontsize=20 )
plt.ylabel('Distancia', fontsize=20)
plt.yticks(fontsize=20, weight='bold') 
plt.xticks(fontsize=20,weight='bold') 
plt.gcf().set_facecolor('#0077c8')

ax.spines['left'].set_color('white')
ax.spines['bottom'].set_color('white')
ax.spines['right'].set_color('white')
ax.spines['top'].set_color('white')
ax.grid(False)

ax.xaxis.label.set_color('white')
ax.yaxis.label.set_color('white')
ax.tick_params(axis='x', colors='white')
ax.tick_params(axis='y', colors='white')
max_d = 3

#Dendograma final
D = dendrogram(Z=Z, labels=df_corrInd.index, color_threshold=None,
  leaf_font_size=15, leaf_rotation=45, link_color_func=lambda x: link_cols[x])
plt.show()


