import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_excel("../Instat/equiposFinal.xlsx")
df.head()

print(df.columns.to_list())

# AÑADIMOS VARIABLES Y NORMALIZAMOS
df2 = df.copy()

#Normalizamos las 3 columnas
df2['AtaquesBandaIzquierda'] = df['AtaquesBandaIzquierda']/(df['AtaquesBandaIzquierda'] 
                            + df['AtaquesZonaCentral'] + df['AtaquesBandaDerecha'])
df2['AtaquesZonaCentral'] = df['AtaquesZonaCentral']/(df['AtaquesBandaIzquierda'] 
                            + df['AtaquesZonaCentral'] + df['AtaquesBandaDerecha'])
df2['AtaquesBandaDerecha'] = df['AtaquesBandaDerecha']/(df['AtaquesBandaIzquierda'] 
                            + df['AtaquesZonaCentral'] + df['AtaquesBandaDerecha'])

#Creamos el dataframe de ataque con las columnas que queremos
dfAtaque = pd.DataFrame(df2, columns = ['AtaquesBandaIzquierda','AtaquesZonaCentral',
            'AtaquesBandaDerecha','JugadasDeGol','Tiros','TirosPuerta','DisputasAtaque','%DisputasAtaqueGanadas','Centros'
            ,'Regates','EntradasCampoRival','EntradasUltimoTercioRival','EntradasAreaRival','AtaquesPosicionales',
            'Contraataques','ABP','DistanciaMediaTiros'])
dfAtaque.head()

#Normalizamos todas las columnas a valores entre 0 y 1
dfAtNorm = (dfAtaque - dfAtaque.min())/(dfAtaque.max()-dfAtaque.min())
dfAtNorm = dfAtNorm.round(2)
dfAtNorm.head()
 
#MATRIZ DE POSICION(COLORES)
dfAtNorm.shape

#Añadimos la columna con el nombre del equipo y la colocamos en primer lugar
dfAtNorm['Equipo'] = df['Equipos']
first_cols = ['Equipo']
last_cols = [col for col in dfAtNorm.columns if col not in first_cols]

dfAtNorm = dfAtNorm[first_cols+last_cols]

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

#Hacemos el PCA con el numero de componentes
def filtradoDataframe(df):
    X, y = df.iloc[:,1:34].values, df.iloc[:,0].values
    sc = StandardScaler()
    X_std = sc.fit_transform(X)
    
    pca = PCA(n_components = 12)
    pca.fit(X_std)
    X_pca = pca.transform(X_std)
    
    exp1 = pca.explained_variance_ratio_
    print('Varianza 4 componentes',sum(exp1[0:4]))
    print('Varianza 8 componentes',sum(exp1[0:8]))
    print('Varianza 12 componentes',sum(exp1[0:12]))
    print(df.shape)
    
    df_pca_resultado = pd.DataFrame(data = X_pca, columns = ["PC1","PC2","PC3","PC4","PC5","PC6","PC7","PC8",
        "PC9","PC10","PC11","PC12"], index=y)
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

# DENDOGRAMA A PARTIR MATRIZ CORRELACION

df_corrInd = df_corr.rename(columns={'Lugo': '1','Real Sporting': '0','CE Sabadell': '5','Cartagena': '2',
    'Ponferradina': '4','UD Logrones': '6','AD Alcorcon': '3','Las Palmas': '7','Malaga': '8',
    'Albacete': '13','Real Zaragoza': '15','Girona FC': '10','CF Fuenlabrada': '11','Real Oviedo': '12',
    'Castellon': '9','Tenerife': '16','Almeria': '14','Leganes': '17','Espanyol': '18','Mirandes': '19',
    'Mallorca': '20','Rayo Vallecano': '21'})
df_corrInd.head()

from scipy.cluster.hierarchy import dendrogram, linkage

Z = linkage(df_corrInd,"ward")

# Personalizacion de colores
dflt_col = "#FFFFFF"   # Unclustered gray
D_leaf_colors = {"0": dflt_col,
                 "1": "#fbfc21", 
                 "2": "#FFFFFF",
                 "3": "#FFFFFF",
                 "4": "#FFFFFF",
                 "5": "#fbfc21",
                 "6": "#fbfc21",
                 "7": "#FFFFFF", 
                 "8": "#fbfc21",
                 "9": "#7cf4d9",
                 "10": "#2630fe", 
                 "11": "#2630fe",
                 "12": "#7cf4d9",
                 "13": "#7cf4d9",
                 "14": "#2630fe",
                 "15": "#7cf4d9",
                 "16": "#7cf4d9", 
                 "17": "#2630fe",
                 "18": "#0404b4",
                 "19": "#2630fe",
                 "20": "#0404b4",
                 "21": "#0404b4",
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

#Dendograma final
D = dendrogram(Z=Z, labels=df_corrInd.index, color_threshold=None,
  leaf_font_size=15, leaf_rotation=45, link_color_func=lambda x: link_cols[x])
plt.show()


