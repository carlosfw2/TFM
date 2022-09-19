import pandas as pd
import numpy as np

df = pd.read_excel("../Wyscout/Jugadores2+2BFinal.xlsx")
df.head()

df2 = df.copy()
#Filtramos por posicion
df2 = df2[(df2.Posicion == "DC") & (df2.MinutosJugados >= 900)]
df2.shape

print(df2.columns.to_list())

#Columnas que vamos a ponderar
subdf2 = pd.DataFrame(df2,columns=['Jugador','xG/90','Remates/90','Remates'
                            ,'xA/90','Desmarques/90'
                            ,'%RegatesRealizados','%DuelosOfensivosGanados'
                            ,'AccionesAtaqueExitosas/90','Regates/90'
                            ,'CarrerasProgresion/90'
                            ,'GolesSinPenalti/90','xG','%DuelosAereosGanados'
                            ,'Goles']) #Varias columnas
subdf2.head()

#Añadimos las columnas calculadas que queramos
subdf2['xG/Remate'] = df2.apply(lambda row: row['xG']/row['Remates'], axis=1).round(2)
subdf2['G-xG'] = df2.apply(lambda row: row['Goles']-row['xG'], axis=1).round(2)
subdf2 = subdf2.drop(['xG','Remates','Remates/90','Goles'], axis=1)

#Normalizamos cada columna por el mayor y damos el valor de la ponderizacion
subdf2['xG/Remate'] = ((subdf2['xG/Remate']/subdf2['xG/Remate'].max())*3).round(2)
subdf2['xG/90'] = ((subdf2['xG/90']/subdf2['xG/90'].max())*3).round(2)
subdf2['G-xG'] = ((subdf2['G-xG']/subdf2['G-xG'].max())*2.5).round(2)
subdf2['AccionesAtaqueExitosas/90'] = ((subdf2['AccionesAtaqueExitosas/90']/subdf2['AccionesAtaqueExitosas/90'].max())*2).round(2)
subdf2['GolesSinPenalti/90'] = ((subdf2['GolesSinPenalti/90']/subdf2['GolesSinPenalti/90'].max())*2).round(2)
subdf2['%RegatesRealizados'] = ((subdf2['%RegatesRealizados']/subdf2['%RegatesRealizados'].max())*1.8).round(2)
subdf2['%DuelosOfensivosGanados'] = ((subdf2['%DuelosOfensivosGanados']/subdf2['%DuelosOfensivosGanados'].max())*1.8).round(2)
subdf2['xA/90'] = ((subdf2['xA/90']/subdf2['xA/90'].max())*1.5).round(2)
subdf2['Desmarques/90'] = ((subdf2['Desmarques/90']/subdf2['Desmarques/90'].max())*1.5).round(2)
subdf2['%DuelosAereosGanados'] = ((subdf2['%DuelosAereosGanados']/subdf2['%DuelosAereosGanados'].max())*1.5).round(2)
subdf2['Regates/90'] = ((subdf2['Regates/90']/subdf2['Regates/90'].max())*1.2).round(2)
subdf2['CarrerasProgresion/90'] = ((subdf2['CarrerasProgresion/90']/subdf2['CarrerasProgresion/90'].max())*1.2).round(2)

#Sumamos las columnas desde la ultima posicion hasta la que queramos para sacar la puntuacion
subdf2['PuntuacionTotal']= subdf2.iloc[:, -13:-1].sum(axis=1)
subdf2 = subdf2.sort_values(by='PuntuacionTotal', ascending=False)
subdf2.head()
subdf2.shape

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

#Hacemos el PCA con el numero de componentes
def filtradoDataframe(df):
    X, y = df.iloc[:,2:13].values, df.iloc[:,0].values
    sc = StandardScaler()
    X_std = sc.fit_transform(X)
    
    pca = PCA(n_components = 10)
    pca.fit(X_std)
    X_pca = pca.transform(X_std)
    
    exp1 = pca.explained_variance_ratio_
    print('Varianza 4 componentes',sum(exp1[0:4]))
    print('Varianza 8 componentes',sum(exp1[0:8]))
    print('Varianza 10 componentes',sum(exp1[0:10]))
    print(df.shape)
    
    df_pca_resultado = pd.DataFrame(data = X_pca, columns = ["PC1","PC2","PC3","PC4","PC5","PC6","PC7","PC8"
                                                  ,"PC9","PC10"], index=y)
    #df_pca_resultado = df_pca_resultado.rename_axis('Equipo')
    return df_pca_resultado.sort_values(by=['PC1'])

df_pca_resultado = filtradoDataframe(subdf2)

#Matriz de correlacion METODO PEARSON
df_corr = df_pca_resultado.T.corr(method='pearson')
df_corr.head()

#Metodo1: Sin Dendogram
df_corr2 = df_corr.filter(like='Dani Rodríguez', axis=0)
df_corr2 =df_corr2.T
df_corr2 = df_corr2*100
df_corr2.sort_values(by=['Dani Rodríguez'],ascending=False).head(11).iloc[1:]

#Quitamos las columnas del dataframe original para añadir las del dataframe con la puntuacion
df2SinColumnas = df2.drop(['xG/90'
                            ,'xA/90','Desmarques/90'
                            ,'%RegatesRealizados','%DuelosOfensivosGanados'
                            ,'AccionesAtaqueExitosas/90','Regates/90'
                            ,'CarrerasProgresion/90'
                            ,'GolesSinPenalti/90','%DuelosAereosGanados'
                            ,'%GolesConvertidos'], axis=1)

#Creamos dataframe con aspectos basicos del jugador para añadir al dataframe final
dfFinal = pd.merge(subdf2,df2SinColumnas, on='Jugador', how='left')
dfFinal.head()
dfFinal.to_csv("puntuacionDelanteros.csv", encoding="utf-8-sig")


