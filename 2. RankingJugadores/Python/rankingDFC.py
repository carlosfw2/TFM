import pandas as pd
import numpy as np

df = pd.read_excel("../Wyscout/jugadoresLimpioConPosiciones.xlsx")
df.head()

df2 = df.copy()

#Filtramos por posicion
df2 = df2[(df2.Posicion == "DFC") & (df2.MinutosJugados >= 900)]
df2.shape

print(df2.columns.tolist())

#Columnas que vamos a ponderar
features = ['Jugador','AccionesDefensivasRealizadas/90','%DuelosDefensivosGanados','%DuelosAereosGanados',
          'Interceptaciones/90','%PrecisionPasesLargos',
          'PasesProgresivos/90','TirosInterceptados/90','GolesCabeza/90']
subdf2 = pd.DataFrame(df2,columns = features) #Varias columnas
subdf2.head()
subdf2.describe()

#Añadimos las columnas calculadas que queramos
#subdf2['xG/RemateContra'] = df2.apply(lambda row: row['xGEnContra']/row['RematesContra'], axis=1).round(2)

#Normalizamos cada columna por el mayor y damos el valor de la ponderizacion
subdf2['%DuelosDefensivosGanados'] = ((subdf2['%DuelosDefensivosGanados']/subdf2['%DuelosDefensivosGanados'].max())*3).round(2)
subdf2['%DuelosAereosGanados'] = ((subdf2['%DuelosAereosGanados']/subdf2['%DuelosAereosGanados'].max())*3).round(2)
subdf2['AccionesDefensivasRealizadas/90'] = ((subdf2['AccionesDefensivasRealizadas/90']/subdf2['AccionesDefensivasRealizadas/90'].max())*2.5).round(2)
subdf2['Interceptaciones/90'] = ((subdf2['Interceptaciones/90']/subdf2['Interceptaciones/90'].max())*2).round(2)
subdf2['TirosInterceptados/90'] = ((subdf2['TirosInterceptados/90']/subdf2['TirosInterceptados/90'].max())*1.5).round(2)
subdf2['%PrecisionPasesLargos'] = ((subdf2['%PrecisionPasesLargos']/subdf2['%PrecisionPasesLargos'].max())*1.2).round(2)
subdf2['PasesProgresivos/90'] = ((subdf2['PasesProgresivos/90']/subdf2['PasesProgresivos/90'].max())*1.2).round(2)
subdf2['GolesCabeza/90'] = ((subdf2['GolesCabeza/90']/subdf2['GolesCabeza/90'].max())*1).round(2)

subdf2.head()

#Sumamos las columnas desde la ultima posicion hasta la que queramos para sacar la puntuacion
subdf2['PuntuacionTotal']= subdf2.iloc[:, -8:-1].sum(axis=1)
subdf2 = subdf2.sort_values(by='PuntuacionTotal', ascending=False)
subdf2.head()

#Quitamos las columnas del dataframe original para añadir las del dataframe con la puntuacion
df2SinColumnas = df2.drop(['AccionesDefensivasRealizadas/90','%DuelosDefensivosGanados','%DuelosAereosGanados',
          'Interceptaciones/90','%PrecisionPasesLargos',
          'PasesProgresivos/90','TirosInterceptados/90','GolesCabeza/90'], axis=1)
df2SinColumnas.head()

#Creamos dataframe con aspectos basicos del jugador para añadir al dataframe final
dfFinal = pd.merge(subdf2,df2SinColumnas, on='Jugador', how='left')
dfFinal.head()
#dfFinal.to_csv("puntuacionPorteros.csv", encoding="utf-8-sig")


