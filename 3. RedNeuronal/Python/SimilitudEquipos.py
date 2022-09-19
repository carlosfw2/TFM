import re
import pandas as pd
import numpy as np
import traceback
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pyvis.network import Network
import networkx as nx

df1 = pd.read_excel("../Instat/equiposSimilitud.xlsx")
df1.head()

def filtradoDataframe(df):
    X, y = df.iloc[:,1:64].values, df.iloc[:,0].values
    sc = StandardScaler()
    X_std = sc.fit_transform(X)
    
    pca = PCA(n_components = 16)
    pca.fit(X_std)
    X_pca = pca.transform(X_std)
    
    exp1 = pca.explained_variance_ratio_
    print('Varianza 4 componentes',sum(exp1[0:4]))
    print('Varianza 8 componentes',sum(exp1[0:8]))
    print('Varianza 12 componentes',sum(exp1[0:12]))
    print('Varianza 16 componentes',sum(exp1[0:16]))
    print(df.shape)
    
    df_pca_resultado = pd.DataFrame(data = X_pca, columns = ["PC1","PC2","PC3","PC4","PC5","PC6","PC7","PC8",
        "PC9","PC10","PC11","PC12","PC13","PC14","PC15","PC16"], index=y)
    #df_pca_resultado = df_pca_resultado.rename_axis('Equipo')
    return df_pca_resultado.sort_values(by=['PC1'])

df_pca_resultado = filtradoDataframe(df1)
df_pca_resultado.head(22)

# calculate the correlation matrix and reshape
df_corr = df_pca_resultado.T.corr(method='pearson').stack().reset_index()

df_corr.head()

# rename the columns
df_corr.columns = ['EquipoInicio', 'EquipoFin', 'Semejanza']

# create a mask to identify rows with duplicate features as mentioned above
mask_dups =  (df_corr['EquipoInicio']==df_corr['EquipoFin']) 
# apply the mask to clean the correlation dataframe
df_corr = df_corr[~mask_dups]
df_corr = df_corr.sort_values(['EquipoInicio','Semejanza'], ascending = (True, False))
df_corr = df_corr.groupby('EquipoInicio').head(2)
#df_corr.to_csv("prueba12.csv",encoding='utf-8-sig')

# set the physics layout of the network
got_net = Network(height="750px", width="100%", bgcolor="#0077c8", font_color="E6FC8C")
got_net.barnes_hut()
sources = df_corr['EquipoInicio']
targets = df_corr['EquipoFin']
weights = df_corr['Semejanza']

edge_data = zip(sources, targets, weights)

for e in edge_data:
    src = e[0]
    dst = e[1]
    w = e[2]

    got_net.add_node(src, src, title=src)
    got_net.add_node(dst, dst, title=dst)
    got_net.add_edge(src, dst, value=w)

neighbor_map = got_net.get_adj_list()

# add neighbor data to node hover data
for node in got_net.nodes:
    node["title"] += " Neighbors:<br>" + "<br>".join(neighbor_map[node["id"]])
    node["value"] = len(neighbor_map[node["id"]])

got_net.show_buttons()
#got_net.show("gameofthrones2.html")


