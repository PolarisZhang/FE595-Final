import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
%matplotlib inline


df = pd.read_csv('drive/Colab Notebooks/textual_10variables.csv')
df.head()

## standardize the data
features = ['v1','v2','v3','v4','v5','v6','v7','v8','v9','v10']
x = df.loc[:, features].values

x = StandardScaler().fit_transform(x)
pd.DataFrame(data = x, columns = features).head()


## Scree plot
## PCA projection 
pca = PCA(n_components=10)
principalComponents = pca.fit_transform(x)
var= pca.explained_variance_ratio_
per_var=np.round(pca.explained_variance_ratio_*100, decimals=1)
labels=['PC'+str(x) for x in range(1,len(per_var)+1)]

plt.bar(x=range(1,len(per_var)+1),height=per_var,tick_label=labels)
plt.ylabel('Percentage of Explained Variance')
plt.xlabel('Principle Component')
plt.title('Scree plot')
plt.show()



#Cumulative Variance explains
pca = PCA(n_components=10)
principalComponents = pca.fit_transform(x)
var1=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
plt.plot(var1)




## loading scores
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents,columns = ['pc1','pc2'])

## PCA1
principalDf.head(5)







