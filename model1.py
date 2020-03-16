import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle
import os
import warnings
path=os.path.abspath(os.curdir)
df=pd.read_csv(path+'/data.csv',index_col=False)
data=df[["diagnosis","radius_worst","texture_worst","perimeter_worst","area_worst","smoothness_worst",
         "compactness_worst","concavity_worst","concave points_worst","symmetry_worst","fractal_dimension_worst"]]
data.head(20)
with warnings.catch_warnings():
  warnings.simplefilter("ignore")
  #cleaning the data Malegnin Cancer is 1 and Benign is 0
  data['diagnosis'] = data['diagnosis'].apply(lambda x: '1' if x == 'M' else '0')

# Fitting Data into training and testing

#splitting the data into training and testing
y=data['diagnosis'].values
x=data.drop('diagnosis',axis=1).values
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=20)
   
#applying the model
M=LogisticRegression()
with warnings.catch_warnings():#lift warnings
  warnings.simplefilter("ignore")
  M.fit(x_train,y_train)#fitting the data
diagnose='cancer_model1.sav'
pickle.dump(M,open(diagnose,'wb'))#saving the model
