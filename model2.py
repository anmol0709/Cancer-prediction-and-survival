import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pickle
import os
import warnings
path=os.path.abspath(os.curdir)
hbr=pd.read_csv(path+'/haberman.csv',delimiter=",")
    #splitting the data into  and testing data
y1=np.array(hbr['Survival_Status'])
x1=np.array(hbr.drop('Survival_Status',axis=1))
x1_train,x1_test,y1_train,y1_test=train_test_split(x1,y1,test_size=0.20,random_state=20)

#calling the model
SVM=SVC()
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    SVM.fit(x1,y1)

survive='cancer_model2.sav'
pickle.dump(SVM,open(survive,'wb'))#saving the model
