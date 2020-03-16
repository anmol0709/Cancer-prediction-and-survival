#importing libraries
from flask import Flask,render_template,request
import pandas as pd
import numpy as np
from sklearn import linear_model
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import time
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import warnings
import os
import pickle
from os import listdir
from flask import Flask, flash, request, redirect, url_for, render_template, send_from_directory, send_file
from flask_mail import Mail,Message
import ast #For stripping the unnecessary symbols
############################################################################################################
app=Flask(__name__)
path=os.path.abspath(os.curdir)

#setting up api for mail
app.config.update(
    DEBUG=False,
    MAIL_USERNAME='myprojects0709@gmail.com',
    MAIL_PASSWORD ='anmoljindal@1',
    MAIL_DEFAULT_SENDER=('Anmol Jindal','anmoljindal0709@gmail.com'),
    MAIL_SERVER='smtp.gmail.com',
    MAIL_PORT=465,
    MAIL_USE_SSL=True
    )
############################################################################################################
mail=Mail(app)
@app.route('/',methods=['GET','POST'])
def info():
    para1=[]
    if request.method=='POST':#Taking inputs for cancer prediction
        a=request.values['Radius']
        para1.append(a)
        b=request.values['Texture']
        para1.append(b)
        c=request.values['Perimeter']
        para1.append(c)
        d=request.values['Area']
        para1.append(d)
        e=request.values['Smoothness']
        para1.append(e)
        f=request.values['compactness']
        para1.append(f)
        g=request.values['Concavity']
        para1.append(g)
        h=request.values['Concavepoints']
        para1.append(h)
        i=request.values['Symmetry']
        para1.append(i)
        j=request.values['Fractal_dimension']
        para1.append(j)
        #The email on which results will be sent
        email=request.form['email']
        #Choice for knowing survival prediction
        choice=request.form['yes_no']

        return redirect(url_for('diagnosis',para1=para1,choice=choice,email=email))
    return render_template('login.html')
    
############################################################################################################

@app.route('/uploads/<para1>/<choice>/<email>')
def diagnosis(para1,choice,email):
    #For stripping the unnecessary symbols
    para1=ast.literal_eval(para1)
    para1=[n.strip() for n in para1]
    #Changing values from string to float
    P1=[]
    for i in para1:
        P1.append(float(i)) 

    #Loading dataset for accuracy calculation
    df=pd.read_csv(path+'/data.csv',index_col=False)
    data=df[["diagnosis","radius_worst","texture_worst","perimeter_worst","area_worst","smoothness_worst",
         "compactness_worst","concavity_worst","concave points_worst","symmetry_worst","fractal_dimension_worst"]]
    data['diagnosis'] = data['diagnosis'].apply(lambda x: '1' if x == 'M' else '0')

    with warnings.catch_warnings():#lift warnings
        warnings.simplefilter("ignore")
        y=data['diagnosis'].values
        x=data.drop('diagnosis',axis=1).values

        #training and testing the data
        x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=20)
        LD=pickle.load(open('cancer_model1.sav','rb'))#loading the model
        prediction=LD.predict(x_test)#predicting the diagnosis

    accuracy=accuracy_score(y_test,prediction)#accuracy of the model

    #cf=confusion_matrix(y_test,prediction)#print and plot the confusion matrix
    #print(cf)

    #predicting cancer for user input 
    accuracy=str(accuracy)
    msg=Message("Cancer Prediction Diagnosis",recipients=[email])
    print(P1)

    #Predicting the result
    A=LD.predict([P1])
    #Attaching the result
    if A==['0']:
        msg.body='The patient does not have posibilty of malignancy; Accuracy ='+ accuracy
    else:
        msg.body='The patient has posibilty of malignancy; Accuracy ='+ accuracy
        
    print('classification_report'+classification_report(y_test,prediction))#print the classification report

    #Sending mail
    mail.send(msg)

    if choice=='Yes':
    	return redirect(url_for('get',email=email))
    else:
        return render_template('sent.html')	

############################################################################################################

@app.route('/<email>',methods=['GET','POST'])
def get(email):
    para2=[]
    if request.method=='POST':#Taking inputs for cancer prediction

        # Taking inputs for Survival prediction
        Age=request.form['age']
        para2.append(Age)
        Operationyear=request.form['operation_year']
        para2.append(Operationyear)
        axillarynodes=request.form['axillarynodes']
        para2.append(axillarynodes)
        return redirect(url_for('survival',para2=para2,email=email))
    return render_template('login1.html')

############################################################################################################
	
@app.route('/uploads/<para2>/<email>')
def survival(para2,email):
    with warnings.catch_warnings():
        para2=ast.literal_eval(para2)
        para2=[n.strip() for n in para2]
        P2=[]
        for i in para2:
            P2.append(float(i)) 
        warnings.simplefilter("ignore")
        #Loading the model
        LS=pickle.load(open('cancer_model2.sav','rb'))
        #Loading dataset for accuracy calculation
        hbr=pd.read_csv(path+'/haberman.csv',delimiter=",")

    #splitting the data into  and testing data
        y1=np.array(hbr['Survival_Status'])
        x1=np.array(hbr.drop('Survival_Status',axis=1))
        x1_train,x1_test,y1_train,y1_test=train_test_split(x1,y1,test_size=0.20,random_state=20)
        pred=LS.predict(x1_test)
        print(pred)
        Accuracy=accuracy_score(y1_test,pred)

    #printing the classification report
    print('classification_report' +classification_report(y1_test,pred))
    
    print(P2)
    Accuracy=str(Accuracy)

    msg=Message('Survival Prediction',recipients=[email])
    B=LS.predict([P2])
    if B==['0']:
        msg.body='Patient will survive less than 5 years ; Accuracy ='+ Accuracy
    else:
        msg.body=' Patient will survive more than 5 years; Accuracy ='+ Accuracy
    mail.send(msg)
    return render_template('sent.html')

############################################################################################################
  
if __name__=='__main__':
    app.run(debug=True)
