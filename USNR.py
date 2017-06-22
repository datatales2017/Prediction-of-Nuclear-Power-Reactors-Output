# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 08:11:29 2017

@author: Home
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_excel('D:\\6june2017\\US_Nuclear_reactors\\appa.xls',encoding = 'ISO-8859-1')
columns = list(data)

data = dataset[['Plant Name, Unit Number','NRC Reactor Unit Web Page ',
'Docket  Number','Location','NRC Region','Licensee','Reactor and Containment Type ',
'Nuclear Steam System Supplier and Design Type ','Architect-Engineer','Contructor',
'Construction Permit Issued','Operating License Issued','Commercial Operation',
'Renewed Operating License Issued','Operating License Expires','Licensed MWt',
'Note','2014 Capacity Factor (Percent)','2013 Capacity Factor (Percent)',
'2012 Capacity Factor (Percent)','2011 Capacity Factor (Percent)',
'2010 Capacity Factor (Percent)','2009 Capacity Factor (Percent)',
'2008 Capacity Factor (Percent)','2005 Capacity Factor (Percent)',
'2004 Capacity Factor (Percent)','2003 Capacity Factor (Percent)']]

nulls = data.apply(lambda x: sum(x.isnull()),axis=0)

#imputing the null values
data['2014 Capacity Factor (Percent)'] = data['2014 Capacity Factor (Percent)'].fillna((data['2014 Capacity Factor (Percent)']).mean(),axis=0)
data['2013 Capacity Factor (Percent)'] = data['2013 Capacity Factor (Percent)'].fillna((data['2013 Capacity Factor (Percent)']).mean(),axis=0)
data['2012 Capacity Factor (Percent)'] = data['2012 Capacity Factor (Percent)'].fillna((data['2012 Capacity Factor (Percent)']).mean(),axis=0)
data['2011 Capacity Factor (Percent)'] = data['2011 Capacity Factor (Percent)'].fillna((data['2011 Capacity Factor (Percent)']).mean(),axis=0)
data['2010 Capacity Factor (Percent)'] = data['2010 Capacity Factor (Percent)'].fillna((data['2010 Capacity Factor (Percent)']).mean(),axis=0)
data['2009 Capacity Factor (Percent)'] = data['2009 Capacity Factor (Percent)'].fillna((data['2009 Capacity Factor (Percent)']).mean(),axis=0)
data['2008 Capacity Factor (Percent)'] = data['2008 Capacity Factor (Percent)'].fillna((data['2008 Capacity Factor (Percent)']).mean(),axis=0)
data['2005 Capacity Factor (Percent)'] = data['2005 Capacity Factor (Percent)'].fillna((data['2005 Capacity Factor (Percent)']).mean(),axis=0)
data['2004 Capacity Factor (Percent)'] = data['2004 Capacity Factor (Percent)'].fillna((data['2004 Capacity Factor (Percent)']).mean(),axis=0)
data['2003 Capacity Factor (Percent)'] = data['2003 Capacity Factor (Percent)'].fillna((data['2003 Capacity Factor (Percent)']).mean(),axis=0)

nulls1 = data.apply(lambda x: sum(x.isnull()),axis=0)

y = data['2014 Capacity Factor (Percent)']
x = data[['2013 Capacity Factor (Percent)','2012 Capacity Factor (Percent)',
'2011 Capacity Factor (Percent)','2010 Capacity Factor (Percent)',
'2009 Capacity Factor (Percent)','2008 Capacity Factor (Percent)',
'2005 Capacity Factor (Percent)','2004 Capacity Factor (Percent)',
'2003 Capacity Factor (Percent)']]
#'Licensed MWt'---------removed
#Creating dummies
x1 = pd.get_dummies(data['Reactor and Containment Type '])
x2 = pd.get_dummies(data['Nuclear Steam System Supplier and Design Type '])
x3 = pd.get_dummies(data['Architect-Engineer'])
x4 = pd.get_dummies(data['Contructor'])
x5 = pd.get_dummies(data['NRC Region'])


X = np.concatenate((x,x1,x2,x3,x4,x5),axis=1)

#Spliting the dataset
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1,random_state=0)

#Creating the model using randomforest
from sklearn.ensemble import RandomForestRegressor
reg_rfr = RandomForestRegressor(max_depth=19)
reg_rfr.fit(X_train,y_train)
y_pred1 = reg_rfr.predict(X_test)
S2 = reg_rfr.score(X_train,y_train)

from sklearn.ensemble import ExtraTreesRegressor
reg_etr = ExtraTreesRegressor(max_depth=20)
reg_etr.fit(X_train,y_train)
y_pred2 = reg_etr.predict(X_test)
S1 = reg_etr.score(X_train,y_train)

from sklearn.svm import SVR
reg_svr = SVR()
reg_svr.fit(X_train,y_train)
y_pred3 = reg_svr.predict(X_test)
S = reg_svr.score(X_train,y_train)

from sklearn.grid_search import GridSearchCV
parameters = [{'max_depth':np.arange(1,21)}]
CV = GridSearchCV(estimator = reg_etr,
                  param_grid = parameters,
                  cv=10)
CV.fit(X_train,y_train)
CV_score = CV.score(X_train,y_train)
best_score = CV.best_score_
CV.best_params_