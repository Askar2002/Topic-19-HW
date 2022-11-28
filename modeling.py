import streamlit as st

# for ML model
import numpy as np 
import pandas as pd
from math import sqrt
from scipy.stats import skew
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

uploaded_file = st.file_uploader("train.csv")
if uploaded_file is not None:
    train = pd.read_csv("train.csv")

    # === encoding ===
    # encode categorical variables
    le = preprocessing.LabelEncoder()
    for name in train.columns:
        if train[name].dtypes == 'O':
         train[name] = train[name].astype(str)
         le.fit(train[name])
         train[name] = le.transform(train[name])
    
    # fill missing values based on probability of occurrence
    for column in train.columns:
        null_vals = train.isnull().values
        a, b = np.unique(train.values[~null_vals], return_counts = 1)
        train.loc[train[column].isna(), column] = np.random.choice(a, train[column].isnull().sum(), p = b / b.sum())

    # apply log transformation to reduce skewness over .75 by taking log(feature + 1)
    skewed_train = train.apply(lambda x: skew(x.dropna()))
    skewed_train = skewed_train[skewed_train > .75]
    train[skewed_train.index] = np.log1p(train[skewed_train.index])

    # === Modeling ===
    #Defisinisikan X dan Y
    X = train.drop(['SalePrice', 'Id'], axis = 1)
    y = train['SalePrice']

    #Train test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 1)
    
    # model
    from sklearn.linear_model import Ridge
    model = Ridge()
    model.fit(X_train, y_train)
    
    # make predictions based on model
    y_pred2 = model.predict(X_test)
    
    # plot
    # alpha helps to show overlapping data
    plt.scatter(y_pred2, y_test, alpha = 0.7, color = 'b')
    plt.xlabel('Predicted Price')
    plt.ylabel('Actual Price')
    plt.title('Linear Regression Model')
    
    st.write(train.head())
    st.pyplot()
