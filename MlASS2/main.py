# This is a sample Python script.
import os

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import pandas as pd
import random
import seaborn as sns
import scipy
from scipy.stats import pearsonr
import sklearn
from sklearn import datasets, linear_model
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from scipy.stats import zscore
import pickle

import pandas as pd
from sklearn.decomposition import PCA


def load_model(model_path):
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)
    return model




if __name__ == '__main__':


    scaler = MinMaxScaler()
    #loads the test data
    all_df = pd.read_excel('testData/TestDatasetExample.xls', sheet_name='Sheet1')
    #loads the regression model
    reg_model = pickle.load(open('Models/reg.pkl', 'rb'))

    #loads the same PCA model used from training
    pca_model_path = 'Models/Pcamodel_pickle.pkl'
    pca_model = load_model(pca_model_path)

    #removes any rows with 999 as value
    df1 = all_df[~all_df.isin([999]).any(axis=1)]
    #appends the remaining rows IDs to a Id list
    ID_list = df1['ID'].tolist()
    #Drops ID column
    df1 = df1.drop('ID', axis=1)

    # Resetting the index
    df1.reset_index(drop=True, inplace=True)
    df1.loc[:, 'Age'] = df1['Age'].round().astype(int)
    Xmri = df1.iloc[:, 10:]
    z_scores = zscore(Xmri)
    # Define a threshold for outlier detection (e.g., 3 standard deviations)
    threshold = 3
    # Create a mask identifying outliers
    outlier_mask = np.abs(z_scores) > threshold
    # Replacing outliers with the mean of the corresponding feature
    Xmri_no_outliers = Xmri.copy()
    Xmri_no_outliers[outlier_mask] = np.nan  # Replace outliers with NaN
    Xmri_no_outliers = Xmri_no_outliers.fillna(Xmri.mean())  # Replace NaN with mean

    #min max normalisation
    Xmri1 = scaler.fit_transform(Xmri_no_outliers)
    Xmri2 = pd.DataFrame(Xmri1, columns=Xmri_no_outliers.columns)
    Xclinical = df1.iloc[:, 0:10]
    Xclinical1 = scaler.fit_transform(Xclinical)
    Xclinical2 = pd.DataFrame(Xclinical1, columns=Xclinical.columns)
    X_1 = pd.concat([Xclinical2, Xmri2], axis=1)
    #Does PCA to reduce features to 23 features
    newdata_transformed = pca_model.transform(X_1)
    # turn it into database
    X_1 = pd.DataFrame(newdata_transformed)



    predictions = []
    for index, row in X_1.iterrows(): #makes prediction on each row of the preprocessed data and appends to a list
        df_row = pd.DataFrame([row])

        prediction  =  reg_model.predict(df_row)

        predictions.append(prediction[0])



    #turns ID list and predictions list into a data frame
    data = {'ID': ID_list, 'prediction': predictions}
    dfprediction = pd.DataFrame(data)


    excel_file_path = 'Output/regression_results.xlsx'

    #exports data frame to a excel file
    dfprediction.to_excel(excel_file_path)
    print("result printed to:" + excel_file_path)
