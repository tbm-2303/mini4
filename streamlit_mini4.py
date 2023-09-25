#!/usr/bin/env python
# coding: utf-8


import streamlit as st
import spacy_streamlit
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeClassifier
from sklearn import model_selection
from sklearn import tree
from sklearn.metrics import accuracy_score
import graphviz
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report



def main():
    st.title("MINI 4: Decision Tree")

    st.header("Upload your CSV data file")
    data_file = st.file_uploader("Upload CSV", type=["csv"])

    if data_file is not None:
        data = pd.read_csv(data_file)
        if data is not None: 
          lst=['Glucose','BloodPressure','SkinThickness','Insulin','BMI']
          for i in lst:
            data[i].replace(0, np.nan, inplace=True)
            data[i].fillna(data[i].median(), inplace=True)

        st.write("Data overview:")
        st.write(data.head())

        X = data.drop("Outcome", axis=1) # Features
        y = data["Outcome"] # Target variable
       
        # Split the dataset into into training and testing sets in proportion 8:2 
        #   80% of it as training data
        #   20% as a validation dataset
        set_prop = 0.2
        #  Initialize seed parameter for the random number generator used for the split
        seed = 7

        # Split
        X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=set_prop, random_state=seed)

        # Build Decision Trees Classifier 
        params = {'max_depth': 5}
        classifier = DecisionTreeClassifier(**params)
        # classifier = RandomForestClassifier(n_estimators = 100, max_depth = 6)
        classifier.fit(X_train, y_train)
        #st.write(classifier.fit(X_train, y_train)) # display classifier parameters

        dot_data = tree.export_graphviz(classifier, out_file=None, 
                         feature_names=data.columns[:8], class_names = True,        
                         filled=True, rounded=True, proportion = False,
                         special_characters=True) 
        st.write("Decision Tree:")
        st.graphviz_chart(dot_data)

        # Predict the labels of the test data
        y_testp = classifier.predict(X_test)
       
        # Calculated the accuracy of the model comparing the observed data and predicted data
        st.write("Accuracy is ", accuracy_score(y_test,y_testp))

        # Create confusion matrix
        confusion_mat = confusion_matrix(y_test,y_testp)
        
        # Visualize confusion matrix
        st.write("Confusion matrix:")
        plt.imshow(confusion_mat, interpolation='nearest')
        plt.title('Confusion matrix')
        plt.colorbar()
        ticks = np.arange(2)
        plt.xticks(ticks, ticks)
        plt.yticks(ticks, ticks)
        plt.ylabel('True labels')
        plt.xlabel('Predicted labels')
        st.pyplot(plt)

        # Visualize Heatmap of confusion matrix
        st.write("Heatmap of confusion matrix:")
        fig, ax = plt.subplots()
        sns.heatmap(confusion_mat, annot=True)
        st.write(fig)

        class_names = ['no-diabetes', 'diabetes']
        # Classifier performance on training dataset
        st.write("Classification report on training set:")
        st.write(classification_report(y_train, classifier.predict(X_train), target_names=class_names, output_dict=True))
        # Classifier performance on test dataset
        st.write("Classification report on test set:")
        st.write(classification_report(y_test, classifier.predict(X_test), target_names=class_names, output_dict=True))
      
        
        

        st.sidebar.header("Visualizations")
        plot_options = ["Bar plot", "Scatter plot", "Histogram", "Box plot"]
        selected_plot = st.sidebar.selectbox("Choose a plot type", plot_options)

        if selected_plot == "Bar plot":
            x_axis = st.sidebar.selectbox("Select x-axis", data.columns)
            y_axis = st.sidebar.selectbox("Select y-axis", data.columns)
            st.write("Bar plot:")
            fig, ax = plt.subplots()
            sns.barplot(x=data[x_axis], y=data[y_axis], ax=ax)
            st.pyplot(fig)

        elif selected_plot == "Scatter plot":
            x_axis = st.sidebar.selectbox("Select x-axis", data.columns)
            y_axis = st.sidebar.selectbox("Select y-axis", data.columns)
            st.write("Scatter plot:")
            fig, ax = plt.subplots()
            sns.scatterplot(x=data[x_axis], y=data[y_axis], ax=ax)
            st.pyplot(fig)

        elif selected_plot == "Histogram":
            column = st.sidebar.selectbox("Select a column", data.columns)
            bins = st.sidebar.slider("Number of bins", 5, 100, 20)
            st.write("Histogram:")
            fig, ax = plt.subplots()
            sns.histplot(data[column], bins=bins, ax=ax)
            st.pyplot(fig)

        elif selected_plot == "Box plot":
            column = st.sidebar.selectbox("Select a column", data.columns)
            st.write("Box plot:")
            fig, ax = plt.subplots()
            sns.boxplot(data[column], ax=ax)
            st.pyplot(fig)


main()

