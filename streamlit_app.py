import streamlit as st
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target
df = pd.DataFrame(X, columns=iris.feature_names)
df['target'] = y

# Display column names for debugging
st.write(df.columns)

# Sidebar for user input
st.sidebar.header('User Input Parameters')

def user_input_features():
    sepal_length = st.sidebar.slider('Sepal length', float(df['sepal length (cm)'].min()), float(df['sepal length (cm)'].max()))
    sepal_width = st.sidebar.slider('Sepal width', float(df['sepal width (cm)'].min()), float(df['sepal width (cm)'].max()))
    petal_length = st.sidebar.slider('Petal length', float(df['petal length (cm)'].min()), float(df['petal length (cm)'].max()))
    petal_width = st.sidebar.slider('Petal width', float(df['petal width (cm)'].min()), float(df['petal width (cm)'].max()))
    data = {'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width}
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Main Panel
st.write("""
# Iris Flower Prediction App
This app predicts the **Iris flower** type!
""")

# Display the user input features
st.subheader('User Input parameters')
st.write(input_df)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Model Training and Evaluation
logreg = LogisticRegression(max_iter=200)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=iris.target_names)

st.subheader('Model Evaluation')
st.write('Accuracy: ', accuracy)
st.text('Classification Report:')
st.text(report)

# Optimizing Model using Grid Search
param_grid = {'C': [0.1, 1, 10, 100]}
grid = GridSearchCV(LogisticRegression(max_iter=200), param_grid, refit=True, verbose=0)
grid.fit(X_train, y_train)

st.subheader('Grid Search Results')
st.write('Best Parameters: ', grid.best_params_)
st.write('Best Estimator: ', grid.best_estimator_)

# Predicting on user input
prediction = grid.predict(input_df)
prediction_proba = grid.predict_proba(input_df)

st.subheader('Prediction')
st.write(iris.target_names[prediction])

st.subheader('Prediction Probability')
st.write(prediction_proba)
