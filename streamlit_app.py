import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load and preprocess data
digits = datasets.load_digits()
X = digits.images
y = digits.target

# Flatten the images for the classifier
n_samples = len(X)
X = X.reshape((n_samples, -1))

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build the model
model = MLPClassifier(hidden_layer_sizes=(64, 64), max_iter=300, random_state=42)
model.fit(X_train, y_train)

# Sidebar for user input
st.sidebar.header('User Input Parameters')
index = st.sidebar.slider('Image index', 0, len(X_test) - 1, 0)

# Main Panel
st.write("""
# Digits Prediction App
This app uses a Neural Network to predict the digit in a given image.
""")

# Display the image
st.subheader('Input Image')
image = X_test[index].reshape(8, 8)
plt.imshow(image, cmap='gray')
plt.axis('off')
st.pyplot(plt)

# Predict the digit
st.subheader('Prediction')
prediction = model.predict([X_test[index]])
predicted_digit = prediction[0]
st.write(f'Predicted Digit: {predicted_digit}')

# Display prediction probabilities
st.subheader('Prediction Probabilities')
probabilities = model.predict_proba([X_test[index]])
st.write(probabilities)
