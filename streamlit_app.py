import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data"
columns = ['class', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
           'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
           'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
           'stalk-surface-below-ring', 'stalk-color-above-ring',
           'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number',
           'ring-type', 'spore-print-color', 'population', 'habitat']
df = pd.read_csv(url, names=columns)

# Data cleaning and preprocessing
encoder = LabelEncoder()
for col in df.columns:
    df[col] = encoder.fit_transform(df[col])

# Define features and target variable
X = df.drop(['class'], axis=1)
y = df['class']

# Building the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Function to predict mushroom edibility
def predict_edibility(cap_shape, cap_surface, cap_color, bruises, odor,
                      gill_attachment, gill_spacing, gill_size, gill_color,
                      stalk_shape, stalk_root, stalk_surface_above_ring,
                      stalk_surface_below_ring, stalk_color_above_ring,
                      stalk_color_below_ring, veil_type, veil_color,
                      ring_number, ring_type, spore_print_color,
                      population, habitat):
    """
    Predicts the edibility of mushrooms based on user input.
    
    Parameters:
    - cap_shape: Shape of the mushroom cap.
    - cap_surface: Surface texture of the mushroom cap.
    - cap_color: Color of the mushroom cap.
    - bruises: Whether the mushroom has bruises.
    - odor: Odor of the mushroom.
    - gill_attachment: Attachment of gills to the stalk.
    - gill_spacing: Spacing of gills.
    - gill_size: Size of gills.
    - gill_color: Color of gills.
    - stalk_shape: Shape of the stalk.
    - stalk_root: Root type of the stalk.
    - stalk_surface_above_ring: Surface texture of the stalk above the ring.
    - stalk_surface_below_ring: Surface texture of the stalk below the ring.
    - stalk_color_above_ring: Color of the stalk above the ring.
    - stalk_color_below_ring: Color of the stalk below the ring.
    - veil_type: Type of veil covering the mushroom.
    - veil_color: Color of the veil.
    - ring_number: Number of rings on the stalk.
    - ring_type: Type of ring on the stalk.
    - spore_print_color: Color of the spore print.
    - population: Population density of mushrooms.
    - habitat: Habitat where the mushroom is found.
    
    Returns:
    - prediction: Predicted class label (0 for edible, 1 for poisonous).
    - probability: Probability of the predicted class.
    """
    input_data = pd.DataFrame([[cap_shape, cap_surface, cap_color, bruises, odor,
                                gill_attachment, gill_spacing, gill_size, gill_color,
                                stalk_shape, stalk_root, stalk_surface_above_ring,
                                stalk_surface_below_ring, stalk_color_above_ring,
                                stalk_color_below_ring, veil_type, veil_color,
                                ring_number, ring_type, spore_print_color,
                                population, habitat]])
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)

    return prediction, probability

# Sidebar for user input
st.sidebar.header('Enter Mushroom Characteristics')
cap_shape = st.sidebar.selectbox('Cap Shape', df['cap-shape'].unique())
cap_surface = st.sidebar.selectbox('Cap Surface', df['cap-surface'].unique())
cap_color = st.sidebar.selectbox('Cap Color', df['cap-color'].unique())
bruises = st.sidebar.selectbox('Bruises', df['bruises'].unique())
odor = st.sidebar.selectbox('Odor', df['odor'].unique())
gill_attachment = st.sidebar.selectbox('Gill Attachment', df['gill-attachment'].unique())
gill_spacing = st.sidebar.selectbox('Gill Spacing', df['gill-spacing'].unique())
gill_size = st.sidebar.selectbox('Gill Size', df['gill-size'].unique())
gill_color = st.sidebar.selectbox('Gill Color', df['gill-color'].unique())
stalk_shape = st.sidebar.selectbox('Stalk Shape', df['stalk-shape'].unique())
stalk_root = st.sidebar.selectbox('Stalk Root', df['stalk-root'].unique())
stalk_surface_above_ring = st.sidebar.selectbox('Stalk Surface Above Ring', df['stalk-surface-above-ring'].unique())
stalk_surface_below_ring = st.sidebar.selectbox('Stalk Surface Below Ring', df['stalk-surface-below-ring'].unique())
stalk_color_above_ring = st.sidebar.selectbox('Stalk Color Above Ring', df['stalk-color-above-ring'].unique())
stalk_color_below_ring = st.sidebar.selectbox('Stalk Color Below Ring', df['stalk-color-below-ring'].unique())
veil_type = st.sidebar.selectbox('Veil Type', df['veil-type'].unique())
veil_color = st.sidebar.selectbox('Veil Color', df['veil-color'].unique())
ring_number = st.sidebar.selectbox('Ring Number', df['ring-number'].unique())
ring_type = st.sidebar.selectbox('Ring Type', df['ring-type'].unique())
spore_print_color = st.sidebar.selectbox('Spore Print Color', df['spore-print-color'].unique())
population = st.sidebar.selectbox('Population', df['population'].unique())
habitat = st.sidebar.selectbox('Habitat', df['habitat'].unique())

# Function to validate inputs are letters 'a' to 'z'
def validate_letter_input(input_value):
    if input_value.isalpha():
        return input_value.lower()
    else:
        st.warning('Please enter alphabetic characters only.')
        return ''

# Convert inputs to lowercase alphabetic characters
cap_shape = validate_letter_input(cap_shape)
cap_surface = validate_letter_input(cap_surface)
cap_color = validate_letter_input(cap_color)
odor = validate_letter_input(odor)
gill_color = validate_letter_input(gill_color)
stalk_shape = validate_letter_input(stalk_shape)
stalk_root = validate_letter_input(stalk_root)
veil_color = validate_letter_input(veil_color)
ring_type = validate_letter_input(ring_type)
spore_print_color = validate_letter_input(spore_print_color)
habitat = validate_letter_input(habitat)

# Main panel
st.title("Mushroom Edibility Prediction App")
st.markdown("""
This app predicts the **edibility** of mushrooms based on various characteristics.
""")

# Predict button for user input
if st.sidebar.button('Predict'):
    prediction, probability = predict_edibility(cap_shape, cap_surface, cap_color, bruises, odor,
                                                gill_attachment, gill_spacing, gill_size, gill_color,
                                                stalk_shape, stalk_root, stalk_surface_above_ring,
                                                stalk_surface_below_ring, stalk_color_above_ring,
                                                stalk_color_below_ring, veil_type, veil_color,
                                                ring_number, ring_type, spore_print_color,
                                                population, habitat)

    # Display prediction
    st.subheader('Prediction')
    edibility = 'Poisonous' if prediction[0] == 1 else 'Edible'
    st.write(f"Based on the provided information, the mushroom is predicted to be **{edibility}**.")

    # Display prediction probabilities
    st.subheader('Prediction Probabilities')
    st.write(f"Probability of Edible: {probability[0][0]:.2f}")
    st.write(f"Probability of Poisonous: {probability[0][1]:.2f}")

# Footer
st.markdown("""
---
App developed by [Your Name]
""")
