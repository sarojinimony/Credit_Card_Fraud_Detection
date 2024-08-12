import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st

# Load and prepare the data
data = pd.read_csv('creditcard.csv')
legit = data[data.Class == 0]
fraud = data[data.Class == 1]

# Balancing the dataset
legit_sample = legit.sample(n=len(fraud), random_state=2)
data = pd.concat([legit_sample, fraud], axis=0)

X = data.drop(columns="Class", axis=1)
y = data["Class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

# Train the model
model = LogisticRegression(max_iter=1000)  # Increase max_iter if convergence issues
model.fit(X_train, y_train)

# Calculate and print accuracy
train_acc = accuracy_score(model.predict(X_train), y_train)
test_acc = accuracy_score(model.predict(X_test), y_test)

print(f"Training accuracy: {train_acc}")
print(f"Testing accuracy: {test_acc}")

# Streamlit app
st.title("Credit Card Fraud Detection")

# Input: Enter feature values
input_features = st.text_input('Enter all required feature values ')

# Button to submit the input
submit = st.button("Submit")

if submit:
    try:
        # Convert input to a list of floats
        input_features_list = list(map(float, input_features.split()))

        # Check if the number of features matches
        if len(input_features_list) != X_train.shape[1]:
            st.write(f"Error: Please enter exactly {X_train.shape[1]} features.")
        else:
            # Predict
            features = np.array(input_features_list, dtype=np.float64).reshape(1, -1)
            prediction = model.predict(features)

            # Output result
            if prediction[0] == 0:
                st.write("Legitimate transaction")
            else:
                st.write("Fraudulent transaction")
    except ValueError:
        st.write("Error: Please enter valid numerical values.")
