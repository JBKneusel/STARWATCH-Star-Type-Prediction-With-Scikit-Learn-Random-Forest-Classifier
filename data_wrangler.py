""" 
    Project Name: Star Spectral Classification Prediction Application Utilizing an Sklearn Random Forest Classifier
    Author: Joseph Kneusel
    Contact: https://github.com/JBKneusel
    License: MIT License
    ML Models: Random Forest Classifier
    Foundation: Python, Kivy, Sklearn, Numpy, Pandas, Matplotlib, Seaborn
"""

""" Libraries """
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix

def dataFrame():
    """Import star CSV and process data"""

    df = pd.read_csv("6 class csv.csv")

    # Check for any missing values
    df.isnull().sum().dropna()
    
    # Encode the target variable 'Spectral Class' with LabelEncoder
    label_enc_spectral = LabelEncoder()

    df['Spectral Class'] = label_enc_spectral.fit_transform(df['Spectral Class'])

    # Store the class names for inverse transformation later
    spectral_classes = label_enc_spectral.classes_
    
    # Encode the 'Star color' feature (to convert from string to numbers)
    label_enc_color = LabelEncoder()
    df['Star color'] = label_enc_color.fit_transform(df['Star color'])

    # Features (X) and target variable (y)
    X = df.drop(columns=['Spectral Class', 'Star type'])  ### Features (excluding non-predictive columns)
    y = df['Spectral Class']  ### Target (classification labels)
    
    return X, y, label_enc_color, label_enc_spectral

def train_model():
    """ Train our RandomForest """
    # Create dataframe
    X, y, label_enc_color, label_enc_spectral = dataFrame()

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and train the Random Forest model
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Make predictions on the test set
    predict = clf.predict(X_test)

    # Calculate accuracy and print classification report
    accuracy = accuracy_score(y_test, predict)
    print(f"Accuracy: {accuracy:.2f}")
    print(classification_report(y_test, predict))

    return clf, label_enc_color, label_enc_spectral, X.columns


def classifier(user_input, clf, label_enc_color, label_enc_spectral, feature_names, X_test, y_test, spectral_classes, predict):
    """Classify star types and make predictions"""
    
    # Encode "Star Color"
    color_str = user_input[0][4]
    user_input[0][4] = label_enc_color.transform([color_str])[0]

    # Predict the user's star
    predicted_class = clf.predict(user_input)
    
    # Display the predicted Spectral Class
    print("Predicted Spectral Class:", spectral_classes[predicted_class[0]])

    # Plot Confusion Matrix Heatmap
    conf_matrix = confusion_matrix(y_test, predict)

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Reds', cbar=False, 
               xticklabels=spectral_classes, yticklabels=spectral_classes)

    plt.title('Confusion Matrix Heatmap')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()

    # Plot Features that were important for prediction
    feature_importances = clf.feature_importances_

    plt.barh(X.columns, feature_importances, color="Red")
    plt.xlabel('Feature Importance')
    plt.title('Feature Importance in Random Forest Classifier')
    plt.show()