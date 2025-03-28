""" Star type prediction with random forest classifier
	Author: Joseph Kneusel
	Model: Random Forest Classfier
"""

"""Libraries"""
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

def dataFrame():
    """Import star CSV and process data"""

    df = pd.read_csv("6 class csv.csv")

    ### Check for any missing values
    df.isnull().sum()
    
    ### Features (X) and target variable (y)
    X = df.drop(columns=['Spectral Class', 'Star type'])  ### Features (excluding non-predictive columns)
    y = df['Spectral Class']  ### Target (classification labels)
    
    ### Encode the target variable 'Spectral Class' with LabelEncoder
    label_enc_spectral = LabelEncoder()
    #print(df["Spectral Class"])
    df['Spectral Class'] = label_enc_spectral.fit_transform(df['Spectral Class'])
    #print(df['Spectral Class'])
    #exit(1)
    ### Store the class names for inverse transformation later
    spectral_classes = label_enc_spectral.classes_
    
    ### Encode the 'Star color' feature (to convert from string to numbers)
    label_enc_color = LabelEncoder()
    df['Star color'] = label_enc_color.fit_transform(df['Star color'])
    
    return X, y, spectral_classes

def main():
    """Classify star types and make predictions"""
    X, y, spectral_classes = dataFrame()

    ### Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    ### Create and train the Random Forest model
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    ### Make predictions on the test set
    predict = clf.predict(X_test)

    ### Calculate accuracy and print classification report
    accuracy = accuracy_score(y_test, predict)
    print(f"Accuracy: {accuracy:.2f}")
    print(classification_report(y_test, predict))
    
    ### Example: Make a prediction for a new star
    new_star = np.array([[6000, 1.5, 1.2, 4.5, 3]])  # Example feature values (adjust as needed)
    predicted_class = clf.predict(new_star)
    
    ### Display the predicted Spectral Class
    print("Predicted Spectral Class:", spectral_classes[predicted_class[0]])

main()