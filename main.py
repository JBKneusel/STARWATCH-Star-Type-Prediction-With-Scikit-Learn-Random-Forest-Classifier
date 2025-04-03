""" 
    Project Name: Star Spectral Classification Prediction Application Utilizing an Sklearn Random Forest Classifier
	Author: Joseph Kneusel
    Contact: https://github.com/JBKneusel
    License: MIT License
	ML Models: Random Forest Classifier
    Foundation: Python, Kivy, Sklearn, Numpy, Pandas, Matplotlib, Seaborn
"""

"""Libraries"""
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.uix.textinput import TextInput
from kivy.uix.scrollview import ScrollView


class StarUI(BoxLayout):
    """Central Class for Tournament UI"""
    def __init__(self, **var_args):
        super(StarUI, self).__init__(**var_args)
        #-------- Application Layout --------
        self.cols = 5 
        self.rows = 2

        #-------- Input Fields Layout --------

        # Observed Star Temperature Input Box
        self.temperature_box = TextInput(hint_text="Enter Calculated Star Temperature (K)", size_hint=(0.7, 1), multiline=False) 
        self.add_widget(self.temperature_box)

        # Observed Star Temperature Input Box
        self.luminosity_box = TextInput(hint_text="Enter Calculated Star Luminosity (L/Lo)", size_hint=(0.7, 1), multiline=False) 
        self.add_widget(self.luminosity_box)

        # Observed Star Temperature Input Box
        self.radius_box = TextInput(hint_text="Enter Estimated Star Radius (R/Ro)", size_hint=(0.7, 1), multiline=False) 
        self.add_widget(self.radius_box)

        # Observed Star Temperature Input Box
        self.magnitude_box = TextInput(hint_text="Enter Calculated Star Absolute magnitude(Mv)", size_hint=(0.7, 1), multiline=False) 
        self.add_widget(self.magnitude_box)

        # Observed Star Temperature Input Box
        self.color_box = TextInput(hint_text="Enter Observed Star Color", size_hint=(0.7, 1), multiline=False) 
        self.add_widget(self.color_box)
        #TODO

        #-------- Button Layout --------

        # Submit Star Button
        self.star_button = Button(text="Submit Star", on_press=self.try_star, size_hint=(0.3, 1), background_color=(0.2, 0.6, 1, 1), bold=True) 
        self.add_widget(self.star_button)

        # Help Button
        self.help_button = Button(text="Usage Instructions", on_press=self.help_menu, size_hint=(0.3, 1), background_color=(0.2, 0.6, 1, 1), bold=True) 
        self.add_widget(self.help_button)
        #TODO

    def help_menu(self, instance):
        """ Help Menu """
        #TODO


    def try_star(self, instance):
        """ Gets Temperature (K), Luminosity(L/Lo), Radius(R/Ro), Absolute magnitude(Mv), Star color """
        Temperature = self.temperature_box.text.strip()
        Luminosity = self.temperature_box.text.strip()
        #TODO

def dataFrame():
    """Import star CSV and process data"""

    df = pd.read_csv("6 class csv.csv")

    # Check for any missing values
    df.isnull().sum().dropna()
    
    # Encode the target variable 'Spectral Class' with LabelEncoder
    label_enc_spectral = LabelEncoder()
    #print(df["Spectral Class"])
    df['Spectral Class'] = label_enc_spectral.fit_transform(df['Spectral Class'])
    #print(df['Spectral Class'])
    #exit(1)
    # Store the class names for inverse transformation later
    spectral_classes = label_enc_spectral.classes_
    
    # Encode the 'Star color' feature (to convert from string to numbers)
    label_enc_color = LabelEncoder()
    df['Star color'] = label_enc_color.fit_transform(df['Star color'])

    # Features (X) and target variable (y)
    X = df.drop(columns=['Spectral Class', 'Star type'])  ### Features (excluding non-predictive columns)
    y = df['Spectral Class']  ### Target (classification labels)
    
    return X, y, spectral_classes

def main(user_input):
    """Classify star types and make predictions"""
    new_star = np.array([[3000, 1.5, 1.2, 4.5, 3]])  # Example feature values (adjust as needed)
    
    # Create dataframe
    X, y, spectral_classes = dataFrame()

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
    
    # Example: Make a prediction for a new star
    
    predicted_class = clf.predict(new_star)
    
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


class StarApp(App):
    def build(self):
        # return StarUI as root widget
        return StarUI()
 
if __name__ == '__main__':
    StarApp().run()