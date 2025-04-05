""" 
    🛠️ Project Name: Star Spectral Classification Prediction Application Utilizing an Sklearn Random Forest Classifier
    🖋️ Author: Joseph Kneusel
    📱 Contact: https://github.com/JBKneusel
    ⚖️ License: MIT License
    🤖 ML Models : Random Forest Classifier 
    🚧 Foundation: Python, Kivy, Sklearn, Numpy, Pandas, Matplotlib, Seaborn
"""

"""File Sharing"""
from data_wrangler import classifier, train_model, dataFrame

""" Libraries """
import numpy as np
from kivy.app import App
from kivy.clock import mainthread
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.uix.textinput import TextInput
from kivy.uix.scrollview import ScrollView
from kivy.uix.anchorlayout import AnchorLayout
from functools import wraps

# Set Kivy Window Size
from kivy.core.window import Window
Window.size = (1000, 500)

class StarUI(BoxLayout):
    """🌟 Central UI Class for StarWatch 🌟"""
    def __init__(self, **var_args):
        super(StarUI, self).__init__(orientation='vertical', padding=10, spacing=2, **var_args)

        #-------- Model Setup --------#

        self.clf, self.label_enc_color, self.label_enc_spectral, self.feature_columns, self.X_test, self.y_test, self.predict = train_model()

        # # ------- Left Side: Image ------- #
        # self.image = Image(source='star_diagram.png', size_hint=(0.5, 1))  # Adjust image size hint
        # self.add_widget(self.image)

        #TODO: Get image working for left side 

        #-------- Input Fields Layout --------#

        # Observed Star Temperature Input Box
        self.temperature_box = TextInput(hint_text="Enter Calculated Star Temperature (K)", size_hint=(0.7, 1), multiline=False) 
        self.add_widget(self.temperature_box)

        # Observed Star Luminosity Input Box
        self.luminosity_box = TextInput(hint_text="Enter Calculated Star Luminosity (L/Lo)", size_hint=(0.7, 1), multiline=False) 
        self.add_widget(self.luminosity_box)

        # Observed Star Radius Input Box
        self.radius_box = TextInput(hint_text="Enter Estimated Star Radius (R/Ro)", size_hint=(0.7, 1), multiline=False) 
        self.add_widget(self.radius_box)

        # Observed Star Absolute Magnitude Input Box
        self.magnitude_box = TextInput(hint_text="Enter Calculated Star Absolute magnitude(Mv)", size_hint=(0.7, 1), multiline=False) 
        self.add_widget(self.magnitude_box)

        # Observed Star Color Input Box
        self.color_box = TextInput(hint_text="Enter Observed Star Color", size_hint=(0.7, 1), multiline=False) 
        self.add_widget(self.color_box)

        #-------- Button Layout --------#

        # Submit Star Button
        self.star_button = Button(text="Submit Star", on_press=self.try_star, size_hint=(0.3, 1), background_color=(0.2, 0.6, 1, 1), bold=True) 
        self.add_widget(self.star_button)

        # Help Button
        self.help_button = Button(text="Usage Instructions", on_press=self.help_menu, size_hint=(0.3, 1), background_color=(0.2, 0.6, 1, 1), bold=True) 
        self.add_widget(self.help_button)

        #-------- Output Fields Layout --------#

        self.result_label = Label(text="Output", font_size=12)
        self.add_widget(self.result_label)

    @mainthread # Runs method of UI thread:
    def display_to_user(self, result):
        """📝 This handles anything we need displayed for the user """
        self.result_label.text = f"Predicted Spectral Class: {result}"

    def safe_action(method):
        """🛡️ Decorator to catch and display errors from UI actions."""
        @wraps(method)
        def wrapper(self, *args, **kwargs):
            try:
                return method(self, *args, **kwargs)
            except ValueError:
                self.display_to_user("Input numerical values for all fields except color.")
            except Exception as err:
                self.display_to_user(f"Error: {err}")
        return wrapper

    def help_menu(self, instance):
        """🧭 Help menu, offers help to Noobies who can't fill out a form """
        self.result_label.text = "\n\nInput Example: 34000,6.3,1.0,1,0,Blue"

    @safe_action
    def try_star(self, instance):
        """🔭 Gets Temperature (K), Luminosity(L/Lo), Radius(R/Ro), Absolute magnitude(Mv), Star color """

        # Strip user inputs and store as floats 
        temperature = float(self.temperature_box.text.strip())
        luminosity = float(self.luminosity_box.text.strip())
        radius = float(self.radius_box.text.strip())
        magnitude = float(self.magnitude_box.text.strip())
            
        # Color is not numerical, so keep it like this.
        color = self.color_box.text.strip()

        # Input values into numpy array for analysis
        new_star = np.array([[temperature, luminosity, radius, magnitude, color]])

        # Call main on our new star to predict its spectral class
        result = classifier(
        new_star,
        self.clf,
        self.label_enc_color,
        self.label_enc_spectral,
        self.feature_columns,
        self.X_test,
        self.y_test,
        spectral_classes=self.label_enc_spectral.classes_,
        predict=self.predict
        )

        # Display result to user screen
        self.display_to_user(result)

class StarApp(App):
    def build(self):
        # return StarUI as root widget
        return StarUI()