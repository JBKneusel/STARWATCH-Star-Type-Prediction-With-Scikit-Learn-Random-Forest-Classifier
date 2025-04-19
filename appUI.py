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
import random
from kivy.app import App
from kivy.clock import mainthread
from kivy.graphics import Color, RoundedRectangle
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.popup import Popup
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.uix.textinput import TextInput
from kivy.uix.scrollview import ScrollView
from kivy.uix.anchorlayout import AnchorLayout
from functools import wraps
from kivy.clock import Clock
from kivy.core.window import Window

class StarWrapper(FloatLayout):
    """🌟 Floating UI Wrapper 🌟"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    #-------- Configure UI Wrapper--------#

        self.ui = StarUI(wrapper=self)
        self.add_widget(self.ui)

    #-------- Loading Gif Layout--------#

        self.loading_gif = Image(source="LoadingGif.gif", anim_delay=0.05, size_hint=(None,None), size=(300,300), pos_hint={'center_x':0.5, 'center_y': 0.5}, opacity=0)
        self.add_widget(self.loading_gif)

    def show_loader(self, show=True):
        """Show the Loader"""
        self.loading_gif.opacity = 1 if show else 0

class StarUI(BoxLayout):
    """🌟 Central UI Class for StarWatch 🌟"""
    def __init__(self, wrapper=None, **var_args):
        super(StarUI, self).__init__(orientation='vertical', padding=10, spacing=2, **var_args)

    #-------- Reference FloatLayout Wrapper--------#

        self.wrapper = wrapper

    #-------- Model Setup --------#

        self.clf, self.label_enc_color, self.label_enc_spectral, self.feature_columns, self.X_test, self.y_test, self.predict = train_model()

    # ------- Central: Image ------- #

        # Render and Image of the Hertzsprung-Russell Diagram
        self.image = Image(source='HR.jfif', size_hint_x=1, size_hint_y=1, pos_hint={'x': 0, 'top': 1})  # Adjust image size hint
        self.add_widget(self.image)

        # 🔧🔧🔧 IMPROVE: Add more image support

    #-------- Input Fields Layout --------#

        # Observed Star Temperature Input Box
        self.temperature_box = TextInput(hint_text="Enter Calculated Star Temperature (K)", size_hint_x=1, size_hint_y=0.1, font_size=12, multiline=False) 
        self.add_widget(self.temperature_box)

        # Observed Star Luminosity Input Box
        self.luminosity_box = TextInput(hint_text="Enter Calculated Star Luminosity (L/Lo)", size_hint_x=1, size_hint_y=0.1, font_size=12, multiline=False) 
        self.add_widget(self.luminosity_box)

        # Observed Star Radius Input Box
        self.radius_box = TextInput(hint_text="Enter Estimated Star Radius (R/Ro)", size_hint_x=1, size_hint_y=0.1, font_size=12, multiline=False) 
        self.add_widget(self.radius_box)

        # Observed Star Absolute Magnitude Input Box
        self.magnitude_box = TextInput(hint_text="Enter Calculated Star Absolute magnitude(Mv)", size_hint_x=1, size_hint_y=0.1, font_size=12, multiline=False) 
        self.add_widget(self.magnitude_box)

        # Observed Star Color Input Box
        self.color_box = TextInput(hint_text="Enter Observed Star Color", size_hint_x=1, size_hint_y=0.1, multiline=False, font_size=12) 
        self.add_widget(self.color_box)

    #-------- Button Layout --------#

        # Submit Star Button
        self.star_button = Button(text="Submit Star", on_press=self.speak_to_wrapper, size_hint_x=1, size_hint_y=0.1, background_color=(0.2, 0.6, 1, 1), bold=True) 
        self.add_widget(self.star_button)

        # Help Button
        self.help_button = Button(text="Usage Instructions", on_press=self.help_menu, size_hint_x=1, size_hint_y=0.1, background_color=(0.2, 0.6, 1, 1), bold=True) 
        self.add_widget(self.help_button)

    #-------- Output Fields Layout --------#

        self.result_label = Label(text="Output", size_hint_x=1, size_hint_y=0.3, font_size=12)
        self.add_widget(self.result_label)

    #-------- Safety Methods --------#

    def safe_action(method):
        """🛡️ Decorator to catch and display errors from UI actions."""
        @wraps(method)
        def wrapper(self, *args, **kwargs):
            try:
                return method(self, *args, **kwargs)
            except ValueError:
                self.display_to_user("Input numerical values.\n For all fields except color.")
            except Exception as err:
                self.display_to_user(f"Error: {err} No or incorrect input.\n See Usage Instructions for more.")
        return wrapper

    def safe_loader(method):
        """🛡️ Decorator to catch and display errors from UI loader behavior."""
        @wraps(method)
        def wrapper(self, *args, **kwargs):
            try:
                if self.wrapper:
                    self.wrapper.show_loader(False) # Kill loading gif if err 
                return method(self, *args, **kwargs)
            except ValueError:
                self.display_to_user("Enter a valid input.") 
            except Exception as err:
                self.display_to_user(f"Error: {err} No or incorrect input.\n See Usage Instructions for more.")
        return wrapper

    #-------- UI Interactions --------#

    @mainthread # Runs method of UI thread:
    def display_to_user(self, result):
        """📝 This handles anything we need displayed for the user """
        self.result_label.text = f"Predicted Spectral Class: {result}"

    @safe_action
    def help_menu(self, instance):
        """🧭 Help menu, offers help to Noobies who can't fill out a form through a popup """
        help_layout = BoxLayout(orientation='vertical', padding=10, spacing=5)
        #help_layout.add_widget(Label())

        help_text = (
        "Enter the following:\n\n"
        "- Temperature (Kelvin), Example: 34000\n\n"
        "- Luminosity (L/Lo), Example: 6.3\n\n"
        "- Radius (R/Ro), Example: 1.0\n\n"
        "- Absolute Magnitude (Mv), Example: 1\n\n"
        "- Color (e.g., Blue, Yellow, Red), Example: Blue\n"
        )

        help_label = Label(text=help_text, font_size=12)
        help_layout.add_widget(help_label)

        # Help menu close button functionality
        close_button = Button(text='Exit', size_hint_y=0.3)
        help_layout.add_widget(close_button)

        # Popup layout
        popup = Popup(
        title="Spectral Watch Usage Instructions",
        content=help_layout,
        size_hint=(None, None),
        size=(320, 400),
        auto_dismiss=False
        )

        close_button.bind(on_press=popup.dismiss)
        popup.open()

    @safe_loader
    def speak_to_wrapper(self, instance):
        """Handles UI Wrapper to Show Loader"""
        if self.wrapper:
            self.wrapper.show_loader(True)

        # Kivy native: schedules noted funct to run in delta-time with a delay of n seconds
        Clock.schedule_once(lambda dt: self.try_star_delayed(dt), random.randint(1,8))

    @safe_loader
    def try_star_delayed(self, dt):
         """This is the method that runs after a 2-second delay."""
         self.try_star(None)  # You can pass `None` since the method doesn't require the event parameter in this case

        # Hide the loader after 2 seconds delay
         if self.wrapper:
             self.wrapper.show_loader(False)

    @safe_action
    def try_star(self, instance):
        """🔭 Gets Temperature (K), Luminosity(L/Lo), Radius(R/Ro), Absolute magnitude(Mv), Star color """

        # Strip user inputs and store as floats 
        temperature = float(self.temperature_box.text.strip())
        luminosity = float(self.luminosity_box.text.strip())
        radius = float(self.radius_box.text.strip())
        magnitude = float(self.magnitude_box.text.strip())
            
        # Dealing with the color input is more tricky. we need to 
        color = self.color_box.text.strip().capitalize()

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

        print("New star array:", new_star)
        # Display result to user screen
        self.display_to_user(result)

class StarApp(App):
    def build(self):
        Window.title = "Spectral Watch"
        Window.size = (500, 720)
        Window.clearcolor = (9/255, 17/255, 56/255, 1)

        icon = '108-1088060_3144-x-3003-12-solar-system-planets-clipart.png'
        # return StarUI as root widget
        return StarWrapper()