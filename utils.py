""" 
    🛠️ Project Name: Star Spectral Classification Prediction Application Utilizing an Sklearn Random Forest Classifier
    🖋️ Author: Joseph Kneusel
    📱 Contact: https://github.com/JBKneusel
    ⚖️ License: MIT License
    🤖 ML Models : Random Forest Classifier 
    🚧 Foundation: Python, Kivy, Sklearn, Numpy, Pandas, Matplotlib, Seaborn
"""
import os
import sys

def resource_path(relative_path):
    """ Resource specific path for building exe """
    if getattr(sys, 'frozen', False):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)