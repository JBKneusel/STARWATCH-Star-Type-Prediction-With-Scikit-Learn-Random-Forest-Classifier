""" Star type prediction with random forest classifier
	Author: Joseph Kneusel
	Model: Random Forest Classfier
"""
import pandas as pd 
import numpy as np 
import matplotlib as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def dataFrame():
	"""Import star CSV"""
	df = pd.read_csv("6 class csv.csv")
	print(df.isnull().sum())
	print(df["Luminosity(L/Lo)"])

	X = df.drop(columns=['Star type'])  # Features
	y = df['Star type']  # Target (classification labels)

def main():
	"""Classify star types"""



dataFrame()