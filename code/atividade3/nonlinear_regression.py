import kagglehub
# https://www.kaggle.com/datasets/mamunhasan2cs/student-academic-performance-synthetic-dataset?select=student_performance.csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras as tf_keras
from sklearn.model_selection import train_test_split 



plt.style.use(['seaborn-v0_8-paper'])

path = kagglehub.dataset_download("mamunhasan2cs/student-academic-performance-synthetic-dataset")
df = pd.read_csv(path + '/student_performance.csv')
df = df.drop(columns=['Student_ID'])
df = df.dropna()


target =['Grade']
features = [i for i in df.columns.values if i not in target] # 'Age', 'Gender', 'Study_Hours', 'Attendance(%)', 'Grade'







pass
