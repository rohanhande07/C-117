import pandas as pd 
df = pd.read_csv("C-117 file.csv")
print(df.head())
from sklearn.model_selection import train_test_split
age = df["age"]
heart_attack = df["target"]
age_train, age_test, heartattack_train train_test_split