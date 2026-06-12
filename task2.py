# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Load Dataset
df  = pd . read_csv("titanic.csv")

# Display first 5 rows
print(df. head())

#Dataset Information
print("\nDataset Info:")
print(df.info())

# check Missing Values
print("\nMissing Values:")
print(df.isnull().sum())

# Fill Missing Values
df["Age"] .fillna(df["Age"].median(),inplace=True)
df["Embarked"].fillna(df["Embarked"].mode()[0],inplace=True)
df.drop("Cabin", axis=1, inplace=True)

#Summary Statistics
print("\nSummary Statistics:")
print(df.describe())

# Survival Count
plt.figure(figsize=(6,4))
sns.countplot(x="Survived",data=df)
plt.title("Survival Count")
plt.xlabel("0 = Died, 1 = Survived")
plt.ylabel("Number of Passengers")
plt.xticks([0,1] , ["Died", "Survived"])
plt.show()

#survival by Gender
plt.figure(figsize=(6,4))
sns.countplot(x="Sex", hue="Survived", data=df)
plt.title("Survival by Gender")
plt.xlabel("Gender")
plt.ylabel("Number of Passengers")
plt.show()

#Age Distribution
plt.figure(figsize=(6,4))
plt.hist(df["Age"], bins=20)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("frequency")
plt.show()

#Fare Distribution
plt.figure(figsize=(8,5))
sns.histplot(df["Fare"], bins=30, kde=True, color="green")
plt.title("Fare Distribution")
plt.xlabel("Fare")
plt.show()

#Survival by Passenger Class
plt.figure(figsize=(8,5))
sns.barplot(x="Pclass", y='Survived', data=df)
plt.title("Survival Rate by Passenger CLASS")
plt.ylabel('Survival Rate')
plt.show()

#Age vs Passenger Class Boxplot
plt.figure(figsize=(8,5))
sns.boxplot(x="Pclass", y="Age", data=df)
plt.title("Age Distribution across Passenger Classes")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(numeric_only=True) , annot=True, cmap="coolwarm")
plt.title("correlation Heatmap")
plt.show()