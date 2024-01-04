import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import classification_report

# Load and preprocess the data
# to re-arrange the table rows and columns and change column name
df = pd.read_csv('bank.csv', delimiter =';')
df.rename(columns={'y': 'deposit'}, inplace=True)
print(df)
print(df.tail(10))
print(df.columns)
print(df.dtypes)
print(df.dtypes.value_counts())
print(df.isna().sum())

# extracting categorical and numerical columns
cat_cols = df.select_dtypes(include='object').columns
print(cat_cols)
print(df.describe(include='object'))

num_cols = df.select_dtypes(exclude='object').columns
print(num_cols)
print(df.describe())

df1 = df.copy()
print(df1)

# Label encoding for categorical var
from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
df_encoded = df1.apply(lb.fit_transform)
print(df_encoded)

# Splitting the data into features (X) and target variable (y)
X = df_encoded.drop("deposit", axis=1)  # Features
y = df_encoded["deposit"]  # Target variable
print(df_encoded.value_counts())

from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LogisticRegression
# Create a logistic regression model
model = LogisticRegression()

# Train the model
model = tree.DecisionTreeClassifier()
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)


from sklearn.metrics import accuracy_score, classification_report

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:")
print(report)

from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import graphviz
import pydotplus
from IPython.display import Image
# Generate the dot data of the decision tree
dot_data = tree.export_graphviz(model, out_file=None,
                                feature_names=X.columns,
                                class_names=["No", "Yes"],
                                filled=True, rounded=True,
                                special_characters=True)

# Create a graph from the dot data
graph = pydotplus.graph_from_dot_data(dot_data)

# Generate the image of the decision tree
image = Image(graph.create_png())
print(image)
