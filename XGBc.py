#Using XGB classifier model

#Imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report


df = pd.read_csv('weatherAUS.csv')
pd.set_option('display.max_columns', 100) #see all columns in dataset
print(df.head())

print(df.shape) #(r,c)

#finding null values
print(df.isnull().sum())

#Droppinf null values
null_values_dropped_df = df.dropna()
print(null_values_dropped_df.shape)


#label RainTomorrow is categorical replace it with 1 and 0 for yes no resp
null_values_dropped_df = null_values_dropped_df.replace({'RainTomorrow':{'Yes':1, 'No':0}})


# store all the numerical columns in one list and all the categorical columns in one list
numerical_col = []
categorical_col = []
for col in null_values_dropped_df.columns:
    if null_values_dropped_df[col].dtype == 'object':
        categorical_col.append(col)
    else:
        numerical_col.append(col)


#numerical column
print(numerical_col)

#correlaton of numerical columns with label RainTomorrow
null_values_dropped_df[numerical_col].corr()
plt.figure(figsize=(15,10))
sns.heatmap(null_values_dropped_df[numerical_col].corr(), annot=True)#heatmap using seaborn
plt.show()

#categorical column
print(null_values_dropped_df[categorical_col])

#we need to convert these categorical columns into numerical columns. To do this we are going to use Label Encoder.

print(null_values_dropped_df[categorical_col].nunique()) #number of unique categorical values

#remove Date
null_values_dropped_df.drop(columns=['Date'], inplace=True)
categorical_col.remove('Date')

#using label encoder
le = LabelEncoder()
for col in categorical_col:
    null_values_dropped_df[col] = le.fit_transform(null_values_dropped_df[col])

print(null_values_dropped_df.head())


#Train_Test_split
X = null_values_dropped_df.drop(columns=['RainTomorrow'])
y = null_values_dropped_df['RainTomorrow']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train.shape, y_train.shape) #((45136, 21), (45136,))
print(X_test.shape, y_test.shape) #((11284, 21), (11284,))


#Building and Evaluating model
xgb = XGBClassifier()
xgb.fit(X_train, y_train)
xgb_prediction = xgb.predict(X_test)

#classification report
print(classification_report(y_test, xgb_prediction))

#Accuracy of the model
xgb_accuracy = accuracy_score(y_test, xgb_prediction)
print("The accuracy of XGB classifier model is ",xgb_accuracy)