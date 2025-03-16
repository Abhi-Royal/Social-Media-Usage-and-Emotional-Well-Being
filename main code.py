from pandas import *
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

train_df = read_csv("train.csv")
val_df = read_csv("val.csv")
test_df = read_csv("test.csv")

print(train_df)
print(train_df.info())
print(train_df.describe())
"""
print(train_df.isnull().sum())
print(val_df.isnull().sum())
print(val_df.isnull().sum())

# Plot age distribution
plt.figure(figsize=(10, 6))
sns.histplot(train_df['Age'], kde=True, bins=30)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Plot gender distribution
plt.figure(figsize=(7, 5))
sns.countplot(x='Gender', data=train_df)
plt.title('Gender Distribution')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()

# Plot average daily usage time by platform
plt.figure(figsize=(12, 6))
sns.barplot(x='Platform', y='Daily_Usage_Time (minutes)', data=train_df, estimator=np.mean)
plt.title('Average Daily Usage Time by Platform')
plt.xlabel('Platform')
plt.ylabel('Daily Usage Time (minutes)')
plt.show()

# Plot dominant emotion distribution
plt.figure(figsize=(10, 6))
sns.countplot(x='Dominant_Emotion', data=train_df)
plt.title('Dominant Emotion Distribution')
plt.xlabel('Dominant Emotion')
plt.ylabel('Count')
plt.show()

"""
#----------------------Pre-Processing of data--------------------------------------------
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

#1 One Hot Encoding
train_df = get_dummies(train_df, columns=['Gender', 'Platform'], drop_first=True)
test_df = get_dummies(test_df, columns=['Gender', 'Platform'], drop_first=True)
val_df = get_dummies(val_df, columns=['Gender', 'Platform'], drop_first=True)

#2 Label Encoding
le = LabelEncoder()
train_df['Dominant_Emotion'] = le.fit_transform(train_df['Dominant_Emotion'])
val_df['Dominant_Emotion'] = le.fit_transform(val_df['Dominant_Emotion'])
test_df['Dominant_Emotion'] = le.fit_transform(test_df['Dominant_Emotion'])

train_df = train_df.drop(columns=['User_ID'])                     # Dropping an unccessary column
train_df['Age'] = to_numeric(train_df['Age'], errors='coerce')    # Converting a string value into NaN 
mean_value_train = train_df['Age'].mean()
train_df['Age'] = train_df['Age'].fillna(mean_value_train)          # filling NaN into median


"""
set_option('display.max_rows',1001)
print(train_df.shape)
print(train_df.columns)  
print(train_df['Dominant_Emotion'])
"""
#----------importing Model---------------------------------

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

X = train_df.drop(columns = 'Dominant_Emotion')
y = train_df['Dominant_Emotion']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

rfc = RandomForestClassifier()
rfc.fit(x_train, y_train) 
pred = rfc.predict(x_test)
print(pred)
print(accuracy_score(pred,y_test))