
from pandas import *
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import random

train_df = read_csv("train.csv")

print(train_df.shape)
print(train_df)
#print(train_df.info())
#print(train_df.describe())
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
#1)__Cleaning of Data__

train_df = train_df.drop(columns=['User_ID'])                     # Dropping an unccessary column
train_df['Age'] = to_numeric(train_df['Age'], errors='coerce')    # Converting a string value into NaN
train_df['Gender'] = to_numeric(train_df['Gender'], errors='coerce').combine_first(train_df['Gender'])  # to convert numeric type str to int(float)
train_df['Gender'] = train_df['Gender'].apply(lambda x: np.nan if isinstance(x, (int, float)) else x)   # to convert numeric value to NaN 

print(train_df.columns)

age_unique = train_df['Age'].unique().tolist()
gender_unique = train_df['Gender'].unique().tolist()
platform_unique = train_df['Platform'].unique().tolist()
daily_unique = train_df['Daily_Usage_Time (minutes)'].unique().tolist()
posts_unique = train_df['Posts_Per_Day'].unique().tolist()
likes_unique = train_df['Likes_Received_Per_Day'].unique().tolist()
comments_unique = train_df['Comments_Received_Per_Day'].unique().tolist()
messages_unique = train_df['Messages_Sent_Per_Day'].unique().tolist()
dominant_unique = train_df['Dominant_Emotion'].unique().tolist()
#print(age_unique),print(daily_unique),print(posts_unique),print(likes_unique),print(comments_unique),print(messages_unique)

#age_unique.remove(float('nan'))
#age_unique.remove(np.nan)
age_unique.pop(0)
gender_unique.pop(0)
platform_unique.pop(0)
daily_unique.pop(0)
posts_unique.pop(0)
likes_unique.pop(0)
comments_unique.pop(0)
messages_unique.pop(0)
dominant_unique.pop(0)
#print(age_unique),print(gender_unique),print(daily_unique),print(posts_unique),print(likes_unique),print(comments_unique),print(messages_unique)

train_df['Age'] = train_df['Age'].apply(lambda x: random.choice(age_unique) if isna(x) else x)
train_df['Gender'] = train_df['Gender'].apply(lambda x: random.choice(gender_unique) if isna(x) else x)
train_df['Platform'] = train_df['Platform'].apply(lambda x: random.choice(platform_unique) if isna(x) else x)
train_df['Daily_Usage_Time (minutes)'] = train_df['Daily_Usage_Time (minutes)'].apply(lambda x: random.choice(daily_unique) if isna(x) else x)
train_df['Posts_Per_Day'] = train_df['Posts_Per_Day'].apply(lambda x: random.choice(posts_unique) if isna(x) else x)
train_df['Likes_Received_Per_Day'] = train_df['Likes_Received_Per_Day'].apply(lambda x: random.choice(likes_unique) if isna(x) else x)
train_df['Comments_Received_Per_Day'] = train_df['Comments_Received_Per_Day'].apply(lambda x: random.choice(comments_unique) if isna(x) else x)
train_df['Messages_Sent_Per_Day'] = train_df['Messages_Sent_Per_Day'].apply(lambda x: random.choice(messages_unique) if isna(x) else x)
train_df['Dominant_Emotion'] = train_df['Dominant_Emotion'].apply(lambda x: random.choice(dominant_unique) if isna(x) else x)

#2)__Creating data__

new_age = random.choices(age_unique, k=4000)
new_gender = random.choices(gender_unique, k=4000)
new_platform = random.choices(platform_unique, k=4000)
new_daily = random.choices(daily_unique, k=4000)
new_posts = random.choices(posts_unique, k=4000)
new_likes = random.choices(likes_unique, k=4000)
new_comments = random.choices(comments_unique, k=4000)
new_messages = random.choices(messages_unique, k=4000)
new_dominant = random.choices(dominant_unique, k=4000)
#print(new_age),print(new_gender),print(new_platform),print(new_daily),print(new_posts),print(new_likes),print(new_comments),print(new_messages)

#)__Adding data__

new_data = DataFrame({
    'Age': new_age,
    'Gender': new_gender,
    'Platform': new_platform,
    'Daily_Usage_Time (minutes)': new_daily,
    'Posts_Per_Day': new_posts,
    'Likes_Received_Per_Day': new_likes,
    'Comments_Received_Per_Day': new_comments,
    'Messages_Sent_Per_Day': new_messages,
    'Dominant_Emotion': new_dominant
})
print(new_data)

finalData = pd.concat([train_df, new_data], ignore_index=True)
print(finalData)
print(finalData.shape)

#4)__Preproccessing techiniques__

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

#1 One Hot Encoding
finalData = get_dummies(finalData, columns=['Gender', 'Platform'], drop_first=True)

#2 Label Encoding
le = LabelEncoder()
finalData['Dominant_Emotion'] = le.fit_transform(finalData['Dominant_Emotion']) 

#set_option('display.max_columns',30)
#set_option('display.max_rows',2004)

#----------importing Model---------------------------------

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

X = finalData.drop(columns = 'Dominant_Emotion')
y = finalData['Dominant_Emotion']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

rfc = RandomForestClassifier(n_estimators=333,max_depth=9,min_samples_split=3,min_samples_leaf=3,
                             max_features='sqrt',bootstrap=True,class_weight='balanced',n_jobs=-6,verbose=6)
rfc.fit(x_train, y_train) 
pred = rfc.predict(x_test)
print(pred)
print(accuracy_score(pred,y_test))