import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("data/mental_health_data.csv")
# Check for missing values
print("\n🛠️ Checking for missing values...\n")
print(df.isnull().sum())

# Drop rows with any missing values
df = df.dropna()
print("\n✅ Cleaned Data: All missing values removed.\n")


# Encode categorical columns
le = LabelEncoder()
df['Mood_Swings'] = le.fit_transform(df['Mood_Swings'])         # Yes/No → 1/0
df['Past_Diagnosis'] = le.fit_transform(df['Past_Diagnosis'])   # Yes/No → 1/0
df['Distraction'] = le.fit_transform(df['Distraction'])         # Low/Med/High
df['Appetite'] = le.fit_transform(df['Appetite'])               # Low/Norm/High

# Input features
X = df.drop(['Disorder_Type', 'Severity'], axis=1)

# Target 1: Disorder_Type
y_disorder = df['Disorder_Type']
X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y_disorder, test_size=0.2, random_state=42)
model_disorder = RandomForestClassifier()
model_disorder.fit(X_train1, y_train1)
pickle.dump(model_disorder, open('model/disorder_model.pkl', 'wb'))

# Target 2: Severity
y_severity = df['Severity']
X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y_severity, test_size=0.2, random_state=42)
model_severity = RandomForestClassifier()
model_severity.fit(X_train2, y_train2)
pickle.dump(model_severity, open('model/severity_model.pkl', 'wb'))

print("✅ Models trained and saved successfully!")
