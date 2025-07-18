# train_and_save.py

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE

# 1. Load your cleaned dataset
df = pd.read_csv("adult 3.csv")

# 2. Preprocessing (repeat your cleaning steps)
df = df[~df['education'].isin(['1st-4th', '5th-6th', 'Preschool'])]
df = df[df['educational-num'] > 5]
df = df[df['occupation'] != 'Armed-Forces']
df = df[df['age'] <= 68]
df.replace('?', pd.NA, inplace=True)
df.dropna(inplace=True)
df.drop('education', axis=1, inplace=True)

df['age_group'] = pd.cut(df['age'], bins=[17, 30, 45, 68], labels=['Young', 'Mid', 'Senior'])
df['is_educated'] = (df['educational-num'] > 12).astype(int)
df = pd.get_dummies(df, drop_first=True)

# 3. Train/test split
X = df.drop('income_>50K', axis=1)
y = df['income_>50K']

# 4. Scale and balance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# 5. Train model
model = MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# 6. Save model, scaler, and features
joblib.dump(model, 'income_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(X.columns.tolist(), 'feature_names.pkl')

print("✅ Training complete. Files saved.")
