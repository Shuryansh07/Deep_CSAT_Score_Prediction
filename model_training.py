import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib

# ==========================================
# 1. DATA WRANGLING & EFFICIENT FORMATTING
# ==========================================
df = pd.read_csv('eCommerce_Customer_support_data.csv')

# Calculate Response Time
df['Issue_reported at'] = pd.to_datetime(df['Issue_reported at'], format='mixed', dayfirst=True)
df['issue_responded'] = pd.to_datetime(df['issue_responded'], format='mixed', dayfirst=True)
df['Response_Time_Minutes'] = (df['issue_responded'] - df['Issue_reported at']).dt.total_seconds() / 60
df['Response_Time_Minutes'] = df['Response_Time_Minutes'].clip(lower=0)
df['Reported_Hour'] = df['Issue_reported at'].dt.hour

# Drop Noise
cols_to_drop = [
    'Unique id', 'Order_id', 'Customer Remarks', 'order_date_time', 'connected_handling_time', 
    'Survey_response_Date', 'Issue_reported at', 'issue_responded', 
    'Customer_City', 'Product_category', 'Item_price',              
    'Agent_name', 'Supervisor', 'Manager'                           
]
df.drop(columns=cols_to_drop, inplace=True)
df.dropna(inplace=True)

# Chart 1: Distribution of Target Variable (CSAT Score)
plt.figure(figsize=(8, 5))
sns.countplot(x='CSAT Score', data=df, palette='viridis')
plt.title('Distribution of CSAT Scores')
plt.xlabel('CSAT Score')
plt.ylabel('Count')
plt.show()

# Chart 2: Average CSAT Score by Channel
plt.figure(figsize=(10, 5))
df.groupby('channel_name')['CSAT Score'].mean().sort_values().plot(kind='bar', color='coral')
plt.title('Average CSAT Score by Channel')
plt.ylabel('Average Score')
plt.xticks(rotation=45)
plt.show()

# Chart 3: Correlation Heatmap
# This shows how the numeric variables relate to each other and to the CSAT score.
plt.figure(figsize=(8, 6))
# Select only numeric columns for the heatmap to avoid errors
numeric_cols = df.select_dtypes(include=[np.number])
sns.heatmap(numeric_cols.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Heatmap of Numeric Variables')
plt.show()

# Chart 4: Response Time vs CSAT Score (Boxplot)
# This proves to the evaluator why 'Response Time' is a highly predictive feature.
plt.figure(figsize=(10, 6))
sns.boxplot(x='CSAT Score', y='Response_Time_Minutes', data=df, palette='Set2')
plt.title('Impact of Response Time on CSAT Score')
plt.yscale('log') # Using log scale because response times can have extreme outliers
plt.ylabel('Response Time (Minutes) - Log Scale')
plt.xlabel('CSAT Score')
plt.show()

# Chart 5: Average CSAT Score by Agent Shift
# Helps identify if customer satisfaction drops during specific times of the day (e.g., Night shifts).
plt.figure(figsize=(10, 5))
sns.barplot(x='Agent Shift', y='CSAT Score', data=df, palette='magma', errorbar=None)
plt.title('Average CSAT Score by Agent Shift')
plt.ylabel('Average CSAT Score')
plt.xlabel('Agent Shift')
plt.show()
# ==========================================
# 2. FEATURE ENGINEERING (SAFE ONE-HOT ENCODING)
# ==========================================
X = df.drop('CSAT Score', axis=1)
y = df['CSAT Score'] - 1 # Neural Networks expect classes to start at 0 (0 to 4)

# Explicitly naming the columns prevents memory crashes!
columns_to_encode = ['channel_name', 'category', 'Sub-category', 'Tenure Bucket', 'Agent Shift']
X_encoded = pd.get_dummies(X, columns=columns_to_encode, drop_first=True)

# Train-Test Split & Scaling
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# 3. ML MODEL: CLASSIFICATION ANN
# ==========================================
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(5, activation='softmax')) # 5 Output classes for CSAT

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print("Training Classification ANN...")
history = model.fit(X_train_scaled, y_train, validation_split=0.2, epochs=20, batch_size=64, verbose=1)

# ==========================================
# 4. EVALUATION & SAVING
# ==========================================
y_pred_probs = model.predict(X_test_scaled)
y_pred_classes = np.argmax(y_pred_probs, axis=1)

accuracy = accuracy_score(y_test, y_pred_classes)
print(f"\n--- CLASSIFICATION MODEL EVALUATION ---")
print(f"Accuracy Score: {accuracy:.4f}")

# Save models for Streamlit
model.save('csat_ann_model.h5')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(list(X_encoded.columns), 'model_columns.pkl')

print("\nModels saved successfully and ready for Streamlit!")