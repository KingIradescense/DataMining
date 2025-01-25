import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("Bicycle_Parking_20241213_AfterConversion.csv")

# Target variable
target = 'BoroCode' 
X = df.drop(columns=[target]) 
y = df[target]

# 80% train data , 20% test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# StartRandom Forest Classifier
classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model on the training data
classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = classifier.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Display classification report (precision, recall, F1-score)
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Feature Importance
importances =  classifier.feature_importances_
cols = X.columns

# Sort feature importances in descending order
sorted_cols = importances.argsort()[::-1]

# Print the feature importance
print("\nFeature Importance:")
for i in sorted_cols:
    print(f"{cols[i]}: {importances[i]:.4f}")

# Visualize Rand Forest Results
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.barh(cols[sorted_cols], importances[sorted_cols], color='skyblue')
plt.xlabel('Feature Importance')
plt.title('Random Forest Results')
plt.tight_layout()
plt.show()