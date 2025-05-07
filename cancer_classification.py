# cancer_classification.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from joblib import dump
from preprocess import preprocess_data

# Preprocess data
X_train, X_test, y_train, y_test, label_encoder = preprocess_data(
    "data.csv", "labels.csv"
)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
dump(model, "cancer_model.joblib")

y_pred = model.predict(X_test)

# Evaluate
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Feature Importance
importances = model.feature_importances_
feature_names = X_train.columns
importance_df = pd.DataFrame({"Gene": feature_names, "Importance": importances})
importance_df = importance_df.sort_values(by="Importance", ascending=False).head(10)

sns.barplot(x="Importance", y="Gene", data=importance_df)
plt.title("Top 10 Important Genes")
plt.xlabel("Importance Score")
plt.ylabel("Gene")
plt.show()
