import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# --------------------------------------------------
# 1. Load the dataset
# --------------------------------------------------

df = pd.read_csv("data/customer_shopping_behavior.csv")


# --------------------------------------------------
# 2. Define target column
# --------------------------------------------------

target = "Subscription Status"


# --------------------------------------------------
# 3. Separate features and target
# --------------------------------------------------

X = df.drop(
    columns=["Customer ID", target]
)

y = df[target]


# --------------------------------------------------
# 4. Define numerical features
# --------------------------------------------------

numerical_features = [
    "Age",
    "Purchase Amount (USD)",
    "Review Rating",
    "Previous Purchases"
]


# --------------------------------------------------
# 5. Define categorical features
# --------------------------------------------------

categorical_features = [
    "Gender",
    "Item Purchased",
    "Category",
    "Location",
    "Size",
    "Color",
    "Season",
    "Shipping Type",
    "Discount Applied",
    "Payment Method",
    "Frequency of Purchases"
]


# --------------------------------------------------
# 6. Create preprocessing pipeline
# --------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[

        (
            "categorical",
            Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(strategy="most_frequent")
                    ),
                    (
                        "encoder",
                        OneHotEncoder(handle_unknown="ignore")
                    )
                ]
            ),
            categorical_features
        ),

        (
            "numerical",
            Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(strategy="median")
                    ),
                    (
                        "scaler",
                        StandardScaler()
                    )
                ]
            ),
            numerical_features
        )
    ]
)


# --------------------------------------------------
# 7. Create ML pipeline
# --------------------------------------------------

model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced"
            )
        )
    ]
)


# --------------------------------------------------
# 8. Split data
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# --------------------------------------------------
# 9. Train the model
# --------------------------------------------------

model.fit(X_train, y_train)


# --------------------------------------------------
# 10. Make predictions
# --------------------------------------------------

y_pred = model.predict(X_test)


# --------------------------------------------------
# 11. Calculate evaluation metrics
# --------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    pos_label="Yes"
)

recall = recall_score(
    y_test,
    y_pred,
    pos_label="Yes"
)

f1 = f1_score(
    y_test,
    y_pred,
    pos_label="Yes"
)


# --------------------------------------------------
# 12. Display results
# --------------------------------------------------

print("\n==============================")
print("MODEL EVALUATION")
print("==============================")

print(f"\nAccuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")


# --------------------------------------------------
# 13. Confusion Matrix
# --------------------------------------------------

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=["No", "Yes"]
)

print("\nConfusion Matrix:")
print(cm)


# --------------------------------------------------
# 14. Classification Report
# --------------------------------------------------

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred
    )
)