import pandas as pd


from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


import joblib



print("\nModel saved successfully.")
# Load the dataset
df = pd.read_csv("data/customer_shopping_behavior.csv")


# Target column
target = "Subscription Status"


# Remove columns that should not be used for prediction
columns_to_drop = [
    "Customer ID",
    target
]

X = df.drop(columns=columns_to_drop)
y = df[target]


# Identify numerical and categorical columns
numerical_features = [
    "Age",
    "Purchase Amount (USD)",
    "Review Rating",
    "Previous Purchases"
]

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


preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore"))
                ]
            ),
            categorical_features
        ),
        (
            "numerical",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            ),
            numerical_features
        )
    ]
)

# Create machine learning pipeline
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced"
            )
        )
    ]
)


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Train the model
model.fit(X_train, y_train)


print("Model training completed successfully.")

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

print("\nTraining target distribution:")
print(y_train.value_counts())

print("\nTesting target distribution:")
print(y_test.value_counts())


model.fit(X_train, y_train)

joblib.dump(
    model,
    "models/subscription_model.pkl"
)