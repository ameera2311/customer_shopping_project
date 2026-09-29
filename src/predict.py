import pandas as pd
import joblib


# --------------------------------------------------
# 1. Load the trained model
# --------------------------------------------------

model = joblib.load(
    "models/subscription_model.pkl"
)

print("Model loaded successfully.")


# --------------------------------------------------
# 2. Create a new customer
# --------------------------------------------------

new_customer = pd.DataFrame([
    {
        "Age": 25,
        "Gender": "Female",
        "Item Purchased": "Blouse",
        "Category": "Clothing",
        "Purchase Amount (USD)": 60,
        "Location": "California",
        "Size": "M",
        "Color": "Blue",
        "Season": "Summer",
        "Review Rating": 4.5,
        "Shipping Type": "Free Shipping",
        "Discount Applied": "Yes",
        "Previous Purchases": 10,
        "Payment Method": "Credit Card",
        "Frequency of Purchases": "Weekly"
    }
])


# --------------------------------------------------
# 3. Make prediction
# --------------------------------------------------

prediction = model.predict(new_customer)


# --------------------------------------------------
# 4. Get prediction probabilities
# --------------------------------------------------

probabilities = model.predict_proba(new_customer)

classes = model.classes_

probability_no = probabilities[0][
    list(classes).index("No")
]

probability_yes = probabilities[0][
    list(classes).index("Yes")
]


# --------------------------------------------------
# 5. Display results
# --------------------------------------------------

print("\n==============================")
print("CUSTOMER SUBSCRIPTION PREDICTION")
print("==============================")

print(
    "\nPredicted Subscription Status:",
    prediction[0]
)

print(
    f"Probability of No Subscription: "
    f"{probability_no * 100:.2f}%"
)

print(
    f"Probability of Yes Subscription: "
    f"{probability_yes * 100:.2f}%"
)