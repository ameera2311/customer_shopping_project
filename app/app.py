import streamlit as st
import pandas as pd
import joblib


# --------------------------------------------------
# 1. Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Customer Subscription Predictor",
    page_icon="🤖",
    layout="wide"
)


# --------------------------------------------------
# 2. Load trained model
# --------------------------------------------------

model = joblib.load(
    "models/subscription_model.pkl"
)


# --------------------------------------------------
# 3. Page title
# --------------------------------------------------

st.title("🤖 Customer Subscription Predictor")

st.write(
    "Enter customer details to predict whether the customer "
    "is likely to have a subscription."
)


# --------------------------------------------------
# 4. Customer information
# --------------------------------------------------

st.header("Customer Information")

col1, col2, col3 = st.columns(3)


with col1:

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=25
    )

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    item_purchased = st.selectbox(
        "Item Purchased",
        [
            "Blouse",
            "Sweater",
            "Jeans",
            "Sandals",
            "Sneakers",
            "Shirt",
            "Shorts",
            "Coat",
            "Handbag",
            "Jacket",
            "Jewelry",
            "T-shirt",
            "Dress",
            "Scarf",
            "Socks",
            "Hoodie",
            "Hat",
            "Backpack",
            "Pants",
            "Skirt"
        ]
    )

    category = st.selectbox(
        "Category",
        [
            "Clothing",
            "Footwear",
            "Accessories",
            "Outerwear"
        ]
    )

    purchase_amount = st.number_input(
        "Purchase Amount (USD)",
        min_value=0.0,
        value=60.0
    )


with col2:

    location = st.selectbox(
        "Location",
        [
            "California",
            "New York",
            "Texas",
            "Florida",
            "Illinois",
            "Pennsylvania",
            "Ohio",
            "Michigan",
            "Georgia",
            "North Carolina",
            "Other"
        ]
    )

    size = st.selectbox(
        "Size",
        ["S", "M", "L", "XL"]
    )

    color = st.selectbox(
        "Color",
        [
            "Black",
            "White",
            "Red",
            "Blue",
            "Green",
            "Yellow",
            "Gray",
            "Brown",
            "Pink",
            "Purple",
            "Orange",
            "Beige",
            "Cyan",
            "Magenta",
            "Olive",
            "Silver",
            "Gold",
            "Violet",
            "Teal",
            "Maroon"
        ]
    )

    season = st.selectbox(
        "Season",
        [
            "Spring",
            "Summer",
            "Fall",
            "Winter"
        ]
    )

    review_rating = st.number_input(
        "Review Rating",
        min_value=1.0,
        max_value=5.0,
        value=4.5,
        step=0.1
    )


with col3:

    shipping_type = st.selectbox(
        "Shipping Type",
        [
            "Free Shipping",
            "Standard",
            "Express",
            "Next Day Air",
            "2-Day Shipping"
        ]
    )

    discount_applied = st.selectbox(
        "Discount Applied",
        ["Yes", "No"]
    )

    previous_purchases = st.number_input(
        "Previous Purchases",
        min_value=0,
        value=10
    )

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Credit Card",
            "Debit Card",
            "Venmo",
            "Cash",
            "PayPal",
            "Bank Transfer"
        ]
    )

    frequency = st.selectbox(
        "Frequency of Purchases",
        [
            "Weekly",
            "Fortnightly",
            "Monthly",
            "Quarterly",
            "Bi-Weekly",
            "Annually",
            "Every 3 Months"
        ]
    )


# --------------------------------------------------
# 5. Create prediction button
# --------------------------------------------------

st.divider()

predict_button = st.button(
    "🔮 Predict Subscription",
    use_container_width=True
)


# --------------------------------------------------
# 6. Make prediction
# --------------------------------------------------

if predict_button:

    new_customer = pd.DataFrame([
        {
            "Age": age,
            "Gender": gender,
            "Item Purchased": item_purchased,
            "Category": category,
            "Purchase Amount (USD)": purchase_amount,
            "Location": location,
            "Size": size,
            "Color": color,
            "Season": season,
            "Review Rating": review_rating,
            "Shipping Type": shipping_type,
            "Discount Applied": discount_applied,
            "Previous Purchases": previous_purchases,
            "Payment Method": payment_method,
            "Frequency of Purchases": frequency
        }
    ])


    # Make prediction
    prediction = model.predict(
        new_customer
    )


    # Get probabilities
    probabilities = model.predict_proba(
        new_customer
    )

    classes = model.classes_


    probability_no = probabilities[0][
        list(classes).index("No")
    ]

    probability_yes = probabilities[0][
        list(classes).index("Yes")
    ]


    # --------------------------------------------------
    # 7. Display results
    # --------------------------------------------------

    st.subheader("Prediction Result")

    result_col1, result_col2 = st.columns(2)


    with result_col1:

        if prediction[0] == "Yes":

            st.success(
                "✅ Predicted Subscription: YES"
            )

        else:

            st.warning(
                "❌ Predicted Subscription: NO"
            )


    with result_col2:

        st.metric(
            "Subscription Probability",
            f"{max(probability_yes, probability_no) * 100:.2f}%"
        )


    # Probability details

    st.write("### Prediction Probabilities")

    probability_data = pd.DataFrame(
        {
            "Subscription Status": [
                "No",
                "Yes"
            ],
            "Probability": [
                probability_no,
                probability_yes
            ]
        }
    )

    st.bar_chart(
        probability_data.set_index(
            "Subscription Status"
        )
    )