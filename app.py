
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Smart Event Planner",
    page_icon="🎉",
    layout="wide"
)

st.title("🎉 Smart Event Planner")
st.caption(
    "Estimate event costs and discover suitable vendors "
    "based on your requirements."
)

MODEL_PATH = "models/event_cost_model.pkl"
VENDOR_PATH = "data/vendors.csv"

# --------------------------------------------------
# 2. LOAD FILES
# --------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_vendors():
    return pd.read_csv(VENDOR_PATH)

if not os.path.exists(MODEL_PATH):
    st.error("Model not found. Run: python train_model.py")
    st.stop()

if not os.path.exists(VENDOR_PATH):
    st.error("Vendor data not found. Run: python generate_data.py")
    st.stop()

model = load_model()
vendors = load_vendors()

# --------------------------------------------------
# 3. USER INPUTS
# --------------------------------------------------

event_types = [
    "Wedding", "Birthday", "Engagement",
    "Corporate", "College Event", "Anniversary"
]

locations = sorted(
    vendors["Location"].dropna().unique().tolist()
)

styles = [
    "Traditional", "Modern", "Minimalist", "Luxury"
]

venue_types = ["Indoor", "Outdoor"]

catering_types = [
    "Vegetarian", "Non-Vegetarian", "Both"
]

with st.sidebar:
    st.header("Plan Your Event")

    event_type = st.selectbox(
        "Event type",
        event_types
    )

    location = st.selectbox(
        "Location",
        locations,
        index=locations.index("Kochi")
        if "Kochi" in locations else 0
    )

    event_date = st.date_input(
        "Event date"
    )

    guests = st.number_input(
        "Expected guests",
        min_value=20,
        max_value=1500,
        value=200,
        step=10
    )

    budget = st.number_input(
        "Total budget (₹)",
        min_value=10000,
        max_value=10000000,
        value=500000,
        step=10000
    )

    style = st.selectbox(
        "Preferred style",
        styles
    )

    venue_type = st.selectbox(
        "Venue type",
        venue_types
    )

    catering = st.selectbox(
        "Catering preference",
        catering_types
    )

    minimum_rating = st.slider(
        "Minimum vendor rating",
        min_value=3.0,
        max_value=5.0,
        value=4.0,
        step=0.1
    )

    st.caption(
        "Sample data is fictional. Vendor prices and "
        "availability have not been verified."
    )

# --------------------------------------------------
# 4. PREDICT EVENT COST
# --------------------------------------------------

input_data = pd.DataFrame([{
    "EventType": event_type,
    "Location": location,
    "GuestCount": int(guests),
    "Budget": float(budget),
    "Style": style,
    "VenueType": venue_type,
    "Catering": catering
}])

predicted_cost = max(
    0,
    float(model.predict(input_data)[0])
)

st.subheader("Your Event Estimate")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Estimated event cost",
        f"₹{predicted_cost:,.0f}"
    )

with col2:
    st.metric(
        "Your budget",
        f"₹{budget:,.0f}"
    )

with col3:
    remaining = float(budget) - predicted_cost
    st.metric(
        "Estimated remaining budget",
        f"₹{remaining:,.0f}",
        delta=(
            "Within budget"
            if remaining >= 0
            else "Over budget"
        ),
        delta_color=(
            "normal" if remaining >= 0 else "inverse"
        )
    )

if predicted_cost > budget:
    st.warning(
        "Your estimated event cost exceeds your budget. "
        "Consider reducing guest count or choosing a "
        "more economical style."
    )
else:
    st.success(
        "The estimated cost is within your selected budget."
    )

st.caption(
    "Prediction is based on a model trained using synthetic "
    "data. It is not a verified market quotation."
)

# --------------------------------------------------
# 5. SUGGESTED BUDGET ALLOCATION
# --------------------------------------------------

st.subheader("Suggested Budget Allocation")

allocations = {
    "Venue": 0.25,
    "Catering": 0.35,
    "Decoration": 0.15,
    "Photography": 0.10,
    "Music": 0.05,
    "Makeup": 0.05,
    "Miscellaneous": 0.05
}

allocation_df = pd.DataFrame({
    "Category": list(allocations.keys()),
    "Percentage": [
        value * 100 for value in allocations.values()
    ],
    "Suggested Amount (₹)": [
        round(float(budget) * value, 2)
        for value in allocations.values()
    ]
})

st.bar_chart(
    allocation_df,
    x="Category",
    y="Suggested Amount (₹)"
)

st.dataframe(
    allocation_df,
    use_container_width=True,
    hide_index=True
)

st.caption(
    "Allocation percentages are illustrative suggestions. "
    "Actual expenses vary by event and vendor."
)

# --------------------------------------------------
# 6. VENDOR RECOMMENDATIONS
# --------------------------------------------------

st.subheader("Recommended Vendors")

filtered = vendors.copy()

# Match the selected city or declared service area.
location_match = (
    filtered["Location"].eq(location)
    | filtered["ServiceAreas"]
        .fillna("")
        .apply(
            lambda areas: location in [
                item.strip() for item in areas.split(",")
            ]
        )
)

filtered = filtered[location_match]

filtered = filtered[
    filtered["Rating"] >= minimum_rating
]

# Prioritize style matches.
filtered["StyleMatch"] = (
    filtered["Style"] == style
).astype(int)

# Prices are indicative, not verified quotations.
category_limits = {
    "Venue": 0.25,
    "Catering": 0.35,
    "Decoration": 0.15,
    "Photography": 0.10,
    "Music": 0.05,
    "Makeup": 0.05
}

for category, share in category_limits.items():

    category_df = filtered[
        filtered["Category"] == category
    ].copy()

    category_budget = float(budget) * share

    # Apply the allocated category budget.
    affordable = category_df[
        category_df["Price"] <= category_budget
    ].copy()

    # If none fits, show closest options separately rather
    # than silently claiming that they are affordable.
    if affordable.empty and not category_df.empty:
        st.markdown(f"### {category}")

        st.info(
            "No vendor matched this category's suggested "
            "budget. The closest options are shown for review."
        )

        category_df["BudgetDifference"] = (
            category_df["Price"] - category_budget
        ).clip(lower=0)

        display_df = category_df.sort_values(
            ["BudgetDifference", "StyleMatch", "Rating"],
            ascending=[True, False, False]
        ).head(3)

    else:
        display_df = affordable.sort_values(
            ["StyleMatch", "Rating", "ReviewCount"],
            ascending=[False, False, False]
        ).head(3)

        if display_df.empty:
            continue

        st.markdown(f"### {category}")

    display_columns = [
        "VendorID",
        "VendorName",
        "Location",
        "Style",
        "PriceMin",
        "PriceMax",
        "Rating",
        "ReviewCount",
        "Capacity",
        "AvailabilityStatus"
    ]

    st.dataframe(
        display_df[display_columns],
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Prices and reviews are synthetic. "
        "Availability is unverified."
    )

if filtered.empty:
    st.warning(
        "No vendors match your location and rating filters. "
        "Try lowering the rating requirement or adding "
        "verified vendors who serve this location."
    )

# --------------------------------------------------
# 7. PREPARATION CHECKLIST
# --------------------------------------------------

st.subheader("Event Preparation Checklist")

checklist = [
    "Confirm the venue and event date",
    "Verify vendor quotations and availability",
    "Finalize catering and guest count",
    "Confirm decoration and photography requirements",
    "Keep a contingency budget",
    "Confirm bookings and payment terms"
]

for task in checklist:
    st.checkbox(task, key=f"task_{task}")

# --------------------------------------------------
# 8. FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Smart Event Planner | ML-based cost estimation "
    "and vendor discovery | Sample data is not live data."
)
 
