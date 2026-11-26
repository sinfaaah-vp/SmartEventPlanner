
import os
import numpy as np
import pandas as pd

# --------------------------------------------------
# 1. SETUP
# --------------------------------------------------

np.random.seed(42)

os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

# --------------------------------------------------
# 2. EVENT AND LOCATION OPTIONS
# --------------------------------------------------

event_types = [
    "Wedding",
    "Birthday",
    "Engagement",
    "Corporate",
    "College Event",
    "Anniversary"
]

# Kerala locations and major Indian cities
locations = [
    "Kozhikode",
    "Kochi",
    "Thiruvananthapuram",
    "Malappuram",
    "Thrissur",
    "Kannur",
    "Palakkad",
    "Kottayam",
    "Alappuzha",
    "Wayanad",
    "Munnar",
    "Thekkady",
    "Kasaragod",
    "Kollam",
    "Pathanamthitta",
    "Bengaluru",
    "Chennai",
    "Coimbatore",
    "Hyderabad",
    "Mysuru",
    "Mumbai",
    "Pune",
    "Delhi",
    "Kolkata"
]

styles = [
    "Traditional",
    "Modern",
    "Minimalist",
    "Luxury"
]

venue_types = ["Indoor", "Outdoor"]

catering_types = [
    "Vegetarian",
    "Non-Vegetarian",
    "Both"
]

# Illustrative location cost multipliers.
# These are assumptions for synthetic data, not market research.
location_multipliers = {
    "Kozhikode": 1.00,
    "Kochi": 1.15,
    "Thiruvananthapuram": 1.10,
    "Malappuram": 0.92,
    "Thrissur": 0.95,
    "Kannur": 0.90,
    "Palakkad": 0.88,
    "Kottayam": 0.96,
    "Alappuzha": 1.00,
    "Wayanad": 1.05,
    "Munnar": 1.20,
    "Thekkady": 1.12,
    "Kasaragod": 0.88,
    "Kollam": 0.98,
    "Pathanamthitta": 0.90,
    "Bengaluru": 1.25,
    "Chennai": 1.20,
    "Coimbatore": 1.05,
    "Hyderabad": 1.15,
    "Mysuru": 1.00,
    "Mumbai": 1.45,
    "Pune": 1.20,
    "Delhi": 1.35,
    "Kolkata": 1.05
}

style_multipliers = {
    "Traditional": 1.15,
    "Modern": 1.10,
    "Minimalist": 0.85,
    "Luxury": 1.60
}

base_costs = {
    "Wedding": 50000,
    "Birthday": 10000,
    "Engagement": 25000,
    "Corporate": 30000,
    "College Event": 20000,
    "Anniversary": 15000
}

cost_per_guest = {
    "Wedding": 1200,
    "Birthday": 350,
    "Engagement": 700,
    "Corporate": 900,
    "College Event": 450,
    "Anniversary": 500
}

# --------------------------------------------------
# 3. GENERATE HISTORICAL EVENT DATA
# --------------------------------------------------

number_of_events = 5000
events = []

for event_id in range(1, number_of_events + 1):

    event_type = np.random.choice(event_types)
    location = np.random.choice(locations)
    style = np.random.choice(styles)
    venue_type = np.random.choice(venue_types)
    catering = np.random.choice(catering_types)

    guests = int(np.random.randint(20, 801))
    budget = int(np.random.randint(30000, 2000001))

    venue_cost = (
        np.random.uniform(20000, 80000)
        if venue_type == "Indoor"
        else np.random.uniform(25000, 100000)
    )

    catering_multiplier = {
        "Vegetarian": 0.85,
        "Non-Vegetarian": 1.00,
        "Both": 1.12
    }[catering]

    estimated_cost = (
        base_costs[event_type]
        + guests * cost_per_guest[event_type]
        + venue_cost
    )

    estimated_cost *= style_multipliers[style]
    estimated_cost *= location_multipliers[location]
    estimated_cost *= catering_multiplier

    # Simulate variation in event costs.
    # This is synthetic noise, not observed market data.
    noise = np.random.normal(1.0, 0.08)

    actual_cost = max(
        10000,
        estimated_cost * noise
    )

    events.append({
        "EventID": event_id,
        "EventType": event_type,
        "Location": location,
        "GuestCount": guests,
        "Budget": budget,
        "Style": style,
        "VenueType": venue_type,
        "Catering": catering,
        "ActualCost": round(actual_cost, 2)
    })

events_df = pd.DataFrame(events)

events_df.to_csv(
    "data/events.csv",
    index=False
)

# --------------------------------------------------
# 4. GENERATE VENDOR DATA
# --------------------------------------------------

vendor_categories = [
    "Venue",
    "Catering",
    "Decoration",
    "Photography",
    "Music",
    "Makeup"
]

# Illustrative base price ranges in INR.
# Replace these assumptions with verified quotations later.
price_ranges = {
    "Venue": (20000, 250000),
    "Catering": (15000, 200000),
    "Decoration": (10000, 150000),
    "Photography": (10000, 120000),
    "Music": (5000, 60000),
    "Makeup": (3000, 40000)
}

number_of_vendors = 1200
vendors = []

for vendor_id in range(1, number_of_vendors + 1):

    category = np.random.choice(vendor_categories)
    location = np.random.choice(locations)
    style = np.random.choice(styles)

    min_price, max_price = price_ranges[category]

    # Generate an indicative vendor price.
    price = int(
        np.random.randint(min_price, max_price + 1)
    )

    # An indicative price range around the generated price.
    price_min = int(price * 0.90)
    price_max = int(price * 1.10)

    rating = round(
        float(np.random.uniform(3.0, 5.0)), 1
    )

    review_count = int(
        np.random.randint(1, 501)
    )

    # Capacity applies to venues.
    # Non-venue capacity is unknown, not zero.
    capacity = (
        int(np.random.randint(50, 1501))
        if category == "Venue"
        else np.nan
    )

    # Example service areas for testing only.
    service_areas = [location]

    if location in [
        "Kozhikode", "Malappuram", "Kannur", "Wayanad"
    ]:
        service_areas.extend(
            ["Kozhikode", "Malappuram", "Kannur", "Wayanad"]
        )

    elif location in [
        "Kochi", "Thrissur", "Kottayam", "Alappuzha"
    ]:
        service_areas.extend(
            ["Kochi", "Thrissur", "Kottayam", "Alappuzha"]
        )

    elif location in [
        "Palakkad", "Coimbatore"
    ]:
        service_areas.extend(
            ["Palakkad", "Coimbatore"]
        )

    # Remove duplicate service-area names.
    service_areas = list(dict.fromkeys(service_areas))

    vendors.append({
        "VendorID": vendor_id,
        "VendorName": (
            f"{location} {style} {category} {vendor_id}"
        ),
        "Category": category,
        "Location": location,
        "Style": style,
        "Price": price,
        "PriceMin": price_min,
        "PriceMax": price_max,
        "Rating": rating,
        "ReviewCount": review_count,
        "Capacity": capacity,
        "ServiceAreas": ", ".join(service_areas),

        # These values are deliberately unverified.
        # Do not treat them as real-time availability.
        "AvailabilityStatus": "Unverified",
        "PriceVerified": False,
        "AvailabilityVerified": False
    })

vendors_df = pd.DataFrame(vendors)

vendors_df.to_csv(
    "data/vendors.csv",
    index=False
)

# --------------------------------------------------
# 5. DISPLAY RESULTS
# --------------------------------------------------

print("\nSample datasets created successfully!")

print(f"Events: {len(events_df)} records")
print(f"Vendors: {len(vendors_df)} records")

print(f"Event locations: {events_df['Location'].nunique()}")
print(f"Vendor locations: {vendors_df['Location'].nunique()}")

print("\nVendor counts by category:")
print(vendors_df["Category"].value_counts())

print("\nFiles saved:")
print(" - data/events.csv")
print(" - data/vendors.csv")

print(
    "\nIMPORTANT: These are fictional sample records. "
    "Prices, ratings, reviews and service areas are "
    "not verified real-world information."
)

