import pickle
import zipfile
from pathlib import Path
from datetime import datetime

import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# SMARTBITE - WEEKLY GROCERY BUDGET & MEAL PLANNER
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_ZIP = BASE_DIR / "SmartBite_Trained_Models.zip"


# ============================================================
# 1. LOAD TRAINED MODELS
# ============================================================

def load_project_files():

    required_files = [
        "price_model.pkl",
        "meal_recommender.pkl",
        "grocery_database.pkl"
    ]

    missing = [
        name for name in required_files
        if not (BASE_DIR / name).exists()
    ]

    if missing and MODEL_ZIP.exists():
        with zipfile.ZipFile(MODEL_ZIP, "r") as zip_file:
            zip_file.extractall(BASE_DIR)

    for name in required_files:
        if not (BASE_DIR / name).exists():
            raise FileNotFoundError(
                f"{name} is missing. Extract SmartBite_Trained_Models.zip."
            )

    with open(BASE_DIR / "price_model.pkl", "rb") as file:
        price_model = pickle.load(file)

    with open(BASE_DIR / "meal_recommender.pkl", "rb") as file:
        meal_recommender = pickle.load(file)

    with open(BASE_DIR / "grocery_database.pkl", "rb") as file:
        groceries = pickle.load(file)

    return price_model, meal_recommender, groceries


@st.cache_resource
def get_models():
    return load_project_files()


# ============================================================
# 2. GENERATE WEEKLY MEAL PLAN
# ============================================================

def create_weekly_plan(
    budget,
    people,
    preference,
    preferred_foods,
    groceries,
    meals,
    vectorizer,
    meal_vectors
):

    query = preference + " " + preferred_foods

    query_vector = vectorizer.transform([query])

    similarities = cosine_similarity(
        query_vector,
        meal_vectors
    )[0]

    meals = meals.copy()
    meals["Similarity"] = similarities

    days = [
        "Monday", "Tuesday", "Wednesday",
        "Thursday", "Friday", "Saturday", "Sunday"
    ]

    categories = ["Breakfast", "Lunch", "Dinner"]

    grocery_quantities = {}
    selected_meals = []

    meal_counts = {
        meal: 0 for meal in meals["Meal"]
    }

    # --------------------------------------------------------
    # SELECT MEALS
    # --------------------------------------------------------

    for day in days:

        for category in categories:

            options = meals[
                meals["Category"] == category
            ].copy()

            if preference != "Any":

                options = options[
                    options["Preference"] == preference
                ]

            best_meal = None
            best_score = None
            best_quantities = None

            for _, meal in options.iterrows():

                new_quantities = grocery_quantities.copy()

                for item, quantity in meal["Ingredients"].items():

                    new_quantities[item] = (
                        new_quantities.get(item, 0)
                        + quantity * people
                    )

                total_cost = 0

                for item, quantity in new_quantities.items():

                    price = groceries.loc[
                        groceries["Item"] == item,
                        "Predicted_Price"
                    ].iloc[0]

                    total_cost += quantity * price

                if total_cost <= budget:

                    score = (
                        (1 - meal["Similarity"]) * 100
                        + meal_counts[meal["Meal"]] * 20
                        + (total_cost / budget) * 10
                    )

                    if best_score is None or score < best_score:

                        best_score = score
                        best_meal = meal
                        best_quantities = new_quantities

            if best_meal is not None:

                grocery_quantities = best_quantities

                meal_counts[best_meal["Meal"]] += 1

                selected_meals.append({
                    "Day": day,
                    "Category": category,
                    "Meal": best_meal["Meal"],
                    "Ingredients": best_meal["Ingredients"],
                    "Preparation Time (min)": best_meal["Preparation_Time"]
                })

            else:

                selected_meals.append({
                    "Day": day,
                    "Category": category,
                    "Meal": "No meal available",
                    "Ingredients": {},
                    "Preparation Time (min)": 0
                })

    # ========================================================
    # 3. GROCERY SHOPPING LIST
    # ========================================================

    grocery_rows = []

    for item, quantity in grocery_quantities.items():

        row = groceries[
            groceries["Item"] == item
        ].iloc[0]

        price = float(row["Predicted_Price"])

        grocery_rows.append({
            "Grocery": item,
            "Quantity": round(quantity, 2),
            "Unit": row["Unit"],
            "Predicted Price (Rs.)": round(price, 2),
            "Total Price (Rs.)": round(quantity * price, 2)
        })

    grocery_df = pd.DataFrame(grocery_rows)

    # ========================================================
    # 4. MEAL INGREDIENTS AND COST
    # ========================================================

    detail_rows = []

    for meal in selected_meals:

        if meal["Meal"] == "No meal available":
            continue

        meal_cost = 0
        ingredients = []

        for item, quantity in meal["Ingredients"].items():

            required_quantity = quantity * people

            price = float(
                groceries.loc[
                    groceries["Item"] == item,
                    "Predicted_Price"
                ].iloc[0]
            )

            unit = groceries.loc[
                groceries["Item"] == item,
                "Unit"
            ].iloc[0]

            meal_cost += required_quantity * price

            ingredients.append(
                f"{item}: {round(required_quantity, 2)} {unit}"
            )

        detail_rows.append({
            "Day": meal["Day"],
            "Category": meal["Category"],
            "Meal": meal["Meal"],
            "Ingredients": ", ".join(ingredients),
            "Estimated Cost (Rs.)": round(meal_cost, 2),
            "Preparation Time (min)": meal["Preparation Time (min)"]
        })

    details_df = pd.DataFrame(detail_rows)

    # ========================================================
    # 5. WEEKLY SCHEDULE
    # ========================================================

    meal_df = pd.DataFrame(selected_meals)

    schedule = meal_df.pivot(
        index="Day",
        columns="Category",
        values="Meal"
    )

    schedule = schedule.reindex(
        index=days,
        columns=categories
    )

    total_cost = (
        grocery_df["Total Price (Rs.)"].sum()
        if not grocery_df.empty else 0
    )

    return {
        "grocery_df": grocery_df,
        "details_df": details_df,
        "schedule": schedule,
        "selected_meals": selected_meals,
        "total_cost": total_cost,
        "remaining": budget - total_cost
    }


# ============================================================
# 6. STREAMLIT PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="SmartBite Meal Planner",
    page_icon="🍽️",
    layout="wide"
)

st.title("🍽️ SmartBite")
st.subheader("Smart Weekly Grocery Budget & Meal Planner")

st.write(
    "Generate a weekly meal schedule and grocery shopping "
    "list according to your budget and food preferences."
)


# ============================================================
# 7. LOAD PROJECT DATA
# ============================================================

try:

    price_model, meal_recommender, groceries = get_models()

    vectorizer = meal_recommender["vectorizer"]
    meal_vectors = meal_recommender["meal_vectors"]
    meals = meal_recommender["meals"]

except Exception as error:

    st.error(str(error))

    st.info(
        "Place your trained model files or "
        "SmartBite_Trained_Models.zip beside app.py."
    )

    st.stop()


# ============================================================
# 8. USER INPUT SIDEBAR
# ============================================================

st.sidebar.header("Weekly Meal Planner")

budget = st.sidebar.number_input(
    "Weekly Budget (Rs.)",
    min_value=1.0,
    value=5000.0,
    step=500.0
)

people = st.sidebar.number_input(
    "Number of People",
    min_value=1,
    value=2,
    step=1
)

preference = st.sidebar.selectbox(
    "Food Preference",
    ["Any", "Vegetarian", "Non-Vegetarian"]
)

preferred_foods = st.sidebar.text_input(
    "Preferred Foods",
    placeholder="Rice, eggs, lentils..."
)

generate = st.sidebar.button(
    "Generate Weekly Plan",
    type="primary",
    use_container_width=True
)


# ============================================================
# 9. GENERATE RESULTS
# ============================================================

if generate:

    if budget <= 0 or people <= 0:

        st.error("Budget and number of people must be positive.")

    else:

        result = create_weekly_plan(
            budget,
            people,
            preference,
            preferred_foods.strip(),
            groceries,
            meals,
            vectorizer,
            meal_vectors
        )

        st.session_state["result"] = result

        st.session_state["inputs"] = {
            "budget": budget,
            "people": people,
            "preference": preference,
            "preferred_foods": preferred_foods
        }


# ============================================================
# 10. DISPLAY DASHBOARD
# ============================================================

if "result" not in st.session_state:

    st.info(
        "Enter your weekly budget and preferences "
        "from the sidebar to generate your meal plan."
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Meal Planning", "7 Days")
    col2.metric("Daily Meals", "3")
    col3.metric("ML Models", "2")

    st.stop()


result = st.session_state["result"]
inputs = st.session_state["inputs"]


# ============================================================
# 11. BUDGET SUMMARY
# ============================================================

st.divider()

st.header("Weekly Plan Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Weekly Budget",
    f"Rs. {inputs['budget']:,.2f}"
)

col2.metric(
    "Grocery Cost",
    f"Rs. {result['total_cost']:,.2f}"
)

col3.metric(
    "Remaining Budget",
    f"Rs. {result['remaining']:,.2f}"
)

col4.metric(
    "People",
    inputs["people"]
)

if result["remaining"] < 0:

    st.warning("Estimated cost exceeds your budget.")

elif result["remaining"] == 0:

    st.success("Your plan uses the full budget.")

else:

    st.success("Your meal plan is within your budget.")

st.caption(
    f"Food Preference: {inputs['preference']} | "
    f"Preferred Foods: {inputs['preferred_foods'] or 'Not specified'}"
)


# ============================================================
# 12. RESULTS TABS
# ============================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "📅 Weekly Meal Schedule",
    "🛒 Grocery Shopping List",
    "🍛 Meal Details",
    "📈 Predicted Prices"
])


# ------------------------------------------------------------
# WEEKLY MEAL SCHEDULE
# ------------------------------------------------------------

with tab1:

    st.subheader("Your 7-Day Meal Schedule")

    st.dataframe(
        result["schedule"],
        use_container_width=True
    )

    planned = sum(
        meal["Meal"] != "No meal available"
        for meal in result["selected_meals"]
    )

    st.write(f"**Meals Planned: {planned}/21**")

    if planned < 21:

        st.warning(
            "Some meal slots could not be filled "
            "within the selected budget or preference."
        )


# ------------------------------------------------------------
# GROCERY SHOPPING LIST
# ------------------------------------------------------------

with tab2:

    st.subheader("Grocery Shopping List")

    if result["grocery_df"].empty:

        st.warning("No groceries selected.")

    else:

        st.dataframe(
            result["grocery_df"],
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "Download Grocery List CSV",
            data=result["grocery_df"].to_csv(
                index=False
            ).encode("utf-8"),
            file_name="SmartBite_Grocery_List.csv",
            mime="text/csv"
        )

        st.write(
            f"**Total Grocery Cost: Rs. "
            f"{result['total_cost']:,.2f}**"
        )


# ------------------------------------------------------------
# MEAL DETAILS
# ------------------------------------------------------------

with tab3:

    st.subheader("Meal Ingredients and Estimated Cost")

    if result["details_df"].empty:

        st.warning("No meals available.")

    else:

        st.dataframe(
            result["details_df"],
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "Download Meal Details CSV",
            data=result["details_df"].to_csv(
                index=False
            ).encode("utf-8"),
            file_name="SmartBite_Meal_Details.csv",
            mime="text/csv"
        )


# ------------------------------------------------------------
# PREDICTED GROCERY PRICES
# ------------------------------------------------------------

with tab4:

    st.subheader("Grocery Price Predictions")

    price_display = groceries[
        ["Item", "Price", "Predicted_Price"]
    ].copy()

    price_display.columns = [
        "Grocery",
        "Current Base Price (Rs.)",
        "Predicted Price (Rs.)"
    ]

    next_month = (datetime.now().month % 12) + 1

    st.write(
        f"Predicted prices for month number {next_month}."
    )

    st.dataframe(
        price_display.round(2),
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Predictions are based on the demonstration "
        "historical price dataset, not live market prices."
    )


# ============================================================
# 13. FOOTER
# ============================================================

st.divider()

st.caption(
    "SmartBite | ML-Based Grocery Price Prediction "
    "and Weekly Meal Recommendation System"
)