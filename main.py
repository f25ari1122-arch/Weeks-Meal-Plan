
# ============================================================
# SMARTBITE - SMART WEEKLY GROCERY BUDGET & MEAL PLANNER
# MACHINE LEARNING PROJECT
# ============================================================

import pandas as pd
import numpy as np
import pickle
import zipfile

from datetime import datetime

from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# 1. GROCERY DATABASE
# ============================================================

groceries = pd.DataFrame([
    ["Rice", "Grains", "kg", 300],
    ["Chicken", "Meat", "kg", 750],
    ["Eggs", "Protein", "piece", 30],
    ["Potatoes", "Vegetables", "kg", 100],
    ["Onions", "Vegetables", "kg", 120],
    ["Tomatoes", "Vegetables", "kg", 150],
    ["Lentils", "Pulses", "kg", 280],
    ["Bread", "Bakery", "piece", 80],
    ["Milk", "Dairy", "litre", 220],
    ["Oil", "Cooking", "litre", 550]
], columns=["Item", "Category", "Unit", "Price"])

# Assign numeric codes to groceries
groceries["Item_Code"] = range(len(groceries))


# ============================================================
# 2. HISTORICAL GROCERY PRICE DATASET
# ============================================================

price_factors = [
    0.90, 0.92, 0.95, 0.97,
    0.96, 1.00, 1.02, 1.05,
    1.03, 1.06, 1.08, 1.10
]

price_history = []

for _, grocery in groceries.iterrows():

    for month in range(1, 13):

        historical_price = (
            grocery["Price"] * price_factors[month - 1]
        )

        price_history.append([
            grocery["Item"],
            grocery["Item_Code"],
            month,
            round(historical_price, 2)
        ])

price_df = pd.DataFrame(
    price_history,
    columns=["Item", "Item_Code", "Month", "Price"]
)


# ============================================================
# 3. TRAIN GROCERY PRICE PREDICTION MODEL
# ============================================================

X = price_df[["Item_Code", "Month"]]
y = price_df["Price"]

price_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

price_model.fit(X, y)

print("Grocery Price Prediction Model Trained!")


# ============================================================
# 4. PREDICT NEXT MONTH'S GROCERY PRICES
# ============================================================

current_month = datetime.now().month

next_month = (current_month % 12) + 1

prediction_input = pd.DataFrame({
    "Item_Code": groceries["Item_Code"],
    "Month": [next_month] * len(groceries)
})

groceries["Predicted_Price"] = np.round(
    price_model.predict(prediction_input),
    2
)

print("Next Month Grocery Prices Predicted!")


# ============================================================
# 5. MEAL DATABASE
# ============================================================

meals = pd.DataFrame([

    [
        "Omelette",
        "Breakfast",
        "Vegetarian",
        {"Eggs": 2, "Onions": 0.05},
        10
    ],

    [
        "Boiled Eggs",
        "Breakfast",
        "Vegetarian",
        {"Eggs": 2},
        10
    ],

    [
        "Bread and Milk",
        "Breakfast",
        "Vegetarian",
        {"Bread": 1, "Milk": 0.25},
        5
    ],

    [
        "Chicken Sandwich",
        "Breakfast",
        "Non-Vegetarian",
        {"Bread": 1, "Chicken": 0.08},
        15
    ],

    [
        "Chicken Rice",
        "Lunch",
        "Non-Vegetarian",
        {"Rice": 0.15, "Chicken": 0.15, "Onions": 0.05},
        30
    ],

    [
        "Potato Curry",
        "Lunch",
        "Vegetarian",
        {"Potatoes": 0.2, "Onions": 0.05, "Tomatoes": 0.05},
        25
    ],

    [
        "Lentil Curry",
        "Lunch",
        "Vegetarian",
        {"Lentils": 0.12, "Onions": 0.05, "Tomatoes": 0.05},
        25
    ],

    [
        "Chicken Biryani",
        "Dinner",
        "Non-Vegetarian",
        {"Rice": 0.15, "Chicken": 0.15, "Potatoes": 0.1},
        40
    ],

    [
        "Egg Curry",
        "Dinner",
        "Vegetarian",
        {"Eggs": 2, "Tomatoes": 0.1, "Onions": 0.05},
        25
    ],

    [
        "Lentil Rice",
        "Dinner",
        "Vegetarian",
        {"Rice": 0.12, "Lentils": 0.1},
        25
    ]

], columns=[
    "Meal",
    "Category",
    "Preference",
    "Ingredients",
    "Preparation_Time"
])


# ============================================================
# 6. TRAIN MEAL RECOMMENDATION MODEL
# ============================================================

meal_text = []

for _, meal in meals.iterrows():

    ingredients = " ".join(meal["Ingredients"].keys())

    text = (
        meal["Meal"] + " " +
        meal["Category"] + " " +
        meal["Preference"] + " " +
        ingredients
    )

    meal_text.append(text)

vectorizer = TfidfVectorizer()

meal_vectors = vectorizer.fit_transform(meal_text)

print("Meal Recommendation Model Trained!")


# ============================================================
# 7. SAVE TRAINED MODELS
# ============================================================

with open("price_model.pkl", "wb") as file:

    pickle.dump(price_model, file)


meal_recommender = {
    "vectorizer": vectorizer,
    "meal_vectors": meal_vectors,
    "meals": meals
}

with open("meal_recommender.pkl", "wb") as file:

    pickle.dump(meal_recommender, file)


with open("grocery_database.pkl", "wb") as file:

    pickle.dump(groceries, file)


price_df.to_csv("price_history.csv", index=False)

print("\nAll trained models saved successfully!")


# ============================================================
# 8. USER INPUT
# ============================================================

budget = float(input("\nEnter weekly budget (Rs): "))

people = int(input("Enter number of people: "))

print("\nFOOD PREFERENCES")
print("1. Any")
print("2. Vegetarian")
print("3. Non-Vegetarian")

choice = input("Select preference: ")

if choice == "1":
    preference = "Any"

elif choice == "2":
    preference = "Vegetarian"

elif choice == "3":
    preference = "Non-Vegetarian"

else:
    preference = "Any"


food_preferences = input(
    "\nEnter preferred foods (e.g. rice, eggs, lentils): "
).strip()


# ============================================================
# 9. GENERATE MEAL RECOMMENDATIONS
# ============================================================

if budget <= 0 or people <= 0:

    print("\nBudget and people must be positive.")

else:

    query = preference + " " + food_preferences

    query_vector = vectorizer.transform([query])

    similarities = cosine_similarity(
        query_vector,
        meal_vectors
    )[0]

    meals["Similarity"] = similarities


    # ========================================================
    # 10. GENERATE SEVEN-DAY MEAL PLAN
    # ========================================================

    days = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday"
    ]

    categories = [
        "Breakfast",
        "Lunch",
        "Dinner"
    ]

    grocery_quantities = {}

    selected_meals = []

    meal_counts = {
        meal: 0 for meal in meals["Meal"]
    }

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

                selected_meals.append([
                    day,
                    category,
                    best_meal["Meal"],
                    best_meal["Ingredients"],
                    best_meal["Preparation_Time"]
                ])

            else:

                selected_meals.append([
                    day,
                    category,
                    "No meal available",
                    {},
                    0
                ])


    # ========================================================
    # 11. GENERATE GROCERY SHOPPING LIST
    # ========================================================

    grocery_list = []

    for item, quantity in grocery_quantities.items():

        row = groceries[
            groceries["Item"] == item
        ].iloc[0]

        grocery_list.append([
            item,
            round(quantity, 2),
            row["Unit"],
            row["Predicted_Price"],
            round(quantity * row["Predicted_Price"], 2)
        ])

    grocery_df = pd.DataFrame(
        grocery_list,
        columns=[
            "Grocery",
            "Quantity",
            "Unit",
            "Predicted Price",
            "Total Price"
        ]
    )


    # ========================================================
    # 12. MEAL INGREDIENTS AND COST
    # ========================================================

    meal_details = []

    for day, category, meal_name, ingredients, prep_time in selected_meals:

        if meal_name == "No meal available":
            continue

        meal_cost = 0
        ingredient_list = []

        for item, quantity in ingredients.items():

            required_quantity = quantity * people

            price = groceries.loc[
                groceries["Item"] == item,
                "Predicted_Price"
            ].iloc[0]

            cost = required_quantity * price

            meal_cost += cost

            ingredient_list.append(
                f"{item}: {round(required_quantity, 2)}"
            )

        meal_details.append([
            day,
            category,
            meal_name,
            ", ".join(ingredient_list),
            round(meal_cost, 2),
            prep_time
        ])

    meal_details_df = pd.DataFrame(
        meal_details,
        columns=[
            "Day",
            "Category",
            "Meal",
            "Ingredients",
            "Estimated Cost",
            "Preparation Time (min)"
        ]
    )


    # ========================================================
    # 13. WEEKLY MEAL SCHEDULE
    # ========================================================

    meal_df = pd.DataFrame(
        selected_meals,
        columns=[
            "Day",
            "Category",
            "Meal",
            "Ingredients",
            "Preparation_Time"
        ]
    )

    schedule = meal_df.pivot(
        index="Day",
        columns="Category",
        values="Meal"
    )

    schedule = schedule.reindex(
        index=days,
        columns=["Breakfast", "Lunch", "Dinner"]
    )


    # ========================================================
    # 14. DISPLAY PREDICTED GROCERY PRICES
    # ========================================================

    print("\n========== SMARTBITE ML ==========")

    print("\nPREDICTED GROCERY PRICES")

    print(
        groceries[
            ["Item", "Price", "Predicted_Price"]
        ].to_string(index=False)
    )


    # ========================================================
    # 15. DISPLAY GROCERY SHOPPING LIST
    # ========================================================

    print("\n========== GROCERY SHOPPING LIST ==========")

    if not grocery_df.empty:
        print(grocery_df.to_string(index=False))

    else:
        print("No groceries selected.")

    total = grocery_df["Total Price"].sum()

    print("\nTotal Grocery Cost: Rs.", round(total, 2))

    print("Remaining Budget: Rs.", round(budget - total, 2))


    # ========================================================
    # 16. DISPLAY WEEKLY MEAL SCHEDULE
    # ========================================================

    print("\n========== WEEKLY MEAL SCHEDULE ==========")

    print(schedule.to_string())


    # ========================================================
    # 17. DISPLAY MEAL DETAILS
    # ========================================================

    print("\n========== MEAL INGREDIENTS AND COST ==========")

    if not meal_details_df.empty:
        print(meal_details_df.to_string(index=False))

    else:
        print("No meals available.")


    # ========================================================
    # 18. FINAL BUDGET SUMMARY
    # ========================================================

    print("\n========== BUDGET SUMMARY ==========")

    print("Weekly Budget: Rs.", budget)

    print("Number of People:", people)

    print("Food Preference:", preference)

    print("Total Expenses: Rs.", round(total, 2))

    print("Remaining Budget: Rs.", round(budget - total, 2))

    if (meal_df["Meal"] == "No meal available").any():

        print("\nSome meal slots could not be filled.")

    else:

        print("\nAll meal slots successfully planned!")


# ============================================================
# 19. CREATE ZIP FILE OF TRAINED MODELS
# ============================================================

with zipfile.ZipFile(
    "SmartBite_Trained_Models.zip",
    "w",
    zipfile.ZIP_DEFLATED
) as zip_file:

    zip_file.write("price_model.pkl")

    zip_file.write("meal_recommender.pkl")

    zip_file.write("grocery_database.pkl")

    zip_file.write("price_history.csv")


print("\n====================================")

print("SMARTBITE TRAINING COMPLETED!")

print("All model files are ready to download.")

print("====================================")


# ============================================================
# 20. DOWNLOAD TRAINED MODEL FILES
# ============================================================

from google.colab import files

files.download("SmartBite_Trained_Models.zip")
