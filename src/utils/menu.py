def display_menu():
    print("Select a dataset option:")
    print("1. Chicago Crime Data")
    print("2. Credit Spending Habits")
    print("3. Superstore Sales")

def get_user_choice():
    return input("Enter the number of your choice: ")

def display_model_menu():
    print("Choose a model:")
    print("1: LightGBM")
    print("2: XGBoost")
    print("3: CatBoost")

def get_model_choice():
    return input("Enter the number of your choice: ")
