def display_menu():
    print("Please select a dataset option:")
    print("1. Chicago Crime Data")
    print("2. Credit Spending Habits")
    print("3. Superstore Sales")

def get_user_choice():
    return input("Enter the number of your choice: ")

def display_model_menu():
    print("Choose a model (1: LSTM, 2: LightGBM, 3: XGBoost):")

def get_model_choice():
    return input("Enter the number of your choice: ")
