def display_menu():
    print("Select a dataset option:")
    print("1. Chicago Crime Data")
    print("2. Credit Spending Habits")
    print("3. Superstore Sales")

def get_user_choice():
    return input("Enter the number of your choice: ")

def display_model_menu():
    print("Choose a model:")
    print("1: LSTM")
    print("2: LightGBM")
    print("3: XGBoost")

def get_model_choice():
    return input("Enter the number of your choice: ")

def display_unlearning_menu():
    print("Choose an unlearning model:") 
    print("1. Feature Masking")
    print("2. Layer Freezing")
    print("3. Knowledge Distillation")
    print("4. Exit Unlearning")

def get_unlearning_choice():
    return input("Enter the number of your choice: ")
