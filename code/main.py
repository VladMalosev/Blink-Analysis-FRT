import cv2
from frt import main as security_test


# We'll implement these later
# from auth import register_user, login_user

def show_menu():
    while True:
        print("\n" + "=" * 30)
        print(" Deepfake Security System ")
        print("=" * 30)
        print("1. Test Code Functionality")
        print("2. Register New User")
        print("3. Login")
        print("4. Interactive Security Test")
        print("5. Exit")

        choice = input("Enter your choice (1-5): ")

        if choice == '1':
            run_demo_mode()
        elif choice == '2':
            register_user()
        elif choice == '3':
            login_user()
        elif choice == '4':
            run_security_test()
        elif choice == '5':
            print("Exiting system...")
            break
        else:
            print("Invalid choice, please try again.")


def run_demo_mode():
    print("\n" + "-" * 30)
    print(" Demo Mode - Basic Functionality Test ")
    print("-" * 30)
    print("Starting webcam feed with basic analysis...")
    security_test(demo_mode=True)


def register_user():
    print("\n" + "-" * 30)
    print(" User Registration ")
    print("-" * 30)
    # Placeholder for actual registration logic
    print("Registration system under development")
    # username = input("Enter username: ")
    # password = input("Enter password: ")
    # Implement actual registration logic later


def login_user():
    print("\n" + "-" * 30)
    print(" User Login ")
    print("-" * 30)
    # Placeholder for actual login logic
    print("Login system under development")


def run_security_test():
    print("\n" + "-" * 30)
    print(" Interactive Security Test ")
    print("-" * 30)
    print("Starting advanced security checks...")
    security_test(security_mode=True)


if __name__ == "__main__":
    show_menu()