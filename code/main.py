from frt import main as security_test



def show_menu():
    while True:
        print("\n" + "=" * 30)
        print(" Deepfake Security System ")
        print("=" * 30)
        print("1. Test Code Functionality")
        print("2. Interactive Security Test")
        print("3. Exit")

        choice = input("Enter your choice (1-3): ")

        if choice == '1':
            run_demo_mode()
        elif choice == '2':
            run_security_test(interactive=True)
        elif choice == '3':
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


def run_security_test(interactive=False):
    print("\n" + "-" * 30)
    print(" Interactive Security Test " if interactive else " Security Test ")
    print("-" * 30)
    if interactive:
        print("\nFollow the on-screen instructions...")
    security_test(interactive_mode=interactive)


if __name__ == "__main__":
    show_menu()