import random
import string

def generate_password(length):
    if length < 1:
        return "Password length must be at least 1."
    
    # Define the character sets
    lowercase = string.ascii_lowercase
    uppercase = string.ascii_uppercase
    digits = string.digits
    special_characters = string.punctuation

    # Combine all characters
    all_characters = lowercase + uppercase + digits + special_characters

    # Generate a random password
    password = ''.join(random.choice(all_characters) for _ in range(length))
    
    return password

def main():
    print("Password Generator")
    
    # Get user input for password length
    try:
        length = int(input("Enter the desired length of the password: "))
    except ValueError:
        print("Invalid input. Please enter a numeric value.")
        return
    
    # Generate and display the password
    password = generate_password(length)
    print(f"Generated Password: {password}")

# Run the main function
if __name__ == "__main__":
    main()
