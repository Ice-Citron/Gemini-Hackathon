# Vulnerable code
def check_password(input_pass):
    # Hardcoded = Bad
    return input_pass == "secret123"

if __name__ == "__main__":
    pass
