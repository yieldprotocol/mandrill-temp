from datetime import datetime
import random
import string
import hashlib

# Get the current datetime as a string
current_datetime_str = str(datetime.now())

# Hash the datetime string to generate a seed
seed = int(hashlib.sha256(current_datetime_str.encode()).hexdigest(), 16)

# Create a random number generator instance with the seed
str_random = random.Random(seed)

def generate_random_string(length):
    # Define the set of characters you want in the random string
    characters = string.ascii_letters + string.digits  # You can include other characters if needed

    # Generate the random string by selecting characters randomly
    random_string = ''.join(str_random.choice(characters) for _ in range(length))

    return random_string