import json
import re

# Read the malformed JSON file
with open('imagenet_class_index.json', 'r') as f:
    data = f.read()

# Step 1: Replace single quotes with double quotes around the values
# This will handle cases like {'key': 'value'} to {"key": "value"}
data = re.sub(r"\'(.*?)\'", r'"\1"', data)

# Step 2: Replace curly quotes with straight quotes (if any)
data = data.replace("“", '"').replace("”", '"')

# Step 3: Ensure all keys are surrounded by double quotes
data = re.sub(r"([^\s{,]+):", r'"\1":', data)

# Step 4: Remove any trailing commas
data = re.sub(r',\s*}', '}', data)
data = re.sub(r',\s*]', ']', data)

# Step 5: Try to load the corrected data into JSON
try:
    class_labels = json.loads(data)
    
    # Save the corrected JSON back to a file
    with open('corrected_imagenet_class_index.json', 'w') as f:
        json.dump(class_labels, f, indent=2)

    print("File fixed and saved as corrected_imagenet_class_index.json.")

except json.JSONDecodeError as e:
    # Output the problematic part of the file to help debug
    print(f"JSON Decode Error: {str(e)}")
    # Assuming the error message gives line and column, we extract them
    error_message = str(e)
    match = re.search(r"line (\d+) column (\d+)", error_message)
    if match:
        line_number, column_number = map(int, match.groups())
        print(f"Problematic part of the file near line {line_number}, column {column_number}:")
        print(data.splitlines()[line_number - 1])
