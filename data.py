import csv
# import pandas as pd

"""
file = "master_csv_test_utf8.csv"

# df = pd.read_csv(file)
# print(df.head())


# Open and read the CSV file
with open(file, mode='r') as f:
    reader = csv.reader(f, delimiter=';')  # Use the correct delimiter ';'

    type_test = {}          # Dictionary to count occurrences of each Type
    narratives_column = []        # List to store Item column values

    # Iterate over each row in the CSV
    for counter, row in enumerate(reader):
        if counter == 0:
            header = row     # Store the header row
            narrative_index = header.index('Narrative')  # Find the index of 'Item' column
            q_1 = header.index('Q_1') 
            q_2 = header.index('Q_2') 
            q_3 = header.index('Q_3') 
            continue         # Skip header processing

        # Extract the 'Item' column
        narratives_column.append(row[narrative_index])  # Add the 'Item' value to the list

        # Count occurrences of the 'Type' column (first column)
        type_value = row[0]  # First column represents 'Type'

        if type_value not in type_test:
            type_test[type_value] = [row[narrative_index], (row[q_1], row[q_2], row[q_3])]



    # print("Type Counts:", type_test)
    # print(type_test.keys())
    # print(type_test)

        
# 102 questions

file_short = 'tests_qa.csv'

with open(file_short, mode='r') as f:
    reader = csv.reader(f)
    
    for row in reader:
        print(row)
"""    
    
import json
# import pandas as pd

"""
# Input and output file paths
csv_file = "Narrative_qs.csv"   # Replace with your CSV file
jsonl_file = "output.jsonl"  # Output JSONL file

# Read CSV and convert to JSONL
with open(csv_file, mode="r", encoding="utf-8") as infile, open(jsonl_file, mode="w", encoding="utf-8") as outfile:
    reader = csv.DictReader(infile)  # Read CSV as dictionaries
    for row in reader:
        json.dump(row, outfile)  # Convert to JSON and write
        outfile.write("\n")  # Newline for JSONL format

print(f"JSONL file saved as {jsonl_file}")


jsonl_file = 'falcon_answers_0.01.jsonl'
excel_file = 'falcon_answers_0.01_18.xlsx'

data = []
with open(jsonl_file, 'r') as f:
    for line in f:
        data.append(json.loads(line))

df = pd.DataFrame(data)
df.to_excel(excel_file, index=False)
"""

# Input JSON file (containing an array)
input_file = 'falcon_answers_0.01-2.jsonl'  # Replace with your JSON file
output_file = "falcon_answers_0.01_4.jsonl"  # Desired JSONL file

# Read JSON array from the file
with open(input_file, "r", encoding="utf-8") as file:
    data = json.load(file)  # Parse the entire JSON array

# Write each JSON object on a new line
with open(output_file, "w", encoding="utf-8") as file:
    for entry in data:
        file.write(json.dumps(entry) + "\n") 