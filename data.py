import csv
# import pandas as pd


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
    