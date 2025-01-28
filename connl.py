    #reads conll files and stores the text as sentence and token classes
# def readCoNLL(filename, labelCol):
""" 
import zipfile
filename = '/Users/momotakamatsu/Desktop/CoNLL_scenarios_prompts_children.numbers'

# Extract the contents of the ZIP archive
with zipfile.ZipFile(filename, 'r') as zip_ref:
    zip_ref.extractall('extracted_numbers_file')

#print("Extracted the contents to 'extracted_numbers_file' folder.")


import plistlib

# Path to the .plist file
plist_path = 'extracted_numbers_file/Metadata/Properties.plist'

# Load the plist data
with open(plist_path, 'rb') as plist_file:
    data = plistlib.load(plist_file)

#print(data)


import os

# Path to the folder containing .iwa files
iwa_folder = 'extracted_numbers_file/Index/Tables'

# Loop through all DataList-*.iwa files
for filename in os.listdir(iwa_folder):
    if filename.startswith("DataList"):
        with open(os.path.join(iwa_folder, filename), 'rb') as f:
            data = f.read()
            print(f"\n--- Content of {filename} ---")
            print(data[:200])  # Print the first 200 bytes
"""


import pandas as pd
import numpy as np
import csv

filepath = 'master_csv_test_utf8.csv'

# df = pd.read_csv(filepath, sep='\t', quoting=csv.QUOTE_NONE)
# print(df)



# Specify the path to your CSV file
# file_path = 'your_file.csv'

# Open the CSV file
with open(filepath, newline='') as csvfile:
    csvreader = csv.reader(csvfile)

    # Print each row in the CSV file
    for row in csvreader:
        print(row)