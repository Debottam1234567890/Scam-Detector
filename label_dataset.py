import csv

# Load the existing dataset
input_file = 'dataset.csv'  # Update if needed
output_file = 'labeled_dataset.csv'

# Open the existing dataset for reading
with open(input_file, mode='r', newline='', encoding='utf-8') as infile:
    reader = csv.reader(infile)
    header = next(reader)  # Skip the header row
    rows = list(reader)

# Define the factor columns
scam_factors_columns = [
    'urgency', 'money_request', 'official_appearance', 'reward_offer', 'celebrity_reference',
    'grammar_issues', 'unusual_contact_method', 'pressure_to_act', 'suspicious_link', 'upfront_payment'
]

# Find the indices of the scam factor columns in the header
factor_indices = [header.index(factor) for factor in scam_factors_columns]

# Add a new column for "Label" at the end of the header
header.append('Label')

# Create a new list for storing updated rows
updated_rows = []

# Define a threshold for labeling a message as a scam
threshold = 3  # You can adjust this threshold as needed

# Process each row to compute the scam score and assign the label
for row in rows:
    try:
        # Calculate the total scam score by summing the values in the scam factor columns
        total_scam_score = sum(
            float(row[i]) if row[i].replace('.', '', 1).isdigit() else 0  # Handle non-numeric values
            for i in factor_indices
        )
        
        # Assign a label based on the total scam score
        label = 1 if total_scam_score >= threshold else 0
        row.append(label)  # Append the label to the row
        
    except Exception as e:
        print(f"Error processing row: {row} - {e}")
        row.append(0)  # Append label as 0 if there's an error in processing the row

    # Add the updated row to the list
    updated_rows.append(row)

# Write the updated rows to a new CSV file
with open(output_file, mode='w', newline='', encoding='utf-8') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(header)  # Write the header
    writer.writerows(updated_rows)  # Write the updated rows

print(f"✅ Labeled dataset saved as '{output_file}'")

