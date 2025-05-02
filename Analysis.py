
"""
File needed to run this: 
    - dataset.csv
        - Can obtain this by running a previous cell that merges all the data from the 3 datasets:
          PTBXL, CPSC2018, Shoaxing

"""
import csv

def save_results(condition_label, code, results):
    """
    Saves the first 10 rows of 'results' to a CSV file named:
       <condition_label>_<code>.csv

    'results' is a list of tuples like [(patient_id, source), ...].
    """
    # Example filename: "false_negative_413444003.csv" or "false_positive_413444003.csv"
    output_filename = f"{condition_label.replace(' ', '_')}_{code}.csv"
    
    with open(output_filename, mode='w', newline='', encoding='utf-8') as out_file:
        writer = csv.writer(out_file)
        
        # Optional: Add a descriptive title row
        writer.writerow([f"{condition_label} for code: {code}"])
        # Header row
        writer.writerow(["PatientID", "Source"])
        
        # Write up to 10 results
        for pid, src in results[:10]:
            writer.writerow([pid, src])
    
    print(f"Results saved to '{output_filename}'\n")

def find_patients_mi_condition(csv_file_path):
    """
    Reads dataset.csv once and lets the user:
      - Enter a code to search (Phys_Codes).
      - Choose a scenario (false negative or false positive).
      - Prints and saves up to 10 matches that meet the chosen condition.
    """
    
    # 1) Read CSV into memory once
    with open(csv_file_path, mode='r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        all_rows = list(reader)
    
    # 2) Loop for multiple queries
    while True:
        # Prompt for a code
        code_to_search = input("Enter a code to search (or press ENTER to quit): ").strip()
        if not code_to_search:
            print("Exiting search.")
            break
        
        # Prompt for the scenario (false negative or false positive)
        print("\nChoose scenario to filter by:")
        print("  1) False Negative (MI_12SL=0, MI_Phys=1)")
        print("  2) False Positive (MI_12SL=1, MI_Phys=0)")
        scenario_choice = input("Enter '1' or '2' (or press ENTER to quit): ").strip()
        if not scenario_choice:
            print("Exiting search.")
            break
        
        # Determine the label and the condition we need
        if scenario_choice == "1":
            condition_label = "false_negative"
            mi_12sl_target = 0
            mi_phys_target = 1
        elif scenario_choice == "2":
            condition_label = "false_positive"
            mi_12sl_target = 1
            mi_phys_target = 0
        else:
            print("Invalid choice. Returning to main menu.\n")
            continue
        
        # 3) Collect matching rows
        matching_patients = []
        
        for row in all_rows:
            # Ensure required columns exist
            required_cols = ['PatientID', 'Source', 'Phys_Codes', 'MI_12SL', 'MI_Phys']
            if any(col not in row for col in required_cols):
                continue  # skip rows missing needed columns
            
            # Extract values
            patient_id = row['PatientID'].strip()
            source = row['Source'].strip()
            phys_codes_str = row['Phys_Codes'].strip()
            
            # Convert MI columns to int
            try:
                mi_12sl_val = int(row['MI_12SL'].strip())
                mi_phys_val = int(row['MI_Phys'].strip())
            except ValueError:
                # If they're invalid or blank, skip
                continue
            
            # Split codes by comma, strip whitespace
            codes_list = [code.strip() for code in phys_codes_str.split(',')]
            
            # Check if code is present
            if code_to_search in codes_list:
                # Check scenario condition: e.g. false negative or false positive
                if mi_12sl_val == mi_12sl_target and mi_phys_val == mi_phys_target:
                    matching_patients.append((patient_id, source))
        
        # 4) Print results in console and save to a CSV file
        if matching_patients:
            print(f"\n{condition_label.title()} matches for code '{code_to_search}':")
            # Show up to 10
            for pid, src in matching_patients[:10]:
                print(f"  PatientID: {pid}, Source: {src}")
            
            # Save results to a file
            save_results(condition_label, code_to_search, matching_patients)
        else:
            print(f"\nNo {condition_label} matches found for code '{code_to_search}'.\n")

if __name__ == "__main__":
    csv_file_path = "dataset.csv"  # Adjust if needed
    find_patients_mi_condition(csv_file_path)

