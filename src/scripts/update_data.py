import pandas as pd
import os

def update_report_checked():
    """
    Update report_checked.csv by adding llm_upper_bound and llm_complexity columns from report.csv
    """
    # Define file paths
    base_dir = "/Users/shimin/Desktop/Repos/BMC-benchmark"
    report_checked_path = os.path.join(base_dir, "report_checked.csv")
    report_path = os.path.join(base_dir, "report.csv")
    
    # Read both CSV files
    print("Reading report_checked.csv...")
    report_checked = pd.read_csv(report_checked_path)
    
    print("Reading report.csv...")
    report = pd.read_csv(report_path)
    
    print(f"report_checked.csv has {len(report_checked)} rows")
    print(f"report.csv has {len(report)} rows")
    
    # Check if the columns already exist in report_checked
    if 'llm_upper_bound' in report_checked.columns and 'llm_complexity' in report_checked.columns:
        print("Columns llm_upper_bound and llm_complexity already exist in report_checked.csv")
        print("Updating with new values from report.csv...")
    else:
        print("Adding new columns llm_upper_bound and llm_complexity...")
    
    # Create a mapping from instance_name to llm_upper_bound and llm_complexity
    llm_data = report[['instance_name', 'llm_upper_bound', 'llm_complexity']].set_index('instance_name')
    
    # Update the columns in report_checked with new values from report.csv
    report_checked['llm_upper_bound'] = report_checked['instance_name'].map(llm_data['llm_upper_bound'])
    report_checked['llm_complexity'] = report_checked['instance_name'].map(llm_data['llm_complexity'])
    
    # Reorder columns to match the desired format
    columns_order = ['instance_name', 'leading_term', 'original_equation', 'type_of_equation', 'confident?', 'llm_upper_bound', 'llm_complexity']
    report_checked = report_checked[columns_order]
    
    # Save the updated file
    print("Saving updated report_checked.csv...")
    # Replace NaN with 'NA' to match the original format
    report_checked['llm_upper_bound'] = report_checked['llm_upper_bound'].fillna('NA')
    report_checked['llm_complexity'] = report_checked['llm_complexity'].fillna('NA')
    report_checked.to_csv(report_checked_path, index=False)
    
    print("Update completed successfully!")
    print(f"Updated llm_upper_bound and llm_complexity columns for {len(report_checked)} rows")
    
    # Show some statistics
    print("\nStatistics:")
    print(f"Rows with llm_upper_bound (not NA): {(report_checked['llm_upper_bound'] != 'NA').sum()}")
    print(f"Rows with llm_complexity (not NA): {(report_checked['llm_complexity'] != 'NA').sum()}")
    print(f"Rows with both (not NA): {((report_checked['llm_upper_bound'] != 'NA') & (report_checked['llm_complexity'] != 'NA')).sum()}")
    print(f"Rows with NA in llm_upper_bound: {(report_checked['llm_upper_bound'] == 'NA').sum()}")
    print(f"Rows with NA in llm_complexity: {(report_checked['llm_complexity'] == 'NA').sum()}")

if __name__ == "__main__":
    update_report_checked()