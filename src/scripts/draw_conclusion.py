import json
from .paths import get_conclusion_path
from .category import get_all_instance_names
import csv
import os

def main():
    interested_names = get_all_instance_names()
    # interested_names = sorted(list(interested_names))

    report = {}
    for name in interested_names:
        conclusion_path = get_conclusion_path(name)
        if not os.path.exists(conclusion_path):
            continue
        with open(conclusion_path, "r") as f:
            conclusion = json.load(f)
        report[name] = {}
        report[name]["type_of_equation"] = conclusion["type_of_equation"]
        report[name]["llm_upper_bound"] = conclusion["llm_upper_bound"]
        report[name]["llm_complexity"] = conclusion["llm_complexity"]
        report[name]["leading_term"] = conclusion["leading_term"]
        report[name]["original_equation"] = conclusion["original_equation"]

    # Convert to CSV format for Google Sheets
    csv_filename = "report.csv"
    with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
        # Define the column headers
        fieldnames = ['instance_name','leading_term','original_equation', 'type_of_equation', 'llm_upper_bound', 'llm_complexity']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write header
        writer.writeheader()
        
        # Write data rows
        for instance_name, data in report.items():
            row = {
                'instance_name': instance_name,
                'leading_term': data['leading_term'],
                'original_equation': data['original_equation'],
                'type_of_equation': data['type_of_equation'],
                'llm_upper_bound': data['llm_upper_bound'],
                'llm_complexity': data['llm_complexity']
            }
            writer.writerow(row)
    
    print(f"Report saved to {csv_filename}")
    pass

if __name__ == "__main__":
    main()