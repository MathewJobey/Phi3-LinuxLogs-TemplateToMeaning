import pandas as pd
import json
import os
from tqdm import tqdm

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def prompt_for_input_file() -> str:
    """Prompt user to paste the full path to the input .xlsx file."""
    path = input("Paste the FULL path to the .xlsx file: ").strip()
    # Remove surrounding quotes if the user copied them (common in Windows)
    return path.strip('"').strip("'")

def make_output_path(input_path: str) -> str:
    """
    Output file will be saved next to the input as:
      <file name> final excel.xlsx
    """
    in_dir = os.path.dirname(os.path.abspath(input_path))
    base = os.path.basename(input_path)
    stem, _ = os.path.splitext(base)
    # Result: "OriginalFile final excel.xlsx"
    return os.path.join(in_dir, f"{stem} final excel.xlsx")

def fill_meaning_from_json(row, meaning_map):
    """
    1. Grabs the 'Event Meaning' using the Template ID.
    2. Parses the 'Parameters' JSON (e.g., {"USER": "root"}).
    3. Replaces <USER> with 'root' in the sentence.
    """
    template_id = row.get('Template ID')
    params_json = row.get('Parameters')
    
    # Get the abstract meaning sentence (e.g. "User <USER> failed login")
    meaning_template = meaning_map.get(template_id)
    
    if not meaning_template:
        return "Error: Template ID not found in Summary"
    
    # If no parameters, return the template as is
    if pd.isna(params_json) or str(params_json).strip() == '{}':
        return meaning_template

    try:
        # Parse JSON and replace placeholders
        params_dict = json.loads(str(params_json))
        final_sentence = meaning_template
        
        for key, value in params_dict.items():
            placeholder = f"<{key}>"
            final_sentence = final_sentence.replace(placeholder, str(value))
            
        return final_sentence

    except json.JSONDecodeError:
        return "Error: Invalid JSON parameters"
    except Exception as e:
        return f"Error: {str(e)}"

# ==========================================
# MAIN LOGIC
# ==========================================
def main():
    # 1. Get Input File
    input_file = prompt_for_input_file()
    
    if not input_file:
        print("Error: No file path provided.")
        return

    if not os.path.exists(input_file):
        print(f"Error: Input file not found at: {input_file}")
        return

    # 2. Define Output Path
    output_file = make_output_path(input_file)

    print(f"\nReading: {input_file}")

    # 3. Load Data
    try:
        df_logs = pd.read_excel(input_file, sheet_name="Log Analysis")
        df_templates = pd.read_excel(input_file, sheet_name="Template Summary")
    except ValueError as e:
        print(f"Error reading sheets: {e}")
        print("Make sure your sheet names are exactly 'Log Analysis' and 'Template Summary'")
        return

    print("Mapping Templates...")
    # Create lookup dictionary: { TemplateID : "Event Meaning String" }
    meaning_map = dict(zip(df_templates['Template ID'], df_templates['Event Meaning']))

    print(f"Processing {len(df_logs)} logs...")
    tqdm.pandas() # Enable progress bar

    # 4. Generate Meanings
    df_logs['Meaning Log'] = df_logs.progress_apply(
        lambda row: fill_meaning_from_json(row, meaning_map), 
        axis=1
    )

    # 5. Reorder Columns (Meaning Log next to Raw Log)
    cols = list(df_logs.columns)
    if 'Raw Log' in cols:
        target_index = cols.index('Raw Log') + 1
        if 'Meaning Log' in cols:
            cols.insert(target_index, cols.pop(cols.index('Meaning Log')))
            df_logs = df_logs[cols]

    # 6. Save
    print(f"Saving to: {output_file}")
    
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        df_logs.to_excel(writer, sheet_name='Log Analysis', index=False)
        df_templates.to_excel(writer, sheet_name='Template Summary', index=False)
        
    print("-" * 30)
    print("DONE! File created successfully.")

if __name__ == "__main__":
    main()