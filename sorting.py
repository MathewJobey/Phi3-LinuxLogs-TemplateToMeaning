import pandas as pd
import os
import json
from dateutil import parser
from tqdm import tqdm

def prompt_for_file():
    path = input("Paste the FULL path to your 'final excel' file: ").strip()
    return path.strip('"').strip("'")

def get_time_from_json(params_str):
    """
    Parses the JSON string: '{"TIMESTAMP": "Jun 14...", ...}'
    Returns a real datetime object.
    """
    try:
        if pd.isna(params_str):
            return pd.NaT
            
        # 1. Parse JSON string to Dictionary
        params = json.loads(str(params_str))
        
        # 2. Get the 'TIMESTAMP' value
        time_str = params.get('TIMESTAMP')
        
        if not time_str:
            return pd.NaT
            
        # 3. Convert string to Datetime
        return parser.parse(time_str)
        
    except (json.JSONDecodeError, TypeError):
        return pd.NaT
    except Exception:
        return pd.NaT

def main():
    # 1. Get Input
    input_file = prompt_for_file()
    
    if not os.path.exists(input_file):
        print(f"Error: File not found at {input_file}")
        return

    print(f"Reading {os.path.basename(input_file)}...")
    try:
        df = pd.read_excel(input_file, sheet_name="Log Analysis")
        df_templates = pd.read_excel(input_file, sheet_name="Template Summary")
    except ValueError as e:
        print(f"Error reading Excel sheets: {e}")
        return

    # Check for Parameters column
    if 'Parameters' not in df.columns:
        print("Error: 'Parameters' column is missing. Cannot extract timestamp.")
        return

    # 2. Extract & Sort
    print("Extracting timestamps from JSON Parameters...")
    tqdm.pandas()
    
    # Create temp column for sorting
    df['Temp_Timestamp'] = df['Parameters'].progress_apply(get_time_from_json)
    
    # Check health
    valid_count = df['Temp_Timestamp'].notna().sum()
    print(f"Parsed {valid_count} timestamps out of {len(df)} logs.")
    
    print("Sorting chronologically...")
    df_sorted = df.sort_values(by='Temp_Timestamp', ascending=True)
    
    # Drop the temp column
    df_sorted = df_sorted.drop(columns=['Temp_Timestamp'])

    # 3. Save Output
    file_dir = os.path.dirname(input_file)
    file_name = os.path.basename(input_file)
    name_stem = os.path.splitext(file_name)[0]
    
    output_file = os.path.join(file_dir, f"{name_stem}_sorted.xlsx")
    
    print(f"Saving sorted file to: {output_file}...")
    
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        df_sorted.to_excel(writer, sheet_name='Log Analysis', index=False)
        df_templates.to_excel(writer, sheet_name='Template Summary', index=False)
        
    print("-" * 50)
    print("DONE! Your logs are now ordered by time.")
    print(f"Output File: {output_file}")

if __name__ == "__main__":
    main()