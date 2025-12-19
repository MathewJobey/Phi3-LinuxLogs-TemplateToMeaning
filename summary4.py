import pandas as pd
import os
import json
from dateutil import parser

# ==========================================
# CONFIGURATION
# ==========================================
CRITICAL_KEYWORDS = ["failed", "error", "refused", "panic", "shut down", "critical", "denied"]

# We ignore these parameters because they change every second and create noise
IGNORE_PARAMS = ["TIMESTAMP", "PID", "UID", "month", "day", "time"] 

def get_unique_filename(base_name):
    if not os.path.exists(base_name): return base_name
    name, ext = os.path.splitext(base_name)
    counter = 1
    while True:
        new_name = f"{name}_{counter}{ext}"
        if not os.path.exists(new_name): return new_name
        counter += 1

def prompt_for_file():
    files = [f for f in os.listdir('.') if f.endswith('_sorted.xlsx')]
    if len(files) >= 1:
        print(f"Auto-detected sorted file: {files[0]}")
        return files[0]
    return input("Paste the FULL path to your '_sorted.xlsx' file: ").strip().strip('"')

def get_time_from_json(params_str):
    try:
        if pd.isna(params_str): return None
        params = json.loads(str(params_str))
        time_str = params.get('TIMESTAMP')
        return parser.parse(time_str) if time_str else None
    except:
        return None

def extract_unique_params(group_df):
    """
    Scans the 'Parameters' column for ALL rows in the group.
    Returns a string like: "Users: [root, admin], RHOST: [192.168.1.1]"
    """
    aggregated = {}
    
    for params_str in group_df['Parameters']:
        try:
            if pd.isna(params_str): continue
            p_dict = json.loads(str(params_str))
            
            for key, val in p_dict.items():
                if key in IGNORE_PARAMS: continue # Skip PIDs and Timestamps
                
                if key not in aggregated:
                    aggregated[key] = set()
                aggregated[key].add(str(val))
        except:
            continue
            
    # Convert sets to nice strings
    result_parts = []
    for key, val_set in aggregated.items():
        # specific fix for your case: meaningful names
        label = key
        if key == "RHOST": label = "Source IPs"
        if key == "USERNAME" or key == "USER": label = "Users"
        
        # If there are too many unique values (like 100 different ports), just show count
        if len(val_set) > 5:
            result_parts.append(f"{label}: {len(val_set)} unique values")
        else:
            clean_vals = ", ".join(sorted(list(val_set)))
            result_parts.append(f"{label}: [{clean_vals}]")
            
    return " | ".join(result_parts)

def main():
    # 1. SETUP
    input_file = prompt_for_file()
    if not os.path.exists(input_file):
        print("Error: File not found.")
        return

    print(f"Reading {input_file}...")
    df = pd.read_excel(input_file, sheet_name="Log Analysis")

    # 2. PARSE TIMES
    print("Aggregating details...")
    df['Real_Time'] = df['Parameters'].apply(get_time_from_json)

    # 3. GROUP & AGGREGATE
    unique_ids_ordered = df['Template ID'].unique()
    timeline_events = []
    total_critical = 0
    
    for t_id in unique_ids_ordered:
        group = df[df['Template ID'] == t_id]
        
        start_time = group['Real_Time'].iloc[0]
        end_time = group['Real_Time'].iloc[-1]
        
        duration_str = "Instant"
        if start_time and end_time and start_time != end_time:
            duration_str = str(end_time - start_time)

        # Grab the "Meaning" but we will enhance it
        base_desc = str(group.iloc[0]['Meaning Log'])
        
        # --- THE NEW MAGIC: Extract details from ALL rows in this group ---
        details_str = extract_unique_params(group)
        
        # Combine them: "Authentication Failed (Users: [root], IPs: [1.2.3.4])"
        full_description = base_desc
        if details_str:
            full_description = f"{base_desc} \n   >>> DETAILS: {details_str}"

        is_critical = any(k in base_desc.lower() for k in CRITICAL_KEYWORDS)
        if is_critical: total_critical += len(group)
        
        event_obj = {
            "event_id": int(t_id),
            "status": "CRITICAL" if is_critical else "ROUTINE",
            "count": len(group),
            "start_time": str(start_time),
            "end_time": str(end_time),
            "duration": duration_str,
            "description": full_description, # Now contains the aggregate data
            "unique_details": details_str    # Also kept separate for JSON
        }
        timeline_events.append(event_obj)

    # 4. SAVE OUTPUTS
    master_report = {
        "report_metadata": {
            "source_file": input_file,
            "total_logs": len(df),
            "critical_events": total_critical
        },
        "event_timeline": timeline_events
    }

    # Save JSON
    json_filename = get_unique_filename("system_report_detailed.json")
    with open(json_filename, "w", encoding="utf-8") as f:
        json.dump(master_report, f, indent=4)

    # Save Text Prompt
    ai_prompt_text = "CHRONOLOGICAL SECURITY REPORT:\n"
    for event in timeline_events:
        ai_prompt_text += f"- [{event['start_time']}] {event['status']} (Count: {event['count']})\n"
        ai_prompt_text += f"  Event: {event['description']}\n"
    
    prompt_filename = get_unique_filename("narrative_prompt_detailed.txt")
    with open(prompt_filename, "w", encoding="utf-8") as f:
        f.write(ai_prompt_text)

    print("-" * 60)
    print("SUCCESS! Details extracted.")
    print(f"Check '{prompt_filename}' - it now lists Users and IPs for every event.")
    print("-" * 60)

if __name__ == "__main__":
    main()