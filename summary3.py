import pandas as pd
import os
import json
from dateutil import parser

# ==========================================
# CONFIGURATION
# ==========================================
CRITICAL_KEYWORDS = ["failed", "error", "refused", "panic", "shut down", "critical", "denied"]

def get_unique_filename(base_name):
    """
    Checks if 'file.txt' exists. If so, tries 'file_1.txt', 'file_2.txt', etc.
    """
    if not os.path.exists(base_name):
        return base_name
    
    name, ext = os.path.splitext(base_name)
    counter = 1
    
    while True:
        new_name = f"{name}_{counter}{ext}"
        if not os.path.exists(new_name):
            return new_name
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

def main():
    # 1. LOAD SORTED FILE
    input_file = prompt_for_file()
    if not os.path.exists(input_file):
        print("Error: File not found.")
        return

    print(f"Reading {input_file}...")
    try:
        df = pd.read_excel(input_file, sheet_name="Log Analysis")
    except Exception as e:
        print(f"Error: {e}")
        return

    # 2. PARSE TIMES
    print("Calculating timeline durations...")
    df['Real_Time'] = df['Parameters'].apply(get_time_from_json)

    # 3. GROUP EVENTS
    unique_ids_ordered = df['Template ID'].unique()
    timeline_events = []
    total_critical = 0
    
    print(f"Processing {len(unique_ids_ordered)} unique event sequences...")
    
    for t_id in unique_ids_ordered:
        group = df[df['Template ID'] == t_id]
        
        start_time = group['Real_Time'].iloc[0]
        end_time = group['Real_Time'].iloc[-1]
        
        duration_str = "Instant"
        if start_time and end_time and start_time != end_time:
            duration_str = str(end_time - start_time)

        example = str(group.iloc[0]['Meaning Log'])
        is_critical = any(k in example.lower() for k in CRITICAL_KEYWORDS)
        if is_critical: total_critical += len(group)
        
        event_obj = {
            "event_id": int(t_id),
            "status": "CRITICAL" if is_critical else "ROUTINE",
            "count": len(group),
            "start_time": str(start_time),
            "end_time": str(end_time),
            "duration": duration_str,
            "description": example
        }
        timeline_events.append(event_obj)

    # 4. PREPARE OUTPUT DATA
    master_report = {
        "report_metadata": {
            "source_file": input_file,
            "total_logs_processed": len(df),
            "total_critical_events": total_critical,
            "timeline_start": str(df['Real_Time'].min()),
            "timeline_end": str(df['Real_Time'].max())
        },
        "event_timeline": timeline_events
    }

    # 5. SAVE WITH AUTO-INCREMENTING NAMES
    json_filename = get_unique_filename("system_report.json")
    with open(json_filename, "w", encoding="utf-8") as f:
        json.dump(master_report, f, indent=4)

    # 6. GENERATE TEXT PROMPT
    ai_prompt_text = f"""SYSTEM LOG TIMELINE SUMMARY
Total Logs: {master_report['report_metadata']['total_logs_processed']}
Criticals: {master_report['report_metadata']['total_critical_events']}
Time Range: {master_report['report_metadata']['timeline_start']} to {master_report['report_metadata']['timeline_end']}

CHRONOLOGICAL EVENT SEQUENCE:
"""
    for event in timeline_events:
        ai_prompt_text += f"- [{event['start_time']}] {event['status']}: {event['description']} (Repeated {event['count']} times over {event['duration']})\n"
    
    ai_prompt_text += "\nINSTRUCTION: Write a professional executive summary based on this timeline."

    prompt_filename = get_unique_filename("narrative_prompt.txt")
    with open(prompt_filename, "w", encoding="utf-8") as f:
        f.write(ai_prompt_text)

    print("-" * 60)
    print("SUCCESS! Output files created:")
    print(f"1. {json_filename}")
    print(f"2. {prompt_filename}")
    print("-" * 60)

if __name__ == "__main__":
    main()