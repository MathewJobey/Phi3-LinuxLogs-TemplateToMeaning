import pandas as pd
import os
import json
from dateutil import parser

# ==========================================
# CONFIGURATION
# ==========================================
# Words that flag an event as "CRITICAL" in the report
CRITICAL_KEYWORDS = ["failed", "error", "refused", "panic", "shut down", "critical", "denied"]

def prompt_for_file():
    # Auto-detects the sorted file if it exists
    files = [f for f in os.listdir('.') if f.endswith('_sorted.xlsx')]
    if len(files) >= 1:
        print(f"Auto-detected sorted file: {files[0]}")
        return files[0]
    return input("Paste the FULL path to your '_sorted.xlsx' file: ").strip().strip('"')

def get_time_from_json(params_str):
    """Safely extracts datetime from JSON string in the 'Parameters' column"""
    try:
        if pd.isna(params_str): return None
        params = json.loads(str(params_str))
        time_str = params.get('TIMESTAMP')
        return parser.parse(time_str) if time_str else None
    except:
        return None

def main():
    # 1. LOAD THE SORTED FILE
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

    # 2. PARSE TIMES (Required for duration calculations)
    print("Calculating timeline durations...")
    # We use the JSON 'Parameters' column for 100% accuracy
    df['Real_Time'] = df['Parameters'].apply(get_time_from_json)

    # 3. GROUP EVENTS (Preserving Chronological Order)
    # Since the file is already sorted, .unique() returns IDs in the order they first appeared
    unique_ids_ordered = df['Template ID'].unique()
    
    timeline_events = []
    total_critical = 0
    
    print(f"Processing {len(unique_ids_ordered)} unique event sequences...")
    
    for t_id in unique_ids_ordered:
        # Get all logs for this specific template
        group = df[df['Template ID'] == t_id]
        
        # Calculate Start/End/Duration
        start_time = group['Real_Time'].iloc[0]
        end_time = group['Real_Time'].iloc[-1]
        
        duration_str = "Instant"
        if start_time and end_time and start_time != end_time:
            duration_str = str(end_time - start_time)

        # Get the human-readable meaning
        example = str(group.iloc[0]['Meaning Log'])
        
        # Check if critical
        is_critical = any(k in example.lower() for k in CRITICAL_KEYWORDS)
        if is_critical: total_critical += len(group)
        
        # Build the Event Object
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

    # 4. CREATE FINAL JSON STRUCTURE
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

    # 5. SAVE JSON FILE
    json_filename = "system_report.json"
    with open(json_filename, "w", encoding="utf-8") as f:
        json.dump(master_report, f, indent=4)

    # 6. GENERATE TEXT PROMPT (For AI usage)
    # Converts the JSON into a clean text list for LLMs
    ai_prompt_text = f"""SYSTEM LOG TIMELINE SUMMARY
Total Logs: {master_report['report_metadata']['total_logs_processed']}
Criticals: {master_report['report_metadata']['total_critical_events']}
Time Range: {master_report['report_metadata']['timeline_start']} to {master_report['report_metadata']['timeline_end']}

CHRONOLOGICAL EVENT SEQUENCE:
"""
    for event in timeline_events:
        ai_prompt_text += f"- [{event['start_time']}] {event['status']}: {event['description']} (Repeated {event['count']} times over {event['duration']})\n"
    
    ai_prompt_text += "\nINSTRUCTION: Write a professional executive summary based on this timeline, highlighting the sequence of attacks or failures."

    with open("narrative_prompt.txt", "w", encoding="utf-8") as f:
        f.write(ai_prompt_text)

    print("-" * 60)
    print("SUCCESS! Generated 2 output files:")
    print(f"1. {json_filename} (Structured data for dashboards/APIs)")
    print("2. narrative_prompt.txt (Text summary to paste into Phi-3 or ChatGPT)")
    print("-" * 60)

if __name__ == "__main__":
    main()