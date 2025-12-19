import pandas as pd
import json
import os
from collections import Counter

# ==========================================
# CONFIGURATION: Define what is "Critical"
# ==========================================
CRITICAL_KEYWORDS = [
    "failed", "failure", "error", "refused", "unauthorized", 
    "panic", "shut down", "critical", "denied", "violation", 
    "invalid", "attack", "bad"
]

def prompt_for_file():
    path = input("Paste the FULL path to your 'final excel' file: ").strip()
    return path.strip('"').strip("'")

def extract_timestamp(row):
    """
    Tries to grab the TIMESTAMP from the Parameters JSON.
    Returns "Unknown Time" if missing.
    """
    try:
        params = row.get('Parameters')
        if pd.isna(params): return "Unknown Time"
        
        p_dict = json.loads(str(params))
        return p_dict.get('TIMESTAMP', "Unknown Time")
    except:
        return "Unknown Time"

def is_critical(text):
    """Checks if the log meaning contains any bad words."""
    if not isinstance(text, str): return False
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in CRITICAL_KEYWORDS)

def main():
    # 1. Load File
    input_file = prompt_for_file()
    if not os.path.exists(input_file):
        print("Error: File not found.")
        return

    print(f"Reading {input_file}...")
    df = pd.read_excel(input_file, sheet_name="Log Analysis")
    
    # Check if 'Meaning Log' exists
    if 'Meaning Log' not in df.columns:
        print("Error: 'Meaning Log' column missing. Did you run the previous script?")
        return

    # 2. Process Data
    total_logs = len(df)
    critical_events = []
    activity_counter = Counter()

    print("Analyzing events...")
    
    for index, row in df.iterrows():
        meaning = str(row['Meaning Log'])
        timestamp = extract_timestamp(row)
        
        # Count for General Summary
        # We limit the meaning length to avoid noise in grouping
        activity_counter[meaning] += 1
        
        # Check Critical
        if is_critical(meaning):
            critical_events.append({
                "line": index + 2, # Excel Row Number
                "time": timestamp,
                "event": meaning
            })

    # 3. Generate Statistics
    top_activities = activity_counter.most_common(10)
    critical_count = len(critical_events)
    
    # 4. Construct Structured Summary (JSON)
    summary_data = {
        "report_overview": {
            "total_logs_analyzed": total_logs,
            "critical_event_count": critical_count,
            "health_status": "Review Required" if critical_count > 0 else "Healthy"
        },
        "critical_incidents": critical_events,  # Detailed list
        "top_recurring_events": [
            {"event": event, "count": count} for event, count in top_activities
        ]
    }

    # 5. Save JSON Output
    base_dir = os.path.dirname(input_file)
    json_path = os.path.join(base_dir, "Log_Summary_Report.json")
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=4)

    # 6. Generate & Print Narrative Paragraph
    print("\n" + "="*60)
    print("EXECUTIVE SUMMARY")
    print("="*60)
    
    paragraph = (
        f"ANALYSIS COMPLETE: The system analyzed {total_logs} log entries. "
        f"A total of {critical_count} critical incidents were detected that require attention. "
        f"The most frequent system activity was '{top_activities[0][0]}' which occurred {top_activities[0][1]} times. "
        f"\n\nCRITICAL HIGHLIGHTS:\n"
    )
    
    # Add first 3 critical events to paragraph as examples
    if critical_events:
        for i, event in enumerate(critical_events[:3]):
            paragraph += f"- At {event['time']}: {event['event']}\n"
        if len(critical_events) > 3:
            paragraph += f"...and {len(critical_events) - 3} more (see JSON report)."
    else:
        paragraph += "No critical errors found."

    print(paragraph)
    print("="*60)
    print(f"Full structured report saved to:\n{json_path}")

if __name__ == "__main__":
    main()