import pandas as pd
import os
import json
from dateutil import parser

# ==========================================
# CONFIGURATION
# ==========================================
# If an event stops for this many seconds, the burst is over.
BURST_GAP_SECONDS = 1000

# Parameters to ignore in the detailed view
IGNORE_PARAMS = ["TIMESTAMP", "PID", "UID", "month", "day", "time", "date"]
CRITICAL_KEYWORDS = ["failed", "error", "refused", "panic", "shut down", "critical", "denied"]

def prompt_for_file():
    files = [f for f in os.listdir('.') if f.endswith('_sorted.xlsx')]
    if len(files) >= 1:
        print(f"Auto-detected sorted file: {files[0]}")
        return files[0]
    return input("Paste path to '_sorted.xlsx' file: ").strip().strip('"')

def get_time(params_str):
    try:
        p = json.loads(str(params_str))
        return parser.parse(p.get('TIMESTAMP'))
    except: return None

def extract_params(rows):
    """
    Harvests unique values (Usernames, IPs) from a group of logs.
    Returns a clean string summary.
    """
    agg = {}
    for p_str in rows['Parameters']:
        try:
            p = json.loads(str(p_str))
            for k, v in p.items():
                if k in IGNORE_PARAMS: continue
                
                # Make labels human-readable
                label = k
                if k == "RHOST": label = "Source IPs"
                if k in ["USERNAME", "USER", "logname"]: label = "Users"
                
                agg.setdefault(label, set()).add(str(v))
        except: continue
        
    parts = []
    for k, v_set in agg.items():
        # Clean up sets (remove 'nan')
        valid = sorted([x for x in v_set if x and str(x).lower() != 'nan'])
        if not valid: continue
        
        if len(valid) > 10:
            parts.append(f"{k}: {len(valid)} unique values")
        else:
            parts.append(f"{k}: {valid}")
            
    return " | ".join(parts)

def main():
    # 1. LOAD
    input_file = prompt_for_file()
    if not os.path.exists(input_file):
        print("Error: File not found.")
        return

    print(f"Reading {input_file}...")
    df = pd.read_excel(input_file, sheet_name="Log Analysis")
    df['Real_Time'] = df['Parameters'].apply(get_time)
    df = df.dropna(subset=['Real_Time'])

    # 2. DETECT BURSTS
    # We process one Template ID at a time to find its bursts
    print(f"Detecting bursts (Gap Threshold: {BURST_GAP_SECONDS}s)...")
    
    all_bursts = []
    
    for t_id in df['Template ID'].unique():
        # Get logs for this template, sorted by time
        t_logs = df[df['Template ID'] == t_id].sort_values('Real_Time')
        
        current_burst = []
        last_time = None
        
        for _, row in t_logs.iterrows():
            curr_time = row['Real_Time']
            
            if last_time is None:
                current_burst.append(row)
            else:
                gap = (curr_time - last_time).total_seconds()
                if gap <= BURST_GAP_SECONDS:
                    current_burst.append(row)
                else:
                    # BURST BROKEN - SAVE IT
                    all_bursts.append(create_burst_record(t_id, current_burst))
                    current_burst = [row] # Start new one
            
            last_time = curr_time
        
        # Save the final one
        if current_burst:
            all_bursts.append(create_burst_record(t_id, current_burst))

    # 3. SORT & REPORT
    # Sort all bursts chronologically by their start time
    all_bursts.sort(key=lambda x: x['start_dt'])
    
    print(f"Found {len(all_bursts)} distinct bursts.")
    
    # Save Text Report
    with open("burst_report.txt", "w", encoding="utf-8") as f:
        f.write(f"BURST ANALYSIS REPORT (Gap: {BURST_GAP_SECONDS}s)\n")
        f.write("==================================================\n")
        
        for b in all_bursts:
            f.write(f"[{b['start_str']} to {b['end_str']}] {b['status']}\n")
            f.write(f"  Template: {b['meaning']}\n")
            f.write(f"  Volume:   {b['count']} logs (Duration: {b['duration']})\n")
            if b['details']:
                f.write(f"  Context:  {b['details']}\n")
            f.write("-" * 50 + "\n")

    print("Done! Created 'burst_report.txt'.")

def create_burst_record(t_id, rows):
    start_dt = rows[0]['Real_Time']
    end_dt = rows[-1]['Real_Time']
    
    # Check Criticality
    meaning = str(rows[0]['Meaning Log'])
    is_critical = any(k in meaning.lower() for k in CRITICAL_KEYWORDS)
    
    # Aggregate Params
    df_chunk = pd.DataFrame(rows)
    details = extract_params(df_chunk)
    
    return {
        "start_dt": start_dt,
        "start_str": start_dt.strftime("%H:%M:%S"),
        "end_str": end_dt.strftime("%H:%M:%S"),
        "duration": str(end_dt - start_dt),
        "count": len(rows),
        "meaning": meaning,
        "details": details,
        "status": "CRITICAL" if is_critical else "INFO"
    }

if __name__ == "__main__":
    main()