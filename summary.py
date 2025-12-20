import pandas as pd
import os
import json
from dateutil import parser
from datetime import timedelta
from sentence_transformers import SentenceTransformer, util

# ==========================================
# CONFIGURATION
# ==========================================
BURST_GAP_SECONDS = 60          # Max silence before a burst breaks
INCIDENT_WINDOW_SECONDS = 300   # Window to link different bursts
SEMANTIC_THRESHOLD = 0.4        # BERT similarity threshold (0.0 - 1.0)
MODEL_NAME = 'all-MiniLM-L6-v2' 

CRITICAL_KEYWORDS = ["failed", "error", "refused", "panic", "shut down", "critical", "denied"]
IGNORE_PARAMS = ["TIMESTAMP", "PID", "UID", "month", "day", "time", "date"]

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

def extract_raw_params(rows):
    """
    Returns a dictionary of sets: {'RHOST': {'1.2.3.4'}, 'USER': {'root'}}
    This allows us to merge details easily later.
    """
    agg = {}
    for p_str in rows['Parameters']:
        try:
            p = json.loads(str(p_str))
            for k, v in p.items():
                if k in IGNORE_PARAMS: continue
                # Normalize keys
                label = "Source IPs" if k == "RHOST" else "Users" if k in ["USERNAME", "USER", "logname"] else k
                
                agg.setdefault(label, set()).add(str(v))
        except: continue
    return agg

def format_details(param_dict):
    """Converts the set dictionary into a readable string"""
    parts = []
    for k, v_set in param_dict.items():
        if len(v_set) > 10: 
            parts.append(f"{k}: {len(v_set)} unique values")
        else: 
            # Filter out empty strings
            valid_vals = [x for x in v_set if x and x != "nan"]
            if valid_vals:
                parts.append(f"{k}: [{', '.join(sorted(valid_vals))}]")
    return " | ".join(parts)

def main():
    # 1. LOAD DATA
    input_file = prompt_for_file()
    if not os.path.exists(input_file): return print("File not found.")
    
    print(f"Reading {input_file}...")
    df = pd.read_excel(input_file, sheet_name="Log Analysis")
    df['Real_Time'] = df['Parameters'].apply(get_time)
    df = df.dropna(subset=['Real_Time'])

    # 2. LOAD BERT
    print("Loading AI Model...")
    model = SentenceTransformer(MODEL_NAME)

    # 3. EMBED TEMPLATES
    print("Embedding Templates...")
    unique_map = df.drop_duplicates('Template ID')[['Template ID', 'Meaning Log']]
    template_embeddings = {}
    for _, row in unique_map.iterrows():
        tid = row['Template ID']
        text = str(row['Meaning Log']).split(',', 1)[-1] # Remove timestamp
        template_embeddings[tid] = model.encode(text, convert_to_tensor=True)

    # 4. DETECT BURSTS (Phase 1)
    print("Detecting Bursts...")
    all_bursts = []
    
    for t_id in df['Template ID'].unique():
        t_logs = df[df['Template ID'] == t_id].sort_values('Real_Time')
        
        burst_rows = []
        burst_start = None
        burst_end = None
        
        for _, row in t_logs.iterrows():
            rt = row['Real_Time']
            if burst_start is None:
                burst_start = rt
                burst_end = rt
                burst_rows.append(row)
            else:
                if (rt - burst_end).total_seconds() <= BURST_GAP_SECONDS:
                    burst_end = rt
                    burst_rows.append(row)
                else:
                    # Capture Burst
                    b_df = pd.DataFrame(burst_rows)
                    all_bursts.append({
                        "start": burst_start,
                        "end": burst_end,
                        "tid": t_id,
                        "count": len(burst_rows),
                        "raw_params": extract_raw_params(b_df), # Keep raw sets
                        "meaning": str(burst_rows[0]['Meaning Log']),
                        "critical": any(k in str(burst_rows[0]['Meaning Log']).lower() for k in CRITICAL_KEYWORDS)
                    })
                    burst_start = rt
                    burst_end = rt
                    burst_rows = [row]
        
        if burst_rows:
            b_df = pd.DataFrame(burst_rows)
            all_bursts.append({
                "start": burst_start,
                "end": burst_end,
                "tid": t_id,
                "count": len(burst_rows),
                "raw_params": extract_raw_params(b_df),
                "meaning": str(burst_rows[0]['Meaning Log']),
                "critical": any(k in str(burst_rows[0]['Meaning Log']).lower() for k in CRITICAL_KEYWORDS)
            })

    # 5. SEMANTIC INCIDENT CORRELATION (Phase 2)
    print("Correlating Incidents...")
    all_bursts.sort(key=lambda x: x['start'])
    
    incidents = []
    if all_bursts:
        current_inc = { "start": all_bursts[0]['start'], "end": all_bursts[0]['end'], "bursts": [all_bursts[0]] }
        
        for i in range(1, len(all_bursts)):
            burst = all_bursts[i]
            window_limit = current_inc['end'] + timedelta(seconds=INCIDENT_WINDOW_SECONDS)
            is_time_correlated = burst['start'] <= window_limit
            
            is_semantic = False
            if is_time_correlated:
                b_emb = template_embeddings[burst['tid']]
                for ib in current_inc['bursts']:
                    if util.cos_sim(b_emb, template_embeddings[ib['tid']]).item() >= SEMANTIC_THRESHOLD:
                        is_semantic = True
                        break

            if is_time_correlated and is_semantic:
                current_inc['bursts'].append(burst)
                if burst['end'] > current_inc['end']: current_inc['end'] = burst['end']
            elif is_time_correlated and not is_semantic:
                incidents.append(current_inc)
                current_inc = { "start": burst['start'], "end": burst['end'], "bursts": [burst] }
            else:
                incidents.append(current_inc)
                current_inc = { "start": burst['start'], "end": burst['end'], "bursts": [burst] }
        incidents.append(current_inc)

    # 6. AGGREGATION & REPORTING (Phase 3 - NEW)
    print(f"Generating Compact Report for {len(incidents)} incidents...")
    
    final_output = []
    
    for idx, inc in enumerate(incidents):
        # --- AGGREGATION LOGIC ---
        # We group bursts by Template ID to summarize them ONCE per incident
        template_summary = {}
        
        for b in inc['bursts']:
            tid = b['tid']
            if tid not in template_summary:
                template_summary[tid] = {
                    "event_name": b['meaning'],
                    "total_count": 0,
                    "burst_count": 0,
                    "merged_params": {}, # Dictionary of sets
                    "start": b['start'],
                    "end": b['end']
                }
            
            # Update counts
            entry = template_summary[tid]
            entry['total_count'] += b['count']
            entry['burst_count'] += 1
            if b['end'] > entry['end']: entry['end'] = b['end'] # Extend time
            
            # Merge Parameters (Sets)
            for k, v_set in b['raw_params'].items():
                if k not in entry['merged_params']: entry['merged_params'][k] = set()
                entry['merged_params'][k].update(v_set)

        # Convert Aggregated Data to List
        events_list = []
        for tid, data in template_summary.items():
            events_list.append({
                "template": data['event_name'],
                "stats": f"{data['total_count']} logs across {data['burst_count']} bursts",
                "duration": f"{data['start'].strftime('%H:%M:%S')} - {data['end'].strftime('%H:%M:%S')}",
                "details": format_details(data['merged_params'])
            })
            
        final_output.append({
            "incident_id": idx + 1,
            "start": str(inc['start']),
            "end": str(inc['end']),
            "total_logs": sum(e['total_count'] for e in template_summary.values()),
            "events": events_list
        })

    # 7. SAVE OUTPUTS
    with open("compact_incidents.json", "w") as f:
        json.dump(final_output, f, indent=4)
        
    with open("compact_narrative_prompt.txt", "w") as f:
        f.write(f"SECURITY INCIDENT REPORT ({len(incidents)} Incidents)\n")
        f.write("===================================================\n\n")
        for inc in final_output:
            f.write(f"INCIDENT #{inc['incident_id']} ({inc['start']} to {inc['end']})\n")
            f.write(f"Total Volume: {inc['total_logs']} logs.\n")
            f.write("Distinct Event Types:\n")
            for e in inc['events']:
                f.write(f"  > {e['template']}\n")
                f.write(f"    - Frequency: {e['stats']}\n")
                f.write(f"    - Timeline:  {e['duration']}\n")
                if e['details']:
                    f.write(f"    - Key Params: {e['details']}\n")
            f.write("\n---------------------------------------------------\n")

    print("Success! Created 'compact_incidents.json' and 'compact_narrative_prompt.txt'")

if __name__ == "__main__":
    main()