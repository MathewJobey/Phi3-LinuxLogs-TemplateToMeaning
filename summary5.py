import pandas as pd
import os
import json
import numpy as np
from dateutil import parser
from datetime import timedelta
from sentence_transformers import SentenceTransformer, util

# ==========================================
# CONFIGURATION
# ==========================================
# 1. TIME SETTINGS
BURST_GAP_SECONDS = 60          # Max gap between logs to be same 'burst'
INCIDENT_WINDOW_SECONDS = 300   # Max overlap window to consider merging bursts

# 2. SEMANTIC SETTINGS
# 0.5 = Loose relation, 0.7 = Strong relation. 
# If two bursts overlap in time BUT similarity is < 0.4, they remain separate incidents.
SEMANTIC_THRESHOLD = 0.4        

# 3. MODEL (Downloads automatically, ~80MB)
MODEL_NAME = 'all-MiniLM-L6-v2' 

CRITICAL_KEYWORDS = ["failed", "error", "refused", "panic", "shut down", "critical", "denied"]
IGNORE_PARAMS = ["TIMESTAMP", "PID", "UID", "month", "day", "time"]

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

def extract_details(rows):
    """Aggregates unique IPs/Users from a burst"""
    agg = {}
    for p_str in rows['Parameters']:
        try:
            p = json.loads(str(p_str))
            for k, v in p.items():
                if k in IGNORE_PARAMS: continue
                agg.setdefault(k, set()).add(str(v))
        except: continue
    
    parts = []
    for k, v_set in agg.items():
        label = "Source IPs" if k == "RHOST" else "Users" if k in ["USERNAME", "USER"] else k
        if len(v_set) > 5: parts.append(f"{label}: {len(v_set)} unique")
        else: parts.append(f"{label}: [{', '.join(sorted(list(v_set)))}]")
    return " | ".join(parts)

def main():
    # 1. LOAD DATA
    input_file = prompt_for_file()
    if not os.path.exists(input_file): return print("File not found.")
    
    print(f"Reading {input_file}...")
    df = pd.read_excel(input_file, sheet_name="Log Analysis")
    df['Real_Time'] = df['Parameters'].apply(get_time)
    df = df.dropna(subset=['Real_Time'])

    # 2. LOAD BERT MODEL
    print(f"Loading BERT model ({MODEL_NAME})...")
    model = SentenceTransformer(MODEL_NAME)

    # 3. GENERATE EMBEDDINGS FOR TEMPLATES
    print("Embedding unique templates...")
    # We embed the "Event Meaning" string from the Template Summary or Log Analysis
    # Let's get unique ID -> Meaning mapping
    unique_map = df.drop_duplicates('Template ID')[['Template ID', 'Meaning Log']]
    
    template_embeddings = {}
    
    # We assume 'Meaning Log' contains the specific instance, but for embedding
    # we ideally want the generalized template. Since we might not have that handy,
    # using the first instance is usually a very good proxy for semantic meaning.
    for _, row in unique_map.iterrows():
        tid = row['Template ID']
        text = str(row['Meaning Log'])
        # Clean text slightly to remove timestamps for better semantic matching
        # (Heuristic: split by comma, usually timestamp is at start)
        clean_text = text.split(',', 1)[-1] if ',' in text else text
        
        template_embeddings[tid] = model.encode(clean_text, convert_to_tensor=True)

    print(f"Embedded {len(template_embeddings)} unique event types.")

    # 4. PHASE 1: DETECT TEMPORAL BURSTS
    print("Detecting temporal bursts...")
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
                    # Close Burst
                    all_bursts.append({
                        "start": burst_start,
                        "end": burst_end,
                        "tid": t_id,
                        "count": len(burst_rows),
                        "details": extract_details(pd.DataFrame(burst_rows)),
                        "meaning": str(burst_rows[0]['Meaning Log']),
                        "critical": any(k in str(burst_rows[0]['Meaning Log']).lower() for k in CRITICAL_KEYWORDS)
                    })
                    burst_start = rt
                    burst_end = rt
                    burst_rows = [row]
        
        # Save last burst
        if burst_rows:
            all_bursts.append({
                "start": burst_start,
                "end": burst_end,
                "tid": t_id,
                "count": len(burst_rows),
                "details": extract_details(pd.DataFrame(burst_rows)),
                "meaning": str(burst_rows[0]['Meaning Log']),
                "critical": any(k in str(burst_rows[0]['Meaning Log']).lower() for k in CRITICAL_KEYWORDS)
            })

    # 5. PHASE 2: SEMANTIC INCIDENT CORRELATION
    print("Correlating bursts using BERT Similarity + Time...")
    all_bursts.sort(key=lambda x: x['start'])
    
    incidents = []
    if all_bursts:
        current_inc = {
            "start": all_bursts[0]['start'],
            "end": all_bursts[0]['end'],
            "bursts": [all_bursts[0]]
        }
        
        for i in range(1, len(all_bursts)):
            burst = all_bursts[i]
            
            # 1. TEMPORAL CHECK
            # Does this burst start within the window of the current incident?
            window_limit = current_inc['end'] + timedelta(seconds=INCIDENT_WINDOW_SECONDS)
            is_time_correlated = burst['start'] <= window_limit
            
            # 2. SEMANTIC CHECK
            # We check similarity against *any* burst currently in the incident
            is_semantic_correlated = False
            if is_time_correlated:
                # Get embedding of current burst
                burst_emb = template_embeddings[burst['tid']]
                
                # Compare with all distinct templates in the current active incident
                for inc_burst in current_inc['bursts']:
                    inc_emb = template_embeddings[inc_burst['tid']]
                    
                    # Calculate Cosine Similarity
                    score = util.cos_sim(burst_emb, inc_emb).item()
                    
                    if score >= SEMANTIC_THRESHOLD:
                        is_semantic_correlated = True
                        break # Found a match, no need to check others

            # 3. DECISION
            if is_time_correlated and is_semantic_correlated:
                # MERGE
                current_inc['bursts'].append(burst)
                if burst['end'] > current_inc['end']:
                    current_inc['end'] = burst['end']
            elif is_time_correlated and not is_semantic_correlated:
                # OVERLAP BUT NOT RELATED -> NEW INCIDENT (or keep separate)
                # For this logic, we close the current and start new, 
                # effectively separating unrelated interleaved events.
                incidents.append(current_inc)
                current_inc = { "start": burst['start'], "end": burst['end'], "bursts": [burst] }
            else:
                # NO TIME OVERLAP -> NEW INCIDENT
                incidents.append(current_inc)
                current_inc = { "start": burst['start'], "end": burst['end'], "bursts": [burst] }
        
        incidents.append(current_inc)

    # 6. OUTPUT
    print(f"Refined {len(all_bursts)} bursts into {len(incidents)} Semantic Incidents.")
    
    output_data = []
    for idx, inc in enumerate(incidents):
        output_data.append({
            "incident_id": idx + 1,
            "start": str(inc['start']),
            "end": str(inc['end']),
            "duration": str(inc['end'] - inc['start']),
            "event_count": sum(b['count'] for b in inc['bursts']),
            "events": [{
                "time": str(b['start']),
                "name": b['meaning'], 
                "details": b['details']
            } for b in inc['bursts']]
        })
        
    with open("semantic_incidents.json", "w") as f:
        json.dump(output_data, f, indent=4)
        
    # Narrative Prompt
    with open("semantic_narrative_prompt.txt", "w") as f:
        f.write(f"SEMANTIC INCIDENT REPORT ({len(incidents)} Groups)\n")
        f.write("Note: Events are grouped by Time AND Meaning (BERT Embedding).\n\n")
        for inc in output_data:
            f.write(f"INCIDENT #{inc['incident_id']} ({inc['start']} - {inc['end']})\n")
            for e in inc['events']:
                f.write(f"  - {e['name']} \n    ({e['details']})\n")
            f.write("\n")

    print("Done. Saved 'semantic_incidents.json' and 'semantic_narrative_prompt.txt'")

if __name__ == "__main__":
    main()