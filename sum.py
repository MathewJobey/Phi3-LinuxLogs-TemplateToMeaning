import pandas as pd
import requests
import json
import os
import ast
from tqdm import tqdm

# ==========================================
# CONFIGURATION
# ==========================================
OLLAMA_API = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3"  # or "mistral", "phi3"

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def prompt_for_file():
    files = [f for f in os.listdir('.') if f.endswith('_sorted.xlsx')]
    if len(files) >= 1:
        print(f"Auto-detected file: {files[0]}")
        return files[0]
    return input("Paste path to '_sorted.xlsx': ").strip().strip('"')

def query_ollama(prompt):
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "format": "json" # Force valid JSON output
    }
    try:
        response = requests.post(OLLAMA_API, json=payload)
        response.raise_for_status()
        return response.json()['response']
    except Exception as e:
        print(f"Error calling Ollama: {e}")
        return None

def extract_parameters(group_df):
    """
    Extracts unique Users, IPs, etc. from a group of logs so the summary is specific.
    """
    aggregated = {}
    
    for params_str in group_df['Parameters']:
        try:
            if pd.isna(params_str): continue
            # Convert string representation of dict to real dict
            if isinstance(params_str, str):
                p_dict = json.loads(params_str)
            else:
                p_dict = params_str

            for key, val in p_dict.items():
                if key in ["TIMESTAMP", "PID", "UID", "month", "day", "time"]: continue
                
                label = key
                if key == "RHOST": label = "IP"
                if key in ["USERNAME", "USER", "logname"]: label = "User"
                
                aggregated.setdefault(label, set()).add(str(val))
        except:
            continue
            
    # Format as string
    parts = []
    for k, v_set in aggregated.items():
        clean_vals = [x for x in v_set if x and x.lower() != 'nan']
        if not clean_vals: continue
        
        if len(clean_vals) > 5:
            parts.append(f"{k}: {len(clean_vals)} unique values")
        else:
            parts.append(f"{k}: {list(clean_vals)}")
            
    return " | ".join(parts)

# ==========================================
# CORE LOGIC
# ==========================================

def pass_1_filter_noise(df_templates):
    """
    Ask AI: "Here is a list of log types. Tell me which ones are NOISE."
    """
    print(f"\n--- PASS 1: Filtering Noise from {len(df_templates)} Templates ---")
    
    template_text = ""
    for _, row in df_templates.iterrows():
        template_text += f"ID {row['Template ID']}: {row['Event Meaning']}\n"

    prompt = f"""
    You are a System Administrator. I have a list of Log Templates.
    I need to summarize "What happened on the server".
    
    Classify each Template ID into two categories:
    1. "NOISE": Useless, repetitive automated background tasks (e.g., cron jobs, debug info, heartbeat messages).
    2. "KEEP": Everything else (User logins, file transfers, errors, warnings, hardware issues, service restarts).

    Here is the list:
    {template_text}

    Return JSON format only:
    {{
        "noise_ids": [list of integers],
        "keep_ids": [list of integers]
    }}
    """
    
    print("Consulting AI for noise filtering...")
    response = query_ollama(prompt)
    
    try:
        data = json.loads(response)
        keep_ids = data.get("keep_ids", [])
        print(f"AI decided to KEEP {len(keep_ids)} template types (User activity, Errors, etc.).")
        print(f"AI decided to IGNORE {len(data.get('noise_ids', []))} template types (Noise).")
        return keep_ids
    except:
        print("AI JSON Error. Defaulting to keeping ALL logs.")
        return df_templates['Template ID'].tolist()

def pass_2_generate_summary(timeline_data):
    """
    Ask AI: "Here is the timeline of relevant events. Write the story."
    """
    print(f"\n--- PASS 2: Generating Executive Summary ---")
    
    prompt = f"""
    You are an Operations Manager. Write a Chronological System Activity Report based on these logs.
    Ignore the raw IDs, focus on the timeline and the story.
    
    INPUT DATA (Chronological Order):
    {timeline_data}
    
    INSTRUCTIONS:
    1. Write a narrative summary: "At 10:00, users X and Y logged in..."
    2. Highlight any errors or anomalies.
    3. Group routine user activity (e.g., "Multiple FTP connections were observed...").
    """
    
    print("Streaming Report from AI...")
    response = query_ollama(prompt)
    return response

def main():
    # 1. Load
    input_file = prompt_for_file()
    if not os.path.exists(input_file): return print("File not found.")
    
    df = pd.read_excel(input_file, sheet_name="Log Analysis")
    df_templates = pd.read_excel(input_file, sheet_name="Template Summary")

    # 2. Filter (Pass 1)
    keep_ids = pass_1_filter_noise(df_templates)
    
    # 3. Aggregate Data (The "Compression" Step)
    # We group by Template ID but maintain chronological order of appearance
    print("Compressing logs for the AI...")
    
    df_filtered = df[df['Template ID'].isin(keep_ids)].copy()
    
    # We create a simple text representation of the timeline
    # We group consecutive events to save space
    timeline_text = ""
    
    # Simple aggregation: Group by Template ID to show volume + context
    # (Note: For a true timeline, we would iterate row by row, but for summary, this is efficient)
    unique_events_in_order = df_filtered['Template ID'].unique()
    
    for t_id in unique_events_in_order:
        group = df_filtered[df_filtered['Template ID'] == t_id]
        
        count = len(group)
        meaning = group.iloc[0]['Meaning Log']
        details = extract_parameters(group)
        
        # Get time range
        # Assuming Parameters has JSON timestamp, let's grab the first one roughly
        try:
            p_first = json.loads(group.iloc[0]['Parameters'])
            p_last = json.loads(group.iloc[-1]['Parameters'])
            time_str = f"{p_first.get('TIMESTAMP')} to {p_last.get('TIMESTAMP')}"
        except:
            time_str = "Unknown Time"

        timeline_text += f"- [{time_str}] {meaning}\n"
        timeline_text += f"  Count: {count} | Context: {details}\n\n"

    # Save the compressed timeline for debugging
    with open("compressed_timeline_data.txt", "w", encoding="utf-8") as f:
        f.write(timeline_text)

    # 4. Summarize (Pass 2)
    final_report = pass_2_generate_summary(timeline_text)
    
    print("\n" + "="*50)
    print(final_report)
    print("="*50)
    
    with open("Final_System_Report.txt", "w", encoding="utf-8") as f:
        f.write(final_report)

if __name__ == "__main__":
    main()