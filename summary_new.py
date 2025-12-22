import pandas as pd
import json
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import re
import os
import sys

def generate_report_txt(file_path, output_txt='Log_Analysis_Report.txt'):
    print(f"Reading data from: {file_path}...")
    
    # 1. Load Data
    try:
        if file_path.lower().endswith('.xlsx'):
            try:
                df_logs = pd.read_excel(file_path, sheet_name='Log Analysis')
            except:
                df_logs = pd.read_excel(file_path, sheet_name=0)
        else:
            try:
                df_logs = pd.read_csv(file_path)
            except UnicodeDecodeError:
                df_logs = pd.read_csv(file_path, encoding='latin1')
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    # 2. Data Cleaning
    print("Processing data...")
    df_logs.columns = [c.strip() for c in df_logs.columns]
    
    # Parse Parameters
    df_logs['params'] = df_logs['Parameters'].apply(lambda x: json.loads(x) if isinstance(x, str) else {})
    df_logs['USERNAME'] = df_logs['params'].apply(lambda x: x.get('USERNAME', 'N/A'))
    df_logs['RHOST'] = df_logs['params'].apply(lambda x: x.get('RHOST', 'N/A'))
    
    # Extract Service Name
    def extract_service(raw):
        try:
            parts = str(raw).split()
            if len(parts) > 4: return re.split(r'\[|:', parts[4])[0]
        except: pass
        return "Unknown"
    df_logs['Service'] = df_logs['Raw Log'].apply(extract_service)

    # --- SMART TIME PARSING ---
    def parse_time(row):
        ts = row['params'].get('TIMESTAMP', '')
        if not ts: 
            parts = str(row['Raw Log']).split()
            if len(parts) >= 3: ts = " ".join(parts[:3])
        # Force a dummy year (2024) internally so Pandas can calculate duration
        # We will HIDE this year in the final graph labels.
        try:
            return pd.to_datetime(f"2024 {ts}", format="%Y %b %d %H:%M:%S")
        except:
            return pd.to_datetime(ts, errors='coerce')
    
    df_logs['datetime'] = df_logs.apply(parse_time, axis=1)
    df_logs = df_logs.dropna(subset=['datetime']) # Drop rows where date failed
    
    if len(df_logs) == 0:
        print("Error: No valid dates found in logs.")
        return

    df_logs['Template ID'] = df_logs['Template ID'].astype(str)

    # --- ADAPTIVE TIMELINE LOGIC ---
    # Calculate how long the logs span
    min_time = df_logs['datetime'].min()
    max_time = df_logs['datetime'].max()
    duration = max_time - min_time
    total_hours = duration.total_seconds() / 3600

    # Decide Resampling & formatting based on duration
    if total_hours < 4:
        # If less than 4 hours, show by MINUTE
        resample_rule = '1T'  # T = Minute
        date_format = '%H:%M' # Show only Time
        xlabel_text = "Time (HH:MM)"
        print(f"Detected short duration ({total_hours:.1f} hrs). Using Minute-level detail.")
        
    elif total_hours < 48:
        # If less than 2 days, show by HOUR
        resample_rule = '1H'
        date_format = '%d %b %H:00' # Show Day + Hour
        xlabel_text = "Time (Day Hour)"
        print(f"Detected medium duration ({total_hours:.1f} hrs). Using Hour-level detail.")
        
    else:
        # If longer, show by DAY
        resample_rule = '1D'
        date_format = '%b %d' # Show Month Day
        xlabel_text = "Date"
        print(f"Detected long duration ({duration.days} days). Using Daily detail.")

    # Helper for Bar Labels
    def add_bar_labels(ax):
        for p in ax.patches:
            if p.get_width() > 0:
                ax.text(p.get_width(), p.get_y() + p.get_height()/2, 
                        f' {int(p.get_width())}', ha='left', va='center')

    # 3. Generate Charts
    print("Generating visualizations...")
    
    # Chart 1: Adaptive Volume Graph
    plt.figure(figsize=(10, 5))
    time_counts = df_logs.resample(resample_rule, on='datetime').size()
    
    if not time_counts.empty:
        ax = time_counts.plot(kind='line', marker='o', color='#1f77b4')
        plt.title(f'Log Volume Over Time (Grouped by {resample_rule.replace("T","Minute").replace("H","Hour").replace("D","Day")})')
        plt.ylabel('Event Count')
        plt.xlabel(xlabel_text)
        plt.grid(True, alpha=0.5)
        
        # Apply the smart format
        ax.xaxis.set_major_formatter(mdates.DateFormatter(date_format))
        plt.tight_layout()
        plt.savefig('1_log_volume.png')
        plt.close()

    # Chart 2: Services
    plt.figure(figsize=(10, 5))
    top_services = df_logs['Service'].value_counts().head(10).sort_values()
    ax = top_services.plot(kind='barh', color='#2ca02c')
    add_bar_labels(ax)
    plt.title('Top System Services')
    plt.tight_layout()
    plt.savefig('2_top_services.png')
    plt.close()

    # Chart 3: Templates
    plt.figure(figsize=(10, 6))
    top_templates = df_logs['Template ID'].value_counts().head(8).sort_values()
    ax = top_templates.plot(kind='barh', color='#ff7f0e')
    ax.set_yticklabels([f"Template {tid}" for tid in top_templates.index])
    add_bar_labels(ax)
    plt.title('Top Log Event Types')
    plt.tight_layout()
    plt.savefig('3_top_templates.png')
    plt.close()

    # Chart 4: Users
    plt.figure(figsize=(10, 5))
    top_users = df_logs[df_logs['USERNAME'] != 'N/A']['USERNAME'].value_counts().head(10).sort_values()
    ax = top_users.plot(kind='barh', color='#9467bd')
    add_bar_labels(ax)
    plt.title('Top Active Users')
    plt.tight_layout()
    plt.savefig('4_top_users.png')
    plt.close()

    # Chart 5: IPs
    plt.figure(figsize=(10, 5))
    top_ips = df_logs[df_logs['RHOST'] != 'N/A']['RHOST'].value_counts().head(10).sort_values()
    ax = top_ips.plot(kind='barh', color='#d62728')
    add_bar_labels(ax)
    plt.title('Top Remote IPs')
    plt.tight_layout()
    plt.savefig('5_top_ips.png')
    plt.close()

    # 4. Prepare Report
    print(f"Writing report to {output_txt}...")
    summary_lines = [
        "========================================",
        "      AUTOMATED LOG ANALYSIS REPORT",
        "========================================",
        f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "1. EXECUTIVE SUMMARY",
        "---------------------",
        f"Total Logs:   {len(df_logs)}",
        f"Time Span:    {min_time.strftime('%b %d %H:%M')} to {max_time.strftime('%b %d %H:%M')}",
        f"Log Duration: {str(duration)}",
        "",
        "KEY FINDINGS:",
        f"* Top Service: {df_logs['Service'].value_counts().idxmax()} ({df_logs['Service'].value_counts().max()} events)",
        f"* Top User:    {df_logs[df_logs['USERNAME'] != 'N/A']['USERNAME'].value_counts().idxmax()}",
        f"* Top IP:      {df_logs[df_logs['RHOST'] != 'N/A']['RHOST'].value_counts().idxmax()}",
        "",
        "2. TEMPLATE KEY (Refers to Chart 3)",
        "------------------------------------",
    ]
    
    top_templates_desc = df_logs['Template ID'].value_counts().head(8)
    for tid in top_templates_desc.index:
        example_log = df_logs[df_logs['Template ID'] == tid]['Meaning Log'].iloc[0]
        example_log = str(example_log).replace('\n', ' ').strip()
        if len(example_log) > 120: example_log = example_log[:117] + "..."
        summary_lines.append(f"[Template {tid}]: {example_log}")

    with open(output_txt, 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary_lines))
            
    print(f"Done! Report saved to '{output_txt}'.")

if __name__ == "__main__":
    print("--- Smart Log Analysis Tool ---")
    log_input = input("Enter file path: ").strip().strip('"')
    if log_input: generate_report_txt(log_input)