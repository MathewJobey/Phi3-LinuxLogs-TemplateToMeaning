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

    # Smart Time Parsing
    def parse_time(row):
        ts = row['params'].get('TIMESTAMP', '')
        if not ts: 
            parts = str(row['Raw Log']).split()
            if len(parts) >= 3: ts = " ".join(parts[:3])
        try:
            return pd.to_datetime(f"2024 {ts}", format="%Y %b %d %H:%M:%S")
        except:
            return pd.to_datetime(ts, errors='coerce')
    
    df_logs['datetime'] = df_logs.apply(parse_time, axis=1)
    df_logs = df_logs.dropna(subset=['datetime'])
    
    if len(df_logs) == 0:
        print("Error: No valid dates found.")
        return

    df_logs['Template ID'] = df_logs['Template ID'].astype(str)

    # --- ADAPTIVE TIMELINE LOGIC ---
    min_time = df_logs['datetime'].min()
    max_time = df_logs['datetime'].max()
    duration = max_time - min_time
    total_hours = duration.total_seconds() / 3600

    if total_hours < 4:
        resample_rule = '1T' 
        date_format = '%H:%M'
        xlabel_text = "Time (HH:MM)"
        time_unit = "Minute"
    elif total_hours < 48:
        resample_rule = '1H'
        date_format = '%d %b %H:00'
        xlabel_text = "Time (Day Hour)"
        time_unit = "Hour"
    else:
        resample_rule = '1D'
        date_format = '%b %d'
        xlabel_text = "Date"
        time_unit = "Day"

    # Helper for Bar Labels
    def add_bar_labels(ax):
        for p in ax.patches:
            if p.get_width() > 0:
                ax.text(p.get_width(), p.get_y() + p.get_height()/2, 
                        f' {int(p.get_width())}', ha='left', va='center')

    # 3. Generate Charts
    print("Generating visualizations...")
    
    # Chart 1: Adaptive Volume
    plt.figure(figsize=(10, 5))
    time_counts = df_logs.resample(resample_rule, on='datetime').size()
    
    if not time_counts.empty:
        ax = time_counts.plot(kind='line', marker='o', color='#1f77b4')
        plt.title(f'Log Volume Over Time (Grouped by {time_unit})')
        plt.ylabel('Event Count')
        plt.xlabel(xlabel_text)
        plt.grid(True, alpha=0.5)
        ax.xaxis.set_major_formatter(mdates.DateFormatter(date_format))
        plt.tight_layout()
        plt.savefig('1_log_volume.png')
        plt.close()

        # Identify Peak Time for Summary
        peak_time = time_counts.idxmax()
        peak_vol = time_counts.max()
        peak_str = peak_time.strftime(date_format)
    else:
        peak_str = "N/A"
        peak_vol = 0

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

    # 4. Prepare DETAILED Report
    print(f"Writing detailed report to {output_txt}...")
    
    # Helper to format Top 3 lists
    def get_top_3_str(series):
        if series.empty: return "None"
        items = []
        total = len(df_logs)
        for name, count in series.head(3).items():
            pct = (count / total) * 100
            items.append(f"{name} ({count}, {pct:.1f}%)")
        return "; ".join(items)

    total_events = len(df_logs)
    unique_templates = df_logs['Template ID'].nunique()
    
    summary_lines = [
        "============================================================",
        "             AUTOMATED LOG ANALYSIS REPORT",
        "============================================================",
        f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "1. EXECUTIVE OVERVIEW",
        "---------------------",
        f"Analysis Period:      {min_time.strftime('%Y-%b-%d %H:%M')} to {max_time.strftime('%Y-%b-%d %H:%M')}",
        f"Duration:             {str(duration)}",
        f"Total Log Entries:    {total_events}",
        f"Unique Event Types:   {unique_templates} distinct templates",
        "",
        "2. ACTIVITY ANALYSIS",
        "---------------------",
        f"Peak Activity Time:   {peak_str} (approx {peak_vol} events/{time_unit.lower()})",
        f"Avg Event Rate:       {total_events / (total_hours if total_hours > 0 else 1):.1f} events per hour",
        "",
        "3. CRITICAL BREAKDOWN (Top 3)",
        "---------------------",
        "Top System Services (Components generating the most noise):",
        f"  -> {get_top_3_str(df_logs['Service'].value_counts())}",
        "",
        "Top Active Users (Accounts triggering the most events):",
        f"  -> {get_top_3_str(df_logs[df_logs['USERNAME'] != 'N/A']['USERNAME'].value_counts())}",
        "",
        "Top Remote Sources (External IPs connecting to the system):",
        f"  -> {get_top_3_str(df_logs[df_logs['RHOST'] != 'N/A']['RHOST'].value_counts())}",
        "",
        "4. EVENT DICTIONARY (Key for Chart 3)",
        "------------------------------------",
        "Detailed meanings for the most frequent Template IDs shown in the graph:",
        ""
    ]
    
    top_templates_desc = df_logs['Template ID'].value_counts().head(8)
    for tid in top_templates_desc.index:
        example_log = df_logs[df_logs['Template ID'] == tid]['Meaning Log'].iloc[0]
        example_log = str(example_log).replace('\n', ' ').strip()
        if len(example_log) > 130: example_log = example_log[:127] + "..."
        summary_lines.append(f"[Template {tid}]: {example_log}")

    with open(output_txt, 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary_lines))
            
    print(f"Done! Report saved to '{output_txt}'.")

if __name__ == "__main__":
    print("--- Detailed Log Analysis Tool ---")
    log_input = input("Enter file path: ").strip().strip('"')
    if log_input: generate_report_txt(log_input)