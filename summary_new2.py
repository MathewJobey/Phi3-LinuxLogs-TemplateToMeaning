import pandas as pd
import json
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import re
import os
import sys
import textwrap

def generate_advanced_report(file_path, output_txt='Log_Analysis_Report.txt'):
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

    # --- STRICT SEVERITY TAGGING ---
    def classify_severity(row):
        text = (str(row['Raw Log']) + " " + str(row['Meaning Log'])).lower()
        
        # STRICT CRITICAL: Must have high-severity words
        if any(x in text for x in ['critical', 'fatal', 'panic', 'emergency', 'alert', 'segmentation fault', 'died']):
            return 'CRITICAL'
        
        # STRICT WARNING: Must have "warning" (excluding simple "fail")
        elif any(x in text for x in ['warning', 'warn']):
            return 'WARNING'
        
        return 'INFO'
            
    df_logs['Severity'] = df_logs.apply(classify_severity, axis=1)

    def classify_security(row):
        text = str(row['Raw Log']).lower()
        if 'illegal' in text: return 'Illegal Access'
        if 'authentication failure' in text: return 'Auth Failure'
        if 'root' in text and 'session' in text: return 'Root Activity'
        if 'session opened' in text or 'logged in' in text or 'accepted' in text: 
            return 'Successful Login'
        return 'Normal'
        
    df_logs['Security_Tag'] = df_logs.apply(classify_security, axis=1)

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
    
    # Chart 1: Volume
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
        peak_str = time_counts.idxmax().strftime(date_format)
        peak_vol = time_counts.max()
    else:
        peak_str, peak_vol = "N/A", 0

    # Chart 2: Services
    plt.figure(figsize=(10, 5))
    ax = df_logs['Service'].value_counts().head(10).sort_values().plot(kind='barh', color='#2ca02c')
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
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
    ax = df_logs[df_logs['USERNAME'] != 'N/A']['USERNAME'].value_counts().head(10).sort_values().plot(kind='barh', color='#9467bd')
    add_bar_labels(ax)
    plt.title('Top Active Users')
    plt.tight_layout()
    plt.savefig('4_top_users.png')
    plt.close()

    # Chart 5: IPs
    plt.figure(figsize=(10, 5))
    ax = df_logs[df_logs['RHOST'] != 'N/A']['RHOST'].value_counts().head(10).sort_values().plot(kind='barh', color='#d62728')
    add_bar_labels(ax)
    plt.title('Top Remote IPs')
    plt.tight_layout()
    plt.savefig('5_top_ips.png')
    plt.close()

    # Chart 6: Security Breakdown
    plt.figure(figsize=(8, 6))
    security_counts = df_logs[df_logs['Security_Tag'] != 'Normal']['Security_Tag'].value_counts()
    
    color_map = {
        'Successful Login': '#99ff99', 
        'Auth Failure': '#ff9999',     
        'Root Activity': '#ffcc99',    
        'Illegal Access': '#c2c2f0',   
        'Normal': '#f0f0f0'
    }
    colors = [color_map.get(x, '#cccccc') for x in security_counts.index]

    if not security_counts.empty:
        security_counts.plot(kind='pie', autopct='%1.1f%%', colors=colors)
        plt.title('Security Event Distribution')
        plt.ylabel('')
    else:
        plt.text(0.5, 0.5, 'No Security Events Detected', ha='center')
    plt.tight_layout()
    plt.savefig('6_security_breakdown.png')
    plt.close()

    # 4. Prepare Report
    print(f"Writing advanced report to {output_txt}...")
    
    def get_top_3_str(series):
        if series.empty: return "None"
        items = []
        total = len(df_logs)
        for name, count in series.head(3).items():
            pct = (count / total) * 100
            items.append(f"{name} ({count}, {pct:.1f}%)")
        return "; ".join(items)

    total_events = len(df_logs)
    
    # Stats
    sev_counts = df_logs['Severity'].value_counts()
    crit_count = sev_counts.get('CRITICAL', 0)
    warn_count = sev_counts.get('WARNING', 0)
    
    root_events = df_logs[df_logs['Security_Tag'] == 'Root Activity']
    auth_fails = df_logs[df_logs['Security_Tag'] == 'Auth Failure']
    success_logins = df_logs[df_logs['Security_Tag'] == 'Successful Login']
    
    summary_lines = [
        "============================================================",
        "             ADVANCED LOG ANALYSIS REPORT",
        "============================================================",
        f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "1. EXECUTIVE OVERVIEW",
        "---------------------",
        f"Analysis Period:      {min_time.strftime('%Y-%b-%d %H:%M')} to {max_time.strftime('%Y-%b-%d %H:%M')}",
        f"Total Log Entries:    {total_events}",
        f"Health Status:        {'⚠️ ATTENTION NEEDED' if crit_count > 0 else '✅ STABLE'}",
        "",
        "2. SECURITY AUDIT",
        "---------------------",
        f"🔴 Critical Events:     {crit_count} events detected",
        f"🟠 Warning Events:      {warn_count} events detected",
        f"🔐 Auth Failures:       {len(auth_fails)} failed login attempts",
        f"⚡ Root Activity:       {len(root_events)} sessions involving root user",
        f"✅ Successful Logins:   {len(success_logins)} sessions established",
        "",
        "3. ACTIVITY ANALYSIS",
        "---------------------",
        f"Peak Activity Time:   {peak_str} (approx {peak_vol} events/{time_unit.lower()})",
        f"Avg Event Rate:       {total_events / (total_hours if total_hours > 0 else 1):.1f} events per hour",
        "",
        "4. CRITICAL BREAKDOWN (Top 3)",
        "---------------------",
        "Top System Services:",
        f"  -> {get_top_3_str(df_logs['Service'].value_counts())}",
        "",
        "Top Active Users:",
        f"  -> {get_top_3_str(df_logs[df_logs['USERNAME'] != 'N/A']['USERNAME'].value_counts())}",
        "",
        "Top Remote Sources:",
        f"  -> {get_top_3_str(df_logs[df_logs['RHOST'] != 'N/A']['RHOST'].value_counts())}",
        ""
    ]

    # --- NEW SECTION: STRICT HIGHLIGHTS (With Meanings) ---
    summary_lines.append("5. RISK EVENT HIGHLIGHTS (Strictly Critical/Warning)")
    summary_lines.append("--------------------------------------------------")
    
    # Filter only STRICT CRITICAL
    crit_groups = df_logs[df_logs['Severity'] == 'CRITICAL'].groupby('Template ID')
    sorted_crit = sorted(crit_groups, key=lambda x: len(x[1]), reverse=True)

    if not sorted_crit:
        summary_lines.append("✅ No 'Critical', 'Fatal', or 'Panic' events found.")
    else:
        summary_lines.append(f"🔴 CRITICAL / FATAL EVENTS ({len(crit_groups)} types):")
        for tid, group in sorted_crit:
            row = group.iloc[0]
            summary_lines.append(f"   [Count: {len(group)}] Template {tid}")
            summary_lines.append(f"   RAW:     {str(row['Raw Log'])[:120]}")
            # Added MEANING Log here
            summary_lines.append(f"   MEANING: {str(row['Meaning Log']).replace('\n', ' ').strip()}")
            summary_lines.append("")

    summary_lines.append("")

    # Filter only STRICT WARNING
    warn_groups = df_logs[df_logs['Severity'] == 'WARNING'].groupby('Template ID')
    sorted_warn = sorted(warn_groups, key=lambda x: len(x[1]), reverse=True)

    if not sorted_warn:
        summary_lines.append("✅ No explicit 'Warning' events found.")
    else:
        summary_lines.append(f"🟠 WARNING EVENTS ({len(warn_groups)} types):")
        for tid, group in sorted_warn:
            row = group.iloc[0]
            summary_lines.append(f"   [Count: {len(group)}] Template {tid}")
            summary_lines.append(f"   RAW:     {str(row['Raw Log'])[:120]}")
            # Added MEANING Log here
            summary_lines.append(f"   MEANING: {str(row['Meaning Log']).replace('\n', ' ').strip()}")
            summary_lines.append("")
    
    summary_lines.append("")
    # -------------------------------

    summary_lines.append("6. EVENT PATTERN DICTIONARY")
    summary_lines.append("------------------------------------")
    summary_lines.append("Definitions for the Top Event Templates:")
    summary_lines.append("")
    
    top_templates_desc = df_logs['Template ID'].value_counts().head(8)
    for tid in top_templates_desc.index:
        if 'Drained Named Log' in df_logs.columns:
            template_pattern = df_logs[df_logs['Template ID'] == tid]['Drained Named Log'].iloc[0]
        else:
            template_pattern = df_logs[df_logs['Template ID'] == tid]['Meaning Log'].iloc[0]
            
        template_pattern = str(template_pattern).replace('\n', ' ').strip()
        wrapped_text = textwrap.fill(f"[Template {tid}]: {template_pattern}", width=90, subsequent_indent="    ")
        summary_lines.append(wrapped_text)
        summary_lines.append("")

    with open(output_txt, 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary_lines))
            
    print(f"Done! Report saved to '{output_txt}'.")

if __name__ == "__main__":
    print("--- Advanced Log Analysis Tool ---")
    log_input = input("Enter file path: ").strip().strip('"')
    if log_input: generate_advanced_report(log_input)