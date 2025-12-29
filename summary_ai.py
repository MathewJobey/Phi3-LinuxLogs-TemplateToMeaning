import justpy as jp
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import sys

# --- GLOBAL DATAFRAME STORAGE ---
# We store the data here so the GUI can access it
APP_DATA = {"df": pd.DataFrame()}

# --- 1. DATA LOADING FUNCTION ---
def load_data(file_path):
    print(f"Attempting to load: {file_path}")
    try:
        # Check if file exists first
        if not os.path.exists(file_path):
            print(f"Error: File not found at {file_path}")
            return pd.DataFrame()

        if file_path.lower().endswith('.xlsx'):
            # Try loading 'Log Analysis' sheet, fallback to first sheet
            try:
                df = pd.read_excel(file_path, sheet_name='Log Analysis')
            except:
                df = pd.read_excel(file_path, sheet_name=0)
        else:
            # CSV Fallback
            try:
                df = pd.read_csv(file_path)
            except UnicodeDecodeError:
                df = pd.read_csv(file_path, encoding='latin1')
        
        # Clean up columns
        df.columns = [c.strip() for c in df.columns]
        
        # Ensure text column exists
        text_col = 'Meaning Log' if 'Meaning Log' in df.columns else 'Raw Log'
        if text_col not in df.columns:
            print(f"Error: Column '{text_col}' not found in file.")
            return pd.DataFrame()

        df[text_col] = df[text_col].fillna('')
        df['Raw Log'] = df['Raw Log'].fillna('')
        
        # Parse Dates (Robust)
        def parse_time(row):
            params = json.loads(row.get('Parameters', '{}')) if isinstance(row.get('Parameters'), str) else {}
            ts = params.get('TIMESTAMP', '')
            if not ts: 
                parts = str(row['Raw Log']).split()
                if len(parts) >= 3: ts = " ".join(parts[:3])
            try:
                # Dummy year 2024 to make sorting work
                return pd.to_datetime(f"2024 {ts}", format="%Y %b %d %H:%M:%S")
            except:
                return pd.to_datetime(ts, errors='coerce')

        df['datetime'] = df.apply(parse_time, axis=1)
        df = df.dropna(subset=['datetime'])
        
        print(f"Successfully loaded {len(df)} logs.")
        return df

    except Exception as e:
        print(f"Critical Error loading file: {e}")
        return pd.DataFrame()

# --- 2. THE JUSTPY APP ---
def log_dashboard():
    df = APP_DATA["df"]
    
    # Create the WebPage
    wp = jp.QuasarPage(tailwind=True) 
    
    # Page Header
    header = jp.Div(a=wp, classes="bg-blue-600 text-white p-4 text-center shadow-md")
    jp.Div(a=header, text="🛡️ Log Analysis Dashboard", classes="text-2xl font-bold")

    # ERROR HANDLING: If data failed to load, show nice error on page
    if df.empty:
        error_container = jp.Div(a=wp, classes="p-10 text-center text-red-600")
        jp.Div(a=error_container, text="❌ Data could not be loaded.", classes="text-3xl font-bold mb-4")
        jp.Div(a=error_container, text="Please check the file path in your console and restart the script.", classes="text-lg")
        return wp

    # Main Grid Layout
    container = jp.Div(a=wp, classes="p-4 grid grid-cols-1 md:grid-cols-2 gap-6")

    # --- SECTION 1: METRICS CARDS ---
    stats_div = jp.Div(a=container, classes="col-span-1 md:col-span-2 grid grid-cols-2 md:grid-cols-4 gap-4")
    
    # Calculate stats safely
    crit_count = len(df[df['Raw Log'].str.contains('Critical|Fatal|Panic', case=False, na=False)])
    unique_ips = df['RHOST'].nunique() if 'RHOST' in df.columns else 0
    
    metrics = [
        ("Total Logs", len(df), "bg-blue-100 text-blue-800"),
        ("Critical Events", crit_count, "bg-red-100 text-red-800"),
        ("Templates", df['Template ID'].nunique(), "bg-green-100 text-green-800"),
        ("Unique IPs", unique_ips, "bg-purple-100 text-purple-800")
    ]
    
    for label, value, color_class in metrics:
        card = jp.Div(a=stats_div, classes=f"p-4 rounded shadow border-l-4 {color_class.replace('text', 'border')}")
        jp.Div(a=card, text=label, classes="text-gray-600 text-sm uppercase tracking-wide")
        jp.Div(a=card, text=str(value), classes="text-3xl font-bold")

    # --- SECTION 2: CHARTS (Matplotlib) ---
    
    # Chart 1: Volume
    chart_card1 = jp.Div(a=container, classes="bg-white p-4 rounded shadow border border-gray-200")
    jp.Div(a=chart_card1, text="Log Activity Over Time", classes="font-bold mb-4 text-lg border-b pb-2")
    
    fig1, ax1 = plt.subplots(figsize=(6, 3.5))
    # Resample to Hourly count
    df.resample('H', on='datetime').size().plot(kind='line', ax=ax1, color='#2563eb', linewidth=2)
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Count")
    plt.tight_layout()
    
    chart1_jp = jp.Matplotlib(a=chart_card1)
    chart1_jp.set_figure(fig1)

    # Chart 2: Top Services
    chart_card2 = jp.Div(a=container, classes="bg-white p-4 rounded shadow border border-gray-200")
    jp.Div(a=chart_card2, text="Top System Services", classes="font-bold mb-4 text-lg border-b pb-2")
    
    fig2, ax2 = plt.subplots(figsize=(6, 3.5))
    if 'Service' not in df.columns:
        # Safe extraction if column missing
        df['Service'] = df['Raw Log'].apply(lambda x: str(x).split()[4].split('[')[0].replace(':','') if len(str(x).split()) > 4 else "Unknown")
    
    df['Service'].value_counts().head(8).sort_values().plot(kind='barh', ax=ax2, color='#10b981')
    ax2.set_xlabel("Count")
    plt.tight_layout()
        
    chart2_jp = jp.Matplotlib(a=chart_card2)
    chart2_jp.set_figure(fig2)

    # --- SECTION 3: DATA GRID ---
    full_width = jp.Div(a=wp, classes="p-4")
    jp.Div(a=full_width, text="📂 Raw Data Explorer", classes="text-xl font-bold mb-2 mt-4")
    
    # Prepare grid data (first 500 rows for performance)
    grid_cols = ['datetime', 'Service', 'Raw Log', 'Meaning Log']
    # Filter only existing cols
    valid_cols = [c for c in grid_cols if c in df.columns]
    grid_df = df[valid_cols].head(500).copy()
    
    # Format datetime as string for JSON serialization
    grid_df['datetime'] = grid_df['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')

    ag_grid = jp.AgGrid(a=full_width)
    ag_grid.load_pandas_frame(grid_df)
    ag_grid.options.pagination = True
    ag_grid.options.paginationAutoPageSize = True
    ag_grid.style = "height: 600px; width: 100%; border: 1px solid #e5e7eb; border-radius: 0.5rem;"
    
    # Footer
    jp.Div(a=wp, classes="p-6 text-center text-gray-400 text-sm", text="Generated by AI Log Analysis Tool")

    return wp

# --- 3. MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    print("--- JustPy Log Dashboard ---")
    
    # INTERACTIVE INPUT: Solves the "File Not Found" error
    # You can paste the full path: C:\Users\Mathe\Downloads\...\your_file.xlsx
    user_path = input("Enter the full path to your Excel/CSV file: ").strip().strip('"')
    
    if user_path:
        APP_DATA["df"] = load_data(user_path)
        
        if not APP_DATA["df"].empty:
            print("Starting Web Server...")
            print("OPEN YOUR BROWSER TO: http://127.0.0.1:8000")
            jp.justpy(log_dashboard)
        else:
            print("Exiting due to data load failure.")
    else:
        print("No file path provided.")