import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
import textwrap

def generate_hybrid_dashboard(file_path):
    print(f"--- Processing: {file_path} ---")

    # 1. LOAD DATA (Both Sheets)
    try:
        # Load Template Summary
        try:
            df_templates = pd.read_excel(file_path, sheet_name='Template Summary')
        except:
            df_templates = pd.read_excel(file_path, sheet_name=1)
            
        # Load Log Analysis
        try:
            df_logs = pd.read_excel(file_path, sheet_name='Log Analysis')
        except:
            df_logs = pd.read_excel(file_path, sheet_name=0)
            
    except Exception as e:
        print(f"CRITICAL ERROR: Could not load data. {e}")
        return

    # Clean Columns
    df_templates.columns = [c.strip() for c in df_templates.columns]
    df_logs.columns = [c.strip() for c in df_logs.columns]

    # Normalize Template IDs for matching
    df_templates['Template ID'] = df_templates['Template ID'].astype(str)
    df_logs['Template ID'] = df_logs['Template ID'].astype(str)
    
    # Fill NA
    df_templates['Event Meaning'] = df_templates['Event Meaning'].fillna("Unknown")
    df_logs['Meaning Log'] = df_logs['Meaning Log'].fillna("")
    df_logs['Raw Log'] = df_logs['Raw Log'].fillna("")

    print(f"-> Loaded {len(df_templates)} templates and {len(df_logs)} logs.")

    # 2. THE BRAIN (Semantic Embedding)
    print("Loading AI Model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Embed Templates
    print("Vectorizing Templates...")
    template_embeddings = model.encode(df_templates['Event Meaning'].tolist())

    # Embed Logs
    # Note: We embed the 'Meaning Log' so they align semantically with the Templates
    print("Vectorizing Logs (this may take a moment)...")
    log_embeddings = model.encode(df_logs['Meaning Log'].tolist(), show_progress_bar=True)

    # 3. PROJECTION (Combining Data for Shared Map)
    print("Projecting all data to 2D Map...")
    
    # We stack them so TSNE runs on the shared vector space
    combined_embeddings = np.vstack([template_embeddings, log_embeddings])
    
    # TSNE
    tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
    combined_coords = tsne.fit_transform(combined_embeddings)
    
    # Split coordinates back apart
    # First N rows are templates, rest are logs
    n_templates = len(df_templates)
    
    df_templates['x'] = combined_coords[:n_templates, 0]
    df_templates['y'] = combined_coords[:n_templates, 1]
    
    df_logs['x'] = combined_coords[n_templates:, 0]
    df_logs['y'] = combined_coords[n_templates:, 1]

    # 4. PREPARE VISUALIZATION
    print("Building Hybrid Dashboard...")
    
    # Create Layout: Map on Left (70%), Table on Right (30%)
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.7, 0.3],
        specs=[[{"type": "scatter"}, {"type": "table"}]],
        horizontal_spacing=0.02,
        subplot_titles=("Semantic Map (Logs + Templates)", "Template Counts")
    )

    # --- WRAPPER HELPERS ---
    def wrap(t): return '<br>'.join(textwrap.wrap(str(t), width=60))

    # --- TRACE 1: LOGS (The background "Cloud") ---
    # We add these first so they are behind the template stars
    
    # Prepare Log Hover Text
    df_logs['Hover_Raw'] = df_logs['Raw Log'].apply(wrap)
    df_logs['Hover_Meaning'] = df_logs['Meaning Log'].apply(wrap)
    
    fig.add_trace(
        go.Scatter(
            x=df_logs['x'],
            y=df_logs['y'],
            mode='markers',
            name='Logs',
            marker=dict(
                size=5, 
                color='lightgrey', # Neutral color to let templates pop? Or color by ID?
                # Let's color by Template ID to show clusters clearly
                color=df_logs['Template ID'].astype("category").cat.codes, 
                colorscale='Jet',
                opacity=0.5
            ),
            # Custom Hover for Logs
            customdata=df_logs[['Hover_Raw', 'Hover_Meaning']].values,
            hovertemplate=(
                "<b>Type:</b> Log Entry<br>" +
                "<b>Raw:</b> %{customdata[0]}<br>" +
                "<b>Meaning:</b> %{customdata[1]}<extra></extra>"
            ),
            showlegend=False
        ),
        row=1, col=1
    )

    # --- TRACE 2: TEMPLATES (The "Anchors") ---
    df_templates['Hover_Meaning'] = df_templates['Event Meaning'].apply(wrap)
    
    fig.add_trace(
        go.Scatter(
            x=df_templates['x'],
            y=df_templates['y'],
            mode='markers+text',
            name='Templates',
            text=df_templates['Template ID'],
            textposition="top center",
            marker=dict(
                symbol='star',
                size=15, # Big stars
                color='black',
                line=dict(width=1, color='white')
            ),
            # Custom Hover for Templates
            customdata=df_templates[['Template ID', 'Occurrences', 'Hover_Meaning']].values,
            hovertemplate=(
                "<b>Type:</b> Template Anchor<br>" +
                "<b>ID:</b> %{customdata[0]}<br>" +
                "<b>Count:</b> %{customdata[1]}<br>" +
                "<b>Def:</b> %{customdata[2]}<extra></extra>"
            )
        ),
        row=1, col=1
    )

    # --- TRACE 3: SIDE PANEL TABLE ---
    df_sorted = df_templates.sort_values(by='Occurrences', ascending=False)
    
    fig.add_trace(
        go.Table(
            header=dict(
                values=["<b>ID</b>", "<b>Count</b>", "<b>Meaning</b>"],
                fill_color='paleturquoise',
                align='left'
            ),
            cells=dict(
                values=[
                    df_sorted['Template ID'], 
                    df_sorted['Occurrences'], 
                    df_sorted['Event Meaning']
                ],
                fill_color='lavender',
                align='left',
                height=30
            )
        ),
        row=1, col=2
    )

    # --- LAYOUT ---
    fig.update_layout(
        title_text="Hybrid Log Analysis: Templates vs. Real Logs",
        template="plotly_white",
        height=900,
        showlegend=True
    )
    
    # Hide axes on map
    fig.update_xaxes(visible=False, row=1, col=1)
    fig.update_yaxes(visible=False, row=1, col=1)

    output_file = 'hybrid_log_dashboard.html'
    fig.write_html(output_file)
    print(f"Done! Dashboard saved to: {output_file}")

if __name__ == "__main__":
    path = input("Enter Excel file path: ").strip().strip('"')
    if path: generate_hybrid_dashboard(path)