import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
import textwrap
import os

def generate_template_dashboard(file_path):
    print(f"--- Processing: {file_path} ---")

    # 1. Load Data
    # We specifically look for the Template Summary data
    try:
        # Check if the user provided the CSV directly or the Excel
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            # Try to read the specific sheet, fallback to index 1 (usually template summary)
            try:
                df = pd.read_excel(file_path, sheet_name='Template Summary')
            except:
                df = pd.read_excel(file_path, sheet_name=1)
    except Exception as e:
        print(f"CRITICAL ERROR: Could not load Template Summary. {e}")
        return

    # Normalize columns
    df.columns = [c.strip() for c in df.columns]
    
    # Required columns check
    required_cols = ['Template ID', 'Event Meaning', 'Occurrences', 'Template Pattern']
    for col in required_cols:
        if col not in df.columns:
            print(f"Error: Missing column '{col}'. Please ensure the file is the Template Summary.")
            return

    # Fill Missing Values
    df['Event Meaning'] = df['Event Meaning'].fillna("Unknown Event")
    df['Occurrences'] = df['Occurrences'].fillna(0)
    df['Template Pattern'] = df['Template Pattern'].fillna("")

    print(f"-> Loaded {len(df)} unique templates.")

    # 2. The Brain: Semantic Analysis
    print("Generating Embeddings for Event Meanings...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # We only embed the 'Event Meaning' as requested
    embeddings = model.encode(df['Event Meaning'].tolist(), show_progress_bar=True)

    # 3. Projection (t-SNE with Cosine Metric)
    print("Projecting to 2D Map...")
    # Perplexity must be less than n_samples. If < 30 samples, adjust perplexity.
    n_samples = len(df)
    perp = min(30, n_samples - 1) if n_samples > 1 else 1
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=perp, metric='cosine', init='random')
    coords = tsne.fit_transform(embeddings)
    
    df['x'] = coords[:, 0]
    df['y'] = coords[:, 1]

    # 4. Visualization Prep
    # Create Hover Text
    df['Hover_Text'] = df.apply(lambda row: f"<b>ID:</b> {row['Template ID']}<br>" +
                                            f"<b>Count:</b> {row['Occurrences']}<br>" +
                                            f"<b>Meaning:</b> {'<br>'.join(textwrap.wrap(str(row['Event Meaning']), width=50))}", axis=1)

    # Scale bubble size based on occurrences (Log scale to prevent massive bubbles)
    # Adding 1 to avoid log(0)
    df['Size'] = np.log1p(df['Occurrences']) 
    # Normalize size for display (min 10, max 40)
    df['Size_Scaled'] = ((df['Size'] - df['Size'].min()) / (df['Size'].max() - df['Size'].min() + 1e-9)) * 30 + 10

    # 5. Build Dashboard (Map + Side Panel)
    print("Building Dashboard...")
    
    # Create a subplot with 1 row, 2 cols. 
    # Column 1 (Map) is 70% width, Column 2 (Table) is 30% width.
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.65, 0.35],
        specs=[[{"type": "scatter"}, {"type": "table"}]],
        horizontal_spacing=0.02,
        subplot_titles=("Semantic Map (Proximity = Similarity)", "Template Data Panel")
    )

    # --- LEFT PANEL: SCATTER PLOT ---
    scatter_trace = go.Scatter(
        x=df['x'],
        y=df['y'],
        mode='markers+text',
        text=df['Template ID'], # Show ID on the dot
        textposition="top center",
        marker=dict(
            size=df['Size_Scaled'],
            color=df['Occurrences'], # Color by count intensity
            colorscale='Viridis',
            showscale=False,
            line=dict(width=1, color='DarkSlateGrey')
        ),
        textfont=dict(size=10, color='black'),
        hoverinfo='text',
        hovertext=df['Hover_Text'],
        name="Templates"
    )
    fig.add_trace(scatter_trace, row=1, col=1)

    # --- RIGHT PANEL: DATA TABLE ---
    # Sort by Occurrences for the table
    df_sorted = df.sort_values(by='Occurrences', ascending=False)
    
    table_trace = go.Table(
        header=dict(
            values=["<b>ID</b>", "<b>Cnt</b>", "<b>Meaning</b>"],
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
            font=dict(size=11),
            height=30
        )
    )
    fig.add_trace(table_trace, row=1, col=2)

    # --- LAYOUT STYLING ---
    fig.update_layout(
        title_text="Template Semantic Analysis Dashboard",
        template="plotly_white",
        height=800, # Tall enough for the table
        showlegend=False
    )
    
    # Remove axes for the map (cleaner look)
    fig.update_xaxes(visible=False, row=1, col=1)
    fig.update_yaxes(visible=False, row=1, col=1)

    output_file = 'template_semantic_map.html'
    fig.write_html(output_file)
    print(f"Done! Dashboard generated: '{output_file}'")

if __name__ == "__main__":
    path = input("Enter path to 'Template Summary' CSV or Excel: ").strip().strip('"')
    if path: generate_template_dashboard(path)