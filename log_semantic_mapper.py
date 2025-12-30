"""
Interactive Semantic Log Cluster Visualizer with GUI
====================================================
A Python-based interactive tool with GUI for visualizing Linux log templates.

Features:
- File selection dialog using tkinter
- Full-screen interactive visualization
- Click blobs to see related logs in a table below
- Light theme for better readability
- Option to reload/input additional data

Dependencies:
    pip install pandas openpyxl sentence-transformers scikit-learn umap-learn plotly numpy dash dash-bootstrap-components
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import umap
import plotly.graph_objects as go
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')

# For GUI file selection
import tkinter as tk
from tkinter import filedialog

# For web-based interactive dashboard
import dash
from dash import dcc, html, dash_table, Input, Output, State
import dash_bootstrap_components as dbc
import webbrowser
from threading import Timer


class SemanticLogVisualizer:
    """
    Main class for semantic log analysis and visualization.
    """
    
    def __init__(self, excel_path: str):
        """Initialize visualizer and load data."""
        print("🔄 Loading data from Excel...")
        self.log_df = pd.read_excel(excel_path, sheet_name='Log Analysis')
        self.template_df = pd.read_excel(excel_path, sheet_name='Template Summary')
        
        # Clean data
        self.template_df['Event Meaning'] = self.template_df['Event Meaning'].fillna('')
        self.template_df['Occurrences'] = self.template_df['Occurrences'].fillna(0)
        
        print(f"✅ Loaded {len(self.log_df)} log entries and {len(self.template_df)} unique templates")
        
        self.model = None
        self.embeddings = None
        self.coordinates_2d = None
        
    def compute_semantic_embeddings(self, model_name: str = 'all-MiniLM-L6-v2'):
        """Compute semantic embeddings for each template's event meaning."""
        print(f"🔄 Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        
        print("🔄 Computing semantic embeddings...")
        meanings = self.template_df['Event Meaning'].tolist()
        self.embeddings = self.model.encode(
            meanings,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        print(f"✅ Generated embeddings with shape: {self.embeddings.shape}")
        
    def compute_2d_layout(self, n_neighbors: int = 15, min_dist: float = 0.1, random_state: int = 42):
        """Use UMAP to project high-dimensional embeddings to 2D."""
        if self.embeddings is None:
            raise ValueError("Must compute embeddings first!")
            
        print("🔄 Computing 2D layout with UMAP...")
        
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=2,
            metric='cosine',
            random_state=random_state
        )
        
        self.coordinates_2d = reducer.fit_transform(self.embeddings)
        
        print(f"✅ Generated 2D coordinates with shape: {self.coordinates_2d.shape}")
        
    def create_interactive_visualization(self) -> go.Figure:
        """Create interactive Plotly visualization with clickable template blobs."""
        if self.coordinates_2d is None:
            raise ValueError("Must compute 2D layout first!")
            
        print("🔄 Creating interactive visualization...")
        
        # Prepare data
        x_coords = self.coordinates_2d[:, 0]
        y_coords = self.coordinates_2d[:, 1]
        
        # Normalize blob sizes
        sizes = self.template_df['Occurrences'].values
        size_min, size_max = 15, 60
        sizes_normalized = size_min + (sizes - sizes.min()) / (sizes.max() - sizes.min() + 1e-10) * (size_max - size_min)
        
        # Create color mapping
        colors = np.arctan2(y_coords, x_coords)
        
        # Prepare hover text
        hover_texts = []
        for _, row in self.template_df.iterrows():
            meaning_short = row['Event Meaning'][:100] + '...' if len(row['Event Meaning']) > 100 else row['Event Meaning']
            hover_text = (
                f"<b>Template ID:</b> {row['Template ID']}<br>"
                f"<b>Occurrences:</b> {row['Occurrences']}<br>"
                f"<b>Pattern:</b> {row['Template Pattern'][:80]}...<br>"
                f"<b>Meaning:</b> {meaning_short}<br>"
                "<i>Click to see details</i>"
            )
            hover_texts.append(hover_text)
        
        # Create figure
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=x_coords,
            y=y_coords,
            mode='markers',
            marker=dict(
                size=sizes_normalized,
                color=colors,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Semantic<br>Region"),
                line=dict(width=1, color='#333333'),
                opacity=0.8
            ),
            text=hover_texts,
            hovertemplate='%{text}<extra></extra>',
            customdata=self.template_df[['Template ID', 'Event Meaning', 'Occurrences', 'Template Pattern']].values,
            name='Templates'
        ))
        
        # Add template ID labels
        fig.add_trace(go.Scatter(
            x=x_coords,
            y=y_coords,
            mode='text',
            text=self.template_df['Template ID'].astype(str),
            textfont=dict(size=9, color='#000000', family='Arial Black'),
            hoverinfo='skip',
            showlegend=False
        ))
        
        # Light theme layout
        fig.update_layout(
            title={
                'text': 'Semantic Log Template Cluster Map<br><sub>Blob size = occurrence count | Distance = semantic similarity | Click blobs for details</sub>',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 22, 'color': '#1a1a1a'}
            },
            xaxis=dict(
                title='Semantic Dimension 1',
                showgrid=True,
                gridcolor='#e0e0e0',
                zeroline=False,
                color='#1a1a1a'
            ),
            yaxis=dict(
                title='Semantic Dimension 2',
                showgrid=True,
                gridcolor='#e0e0e0',
                zeroline=False,
                color='#1a1a1a'
            ),
            hovermode='closest',
            height=700,
            plot_bgcolor='#ffffff',
            paper_bgcolor='#f8f9fa',
            font=dict(color='#1a1a1a', family='Arial'),
            showlegend=False,
            margin=dict(l=50, r=50, t=100, b=50)
        )
        
        print("✅ Visualization created successfully!")
        return fig
    
    def get_template_details(self, template_id: int) -> pd.DataFrame:
        """Get all log entries for a specific template."""
        filtered = self.log_df[self.log_df['Template ID'] == template_id].copy()
        
        if len(filtered) == 0:
            return pd.DataFrame(columns=['Raw Log', 'Meaning Log', 'Parameters'])
        
        result = filtered[['Raw Log', 'Meaning Log', 'Parameters']].reset_index(drop=True)
        return result


def select_file():
    """Open file dialog to select Excel file."""
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    root.attributes('-topmost', True)  # Bring dialog to front
    
    file_path = filedialog.askopenfilename(
        title="Select Log Analysis Excel File",
        filetypes=[
            ("Excel files", "*.xlsx *.xls"),
            ("All files", "*.*")
        ]
    )
    
    root.destroy()
    return file_path


def create_dash_app(visualizer: SemanticLogVisualizer):
    """Create Dash web application for interactive visualization."""
    
    # Initialize Dash app with Bootstrap theme
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    
    # Create initial figure
    fig = visualizer.create_interactive_visualization()
    
    # App layout
    app.layout = dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H1("🔍 Semantic Log Cluster Visualizer", 
                       className="text-center mb-3 mt-3",
                       style={'color': '#1a1a1a'}),
                html.P("Click on any blob to view detailed logs below",
                      className="text-center text-muted mb-4"),
            ])
        ]),
        
        dbc.Row([
            dbc.Col([
                dbc.Button("📁 Load New File", id="load-file-btn", color="primary", className="me-2"),
                dbc.Button("🔄 Refresh Visualization", id="refresh-btn", color="secondary", className="me-2"),
                html.Span(id="file-status", className="ms-3 text-muted"),
            ], className="mb-3")
        ]),
        
        dbc.Row([
            dbc.Col([
                dcc.Graph(
                    id='cluster-graph',
                    figure=fig,
                    config={
                        'displayModeBar': True,
                        'displaylogo': False,
                        'modeBarButtonsToAdd': ['drawopenpath', 'eraseshape'],
                    },
                    style={'height': '700px'}
                )
            ])
        ]),
        
        dbc.Row([
            dbc.Col([
                html.Hr(),
                html.H3("📊 Template Details", className="mt-4 mb-3"),
                html.Div(id="template-info", className="mb-3"),
                html.Div([
                    dash_table.DataTable(
                        id='log-table',
                        columns=[
                            {'name': 'Raw Log', 'id': 'Raw Log'},
                            {'name': 'Meaning Log', 'id': 'Meaning Log'},
                            {'name': 'Parameters', 'id': 'Parameters'},
                        ],
                        data=[],
                        style_table={
                            'overflowX': 'auto',
                            'overflowY': 'auto',
                            'maxHeight': '400px',
                        },
                        style_cell={
                            'textAlign': 'left',
                            'padding': '10px',
                            'fontSize': '13px',
                            'fontFamily': 'Arial',
                            'whiteSpace': 'normal',
                            'height': 'auto',
                        },
                        style_header={
                            'backgroundColor': '#007bff',
                            'color': 'white',
                            'fontWeight': 'bold',
                            'fontSize': '14px',
                        },
                        style_data={
                            'backgroundColor': '#ffffff',
                            'color': '#1a1a1a',
                        },
                        style_data_conditional=[
                            {
                                'if': {'row_index': 'odd'},
                                'backgroundColor': '#f8f9fa',
                            }
                        ],
                        page_size=10,
                        sort_action='native',
                        filter_action='native',
                    )
                ])
            ])
        ]),
        
        html.Div(id='click-data-store', style={'display': 'none'}),
        
    ], fluid=True, style={'backgroundColor': '#ffffff', 'minHeight': '100vh', 'padding': '20px'})
    
    # Callback for click events
    @app.callback(
        [Output('log-table', 'data'),
         Output('template-info', 'children')],
        [Input('cluster-graph', 'clickData')]
    )
    def display_click_data(clickData):
        if clickData is None:
            return [], html.P("Click on a blob to see its logs", className="text-muted")
        
        # Extract template ID from click
        point = clickData['points'][0]
        template_id = int(point['customdata'][0])
        event_meaning = point['customdata'][1]
        occurrences = int(point['customdata'][2])
        pattern = point['customdata'][3]
        
        # Get template details
        details_df = visualizer.get_template_details(template_id)
        
        # Create info card
        info_card = dbc.Card([
            dbc.CardBody([
                html.H5(f"Template ID: {template_id}", className="card-title"),
                html.P([
                    html.Strong("Occurrences: "), f"{occurrences} logs",
                ], className="mb-2"),
                html.P([
                    html.Strong("Pattern: "), pattern[:150] + "..." if len(pattern) > 150 else pattern,
                ], className="mb-2"),
                html.P([
                    html.Strong("Event Meaning: "), event_meaning,
                ], className="mb-0"),
            ])
        ], color="light", className="mb-3")
        
        # Convert to dict for table
        table_data = details_df.to_dict('records')
        
        return table_data, info_card
    
    # Callback for loading new file
    @app.callback(
        Output('file-status', 'children'),
        [Input('load-file-btn', 'n_clicks')],
        prevent_initial_call=True
    )
    def load_new_file(n_clicks):
        if n_clicks:
            return html.Span("⚠️ Please restart the application to load a new file", 
                           style={'color': '#ff6b6b'})
        return ""
    
    return app


def main():
    """Main execution flow."""
    print("="*100)
    print("SEMANTIC LOG CLUSTER VISUALIZER - GUI VERSION")
    print("="*100)
    
    # Step 1: File selection
    print("\n📁 Opening file selection dialog...")
    file_path = select_file()
    
    if not file_path:
        print("❌ No file selected. Exiting...")
        return
    
    print(f"✅ Selected file: {file_path}")
    
    # Step 2: Initialize visualizer
    visualizer = SemanticLogVisualizer(file_path)
    
    # Step 3: Compute embeddings
    visualizer.compute_semantic_embeddings(model_name='all-MiniLM-L6-v2')
    
    # Step 4: Compute 2D layout
    visualizer.compute_2d_layout(n_neighbors=15, min_dist=0.1)
    
    # Step 5: Create and launch web app
    print("\n🚀 Launching interactive dashboard...")
    print("="*100)
    print("📊 Dashboard will open in your default web browser")
    print("🔗 URL: http://127.0.0.1:8050")
    print("⌨️  Press Ctrl+C in terminal to stop the server")
    print("="*100)
    
    app = create_dash_app(visualizer)
    
    # Auto-open browser after 1.5 seconds
    def open_browser():
        webbrowser.open('http://127.0.0.1:8050')
    
    Timer(1.5, open_browser).start()
    
    # Run server
    app.run(debug=False, port=8050)


if __name__ == "__main__":
    main()