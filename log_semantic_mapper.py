"""
Semantic Log Cluster Visualizer
=================================
A Python-based interactive tool for visualizing Linux log templates as semantic clusters.

Architecture:
1. Data Loading: Reads Excel with two sheets (Log Analysis + Template Summary)
2. Semantic Embedding: Uses sentence-transformers to encode event meanings
3. Dimensionality Reduction: UMAP for 2D projection preserving semantic distance
4. Interactive Visualization: Plotly for clickable blob-based cluster map
5. Detail Panel: Shows filtered logs when clicking a template blob

Dependencies:
    pip install pandas openpyxl sentence-transformers scikit-learn umap-learn plotly numpy
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import umap
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')


class SemanticLogVisualizer:
    """
    Main class for semantic log analysis and visualization.
    
    Attributes:
        log_df: DataFrame containing individual log entries
        template_df: DataFrame containing unique templates with meanings
        embeddings: Semantic embeddings for each template
        coordinates_2d: 2D coordinates for visualization
        model: Sentence transformer model for encoding
    """
    
    def __init__(self, excel_path: str):
        """
        Initialize visualizer and load data.
        
        Args:
            excel_path: Path to Excel file with 'Log Analysis' and 'Template Summary' sheets
        """
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
        """
        Compute semantic embeddings for each template's event meaning.
        
        Args:
            model_name: Name of sentence-transformer model to use
                       'all-MiniLM-L6-v2' - Fast, good quality (default)
                       'all-mpnet-base-v2' - Higher quality, slower
        """
        print(f"🔄 Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        
        print("🔄 Computing semantic embeddings...")
        # Encode all event meanings
        meanings = self.template_df['Event Meaning'].tolist()
        self.embeddings = self.model.encode(
            meanings,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        print(f"✅ Generated embeddings with shape: {self.embeddings.shape}")
        
    def compute_2d_layout(self, n_neighbors: int = 15, min_dist: float = 0.1, random_state: int = 42):
        """
        Use UMAP to project high-dimensional embeddings to 2D while preserving semantic distance.
        
        Args:
            n_neighbors: UMAP parameter - larger values preserve more global structure
            min_dist: UMAP parameter - minimum distance between points
            random_state: Random seed for reproducibility
        """
        if self.embeddings is None:
            raise ValueError("Must compute embeddings first!")
            
        print("🔄 Computing 2D layout with UMAP...")
        
        # UMAP preserves both local and global structure better than t-SNE
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=2,
            metric='cosine',  # Use cosine distance for semantic similarity
            random_state=random_state
        )
        
        self.coordinates_2d = reducer.fit_transform(self.embeddings)
        
        print(f"✅ Generated 2D coordinates with shape: {self.coordinates_2d.shape}")
        
    def create_interactive_visualization(self) -> go.Figure:
        """
        Create interactive Plotly visualization with clickable template blobs.
        
        Returns:
            Plotly Figure object with interactive scatter plot
        """
        if self.coordinates_2d is None:
            raise ValueError("Must compute 2D layout first!")
            
        print("🔄 Creating interactive visualization...")
        
        # Prepare data for visualization
        x_coords = self.coordinates_2d[:, 0]
        y_coords = self.coordinates_2d[:, 1]
        
        # Normalize blob sizes (scale by occurrences)
        sizes = self.template_df['Occurrences'].values
        # Scale to reasonable marker sizes (10 to 60 pixels)
        size_min, size_max = 15, 60
        sizes_normalized = size_min + (sizes - sizes.min()) / (sizes.max() - sizes.min() + 1e-10) * (size_max - size_min)
        
        # Create color mapping based on semantic clusters
        # Use the coordinates themselves to derive colors
        colors = np.arctan2(y_coords, x_coords)  # Angle-based coloring
        
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
        
        # Create the main scatter plot
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
                line=dict(width=1, color='white'),
                opacity=0.8
            ),
            text=hover_texts,
            hovertemplate='%{text}<extra></extra>',
            customdata=self.template_df[['Template ID', 'Event Meaning', 'Occurrences', 'Template Pattern']].values,
            name='Templates'
        ))
        
        # Add template ID labels for clarity
        fig.add_trace(go.Scatter(
            x=x_coords,
            y=y_coords,
            mode='text',
            text=self.template_df['Template ID'].astype(str),
            textfont=dict(size=8, color='white'),
            hoverinfo='skip',
            showlegend=False
        ))
        
        # Update layout
        fig.update_layout(
            title={
                'text': 'Semantic Log Template Cluster Map<br><sub>Blob size = occurrence count | Distance = semantic similarity | Click blobs for details</sub>',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20}
            },
            xaxis=dict(title='Semantic Dimension 1', showgrid=False, zeroline=False),
            yaxis=dict(title='Semantic Dimension 2', showgrid=False, zeroline=False),
            hovermode='closest',
            width=1400,
            height=800,
            plot_bgcolor='#0a0a0a',
            paper_bgcolor='#1a1a1a',
            font=dict(color='white'),
            showlegend=False
        )
        
        print("✅ Visualization created successfully!")
        return fig
    
    def get_template_details(self, template_id: int) -> pd.DataFrame:
        """
        Get all log entries for a specific template.
        
        Args:
            template_id: Template ID to filter by
            
        Returns:
            DataFrame with Raw Log, Meaning Log, and Parameters for the template
        """
        filtered = self.log_df[self.log_df['Template ID'] == template_id].copy()
        
        if len(filtered) == 0:
            return pd.DataFrame(columns=['Raw Log', 'Meaning Log', 'Parameters'])
        
        # Select relevant columns
        result = filtered[['Raw Log', 'Meaning Log', 'Parameters']].reset_index(drop=True)
        return result
    
    def print_template_details(self, template_id: int):
        """
        Pretty print template details to console.
        
        Args:
            template_id: Template ID to display
        """
        # Get template info
        template_info = self.template_df[self.template_df['Template ID'] == template_id]
        
        if len(template_info) == 0:
            print(f"❌ Template ID {template_id} not found!")
            return
        
        template_info = template_info.iloc[0]
        
        print("\n" + "="*100)
        print(f"📋 TEMPLATE ID: {template_id}")
        print("="*100)
        print(f"🔢 Occurrences: {template_info['Occurrences']}")
        print(f"📝 Pattern: {template_info['Template Pattern']}")
        print(f"💡 Event Meaning: {template_info['Event Meaning']}")
        print("="*100)
        
        # Get log entries
        logs = self.get_template_details(template_id)
        
        print(f"\n📊 Showing {len(logs)} log entries:\n")
        
        for idx, row in logs.iterrows():
            print(f"--- Entry {idx + 1} ---")
            print(f"Raw Log: {row['Raw Log']}")
            print(f"Meaning: {row['Meaning Log']}")
            if pd.notna(row['Parameters']):
                print(f"Parameters: {row['Parameters']}")
            print()
    
    def analyze_semantic_distances(self, top_k: int = 5):
        """
        Analyze and print semantic similarity between templates.
        
        Args:
            top_k: Number of most similar templates to show for each
        """
        if self.embeddings is None:
            raise ValueError("Must compute embeddings first!")
            
        print("\n🔍 Computing pairwise semantic similarities...\n")
        
        # Compute cosine similarity matrix
        similarity_matrix = cosine_similarity(self.embeddings)
        
        # For each template, find most similar ones
        for idx, row in self.template_df.iterrows():
            template_id = row['Template ID']
            
            # Get similarities (excluding self)
            sims = similarity_matrix[idx].copy()
            sims[idx] = -1  # Exclude self
            
            # Get top K most similar
            top_indices = np.argsort(sims)[-top_k:][::-1]
            
            print(f"Template {template_id}: {row['Event Meaning'][:80]}...")
            print(f"  Most similar templates:")
            
            for rank, sim_idx in enumerate(top_indices, 1):
                sim_template = self.template_df.iloc[sim_idx]
                similarity = sims[sim_idx]
                print(f"    {rank}. Template {sim_template['Template ID']} (similarity: {similarity:.3f})")
                print(f"       {sim_template['Event Meaning'][:70]}...")
            print()
    
    def save_interactive_html(self, output_path: str = 'log_cluster_map.html'):
        """
        Save the interactive visualization as standalone HTML.
        
        Args:
            output_path: Path to save HTML file
        """
        fig = self.create_interactive_visualization()
        fig.write_html(output_path)
        print(f"💾 Saved interactive visualization to: {output_path}")
        print(f"   Open this file in a web browser to interact with it.")


def main():
    """
    Main execution flow demonstrating the complete pipeline.
    """
    # Configuration
    # Use raw string (r'...') to handle Windows paths with backslashes
    EXCEL_FILE = r'C:\Users\Mathe\Downloads\Phi3-LinuxLogs-TemplateToMeaning\outputs\Linux_20k_clean_analysis_meaning final excel_sorted.xlsx'
    
    # Alternative options:
    # 1. Use forward slashes: 'C:/Users/Mathe/Downloads/...'
    # 2. Use double backslashes: 'C:\\Users\\Mathe\\Downloads\\...'
    # 3. Or just the filename if running from same directory: 'Linux_2k_clean_analysis_meaning final excel_sorted.xlsx'
    
    print("="*100)
    print("SEMANTIC LOG CLUSTER VISUALIZER")
    print("="*100)
    
    # Step 1: Initialize
    visualizer = SemanticLogVisualizer(EXCEL_FILE)
    
    # Step 2: Compute semantic embeddings
    # Options: 'all-MiniLM-L6-v2' (fast), 'all-mpnet-base-v2' (better quality)
    visualizer.compute_semantic_embeddings(model_name='all-MiniLM-L6-v2')
    
    # Step 3: Compute 2D layout
    visualizer.compute_2d_layout(n_neighbors=15, min_dist=0.1)
    
    # Step 4: Analyze semantic relationships (optional)
    visualizer.analyze_semantic_distances(top_k=3)
    
    # Step 5: Create and show visualization
    fig = visualizer.create_interactive_visualization()
    
    # Save as HTML (can be opened in any browser)
    visualizer.save_interactive_html('log_cluster_map.html')
    
    # Show in browser
    fig.show()
    
    # Step 6: Example of accessing template details
    print("\n" + "="*100)
    print("EXAMPLE: Detailed view of Template 10")
    print("="*100)
    visualizer.print_template_details(10)
    
    # Interactive mode (optional)
    print("\n" + "="*100)
    print("INTERACTIVE MODE")
    print("="*100)
    print("Enter a Template ID to see details (or 'quit' to exit):")
    
    while True:
        user_input = input("\nTemplate ID: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            break
            
        try:
            template_id = int(user_input)
            visualizer.print_template_details(template_id)
        except ValueError:
            print("❌ Please enter a valid Template ID number")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n✅ Visualization complete! Check 'log_cluster_map.html' for the interactive map.")


if __name__ == "__main__":
    main()