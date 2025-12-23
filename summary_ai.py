import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
import re
import textwrap

def generate_cosine_map(file_path):
    print(f"Reading data from: {file_path}...")
    
    # 1. Load Data
    try:
        if file_path.lower().endswith('.xlsx'):
            df = pd.read_excel(file_path, sheet_name=0)
        else:
            try:
                df = pd.read_csv(file_path)
            except UnicodeDecodeError:
                df = pd.read_csv(file_path, encoding='latin1')
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    # Use Meaning Log for AI, fallback to Raw Log
    text_col = 'Meaning Log' if 'Meaning Log' in df.columns else 'Raw Log'
    df[text_col] = df[text_col].fillna('')
    
    # 2. Generate Embeddings (AI)
    print("Loading AI Model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print("Vectorizing logs (Thinking)...")
    # CRITICAL CHANGE: normalize_embeddings=True makes Cosine and Euclidean comparable visually
    embeddings = model.encode(df[text_col].tolist(), show_progress_bar=True, normalize_embeddings=True)

    # 3. DBSCAN Clustering (Cosine Metric)
    print("Discovering natural islands (DBSCAN with Cosine Similarity)...")
    
    # eps=0.15: This means "Similarity must be > 85% to be in the same cluster"
    # metric='cosine': Uses angular distance instead of straight line
    dbscan = DBSCAN(eps=0.15, min_samples=10, metric='cosine')
    df['cluster'] = dbscan.fit_predict(embeddings)
    
    # Count clusters
    n_clusters = len(set(df['cluster'])) - (1 if -1 in df['cluster'] else 0)
    n_noise = list(df['cluster']).count(-1)
    print(f"-> Found {n_clusters} semantic clusters.")
    print(f"-> Identified {n_noise} outliers (Noise).")

    # 4. Dimension Reduction (Map Making)
    print("Projecting to 2D...")
    # t-SNE with 'cosine' metric preserves the angular relationships in the 2D plot
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000, metric='cosine')
    coords = tsne.fit_transform(embeddings)
    df['x'] = coords[:, 0]
    df['y'] = coords[:, 1]

    # 5. Name the Islands
    print("Naming the islands...")
    cluster_labels = {}
    
    for c_id in sorted(df['cluster'].unique()):
        if c_id == -1:
            cluster_labels[c_id] = "ANOMALY (Noise)"
            continue
            
        # Get sample logs
        samples = df[df['cluster'] == c_id][text_col].head(50).tolist()
        
        # Heuristic Naming
        if 'Template ID' in df.columns:
            top_tpl = df[df['cluster'] == c_id]['Template ID'].mode()
            if not top_tpl.empty:
                label = f"Group {c_id} (Tpl {top_tpl[0]})"
            else:
                label = f"Group {c_id}"
        else:
            if 'Service' not in df.columns:
                def get_svc(x):
                    try: return str(x).split()[4].split('[')[0].split(':')[0]
                    except: return "Unknown"
                df['Service'] = df['Raw Log'].apply(get_svc)
            
            top_svc = df[df['cluster'] == c_id]['Service'].mode()[0]
            label = f"Group {c_id} ({top_svc})"
            
        cluster_labels[c_id] = label

    df['Label'] = df['cluster'].map(cluster_labels)

    # 6. Plotting
    print("Generating map...")
    plt.figure(figsize=(16, 10))
    
    # Plot Normal Clusters
    sns.scatterplot(
        data=df[df['cluster'] != -1],
        x='x', y='y',
        hue='Label',
        palette='turbo',
        alpha=0.6,
        s=50,
        legend='full'
    )
    
    # Plot Noise (Anomalies)
    noise = df[df['cluster'] == -1]
    if not noise.empty:
        plt.scatter(
            noise['x'], noise['y'],
            color='black', marker='x', s=100, linewidths=1.5,
            label='ANOMALY (Unclustered)'
        )

    plt.title(f'AI Semantic Map (Cosine Similarity)\nFound {n_clusters} Topics + {n_noise} Anomalies', fontsize=16)
    plt.xlabel('Semantic Dimension 1')
    plt.ylabel('Semantic Dimension 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Discovered Topics')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    output_file = 'cosine_log_map.png'
    plt.savefig(output_file)
    print(f"Done! Map saved to '{output_file}'")
    plt.show()

if __name__ == "__main__":
    print("--- AI Auto-Discovery Tool (Cosine Edition) ---")
    path = input("Enter log file path: ").strip().strip('"')
    if path: generate_cosine_map(path)