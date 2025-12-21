import json
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from dataset_metrics import ImbalancePredictor

# ============= Load Domain Queries =============

def load_domain_queries(queries_file="domain_queries.json"):
    """Load domain queries from JSON file"""
    # Try multiple possible paths
    possible_paths = [
        Path(__file__).parent / queries_file,  # Same directory as this file
        Path(queries_file),  # Current working directory
        Path("data_analysis_pipeline") / queries_file,  # From project root
    ]
    
    for queries_path in possible_paths:
        if queries_path.exists():
            with open(queries_path, 'r') as f:
                return json.load(f)
    
    raise FileNotFoundError(f"Could not find {queries_file} in any of: {possible_paths}")

# ============= Generation Functions =============

def generate_high_repetition(n_samples=30, seq_length=50):
    """
    Expected: VERY HIGH imbalance
    Strategy: Repeat same token/phrase many times
    """
    datasets = []
    
    templates = [
        "the " * seq_length,
        "hello world " * (seq_length // 2),
        "Python programming " * (seq_length // 2),
        "1 2 3 4 5 " * (seq_length // 5),
    ]
    
    for i in range(n_samples):
        text = templates[i % len(templates)]
        datasets.append({
            "id": f"high_rep_{i}",
            "text": text.strip(),
            "category": "high_repetition"
        })
    
    return datasets

def generate_single_domain_dataset(domain_queries, domain, n_samples=30):
    """
    Expected: HIGH imbalance
    Strategy: All prompts from same domain (high semantic similarity)
    """
    datasets = []
    queries = domain_queries.get(domain, [])
    
    if not queries:
        return []
    
    # Sample queries from the domain (with replacement if needed)
    for i in range(n_samples):
        query = queries[i % len(queries)]
        datasets.append({
            "id": f"single_{domain}_{i}",
            "text": query,
            "category": f"single_domain_{domain}"
        })
    
    return datasets

def generate_multi_domain_concentrated(domain_queries, n_domains=2, n_samples=30):
    """
    Expected: MEDIUM-HIGH imbalance
    Strategy: Mix a few domains (2-3), creating some concentration
    """
    datasets = []
    all_domains = list(domain_queries.keys())
    
    # Select a small subset of domains
    selected_domains = random.sample(all_domains, min(n_domains, len(all_domains)))
    
    for i in range(n_samples):
        # Alternate between selected domains
        domain = selected_domains[i % len(selected_domains)]
        queries = domain_queries[domain]
        query = queries[i % len(queries)]
        
        datasets.append({
            "id": f"multi_concentrated_{i}",
            "text": query,
            "category": f"multi_domain_{n_domains}"
        })
    
    return datasets

def generate_mixed_domains_per_query(domain_queries, n_domains_per_query=3, n_samples=30):
    """
    Expected: MEDIUM imbalance
    Strategy: Each query combines multiple domains
    """
    datasets = []
    all_domains = list(domain_queries.keys())
    
    for i in range(n_samples):
        selected_domains = random.sample(all_domains, min(n_domains_per_query, len(all_domains)))
        mixed_queries = [
            random.choice(domain_queries[d]) 
            for d in selected_domains
        ]
        mixed_text = " ".join(mixed_queries)
        
        datasets.append({
            "id": f"mixed_per_query_{i}",
            "text": mixed_text,
            "category": f"mixed_{n_domains_per_query}_domains"
        })
    
    return datasets

def generate_diverse_all_domains(domain_queries, n_samples=30):
    """
    Expected: LOW imbalance
    Strategy: Maximize diversity - one query per domain, cycle through all domains
    """
    datasets = []
    all_domains = list(domain_queries.keys())
    
    # Create a pool of queries from all domains
    all_queries = []
    for domain in all_domains:
        all_queries.extend([(domain, q) for q in domain_queries[domain]])
    
    # Sample without replacement for maximum diversity
    sampled = random.sample(all_queries, min(n_samples, len(all_queries)))
    
    for i, (domain, query) in enumerate(sampled):
        datasets.append({
            "id": f"diverse_{i}",
            "text": query,
            "category": f"diverse_all_domains"
        })
    
    return datasets

def generate_repeated_structure(domain_queries, n_samples=30):
    """
    Expected: HIGH imbalance
    Strategy: Same syntactic structure, different content
    """
    datasets = []
    
    # Get queries from various domains
    all_queries = []
    for queries in domain_queries.values():
        all_queries.extend(queries[:5])  # Take first 5 from each domain
    
    template = "Explain the concept of {} in simple terms"
    
    # Extract concepts from queries (simple extraction)
    concepts = []
    for query in all_queries[:20]:
        # Try to extract a key concept (simplified)
        words = query.lower().split()
        if len(words) > 3:
            concepts.append(" ".join(words[3:6]))  # Take middle words
    
    for i in range(n_samples):
        if concepts:
            concept = concepts[i % len(concepts)]
            text = template.format(concept)
        else:
            text = template.format(f"concept_{i}")
        
        datasets.append({
            "id": f"repeated_struct_{i}",
            "text": text,
            "category": "repeated_structure"
        })
    
    return datasets

def generate_domain_clusters(domain_queries, cluster_size=3, n_clusters=5, n_samples=30):
    """
    Expected: MEDIUM-HIGH imbalance
    Strategy: Create clusters of similar domains
    """
    datasets = []
    all_domains = list(domain_queries.keys())
    
    # Create clusters of related domains
    clusters = []
    for i in range(n_clusters):
        start_idx = (i * cluster_size) % len(all_domains)
        cluster = all_domains[start_idx:start_idx + cluster_size]
        if len(cluster) < cluster_size:
            cluster.extend(all_domains[:cluster_size - len(cluster)])
        clusters.append(cluster)
    
    queries_per_cluster = n_samples // len(clusters)
    
    for cluster_idx, cluster_domains in enumerate(clusters):
        for i in range(queries_per_cluster):
            domain = cluster_domains[i % len(cluster_domains)]
            queries = domain_queries[domain]
            query = queries[i % len(queries)]
            
            datasets.append({
                "id": f"cluster_{cluster_idx}_{i}",
                "text": query,
                "category": f"cluster_{cluster_idx}"
            })
    
    # Fill remaining samples
    while len(datasets) < n_samples:
        cluster = random.choice(clusters)
        domain = random.choice(cluster)
        queries = domain_queries[domain]
        query = random.choice(queries)
        datasets.append({
            "id": f"cluster_fill_{len(datasets)}",
            "text": query,
            "category": "cluster_fill"
        })
    
    return datasets[:n_samples]

def generate_varying_concentration(domain_queries, n_samples=30):
    """
    Expected: WIDE RANGE of imbalance
    Strategy: Systematically vary domain concentration from 1 to many
    """
    datasets = []
    all_domains = list(domain_queries.keys())
    
    for i in range(n_samples):
        # Vary concentration: start with single domain, gradually increase
        n_domains = 1 + (i * len(all_domains)) // n_samples
        
        if n_domains == 1:
            # Single domain - high imbalance
            domain = all_domains[i % len(all_domains)]
            queries = domain_queries[domain]
            query = queries[i % len(queries)]
        else:
            # Multiple domains - lower imbalance
            selected_domains = random.sample(all_domains, min(n_domains, len(all_domains)))
            domain = selected_domains[i % len(selected_domains)]
            queries = domain_queries[domain]
            query = queries[i % len(queries)]
        
        datasets.append({
            "id": f"varying_conc_{i}",
            "text": query,
            "category": f"varying_concentration_{n_domains}_domains"
        })
    
    return datasets

# ============= Main Generation =============

def generate_all_datasets(output_dir="datasets", graph_dir="graphs", n_datasets=50):
    """
    Generate datasets with varying imbalance scores and save them.
    
    Args:
        output_dir: Directory to save dataset files
        graph_dir: Directory to save distribution graph
        n_datasets: Number of datasets to generate
    """
    # Load domain queries
    print("Loading domain queries...")
    domain_queries = load_domain_queries()
    print(f"Loaded {len(domain_queries)} domains\n")
    
    # Initialize imbalance predictor
    print("Initializing ImbalancePredictor...")
    predictor = ImbalancePredictor()
    print("Ready!\n")
    
    # Create output directories
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    graph_path = Path(graph_dir)
    graph_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating datasets in: {output_path.absolute()}")
    print(f"Saving graphs to: {graph_path.absolute()}\n")
    
    # Generation strategies designed to create large dispersion
    generation_strategies = [
        ("high_repetition", lambda: generate_high_repetition(n_samples=30)),
        ("single_domain_programming", lambda: generate_single_domain_dataset(domain_queries, "programming", n_samples=30)),
        ("single_domain_science", lambda: generate_single_domain_dataset(domain_queries, "science", n_samples=30)),
        ("single_domain_cooking", lambda: generate_single_domain_dataset(domain_queries, "cooking", n_samples=30)),
        ("single_domain_music", lambda: generate_single_domain_dataset(domain_queries, "music", n_samples=30)),
        ("single_domain_astronomy", lambda: generate_single_domain_dataset(domain_queries, "astronomy", n_samples=30)),
        ("multi_domain_2", lambda: generate_multi_domain_concentrated(domain_queries, n_domains=2, n_samples=30)),
        ("multi_domain_3", lambda: generate_multi_domain_concentrated(domain_queries, n_domains=3, n_samples=30)),
        ("mixed_per_query_3", lambda: generate_mixed_domains_per_query(domain_queries, n_domains_per_query=3, n_samples=30)),
        ("mixed_per_query_5", lambda: generate_mixed_domains_per_query(domain_queries, n_domains_per_query=5, n_samples=30)),
        ("diverse_all", lambda: generate_diverse_all_domains(domain_queries, n_samples=30)),
        ("repeated_structure", lambda: generate_repeated_structure(domain_queries, n_samples=30)),
        ("domain_clusters", lambda: generate_domain_clusters(domain_queries, cluster_size=3, n_clusters=5, n_samples=30)),
        ("varying_concentration", lambda: generate_varying_concentration(domain_queries, n_samples=30)),
    ]
    
    # Generate datasets
    all_datasets = []
    imbalance_scores = []
    dataset_index = 0
    
    print("Generating datasets and computing imbalance scores...")
    print("=" * 70)
    
    # Generate from strategies
    for strategy_name, strategy_func in generation_strategies:
        try:
            dataset = strategy_func()
            if not dataset:
                continue
            
            # Compute imbalance score
            texts = [item['text'] for item in dataset]
            score = predictor.predict(texts, use_semantic=True)
            
            # Format score for filename (4 decimal places)
            score_str = f"{score:.4f}"
            
            # Increment dataset index and format with zero-padding
            dataset_index += 1
            index_str = f"{dataset_index:03d}"  # 3-digit zero-padded index
            
            # Save dataset with index prefix
            filepath = output_path / f"{index_str}_{score_str}.jsonl"
            with open(filepath, 'w') as f:
                for item in dataset:
                    f.write(json.dumps(item) + '\n')
            
            all_datasets.append({
                'index': dataset_index,
                'score': score,
                'score_str': score_str,
                'filepath': str(filepath),
                'strategy': strategy_name,
                'n_samples': len(dataset)
            })
            imbalance_scores.append(score)
            
            print(f"✓ [{dataset_index:3d}] {strategy_name:30s} → Score: {score:.4f} → {filepath.name}")
            
        except Exception as e:
            print(f"✗ {strategy_name:30s} → Error: {e}")
            continue
    
    # Generate additional random combinations to fill up to n_datasets
    print("\nGenerating additional random combinations...")
    print("=" * 70)
    
    while len(all_datasets) < n_datasets:
        # Random strategy: vary number of domains
        n_domains = random.choice([1, 2, 3, 5, 10, 15, 20])
        all_domains = list(domain_queries.keys())
        
        if n_domains == 1:
            # Single domain
            domain = random.choice(all_domains)
            queries = domain_queries[domain]
            dataset = [{
                "id": f"random_single_{len(all_datasets)}_{i}",
                "text": queries[i % len(queries)],
                "category": f"random_single_{domain}"
            } for i in range(30)]
        else:
            # Multiple domains
            selected_domains = random.sample(all_domains, min(n_domains, len(all_domains)))
            dataset = []
            for i in range(30):
                domain = selected_domains[i % len(selected_domains)]
                queries = domain_queries[domain]
                query = queries[i % len(queries)]
                dataset.append({
                    "id": f"random_multi_{len(all_datasets)}_{i}",
                    "text": query,
                    "category": f"random_multi_{n_domains}_domains"
                })
        
        try:
            # Compute imbalance score
            texts = [item['text'] for item in dataset]
            score = predictor.predict(texts, use_semantic=True)
            
            # Format score for filename
            score_str = f"{score:.4f}"
            
            # Increment dataset index and format with zero-padding
            dataset_index += 1
            index_str = f"{dataset_index:03d}"  # 3-digit zero-padded index
            
            # Save dataset with index prefix (overwrite if same score exists)
            filepath = output_path / f"{index_str}_{score_str}.jsonl"
            with open(filepath, 'w') as f:
                for item in dataset:
                    f.write(json.dumps(item) + '\n')
            
            all_datasets.append({
                'index': dataset_index,
                'score': score,
                'score_str': score_str,
                'filepath': str(filepath),
                'strategy': f'random_{n_domains}_domains',
                'n_samples': len(dataset)
            })
            imbalance_scores.append(score)
            
            print(f"✓ [{dataset_index:3d}] random_{n_domains}_domains → Score: {score:.4f} → {filepath.name}")
            
        except Exception as e:
            print(f"✗ random generation → Error: {e}")
            continue
    
    # Sort by score
    all_datasets.sort(key=lambda x: x['score'])
    imbalance_scores = sorted(imbalance_scores)
    
    # Print summary
    print("\n" + "=" * 70)
    print("Dataset Generation Summary:")
    print("=" * 70)
    print(f"Total datasets generated: {len(all_datasets)}")
    print(f"Imbalance score range: {min(imbalance_scores):.4f} - {max(imbalance_scores):.4f}")
    print(f"Imbalance score mean: {np.mean(imbalance_scores):.4f}")
    print(f"Imbalance score std: {np.std(imbalance_scores):.4f}")
    print("=" * 70)
    
    # Create distribution plot
    print("\nCreating imbalance score distribution plot...")
    plt.figure(figsize=(12, 6))
    
    # Histogram
    plt.subplot(1, 2, 1)
    plt.hist(imbalance_scores, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    plt.xlabel('Induced Imbalance Score', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Distribution of Induced Imbalance Scores', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Box plot
    plt.subplot(1, 2, 2)
    plt.boxplot(imbalance_scores, vert=True)
    plt.ylabel('Induced Imbalance Score', fontsize=12)
    plt.title('Box Plot of Imbalance Scores', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot (overwrite existing)
    plot_path = graph_path / "induced_imbalance_distribution.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved distribution plot → {plot_path}")
    
    # Save metadata
    metadata_path = output_path / "dataset_metadata.json"
    metadata = {
        'n_datasets': len(all_datasets),
        'score_range': [float(min(imbalance_scores)), float(max(imbalance_scores))],
        'score_mean': float(np.mean(imbalance_scores)),
        'score_std': float(np.std(imbalance_scores)),
        'datasets': all_datasets
    }
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Saved metadata → {metadata_path}")
    
    return all_datasets, imbalance_scores

if __name__ == "__main__":
    random.seed(42)  # Reproducibility
    np.random.seed(42)
    
    # Generate 50 datasets
    datasets, scores = generate_all_datasets(
        output_dir="../datasets",
        graph_dir="../graphs",
        n_datasets=50
    )
    
    print("\n" + "=" * 70)
    print("Generation complete!")
    print("=" * 70)
