import json
import numpy as np
from collections import Counter
import math
from transformers import AutoTokenizer
from typing import List, Optional

# Optional imports for semantic metrics
try:
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.feature_extraction.text import TfidfVectorizer
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

class ImbalancePredictor:
    """
    Predicts routing imbalance level for MoE models based on input characteristics.
    
    This predictor estimates imbalance by analyzing token-level and semantic-level patterns
    that correlate with uneven expert load distribution. The score ranges from 0 (balanced)
    to 1 (highly imbalanced).
    
    Note: Actual MoE imbalance is measured as Coefficient of Variation (CV) of expert loads
    (see get_real_imbalance.py). This predictor uses heuristics that correlate with CV:
    - High token repetition/concentration → likely imbalanced routing
    - High semantic similarity → similar queries route to same experts → imbalance
    - Low vocabulary diversity → concentrated token patterns → imbalance
    
    Usage:
        predictor = ImbalancePredictor()
        score = predictor.predict(texts)  # texts is list of strings
        # Use use_semantic=False for faster computation without semantic metrics
    """
    
    def __init__(self, tokenizer_name="mistralai/Mixtral-8x7B-v0.1"):
        """Initialize with tokenizer"""
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    def predict(self, texts, use_semantic=True, return_components=False):
        """
        Compute predicted imbalance score for a list of text samples.
        
        Args:
            texts: List of strings OR list of dicts with 'text' field
            use_semantic: Whether to use semantic similarity metrics (slower but more accurate)
            return_components: If True, return dict with score and components; if False, return float
        
        Returns:
            float or dict: 
                - If return_components=False: Imbalance score in [0, 1], where higher = more imbalanced
                - If return_components=True: Dict with 'imbalance_score' and 'components' keys
        """
        # Handle both raw strings and dicts with 'text' field
        original_texts = texts
        if isinstance(texts[0], dict):
            texts = [sample['text'] for sample in texts]
        
        # Tokenize all texts
        all_tokens = []
        for text in texts:
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            all_tokens.extend(tokens)
        
        if len(all_tokens) == 0:
            if return_components:
                return {
                    'imbalance_score': 0.0,
                    'components': {}
                }
            return 0.0
        
        # Compute component metrics
        metrics = self._compute_metrics(all_tokens, texts if use_semantic else None)
        
        # Aggregate into single score
        score = self._aggregate_score(metrics)
        
        if return_components:
            # Extract individual component contributions
            components = self._extract_components(metrics, score)
            return {
                'imbalance_score': score,
                'components': components,
                'raw_metrics': metrics
            }
        
        return score
    
    def predict_with_breakdown(self, texts, use_semantic=True):
        """
        Compute predicted imbalance score with detailed metric breakdown.
        
        Args:
            texts: List of strings OR list of dicts with 'text' field
            use_semantic: Whether to use semantic similarity metrics (slower but more accurate)
        
        Returns:
            dict: {
                'imbalance_score': float,
                'components': dict of individual component contributions to the score,
                'raw_metrics': dict of raw computed metrics
            }
        """
        # Use predict with return_components=True
        return self.predict(texts, use_semantic=use_semantic, return_components=True)
    
    def _compute_metrics(self, all_tokens, texts=None):
        """Compute all component metrics"""
        total_tokens = len(all_tokens)
        unique_tokens = len(set(all_tokens))
        counter = Counter(all_tokens)
        
        metrics = {
            'total_tokens': total_tokens,
            'unique_tokens': unique_tokens,
            'token_repetition_rate': 1.0 - (unique_tokens / total_tokens),
            'top10_concentration': self._top_k_concentration(all_tokens, counter, k=10),
            'vocabulary_diversity': unique_tokens / total_tokens,
            'unigram_entropy': self._unigram_entropy(counter, total_tokens),
            'gini_coefficient': self._gini_coefficient(counter),
        }
        
        # Add semantic similarity metrics if texts are provided
        if texts is not None and len(texts) > 1:
            semantic_metrics = self._compute_semantic_metrics(texts)
            metrics.update(semantic_metrics)
        
        # Add n-gram concentration metrics
        metrics.update(self._compute_ngram_metrics(all_tokens, counter))
        
        return metrics
    
    def _aggregate_score(self, metrics):
        """
        Combine metrics into single imbalance score.
        
        Weights are based on expected correlation with routing imbalance:
        - High repetition → high imbalance
        - High concentration → high imbalance
        - Low diversity → high imbalance
        - Low entropy → high imbalance
        - High Gini → high imbalance
        - High semantic similarity → high imbalance (similar queries route to same experts)
        - High n-gram concentration → high imbalance
        """
        score = 0.0
        total_weight = 0.0
        
        # Token repetition (higher = more imbalance)
        weight = 0.25
        score += metrics['token_repetition_rate'] * weight
        total_weight += weight
        
        # Top-10 concentration (higher = more imbalance)
        weight = 0.20
        score += metrics['top10_concentration'] * weight
        total_weight += weight
        
        # Inverse of vocabulary diversity (lower diversity = more imbalance)
        weight = 0.15
        score += (1 - metrics['vocabulary_diversity']) * weight
        total_weight += weight
        
        # Inverse of normalized entropy (lower entropy = more imbalance)
        weight = 0.10
        max_entropy = math.log2(max(metrics['unique_tokens'], 2))
        normalized_entropy = metrics['unigram_entropy'] / max_entropy if max_entropy > 0 else 0
        score += (1 - normalized_entropy) * weight
        total_weight += weight
        
        # Gini coefficient (higher = more imbalance)
        weight = 0.10
        score += metrics['gini_coefficient'] * weight
        total_weight += weight
        
        # Bigram concentration (higher = more imbalance)
        if 'bigram_concentration' in metrics:
            weight = 0.10
            score += metrics['bigram_concentration'] * weight
            total_weight += weight
        
        # Semantic similarity (higher = more imbalance - similar queries route to same experts)
        if 'semantic_similarity_mean' in metrics:
            weight = 0.10
            # Normalize semantic similarity to [0, 1] range (cosine similarity is [-1, 1])
            normalized_sim = (metrics['semantic_similarity_mean'] + 1) / 2
            score += normalized_sim * weight
            total_weight += weight
        
        # Normalize by total weight to keep score in [0, 1]
        if total_weight > 0:
            score = score / total_weight
        
        return min(score, 1.0)  # Clamp to [0, 1]
    
    def _top_k_concentration(self, tokens, counter, k=10):
        """Proportion of tokens covered by top-k most frequent"""
        top_k_count = sum([count for _, count in counter.most_common(k)])
        return top_k_count / len(tokens)
    
    def _unigram_entropy(self, counter, total):
        """Shannon entropy of token distribution"""
        entropy = 0.0
        for count in counter.values():
            p = count / total
            entropy -= p * math.log2(p)
        return entropy
    
    def _gini_coefficient(self, counter):
        """Gini coefficient of token frequency distribution"""
        frequencies = sorted(counter.values())
        n = len(frequencies)
        
        if n == 0:
            return 0.0
        
        cumsum = np.cumsum(frequencies)
        gini = (2 * np.sum((np.arange(1, n + 1) * frequencies))) / (n * np.sum(frequencies)) - (n + 1) / n
        
        return gini
    
    def _compute_semantic_metrics(self, texts):
        """
        Compute semantic similarity metrics between texts.
        Higher similarity suggests queries will route to similar experts.
        """
        if not HAS_SKLEARN:
            return {}
        
        try:
            # Use TF-IDF as a lightweight semantic measure
            # (Could be upgraded to use sentence transformers for better accuracy)
            vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1, 2))
            vectors = vectorizer.fit_transform(texts)
            
            # Compute pairwise cosine similarities
            similarity_matrix = cosine_similarity(vectors)
            
            # Get upper triangle (excluding diagonal)
            n = len(texts)
            upper_triangle = []
            for i in range(n):
                for j in range(i + 1, n):
                    upper_triangle.append(similarity_matrix[i, j])
            
            if len(upper_triangle) == 0:
                return {}
            
            return {
                'semantic_similarity_mean': float(np.mean(upper_triangle)),
                'semantic_similarity_std': float(np.std(upper_triangle)),
                'semantic_similarity_max': float(np.max(upper_triangle)),
            }
        except Exception as e:
            # Fallback if TF-IDF fails (e.g., too few texts)
            return {}
    
    def _compute_ngram_metrics(self, all_tokens, counter):
        """
        Compute n-gram concentration metrics.
        High n-gram concentration suggests repetitive patterns that route to same experts.
        """
        if len(all_tokens) < 2:
            return {}
        
        # Compute bigrams
        bigrams = []
        for i in range(len(all_tokens) - 1):
            bigram = (all_tokens[i], all_tokens[i + 1])
            bigrams.append(bigram)
        
        bigram_counter = Counter(bigrams)
        
        if len(bigrams) == 0:
            return {}
        
        # Top-10 bigram concentration
        top10_bigrams = sum([count for _, count in bigram_counter.most_common(10)])
        bigram_concentration = top10_bigrams / len(bigrams)
        
        return {
            'bigram_concentration': bigram_concentration,
            'unique_bigrams': len(bigram_counter),
            'bigram_diversity': len(bigram_counter) / len(bigrams) if len(bigrams) > 0 else 0,
        }
    
    def _extract_components(self, metrics, total_score):
        """
        Extract individual component contributions to the total imbalance score.
        
        Args:
            metrics: Dict of raw metrics
            total_score: Total aggregated imbalance score
        
        Returns:
            dict: Component contributions to the score
        """
        components = {}
        total_weight = 0.0
        
        # Token repetition component
        weight = 0.25
        components['token_repetition'] = {
            'value': metrics['token_repetition_rate'],
            'weight': weight,
            'contribution': metrics['token_repetition_rate'] * weight
        }
        total_weight += weight
        
        # Top-10 concentration component
        weight = 0.20
        components['top10_concentration'] = {
            'value': metrics['top10_concentration'],
            'weight': weight,
            'contribution': metrics['top10_concentration'] * weight
        }
        total_weight += weight
        
        # Vocabulary diversity component (inverse)
        weight = 0.15
        vocab_contribution = (1 - metrics['vocabulary_diversity']) * weight
        components['vocabulary_diversity'] = {
            'value': metrics['vocabulary_diversity'],
            'inverse_value': 1 - metrics['vocabulary_diversity'],
            'weight': weight,
            'contribution': vocab_contribution
        }
        total_weight += weight
        
        # Normalized entropy component (inverse)
        weight = 0.10
        max_entropy = math.log2(max(metrics['unique_tokens'], 2))
        normalized_entropy = metrics['unigram_entropy'] / max_entropy if max_entropy > 0 else 0
        entropy_contribution = (1 - normalized_entropy) * weight
        components['normalized_entropy'] = {
            'value': metrics['unigram_entropy'],
            'normalized_value': normalized_entropy,
            'inverse_normalized_value': 1 - normalized_entropy,
            'weight': weight,
            'contribution': entropy_contribution
        }
        total_weight += weight
        
        # Gini coefficient component
        weight = 0.10
        components['gini_coefficient'] = {
            'value': metrics['gini_coefficient'],
            'weight': weight,
            'contribution': metrics['gini_coefficient'] * weight
        }
        total_weight += weight
        
        # Bigram concentration component
        if 'bigram_concentration' in metrics:
            weight = 0.10
            components['bigram_concentration'] = {
                'value': metrics['bigram_concentration'],
                'weight': weight,
                'contribution': metrics['bigram_concentration'] * weight
            }
            total_weight += weight
        
        # Semantic similarity component
        if 'semantic_similarity_mean' in metrics:
            weight = 0.10
            normalized_sim = (metrics['semantic_similarity_mean'] + 1) / 2
            sim_contribution = normalized_sim * weight
            components['semantic_similarity'] = {
                'value': metrics['semantic_similarity_mean'],
                'normalized_value': normalized_sim,
                'weight': weight,
                'contribution': sim_contribution
            }
            total_weight += weight
        
        # Normalize contributions by total weight and calculate percentages
        if total_weight > 0:
            for comp_name in components:
                components[comp_name]['normalized_contribution'] = components[comp_name]['contribution'] / total_weight
                components[comp_name]['percentage'] = (components[comp_name]['contribution'] / total_weight) * 100
        
        return components


# ============= Convenience Functions =============

def load_dataset(filepath):
    """
    Load a dataset from JSONL file.
    
    Args:
        filepath: Path to .jsonl file
    
    Returns:
        List of dicts with 'text' field
    """
    samples = []
    with open(filepath, 'r') as f:
        for line in f:
            samples.append(json.loads(line))
    return samples


def predict_imbalance(texts, tokenizer_name="mistralai/Mixtral-8x7B-v0.1"):
    """
    Quick function to predict imbalance score.
    
    Args:
        texts: List of strings or list of dicts with 'text' field
        tokenizer_name: HuggingFace model name for tokenizer
    
    Returns:
        float: Imbalance score in [0, 1]
    """
    predictor = ImbalancePredictor(tokenizer_name=tokenizer_name)
    return predictor.predict(texts)


# ============= Example Usage =============

if __name__ == "__main__":
    # Example 1: Direct usage with text list
    predictor = ImbalancePredictor()
    
    texts = [
        "the the the the the",
        "Python programming Python programming",
        "hello world hello world"
    ]
    
    score = predictor.predict(texts)
    print(f"Imbalance score: {score:.3f}")
    
    # Example 2: With breakdown
    result = predictor.predict_with_breakdown(texts)
    print(f"\nDetailed breakdown:")
    print(f"  Imbalance score: {result['imbalance_score']:.3f}")
    print(f"\n  Component contributions:")
    for comp_name, comp_data in result['components'].items():
        print(f"    {comp_name}:")
        print(f"      Value: {comp_data['value']:.4f}")
        print(f"      Weight: {comp_data['weight']:.2f}")
        print(f"      Contribution: {comp_data['contribution']:.4f} ({comp_data['percentage']:.1f}%)")
    
    # Example 3: Using predict with return_components
    result2 = predictor.predict(texts, return_components=True)
    print(f"\n  Alternative: predict() with return_components=True")
    print(f"  Same result: {result2['imbalance_score']:.3f}")
    
    # Example 3: Load from file
    # samples = load_dataset("datasets/high_repetition.jsonl")
    # score = predictor.predict(samples)
    # print(f"Dataset imbalance score: {score:.3f}")