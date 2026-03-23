#!/usr/bin/env python3
"""
Generate a shared test corpus for Go/Python BM25 cross-validation.
Outputs JSON files: corpus.json, queries.json, expected_scores.json
"""

import json
import math
import random
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Fixed vocabulary — matches the Go benchmark vocabulary exactly.
all_words = [
    # tech
    "algorithm", "binary", "cache", "database", "encryption", "framework",
    "gateway", "hardware", "interface", "javascript", "kernel", "latency",
    "middleware", "network", "optimization", "protocol", "queue", "runtime",
    "server", "throughput", "unicode", "virtualization", "websocket", "xml",
    "kubernetes", "container", "microservice", "pipeline", "deployment", "cluster",
    "terraform", "ansible", "docker", "monitoring", "observability", "tracing",
    "distributed", "consensus", "replication", "sharding", "partition", "failover",
    "load", "balancer", "proxy", "reverse", "certificate", "authentication",
    "authorization", "token", "session", "middleware", "router", "handler",
    # science
    "hypothesis", "experiment", "observation", "analysis", "conclusion", "variable",
    "control", "sample", "population", "correlation", "causation", "statistical",
    "significance", "deviation", "regression", "distribution", "probability", "theorem",
    "quantum", "relativity", "entropy", "molecule", "catalyst", "reaction",
    "synthesis", "compound", "element", "isotope", "spectrum", "wavelength",
    "frequency", "amplitude", "oscillation", "resonance", "diffraction", "polarization",
    "electron", "proton", "neutron", "photon", "quark", "boson", "fermion",
    "gravity", "acceleration", "velocity", "momentum", "force", "energy", "mass",
    # business
    "revenue", "profit", "margin", "investment", "portfolio", "dividend",
    "acquisition", "merger", "stakeholder", "shareholder", "equity", "valuation",
    "strategy", "competitive", "advantage", "innovation", "disruption", "scalability",
    "market", "segment", "penetration", "expansion", "retention", "acquisition",
    "customer", "engagement", "satisfaction", "loyalty", "brand", "positioning",
    "supply", "chain", "logistics", "procurement", "inventory", "warehouse",
    "forecast", "budget", "quarterly", "annual", "fiscal", "compliance",
    "regulation", "governance", "audit", "transparency", "accountability", "sustainability",
    # common
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could", "should",
    "may", "might", "must", "shall", "can", "need", "dare", "ought", "used",
    "to", "of", "in", "for", "on", "with", "at", "by", "from", "as",
    "into", "through", "during", "before", "after", "above", "below", "between",
    "and", "but", "or", "nor", "not", "so", "yet", "both", "either", "neither",
    "this", "that", "these", "those", "each", "every", "all", "any", "few",
    "more", "most", "other", "some", "such", "no", "only", "own", "same",
    "than", "too", "very", "just", "about", "also", "how", "when", "where",
    "which", "while", "who", "whom", "why", "what", "new", "many", "much",
    "system", "process", "performance", "data", "information", "result", "method",
    "approach", "solution", "problem", "design", "implementation", "application",
    "development", "management", "service", "resource", "model", "framework",
    "structure", "function", "feature", "component", "module", "platform",
]


def generate():
    rng = random.Random(12345)
    num_docs = 20
    corpus_strings = []  # raw strings for Go
    corpus_tokenized = []  # tokenized for Python rank_bm25

    for _ in range(num_docs):
        n_words = 30 + rng.randint(0, 20)  # 30-50 words
        tokens = [rng.choice(all_words) for _ in range(n_words)]
        corpus_strings.append(" ".join(tokens))
        corpus_tokenized.append(tokens)

    queries = [
        ["algorithm", "optimization"],
        ["quantum", "energy", "mass"],
        ["revenue", "profit"],
        ["the", "is", "a"],  # very common terms (low/zero IDF)
        ["kubernetes", "docker", "container"],
    ]

    # Compute expected scores using our own BM25 Okapi implementation
    # to avoid depending on rank_bm25's specific formula.
    # Formula: score = sum over q in query: idf(q) * tf(q,d) / (tf(q,d) + k)
    # where k = k1 * (1 - b + b * docLen / avgDocLen)
    # and idf(q) = log(N / df(q))

    k1 = 1.2
    b = 0.75
    N = num_docs

    # Compute doc frequencies
    doc_freqs = {}
    for tokens in corpus_tokenized:
        seen = set()
        for t in tokens:
            if t not in seen:
                doc_freqs[t] = doc_freqs.get(t, 0) + 1
                seen.add(t)

    # Compute IDF
    idf_map = {}
    for term, df in doc_freqs.items():
        v = math.log(N / df)
        if v > 0:
            idf_map[term] = v

    avg_doc_len = sum(len(t) for t in corpus_tokenized) / N

    # Compute scores for each query
    all_scores = []
    all_top3 = []
    for query in queries:
        scores = [0.0] * num_docs
        for q in query:
            idf = idf_map.get(q, 0.0)
            if idf == 0:
                continue
            for d_idx, tokens in enumerate(corpus_tokenized):
                tf = tokens.count(q)
                doc_len = len(tokens)
                k = k1 * (1 - b + b * doc_len / avg_doc_len)
                scores[d_idx] += idf * (tf / (tf + k))
        all_scores.append(scores)
        # Top 3 indices
        indexed = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        all_top3.append(indexed[:3])

    # Write files
    with open(os.path.join(SCRIPT_DIR, "corpus.json"), "w") as f:
        json.dump(corpus_strings, f, indent=2)

    with open(os.path.join(SCRIPT_DIR, "queries.json"), "w") as f:
        json.dump(queries, f, indent=2)

    with open(os.path.join(SCRIPT_DIR, "expected_okapi.json"), "w") as f:
        json.dump({
            "k1": k1,
            "b": b,
            "scores": all_scores,
            "top3": all_top3,
        }, f, indent=2)

    print(f"Generated {num_docs} docs, {len(queries)} queries")
    print(f"Avg doc length: {avg_doc_len:.1f} words")
    print(f"Unique terms with IDF>0: {len(idf_map)}")


if __name__ == "__main__":
    generate()
