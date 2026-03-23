#!/usr/bin/env python3
"""
Generate benchmark corpora of various sizes as JSON files.
Both Go and Python benchmarks load these exact files.
"""

import json
import os
import random

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

all_words = [
    "algorithm", "binary", "cache", "database", "encryption", "framework",
    "gateway", "hardware", "interface", "javascript", "kernel", "latency",
    "middleware", "network", "optimization", "protocol", "queue", "runtime",
    "server", "throughput", "unicode", "virtualization", "websocket", "xml",
    "kubernetes", "container", "microservice", "pipeline", "deployment", "cluster",
    "terraform", "ansible", "docker", "monitoring", "observability", "tracing",
    "distributed", "consensus", "replication", "sharding", "partition", "failover",
    "load", "balancer", "proxy", "reverse", "certificate", "authentication",
    "authorization", "token", "session", "middleware", "router", "handler",
    "hypothesis", "experiment", "observation", "analysis", "conclusion", "variable",
    "control", "sample", "population", "correlation", "causation", "statistical",
    "significance", "deviation", "regression", "distribution", "probability", "theorem",
    "quantum", "relativity", "entropy", "molecule", "catalyst", "reaction",
    "synthesis", "compound", "element", "isotope", "spectrum", "wavelength",
    "frequency", "amplitude", "oscillation", "resonance", "diffraction", "polarization",
    "electron", "proton", "neutron", "photon", "quark", "boson", "fermion",
    "gravity", "acceleration", "velocity", "momentum", "force", "energy", "mass",
    "revenue", "profit", "margin", "investment", "portfolio", "dividend",
    "acquisition", "merger", "stakeholder", "shareholder", "equity", "valuation",
    "strategy", "competitive", "advantage", "innovation", "disruption", "scalability",
    "market", "segment", "penetration", "expansion", "retention", "acquisition",
    "customer", "engagement", "satisfaction", "loyalty", "brand", "positioning",
    "supply", "chain", "logistics", "procurement", "inventory", "warehouse",
    "forecast", "budget", "quarterly", "annual", "fiscal", "compliance",
    "regulation", "governance", "audit", "transparency", "accountability", "sustainability",
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


def gen_corpus(num_docs, seed=42):
    rng = random.Random(seed)
    docs = []
    for _ in range(num_docs):
        n = 300 + rng.randint(0, 100)
        docs.append(" ".join(rng.choice(all_words) for _ in range(n)))
    return docs


def gen_query(n, seed=99):
    rng = random.Random(seed)
    return [rng.choice(all_words) for _ in range(n)]


def main():
    # Corpus sizes for scaling benchmarks
    for size in [50, 100, 500, 1000]:
        corpus = gen_corpus(size)
        path = os.path.join(SCRIPT_DIR, f"bench_corpus_{size}.json")
        with open(path, "w") as f:
            json.dump(corpus, f)
        print(f"  {path}: {size} docs, {sum(len(d.split()) for d in corpus)} total words")

    # Queries
    for n in [1, 3, 5, 10]:
        q = gen_query(n)
        path = os.path.join(SCRIPT_DIR, f"bench_query_{n}.json")
        with open(path, "w") as f:
            json.dump(q, f)
        print(f"  {path}: {q}")


if __name__ == "__main__":
    main()
