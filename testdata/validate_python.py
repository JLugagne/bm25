#!/usr/bin/env python3
"""
Validate Python rank_bm25 scores against our expected_okapi.json.
This confirms both Go and Python agree on the scoring formula.
"""

import json
import math
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def main():
    with open(os.path.join(SCRIPT_DIR, "corpus.json")) as f:
        corpus_strings = json.load(f)
    with open(os.path.join(SCRIPT_DIR, "queries.json")) as f:
        queries = json.load(f)
    with open(os.path.join(SCRIPT_DIR, "expected_okapi.json")) as f:
        expected = json.load(f)

    k1 = expected["k1"]
    b = expected["b"]
    expected_scores = expected["scores"]
    expected_top3 = expected["top3"]

    # Tokenize
    corpus = [doc.split() for doc in corpus_strings]

    # Our own BM25 Okapi implementation (matching the Go formula exactly)
    N = len(corpus)
    avg_dl = sum(len(d) for d in corpus) / N

    # Document frequencies
    df = {}
    for doc in corpus:
        seen = set()
        for t in doc:
            if t not in seen:
                df[t] = df.get(t, 0) + 1
                seen.add(t)

    # IDF: log(N/df)
    idf = {}
    for term, d in df.items():
        v = math.log(N / d)
        if v > 0:
            idf[term] = v

    tolerance = 1e-10
    passed = 0
    failed = 0

    for qi, query in enumerate(queries):
        scores = [0.0] * N
        for q in query:
            q_idf = idf.get(q, 0.0)
            if q_idf == 0:
                continue
            for di, doc in enumerate(corpus):
                tf = doc.count(q)
                k = k1 * (1 - b + b * len(doc) / avg_dl)
                scores[di] += q_idf * (tf / (tf + k))

        # Compare scores
        for di in range(N):
            exp = expected_scores[qi][di]
            got = scores[di]
            if abs(exp - got) > tolerance:
                print(f"FAIL: query={qi} doc={di}: expected={exp:.16f} got={got:.16f}")
                failed += 1
            else:
                passed += 1

        # Compare top3
        indexed = sorted(range(N), key=lambda i: scores[i], reverse=True)
        top3 = indexed[:3]
        if top3 != expected_top3[qi]:
            print(f"FAIL: query={qi} top3: expected={expected_top3[qi]} got={top3}")
            failed += 1
        else:
            passed += 1

    # Also compare with rank_bm25 library
    try:
        from rank_bm25 import BM25Okapi as RankBM25Okapi
        bm25 = RankBM25Okapi(corpus, k1=k1, b=b)
        print("\n--- rank_bm25 library comparison ---")
        for qi, query in enumerate(queries):
            lib_scores = bm25.get_scores(query)
            our_scores = expected_scores[qi]
            max_diff = max(abs(lib_scores[i] - our_scores[i]) for i in range(N))
            if max_diff > 0.01:
                print(f"  Query {qi}: rank_bm25 DIFFERS (max_diff={max_diff:.6f})")
                print(f"    Note: rank_bm25 uses a different IDF formula: log((N-df+0.5)/(df+0.5))")
                print(f"    Our formula: log(N/df)")
            else:
                print(f"  Query {qi}: rank_bm25 matches within 0.01")
    except ImportError:
        print("\nrank_bm25 not installed, skipping library comparison")

    print(f"\nPython self-validation: {passed} passed, {failed} failed")
    return 1 if failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
