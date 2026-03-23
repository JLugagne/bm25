#!/usr/bin/env python3
"""
BM25 Python benchmark — loads the exact same corpus files as the Go benchmarks.
"""

import json
import os
import statistics
import time

from rank_bm25 import BM25Okapi, BM25L, BM25Plus

TESTDATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "testdata")


def load_corpus(n):
    with open(os.path.join(TESTDATA, f"bench_corpus_{n}.json")) as f:
        return [doc.split() for doc in json.load(f)]


def load_query(n):
    with open(os.path.join(TESTDATA, f"bench_query_{n}.json")) as f:
        return json.load(f)


def bench(name, fn, target_secs=2.0):
    fn()  # warm up
    t0 = time.perf_counter_ns()
    fn()
    single = max(time.perf_counter_ns() - t0, 1)
    iterations = max(int(target_secs * 1e9 / single), 5)

    times = []
    for _ in range(iterations):
        t0 = time.perf_counter_ns()
        fn()
        times.append(time.perf_counter_ns() - t0)

    avg = statistics.mean(times)
    mn = min(times)
    print(f"  {name:<60s} {iterations:>6d} iters  avg {avg:>12,.0f} ns/op  min {mn:>12,.0f} ns/op")
    return avg


def main():
    print("=" * 110)
    print("Python BM25 Benchmark — shared testdata corpus")
    print("=" * 110)

    # --- Construction ---
    print("\n── Construction ──")
    for size in [50, 100, 500, 1000]:
        corpus = load_corpus(size)
        bench(f"BM25Okapi construction ({size} docs)", lambda c=corpus: BM25Okapi(c))

    # --- GetScores: variant comparison (50 docs, 3 terms) ---
    print("\n── GetScores — 50 docs, 3-term query ──")
    c50 = load_corpus(50)
    q3 = load_query(3)
    bm_okapi_50 = BM25Okapi(c50)
    bm_l_50 = BM25L(c50)
    bm_plus_50 = BM25Plus(c50)
    bench("BM25Okapi.get_scores  (50 docs, 3 terms)", lambda: bm_okapi_50.get_scores(q3))
    bench("BM25L.get_scores      (50 docs, 3 terms)", lambda: bm_l_50.get_scores(q3))
    bench("BM25Plus.get_scores   (50 docs, 3 terms)", lambda: bm_plus_50.get_scores(q3))

    # --- GetScores: 500 docs ---
    print("\n── GetScores — 500 docs, 3-term query ──")
    c500 = load_corpus(500)
    bm_okapi_500 = BM25Okapi(c500)
    bench("BM25Okapi.get_scores  (500 docs, 3 terms)", lambda: bm_okapi_500.get_scores(q3))
    bench("BM25Okapi.get_top_n   (500 docs, 3 terms, n=10)", lambda: bm_okapi_500.get_top_n(q3, c500, n=10))

    # --- GetScores: 1000 docs, 5 terms ---
    print("\n── GetScores — 1000 docs, 5-term query ──")
    c1000 = load_corpus(1000)
    q5 = load_query(5)
    bm_okapi_1000 = BM25Okapi(c1000)
    bm_l_1000 = BM25L(c1000)
    bm_plus_1000 = BM25Plus(c1000)
    bench("BM25Okapi.get_scores  (1000 docs, 5 terms)", lambda: bm_okapi_1000.get_scores(q5))
    bench("BM25L.get_scores      (1000 docs, 5 terms)", lambda: bm_l_1000.get_scores(q5))
    bench("BM25Plus.get_scores   (1000 docs, 5 terms)", lambda: bm_plus_1000.get_scores(q5))
    bench("BM25Okapi.get_top_n   (1000 docs, 5 terms, n=10)", lambda: bm_okapi_1000.get_top_n(q5, c1000, n=10))

    # --- Corpus scaling ---
    print("\n── Corpus scaling (BM25Okapi, 3-term query) ──")
    for size in [50, 100, 500, 1000]:
        c = load_corpus(size)
        bm = BM25Okapi(c)
        bench(f"BM25Okapi.get_scores  ({size:>4d} docs, 3 terms)", lambda bm=bm: bm.get_scores(q3))

    # --- Query scaling ---
    print("\n── Query scaling (BM25Okapi, 500 docs) ──")
    c500 = load_corpus(500)
    bm_500 = BM25Okapi(c500)
    for n in [1, 3, 5, 10]:
        q = load_query(n)
        bench(f"BM25Okapi.get_scores  (500 docs, {n:>2d} terms)", lambda q=q: bm_500.get_scores(q))

    print("\n" + "=" * 110)
    print("Done.")


if __name__ == "__main__":
    main()
