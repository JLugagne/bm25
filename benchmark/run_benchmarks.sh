#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

DIVIDER="$(printf '=%.0s' {1..110})"
SUBDIV="$(printf '-%.0s' {1..110})"

echo "$DIVIDER"
echo "  BM25 Benchmark: Go vs Python"
echo "  $(date)"
echo "$DIVIDER"

# ── Check prerequisites ──
echo
echo "Checking prerequisites..."

if ! command -v go &>/dev/null; then
    echo "ERROR: go not found in PATH" >&2
    exit 1
fi

if ! command -v python3 &>/dev/null; then
    echo "ERROR: python3 not found in PATH" >&2
    exit 1
fi

if ! python3 -c "import rank_bm25" &>/dev/null; then
    echo "ERROR: rank_bm25 Python package not installed (pip install rank-bm25)" >&2
    exit 1
fi

echo "  Go:      $(go version)"
echo "  Python:  $(python3 --version)"
echo "  rank_bm25: $(python3 -c 'import rank_bm25; print(rank_bm25.__version__)' 2>/dev/null || echo 'unknown')"

# ── Generate test corpora if missing ──
if [ ! -f testdata/bench_corpus_50.json ]; then
    echo
    echo "Generating test corpora..."
    python3 testdata/gen_bench_corpus.py
fi

# ── Run Go benchmarks ──
echo
echo "$DIVIDER"
echo "  Go Benchmarks"
echo "$DIVIDER"
echo

GO_OUTPUT=$(go test -bench=. -benchmem -benchtime=2s -count=1 -run='^$' ./... 2>&1)
echo "$GO_OUTPUT"

# ── Run Python benchmarks ──
echo
echo "$DIVIDER"
echo "  Python Benchmarks"
echo "$DIVIDER"
echo

python3 benchmark/bench_python.py

# ── Summary comparison ──
echo
echo "$DIVIDER"
echo "  Summary"
echo "$DIVIDER"
echo
echo "Key comparisons (lower ns/op is better):"
echo "$SUBDIV"
printf "%-50s %15s %15s %10s\n" "Benchmark" "Go (ns/op)" "Python (ns/op)" "Speedup"
echo "$SUBDIV"

# Parse Go results into associative array
declare -A GO_RESULTS

while IFS= read -r line; do
    # Construction benchmarks
    if [[ "$line" =~ BenchmarkConstruction/BM25Okapi/([0-9]+)_docs ]]; then
        size="${BASH_REMATCH[1]}"
        ns=$(echo "$line" | awk '{for(i=1;i<=NF;i++) if($(i+1)=="ns/op") print $i}')
        GO_RESULTS["construction_${size}"]="$ns"
    fi
    # GetScores 50 docs 3 terms
    if [[ "$line" =~ BenchmarkGetScores_50docs_3terms/BM25Okapi[[:space:]] ]]; then
        ns=$(echo "$line" | awk '{for(i=1;i<=NF;i++) if($(i+1)=="ns/op") print $i}')
        GO_RESULTS["scores_50_3"]="$ns"
    fi
    # GetScores 500 docs 3 terms
    if [[ "$line" =~ BenchmarkGetScores_500docs_3terms[[:space:]] ]] && [[ ! "$line" =~ "/" ]]; then
        ns=$(echo "$line" | awk '{for(i=1;i<=NF;i++) if($(i+1)=="ns/op") print $i}')
        GO_RESULTS["scores_500_3"]="$ns"
    fi
    # GetScores 1000 docs 5 terms (Okapi)
    if [[ "$line" =~ BenchmarkGetScores_1000docs_5terms/BM25Okapi[[:space:]] ]]; then
        ns=$(echo "$line" | awk '{for(i=1;i<=NF;i++) if($(i+1)=="ns/op") print $i}')
        GO_RESULTS["scores_1000_5"]="$ns"
    fi
    # Corpus scaling
    if [[ "$line" =~ BenchmarkCorpusScaling/([0-9]+)_docs ]]; then
        size="${BASH_REMATCH[1]}"
        ns=$(echo "$line" | awk '{for(i=1;i<=NF;i++) if($(i+1)=="ns/op") print $i}')
        GO_RESULTS["scaling_${size}"]="$ns"
    fi
    # Query scaling
    if [[ "$line" =~ BenchmarkQueryScaling/([0-9]+)_terms ]]; then
        terms="${BASH_REMATCH[1]}"
        ns=$(echo "$line" | awk '{for(i=1;i<=NF;i++) if($(i+1)=="ns/op") print $i}')
        GO_RESULTS["query_${terms}"]="$ns"
    fi
done <<< "$GO_OUTPUT"

# Run Python benchmarks again, capturing output for parsing
PY_OUTPUT=$(python3 -c "
import json, os, statistics, time
from rank_bm25 import BM25Okapi

TESTDATA = os.path.join('testdata')

def load_corpus(n):
    with open(os.path.join(TESTDATA, f'bench_corpus_{n}.json')) as f:
        return [doc.split() for doc in json.load(f)]

def load_query(n):
    with open(os.path.join(TESTDATA, f'bench_query_{n}.json')) as f:
        return json.load(f)

def bench(fn, target_secs=2.0):
    fn()
    t0 = time.perf_counter_ns()
    fn()
    single = max(time.perf_counter_ns() - t0, 1)
    iterations = max(int(target_secs * 1e9 / single), 5)
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter_ns()
        fn()
        times.append(time.perf_counter_ns() - t0)
    return statistics.mean(times)

# Construction
for size in [50, 100, 500, 1000]:
    c = load_corpus(size)
    avg = bench(lambda c=c: BM25Okapi(c))
    print(f'construction_{size} {avg:.0f}')

# Scores 50/3
c50 = load_corpus(50)
q3 = load_query(3)
bm50 = BM25Okapi(c50)
avg = bench(lambda: bm50.get_scores(q3))
print(f'scores_50_3 {avg:.0f}')

# Scores 500/3
c500 = load_corpus(500)
bm500 = BM25Okapi(c500)
avg = bench(lambda: bm500.get_scores(q3))
print(f'scores_500_3 {avg:.0f}')

# Scores 1000/5
c1000 = load_corpus(1000)
q5 = load_query(5)
bm1000 = BM25Okapi(c1000)
avg = bench(lambda: bm1000.get_scores(q5))
print(f'scores_1000_5 {avg:.0f}')

# Corpus scaling
for size in [50, 100, 500, 1000]:
    c = load_corpus(size)
    bm = BM25Okapi(c)
    avg = bench(lambda bm=bm: bm.get_scores(q3))
    print(f'scaling_{size} {avg:.0f}')

# Query scaling
for n in [1, 3, 5, 10]:
    q = load_query(n)
    avg = bench(lambda q=q: bm500.get_scores(q))
    print(f'query_{n} {avg:.0f}')
")

declare -A PY_RESULTS
while IFS=' ' read -r key val; do
    PY_RESULTS["$key"]="$val"
done <<< "$PY_OUTPUT"

# Print comparison table
print_row() {
    local label="$1" key="$2"
    local go_ns="${GO_RESULTS[$key]:-}"
    local py_ns="${PY_RESULTS[$key]:-}"

    if [[ -n "$go_ns" && -n "$py_ns" ]]; then
        # Remove commas for arithmetic
        local go_clean="${go_ns//,/}"
        local py_clean="${py_ns//,/}"
        # Use awk to handle floating-point division
        local speedup
        speedup=$(awk "BEGIN { printf \"%.1fx\", $py_clean / $go_clean }")
        printf "%-50s %15s %15s %10s\n" "$label" "$go_ns" "$py_ns" "$speedup"
    fi
}

print_row "Construction (50 docs)"     "construction_50"
print_row "Construction (100 docs)"    "construction_100"
print_row "Construction (500 docs)"    "construction_500"
print_row "Construction (1000 docs)"   "construction_1000"
echo "$SUBDIV"
print_row "GetScores (50 docs, 3q)"    "scores_50_3"
print_row "GetScores (500 docs, 3q)"   "scores_500_3"
print_row "GetScores (1000 docs, 5q)"  "scores_1000_5"
echo "$SUBDIV"
print_row "Scaling: 50 docs"           "scaling_50"
print_row "Scaling: 100 docs"          "scaling_100"
print_row "Scaling: 500 docs"          "scaling_500"
print_row "Scaling: 1000 docs"         "scaling_1000"
echo "$SUBDIV"
print_row "Query: 1 term"              "query_1"
print_row "Query: 3 terms"             "query_3"
print_row "Query: 5 terms"             "query_5"
print_row "Query: 10 terms"            "query_10"
echo "$SUBDIV"

echo
echo "Done."
