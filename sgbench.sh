#!/bin/bash
# Multi-prompt parallel benchmark — different prompts per stream
# Usage: ./bench_multi.sh [-v] [N]  (N = 1-8, default 1)

VERBOSE=0
if [ "$1" = "-v" ]; then
  VERBOSE=1
  shift
fi

PARALLEL="${1:-1}"

if [ "$PARALLEL" -gt 8 ] || [ "$PARALLEL" -lt 1 ]; then
  echo "Error: N must be 1-8 (got $PARALLEL)"
  exit 1
fi

VLLM_URL="http://localhost:30000/v1/models"
API="http://localhost:30000/v1/chat/completions"

vllm_model=$(curl -s "$VLLM_URL" | jq -r '.data[0].id')
if [ -z "$vllm_model" ] || [ "$vllm_model" = "null" ]; then
  echo "Error: Could not determine active model from $VLLM_URL"
  exit 1
fi

MODEL="$vllm_model"
echo "Active model: $MODEL"
echo "Parallel streams: $PARALLEL"
echo ""

# ── Prompt pools (8 unique variants per test) ────────────────────────────
qa_prompts=(
  "What are the main differences between TCP and UDP? Be concise."
  "Explain the OSI model layers. What does each layer do? Be succinct."
  "What is the difference between HTTP/1.1 and HTTP/2? List the key improvements. Be brief."
  "Compare synchronous and asynchronous programming. When would you use each? Be concise."
  "What is the CAP theorem? Explain why you can only pick two of consistency, availability, and partition tolerance."
  "Explain how DNS resolution works step by step, from typing a URL to loading the page."
  "What are the differences between OAuth 2.0 and OIDC? When should you use each?"
  "Compare and contrast REST, GraphQL, and gRPC. What are the tradeoffs of each?"
)

code_prompts=(
  "Write a Python function that implements binary search on a sorted list. Include type hints and docstring."
  "Write a Python function that merges two sorted lists into one sorted list. Include type hints and docstring."
  "Write a Python function that finds the longest palindromic substring in a string. Include type hints and docstring."
  "Write a Python function that implements a queue using two stacks. Include type hints and docstring."
  "Write a Python function that implements the quicksort algorithm with random pivot selection. Include type hints and docstring."
  "Write a Python function that flattens a nested dictionary of arbitrary depth. Include type hints and docstring."
  "Write a Python function that generates all valid combinations of N pairs of parentheses. Include type hints and docstring."
  "Write a Python function that implements the LRU cache from scratch with O(1) get and put. Include type hints and docstring."
)

json_prompts=(
  "Generate a JSON array of 10 fictional employees with fields: name, age, department, salary, email, skills (array of 3). Output ONLY valid JSON, no explanation."
  "Generate a JSON object representing a library catalog of 8 books with fields: title, author, isbn, genre, pages, rating, yearPublished, tags (array of 3). Output ONLY valid JSON, no explanation."
  "Generate a JSON array of 6 courses with fields: name, instructor, duration, difficulty, modules (array of 4 strings), prerequisites (array of 3). Output ONLY valid JSON, no explanation."
  "Generate a JSON array of 5 countries with fields: name, capital, population, language, continent. Output ONLY valid JSON, no explanation."
  "Generate a JSON array of 7 cities with fields: name, country, population, area_km2, climate, landmarks (array of 3), and timezone_offset. Output ONLY valid JSON, no explanation."
  "Generate a JSON object representing a project management tool state with 5 tasks. Each task has: id, title, assignee, status, due_date, tags (array of 2). Output ONLY valid JSON, no explanation."
  "Generate a JSON array of 8 restaurants with fields: name, cuisine, rating, price_range, location (object with city and state), phone, and popular_dishes (array of 3). Output ONLY valid JSON, no explanation."
  "Generate a JSON array of 5 movies with fields: title, director, release_year, genre, duration_minutes, imdb_rating, cast (array of 3 names), and streaming_platform. Output ONLY valid JSON, no explanation."
)

math_prompts=(
  "What is 7823 * 4519? Show only the answer."
  "What is 3947 + 8265? Show only the answer."
  "What is 6247 * 3981? Show only the answer."
  "What is 5192 * 7364? Show only the answer."
  "What is 9841 / 27? Show only the answer."
  "What is 6982 / 43? Show only the answer."
  "What is 55555 / 15? Show only the answer."
  "What is 8742 / 26? Show only the answer."
)

longcode_prompts=(
  "Write a complete Python implementation of a red-black tree with insert, delete, search, and in-order traversal. Include all rotation methods."
  "Write a complete Python implementation of a singly-linked list with head/tail, insert at end, delete by value, reverse, and find middle node. Handle edge cases."
  "Write a complete Python implementation of a hash map using separate chaining. Include: put, get, delete, resize on load factor 0.75, hash function."
  "Write a complete Python implementation of a Trie including insert, search, startsWith, delete, and auto-complete. Include unit tests."
  "Write a complete Python implementation of a file system in user space using a dictionary-based virtual FS. Include: mkdir, write, read, ls, cat, rm. Handle relative and absolute paths."
  "Write a complete Python HTTP server from scratch using only the socket module. Include: GET and POST handling, routing, response codes, static file serving, and content-type negotiation."
  "Write a complete Python implementation of a concurrent web scraper. Include: multi-threaded downloads, rate limiting, retry with exponential backoff, robots.txt parsing, and results saving to JSON."
  "Write a complete Python log aggregator. Include: file watcher for multiple log files parsing syslog format, pattern filtering with regex, deduplication, time-windowed aggregation, and console output."
)

# ── Single bench ─────────────────────────────────────────────────────────
bench() {
  local name="$1"
  local prompt="$2"
  local max_tokens="${3:-512}"

  local start_ns=$(date +%s%N)
  local response=$(curl -s "$API" \
    -H "Content-Type: application/json" \
    -d "{
      \"model\": \"$MODEL\",
      \"messages\": [{\"role\": \"user\", \"content\": \"$prompt\"}],
      \"max_tokens\": $max_tokens,
      \"temperature\": 0.0
    }")
  local end_ns=$(date +%s%N)

  local elapsed=$(echo "scale=2; ($end_ns - $start_ns) / 1000000000" | bc)
  local prompt_tokens=$(echo "$response" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['usage']['prompt_tokens'])" 2>/dev/null)
  local completion_tokens=$(echo "$response" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['usage']['completion_tokens'])" 2>/dev/null)

  if [ -z "$completion_tokens" ] || [ "$completion_tokens" = "0" ]; then
    echo "  [$name] FAILED — no completion tokens"
    echo "$response" | python3 -m json.tool 2>/dev/null | head -10
    return 1
  fi

  local toks=$(echo "scale=1; $completion_tokens / $elapsed" | bc)
  echo "  [$name] ${completion_tokens} tokens in ${elapsed}s = ${toks} tok/s (prompt: ${prompt_tokens})"
  return 0
}

# ── Parallel bench ───────────────────────────────────────────────────────
bench_parallel() {
  local name="$1"
  local n="$2"
  shift 2
  local prompt_list=("$@")

  local tmpdir=$(mktemp -d)
  local pids=()
  local fail=0
  local block_start block_end elapsed_total

  block_start=$(date +%s%N)

  for ((i = 0; i < n; i++)); do
    local prompt="${prompt_list[$i]}"
    (
      _bench_start_ns=$(date +%s%N)
      _bench_response=$(curl -s "$API" \
        -H "Content-Type: application/json" \
        -d "{
          \"model\": \"$MODEL\",
          \"messages\": [{\"role\": \"user\", \"content\": \"$prompt\"}],
          \"max_tokens\": 512,
          \"temperature\": 0.0
        }")
      _bench_end_ns=$(date +%s%N)
      export _BENCH_ELAPSED=$(echo "scale=2; ($_bench_end_ns - $_bench_start_ns) / 1000000000" | bc)
      python3 -c "
import json, os, sys
d = json.load(sys.stdin)
pt = d['usage']['prompt_tokens']
ct = d['usage']['completion_tokens']
if ct == 0:
    sys.exit(1)
print(f'{pt} {ct} {os.environ[\"_BENCH_ELAPSED\"]}')
" <<< "$_bench_response" > "$tmpdir/stream_$i" 2>/dev/null
      unset _bench_start_ns _bench_end_ns _bench_response _BENCH_ELAPSED
    ) &
    pids+=($!)
  done

  for pid in "${pids[@]}"; do
    wait "$pid" || fail=$((fail + 1))
  done

  block_end=$(date +%s%N)
  elapsed_total=$(echo "scale=2; ($block_end - $block_start) / 1000000000" | bc)

  local total_completion=0
  local total_prompt=0

  for ((i = 0; i < n; i++)); do
    if [ -f "$tmpdir/stream_$i" ]; then
      read pt ct elapsed < "$tmpdir/stream_$i"
      total_prompt=$((total_prompt + pt))
      total_completion=$((total_completion + ct))
    else
      fail=$((fail + 1))
    fi
  done

  local tokp_s="N/A"
  if [ "$total_completion" -gt 0 ]; then
    tokp_s=$(echo "scale=1; $total_completion / $elapsed_total" | bc)
  fi

  if [ "$VERBOSE" -eq 1 ]; then
    echo "  Stream breakdown:"
    for ((i = 0; i < n; i++)); do
      if [ -f "$tmpdir/stream_$i" ]; then
        read pt ct elapsed < "$tmpdir/stream_$i"
        local stoks
        stoks=$(echo "scale=1; $ct / $elapsed" | bc)
        printf "    [%s/%d] %4d tokens in %5ss = %7s tok/s (prompt: %3d)\n" \
          "$name" "$((i+1))" "$ct" "$elapsed" "$stoks" "$pt"
      else
        printf "    [%s/%d] FAILED\n" "$name" "$((i+1))"
      fi
    done
    echo "  ──"
  fi
  echo "  [$name/1-$n] ${total_completion} tokens in ${elapsed_total}s = ${tokp_s} tok/s (prompt: ${total_prompt} across ${n} streams)"

  rm -rf "$tmpdir"

  if [ "$fail" -gt 0 ]; then
    echo "( ${fail} stream(s) failed )"
  fi
}

# ── Unified test runner ──────────────────────────────────────────────────
run_test() {
  local name="$1"
  shift
  local max_tokens="$1"
  shift
  local prompts=("$@")

  if [ "$PARALLEL" -eq 1 ]; then
    bench "$name" "${prompts[0]}" "$max_tokens"
  else
    bench_parallel "$name" "$PARALLEL" "${prompts[@]}"
  fi
}

# ── Scaling summary ──────────────────────────────────────────────────────
print_scaling() {
  echo ""
  echo "── Scaling summary ───────────────────────────"
  printf "  %-10s %8s %8s %8s %8s %6s %6s\n" "" "N=1x" "N=2x" "N=4x" "N=8x" "2/1x" "4/2x"
  echo "  ──────────────────────────────────────────────────────────"
}

# ── Main benchmark loop ──────────────────────────────────────────────────
local_width=62
echo "╔$(printf '═%.0s' $(seq 1 $local_width))╗"
printf "║%-${local_width}s║\n" "  Parallel Benchmark"
printf "║%-${local_width}s║\n" "  Model: $MODEL"
printf "║%-${local_width}s║\n" "  $(date)"
printf "║%-${local_width}s║\n" "  Streams: $PARALLEL"
echo "╚$(printf '═%.0s' $(seq 1 $local_width))╝"
echo ""

for RUN in 1 2; do
  echo "── Run $RUN/2 ──────────────────────────────────────"

  qa_pool=()
  code_pool=()
  json_pool=()
  math_pool=()
  longcode_pool=()

  for ((i = 0; i < PARALLEL; i++)); do
    idx=$((i % 8))
    qa_pool+=("${qa_prompts[$idx]}")
    code_pool+=("${code_prompts[$idx]}")
    json_pool+=("${json_prompts[$idx]}")
    math_pool+=("${math_prompts[$idx]}")
    longcode_pool+=("${longcode_prompts[$idx]}")
  done

  run_test "Q&A"       256  "${qa_pool[@]}"
  run_test "Code"      512  "${code_pool[@]}"
  run_test "JSON"     1024  "${json_pool[@]}"
  run_test "Math"      64  "${math_pool[@]}"
  run_test "LongCode" 2048  "${longcode_pool[@]}"

  echo ""
done

echo "=== Done ==="
