# 📋 DSA Cheatsheet — Complete Quick Reference

> **Print this out or keep it open** while practicing. Covers every data structure, every pattern, and every complexity.

---

## 🕐 Time Complexity Quick Reference

```
O(1)       → Hash lookup, array access, stack push/pop
O(log n)   → Binary search, balanced BST, heap insert/remove
O(n)       → Single loop, linear scan, BFS/DFS
O(n log n) → Merge sort, quick sort, heap sort
O(n²)      → Nested loops, brute force pairs
O(2ⁿ)      → Subsets, recursion without memoization
O(n!)      → Permutations
```

### Constraint → Complexity Guide

```
n ≤ 10      → O(n!) OK         → Brute force, backtracking
n ≤ 20      → O(2ⁿ) OK        → Bitmask, backtracking
n ≤ 500     → O(n³) OK         → 3 nested loops
n ≤ 5,000   → O(n²) OK         → 2 nested loops
n ≤ 100,000 → O(n log n) needed → Sort, binary search, heap
n ≤ 10⁷     → O(n) needed      → Single pass, HashMap
n > 10⁷     → O(log n) needed  → Math, binary search
```

---

## 📊 Data Structures at a Glance

| Data Structure | Access | Search | Insert | Delete | Best Use Case |
|---------------|--------|--------|--------|--------|---------------|
| **Array** | O(1) | O(n) | O(n) | O(n) | Random access, cache-friendly |
| **String** | O(1) | O(n) | O(n)* | O(n)* | Text processing (*immutable: creates new) |
| **HashMap** | — | O(1)† | O(1)† | O(1)† | Counting, lookup, caching |
| **HashSet** | — | O(1)† | O(1)† | O(1)† | Unique elements, membership |
| **Linked List** | O(n) | O(n) | O(1)‡ | O(1)‡ | Frequent insert/delete (‡at known pos) |
| **Stack** | O(n) | O(n) | O(1) | O(1) | Undo, matching, DFS |
| **Queue** | O(n) | O(n) | O(1) | O(1) | BFS, task scheduling |
| **BST** | O(log n) | O(log n) | O(log n) | O(log n) | Ordered data, range queries |
| **Heap (PQ)** | O(1)§ | O(n) | O(log n) | O(log n) | Top-K, scheduling (§min/max only) |
| **Trie** | — | O(L) | O(L) | O(L) | Prefix search, autocomplete |
| **Graph** | — | O(V+E) | O(1) | O(V+E) | Networks, paths, cycles |
| **Union-Find** | — | O(α) | — | — | Connected components, merging |

†Average case | ‡At known position | §Min/Max only | L = word length | α ≈ constant

---

## 🧩 All Patterns — Complete Lookup

### Day 1: Fundamental Patterns

| # | Pattern | Key Idea | When to Use | Template |
|---|---------|----------|-------------|----------|
| 1 | **Two Pointers** | Converge from both ends | Sorted data, pairs, partition | `while L < R` |
| 2 | **Sliding Window** | Expand right, shrink left | Contiguous subarray/substring | `for R: while invalid: L+=1` |
| 3 | **Prefix Sum** | Cumulative sums | Subarray sum queries | `prefix[j+1] - prefix[i]` |
| 4 | **Kadane's** | Extend or start fresh | Maximum subarray | `curr = max(num, curr+num)` |
| 5 | **Char Frequency** | Count character occurrences | Anagrams, permutations | `Counter(s)` |
| 6 | **Palindrome Expand** | Expand from center | Palindromic substrings | `while s[l]==s[r]: l--,r++` |
| 7 | **HashMap Lookup** | O(1) existence check | "Have I seen X?" | `if X in seen` |
| 8 | **Slow/Fast Ptrs** | Different speeds | Cycle, middle of list | `slow=1step, fast=2steps` |
| 9 | **Reverse LL** | 3-pointer swap | Restructuring linked lists | `save→reverse→advance` |
| 10 | **Stack Matching** | Push open, pop close | Nesting, brackets | `push/pop with match` |
| 11 | **Monotonic Stack** | Maintain sorted order | Next greater/smaller | `while violates: pop` |
| 12 | **Binary Search** | Halve search space | Sorted data, answer range | `while lo<=hi: mid` |

### Day 2: Advanced Patterns

| # | Pattern | Key Idea | When to Use | Template |
|---|---------|----------|-------------|----------|
| 13 | **Subsets** | Include or exclude | All subsets/combinations | `take; recurse; undo` |
| 14 | **Permutations** | Choose from remaining | All orderings | `for x in remaining: try` |
| 15 | **Tree Traversal** | DFS (in/pre/post) + BFS | Explore tree structure | `recurse(left); process; recurse(right)` |
| 16 | **Recursive Tree** | Left + right → combine | Tree properties | `return f(solve(L), solve(R))` |
| 17 | **Heap / Top-K** | Min/max for streaming | Top-K, merge streams | `heappush/heappop` |
| 18 | **Trie** | Prefix tree for strings | Prefix match, autocomplete | `node.children[c]` |
| 19 | **BFS** | Level-by-level | Shortest path, levels | `queue + visited` |
| 20 | **DFS** | Go deep, backtrack | All paths, cycles | `stack/recursion + visited` |
| 21 | **Union-Find** | Track/merge groups | Connected components | `find(x), union(a,b)` |
| 22 | **Greedy** | Local optimal choice | Intervals, scheduling | `sort + greedy select` |
| 23 | **1D DP** | Linear optimization | Sequence problems | `dp[i] = f(dp[i-1]...)` |
| 24 | **2D DP** | Grid/string comparison | Two sequences, grids | `dp[i][j] = f(neighbors)` |
| 25 | **Knapsack DP** | Items + capacity | Subset sum, coin change | `dp[i][w] = take/leave` |

---

## 🎯 Pattern Recognition Flowchart

```
GIVEN A PROBLEM, ASK:

 1. Is the input SORTED or can I SORT it?
    → Binary Search, Two Pointers, Merge

 2. Does it ask about SUBARRAY / SUBSTRING?
    → Sliding Window (contiguous), Prefix Sum (sum queries)

 3. Does it ask about FREQUENCY / COUNTING / LOOKUP?
    → HashMap / HashSet

 4. Does it ask about PALINDROMES?
    → Expand Around Center, Two Pointers, DP

 5. Does it ask about TOP K / MIN / MAX from streaming data?
    → Heap (Priority Queue)

 6. Does it ask about VALID SEQUENCE / MATCHING?
    → Stack

 7. Does it ask about NEXT GREATER / SMALLER?
    → Monotonic Stack

 8. Does it ask about TREE traversal / properties?
    → DFS (recursive) or BFS (level-order)

 9. Does it ask about SHORTEST PATH or LEVEL-BY-LEVEL?
    → BFS (unweighted), Dijkstra (weighted)

10. Does it ask about ALL PATHS / CONNECTIVITY / CYCLES?
    → DFS (directed), Union-Find (undirected)

11. Does it ask about CONNECTED COMPONENTS / GROUPING?
    → Union-Find or BFS/DFS

12. Does it ask about OPTIMIZATION (min/max) with choices?
    → Dynamic Programming

13. Does it ask about ALL COMBINATIONS / SUBSETS / PERMUTATIONS?
    → Backtracking

14. Does it ask about ORDERING / DEPENDENCIES?
    → Topological Sort (DFS postorder)

15. Does it ask about SCHEDULING / INTERVALS?
    → Sort + Greedy, or Heap

16. Does it ask about PREFIX MATCHING / AUTOCOMPLETE?
    → Trie
```

---

## 🔥 Top 30 Must-Know Problems

### 🟢 Easy (10) — Absolute Must-Know

```
#1    Two Sum                (HashMap)
#20   Valid Parentheses      (Stack)
#21   Merge Two Sorted Lists (Linked List)
#70   Climbing Stairs        (DP)
#104  Max Depth Binary Tree  (Tree DFS)
#121  Best Time Buy/Sell     (Linear Scan)
#125  Valid Palindrome       (Two Pointers)
#136  Single Number          (Bit XOR)
#141  Linked List Cycle      (Slow/Fast)
#206  Reverse Linked List    (Three Pointers)
```

### 🟡 Medium (15) — Core Interview Questions

```
#3    Longest Substring No Repeat  (Sliding Window)
#5    Longest Palindromic Substr   (Expand Center)
#11   Container With Most Water    (Two Pointers)
#15   3Sum                         (Sort + Two Pointers)
#33   Search Rotated Array         (Binary Search)
#49   Group Anagrams               (HashMap)
#53   Maximum Subarray             (Kadane's)
#55   Jump Game                    (Greedy)
#56   Merge Intervals              (Sort + Sweep)
#78   Subsets                      (Backtracking)
#102  Level Order Traversal        (Tree BFS)
#200  Number of Islands            (BFS/DFS)
#207  Course Schedule              (Topo Sort)
#236  Lowest Common Ancestor       (Tree DFS)
#322  Coin Change                  (DP)
```

### 🔴 Hard (5) — Stand Out

```
#23   Merge K Sorted Lists   (Heap + LL)
#42   Trapping Rain Water    (Two Pointers)
#76   Min Window Substring   (Sliding Window)
#84   Largest Rectangle      (Monotonic Stack)
#124  Max Path Sum Tree      (Tree DFS)
```

---

## 📏 Common Python Tricks

```python
# Infinity
float('inf'), float('-inf')

# Swap
a, b = b, a

# List comprehension
squares = [x**2 for x in range(10)]

# defaultdict (no KeyError)
from collections import defaultdict
graph = defaultdict(list)

# Counter
from collections import Counter
freq = Counter([1, 2, 2, 3])  # {2: 2, 1: 1, 3: 1}

# deque (O(1) both ends)
from collections import deque
q = deque([1, 2, 3])
q.appendleft(0); q.popleft()

# heapq (min-heap)
import heapq
heapq.heappush(heap, val)
heapq.heappop(heap)
heapq.nlargest(k, iterable)
# Max-heap trick: negate values
heapq.heappush(heap, -val)

# Sort with key
arr.sort(key=lambda x: x[1])

# Enumerate
for i, val in enumerate(arr):

# Zip
for a, b in zip(arr1, arr2):

# String building (O(n) not O(n²))
result = ''.join(char_list)  # ✅
# NOT: result += char         # ❌ O(n²)

# Ceiling division
import math
math.ceil(a / b)
# Or: (a + b - 1) // b

# Binary — useful for bit manipulation
bin(10)     # '0b1010'
10 & 1      # 0 (even)
10 | 1      # 11
10 ^ 10     # 0 (XOR cancels)
10 >> 1     # 5 (divide by 2)
10 << 1     # 20 (multiply by 2)

# Union-Find quick check
# parent[x] = x means x is a root
```

---

[← Day 2 Practice](day2-practice.md) | [Back to Schedule](README.md) | [Next: Interview Playbook →](interview-playbook.md)
