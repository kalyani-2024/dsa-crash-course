# 📋 DSA Cheatsheet — Quick Reference Card

> **Print this out or keep it open** while practicing. Covers every pattern, complexity, and data structure.

---

## 🕐 Time Complexity Quick Reference

```
O(1)       → Hash lookup, array access, stack push/pop
O(log n)   → Binary search, balanced BST operations
O(n)       → Single loop, linear scan, BFS/DFS
O(n log n) → Merge sort, quick sort, heap sort
O(n²)      → Nested loops, brute force pairs
O(2ⁿ)      → Subsets, recursion without memoization
O(n!)      → Permutations
```

---

## 📊 Data Structures at a Glance

| Data Structure | Access | Search | Insert | Delete | Use Case |
|---------------|--------|--------|--------|--------|----------|
| **Array** | O(1) | O(n) | O(n) | O(n) | Random access, cache-friendly |
| **Linked List** | O(n) | O(n) | O(1) | O(1) | Frequent insert/delete at head |
| **Stack** | O(n) | O(n) | O(1) | O(1) | Undo, matching, DFS |
| **Queue** | O(n) | O(n) | O(1) | O(1) | BFS, task scheduling |
| **HashMap** | — | O(1)* | O(1)* | O(1)* | Counting, lookup, caching |
| **HashSet** | — | O(1)* | O(1)* | O(1)* | Unique elements, membership |
| **BST** | O(log n) | O(log n) | O(log n) | O(log n) | Ordered data, range queries |
| **Heap (PQ)** | O(1)† | O(n) | O(log n) | O(log n) | Top-K, scheduling, median |
| **Trie** | — | O(L) | O(L) | O(L) | Prefix search, autocomplete |
| **Graph** | — | O(V+E) | O(1) | O(V+E) | Networks, paths, cycles |

*Average case | †Min/Max only

---

## 🧩 All 24 Patterns — Quick Lookup

### Arrays & Strings
| # | Pattern | Key Idea | Template |
|---|---------|----------|----------|
| 1 | **Linear Scan** | Track max/min/count in one pass | `for x in arr: update(x)` |
| 2 | **Prefix Sum** | Pre-compute sums for range queries | `prefix[i] = prefix[i-1] + arr[i]` |
| 3 | **Frequency Map** | Count occurrences with HashMap | `counter[x] += 1` |
| 4 | **Two Pointers (opposite)** | Converge from both ends | `while L < R` |
| 5 | **Two Pointers (same dir)** | Fast/slow or read/write | `slow, fast` |
| 6 | **Sliding Window** | Expand right, shrink left | `for R: while invalid: L += 1` |
| 7 | **Binary Search** | Halve search space | `while lo <= hi: mid = (lo+hi)//2` |
| 8 | **BS on Answer** | Search range of possible answers | `while lo < hi: check(mid)` |

### Recursion & Backtracking
| # | Pattern | Key Idea | Template |
|---|---------|----------|----------|
| 9 | **Subsets** | Include or exclude each element | `take(x); recurse; undo` |
| 10 | **Permutations** | Choose from remaining | `for x in remaining: try(x)` |
| 11 | **Constraint Satisfaction** | Place, validate, backtrack | `if valid: place → recurse → remove` |

### Linked Lists
| # | Pattern | Key Idea | Template |
|---|---------|----------|----------|
| 12 | **Slow/Fast Pointers** | Different speeds detect cycles/middle | `slow=1step, fast=2steps` |
| 13 | **Reverse** | Three-pointer reversal | `prev, curr, next` |

### Stacks & Queues
| # | Pattern | Key Idea | Template |
|---|---------|----------|----------|
| 14 | **Stack Matching** | Push open, pop for close | `push/pop with match check` |
| 15 | **Monotonic Stack** | Maintain increasing/decreasing order | `while stack and violates_order: pop` |

### Trees
| # | Pattern | Key Idea | Template |
|---|---------|----------|----------|
| 16 | **Tree Traversal** | DFS (in/pre/post) or BFS | `traverse(left); process; traverse(right)` |
| 17 | **Recursive Properties** | Compute via left/right subtrees | `return f(left, right)` |
| 18 | **Heap / Top-K** | Min/max heap for streaming data | `heapq.push/pop` |

### Graphs
| # | Pattern | Key Idea | Template |
|---|---------|----------|----------|
| 19 | **BFS** | Level-by-level, shortest path | `queue + visited` |
| 20 | **DFS** | Go deep, explore all paths | `stack/recursion + visited` |

### Dynamic Programming
| # | Pattern | Key Idea | Template |
|---|---------|----------|----------|
| 21 | **1D DP** | Linear sequence optimization | `dp[i] = f(dp[i-1], dp[i-2])` |
| 22 | **2D DP (Grid)** | Grid paths, two variables | `dp[i][j] = f(neighbors)` |
| 23 | **Subsequence DP** | Two string comparison | `dp[i][j] = match/skip` |
| 24 | **Knapsack DP** | Select items with constraint | `dp[i][w] = take/leave` |

---

## 🔥 Top 30 Must-Know Problems (LeetCode #)

### 🟢 Easy (Absolute Must-Know)
```
#1    Two Sum               (HashMap)
#21   Merge Two Sorted Lists (Linked List)
#70   Climbing Stairs        (DP)
#104  Max Depth Binary Tree  (Tree DFS)
#121  Best Time Buy/Sell     (Linear Scan)
#125  Valid Palindrome       (Two Pointers)
#136  Single Number          (Bit XOR)
#141  Linked List Cycle      (Fast/Slow)
#206  Reverse Linked List    (Three Pointers)
#226  Invert Binary Tree     (Tree DFS)
```

### 🟡 Medium (Core Interview Questions)
```
#3    Longest Substring No Repeat  (Sliding Window)
#5    Longest Palindromic Substr   (Expand Center)
#11   Container With Most Water    (Two Pointers)
#15   3Sum                         (Sort + Two Pointers)
#33   Search Rotated Array         (Binary Search)
#49   Group Anagrams               (HashMap)
#53   Maximum Subarray             (Kadane's)
#56   Merge Intervals              (Sort + Sweep)
#102  Level Order Traversal        (Tree BFS)
#128  Longest Consecutive Seq      (HashSet)
#198  House Robber                 (DP)
#200  Number of Islands            (BFS/DFS)
#207  Course Schedule              (Topo Sort)
#215  Kth Largest Element          (Heap)
#236  Lowest Common Ancestor       (Tree DFS)
#322  Coin Change                  (DP)
#560  Subarray Sum Equals K        (Prefix Sum)
```

### 🔴 Hard (Stand Out in Interviews)
```
#4    Median Two Sorted Arrays (Binary Search)
#42   Trapping Rain Water      (Two Pointers)
#76   Min Window Substring     (Sliding Window)
#84   Largest Rectangle Hist   (Monotonic Stack)
#124  Max Path Sum Binary Tree (Tree DFS)
#297  Serialize/Deserialize    (Tree BFS/DFS)
```

---

## 🎯 Pattern Recognition Flowchart

```
GIVEN A PROBLEM, ASK:

1. Is the input SORTED or can I SORT it?
   → Binary Search, Two Pointers, Merge

2. Does it ask about SUBARRAY/SUBSTRING?
   → Sliding Window, Prefix Sum

3. Does it ask about FREQUENCY/COUNTING/LOOKUP?
   → HashMap / HashSet

4. Does it ask about TOP K / MIN / MAX with streaming data?
   → Heap (Priority Queue)

5. Does it ask about VALID SEQUENCE / MATCHING?
   → Stack

6. Does it ask about TREE traversal / properties?
   → DFS (recursive) or BFS (level-order)

7. Does it ask about SHORTEST PATH or LEVEL-BY-LEVEL?
   → BFS

8. Does it ask about ALL PATHS / CONNECTIVITY / CYCLES?
   → DFS

9. Does it ask about OPTIMIZATION (min/max) with choices?
   → Dynamic Programming

10. Does it ask about ALL COMBINATIONS / SUBSETS?
    → Backtracking / Recursion

11. Does it ask about ORDERING / DEPENDENCIES?
    → Topological Sort

12. Does it ask about PREFIX MATCHING / AUTOCOMPLETE?
    → Trie
```

---

## 📏 Common Python Tricks

```python
# Infinity
float('inf'), float('-inf')

# Swap without temp
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
q.appendleft(0)
q.popleft()

# heapq (min-heap)
import heapq
heapq.heappush(heap, val)
heapq.heappop(heap)
heapq.nlargest(k, iterable)

# Sort with key
arr.sort(key=lambda x: x[1])  # Sort by second element

# Enumerate
for i, val in enumerate(arr):

# Zip
for a, b in zip(arr1, arr2):

# Ceiling division
import math
math.ceil(a / b)
# Or: (a + b - 1) // b
```

---

[← Day 2 Practice](day2-practice.md) | [Back to Schedule](README.md) | [Next: Interview Playbook →](interview-playbook.md)
