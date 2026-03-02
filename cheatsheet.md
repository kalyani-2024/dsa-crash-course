# 09: DSA Cheatsheet — Complete Quick Reference

Keep this open while practicing. Covers every data structure, every pattern, and every complexity.

> 🔗 **Solutions repo:** [AlgoMaster-io/leetcode-solutions](https://github.com/AlgoMaster-io/leetcode-solutions/tree/main) | **Practice patterns:** [algomaster.io](https://algomaster.io/practice/dsa-patterns)

---

## Time Complexity Quick Reference

```
O(1)       -> Hash lookup, array access, stack push/pop
O(log n)   -> Binary search, balanced BST, heap insert/remove
O(n)       -> Single loop, linear scan, BFS/DFS
O(n log n) -> Merge sort, quick sort, heap sort
O(n^2)     -> Nested loops, brute force pairs
O(2^n)     -> Subsets, recursion without memoization
O(n!)      -> Permutations
```

### Constraint to Complexity Guide

```
n <= 10      -> O(n!) OK         -> Brute force, backtracking
n <= 20      -> O(2^n) OK        -> Bitmask, backtracking
n <= 500     -> O(n^3) OK        -> 3 nested loops
n <= 5,000   -> O(n^2) OK        -> 2 nested loops
n <= 100,000 -> O(n log n) needed -> Sort, binary search, heap
n <= 10^7    -> O(n) needed      -> Single pass, HashMap
n > 10^7     -> O(log n) needed  -> Math, binary search
```

---

## Data Structures at a Glance

| Data Structure | Access | Search | Insert | Delete | Best Use Case |
|---------------|--------|--------|--------|--------|---------------|
| **[Array](https://www.geeksforgeeks.org/array-data-structure/)** | O(1) | O(n) | O(n) | O(n) | Random access, cache-friendly |
| **[String](https://www.geeksforgeeks.org/string-data-structure/)** | O(1) | O(n) | O(n)* | O(n)* | Text processing (*immutable: creates new) |
| **[HashMap](https://www.geeksforgeeks.org/hashing-data-structure/)** | — | O(1)+ | O(1)+ | O(1)+ | Counting, lookup, caching |
| **[HashSet](https://www.geeksforgeeks.org/hashset-in-python/)** | — | O(1)+ | O(1)+ | O(1)+ | Unique elements, membership |
| **[Linked List](https://www.geeksforgeeks.org/data-structures/linked-list/)** | O(n) | O(n) | O(1)++ | O(1)++ | Frequent insert/delete (++at known pos) |
| **[Stack](https://www.geeksforgeeks.org/stack-data-structure/)** | O(n) | O(n) | O(1) | O(1) | Undo, matching, DFS |
| **[Queue](https://www.geeksforgeeks.org/queue-data-structure/)** | O(n) | O(n) | O(1) | O(1) | BFS, task scheduling |
| **[BST](https://www.geeksforgeeks.org/binary-search-tree-data-structure/)** | O(log n) | O(log n) | O(log n) | O(log n) | Ordered data, range queries |
| **[Heap (PQ)](https://www.geeksforgeeks.org/heap-data-structure/)** | O(1)* | O(n) | O(log n) | O(log n) | Top-K, scheduling (*min/max only) |
| **[Trie](https://www.geeksforgeeks.org/trie-insert-and-search/)** | — | O(L) | O(L) | O(L) | Prefix search, autocomplete |
| **[Graph](https://www.geeksforgeeks.org/graph-data-structure-and-algorithms/)** | — | O(V+E) | O(1) | O(V+E) | Networks, paths, cycles |
| **[Union-Find](https://www.geeksforgeeks.org/union-find/)** | — | O(α) | — | — | Connected components, merging |

+Average case | ++At known position | *Min/Max only | L = word length | α is nearly constant

---

## All Patterns — Complete Lookup

### Day 1 and Day 2: Fundamental Patterns

| # | Pattern | Key Idea | When to Use | Template |
|---|---------|----------|-------------|----------|
| 1 | **[Two Pointers](https://www.geeksforgeeks.org/two-pointers-technique/)** | Converge from both ends | Sorted data, pairs, partition | `while L < R` |
| 2 | **[Sliding Window](https://www.geeksforgeeks.org/window-sliding-technique/)** | Expand right, shrink left | Contiguous subarray/substring | `for R: while invalid: L+=1` |
| 3 | **[Prefix Sum](https://www.geeksforgeeks.org/prefix-sum-array-implementation-applications-competitive-programming/)** | Cumulative sums | Subarray sum queries | `prefix[j+1] - prefix[i]` |
| 4 | **[Kadane's](https://www.geeksforgeeks.org/largest-sum-contiguous-subarray/)** | Extend or start fresh | Maximum subarray | `curr = max(num, curr+num)` |
| 5 | **[Char Frequency](https://www.geeksforgeeks.org/print-characters-frequencies-order-occurrence/)** | Count character occurrences | Anagrams, permutations | `Counter(s)` |
| 6 | **[Palindrome Expand](https://www.geeksforgeeks.org/longest-palindromic-substring/)** | Expand from center | Palindromic substrings | `while s[l]==s[r]: l--,r++` |
| 7 | **[HashMap Lookup](https://www.geeksforgeeks.org/hashing-data-structure/)** | O(1) existence check | "Have I seen X?" | `if X in seen` |
| 8 | **[Slow/Fast Ptrs](https://www.geeksforgeeks.org/floyds-cycle-finding-algorithm/)** | Different speeds | Cycle, middle of list | `slow=1step, fast=2steps` |
| 9 | **[Reverse LL](https://www.geeksforgeeks.org/reverse-a-linked-list/)** | 3-pointer swap | Restructuring linked lists | `save, reverse, advance` |
| 10 | **[Stack Matching](https://www.geeksforgeeks.org/check-for-balanced-parentheses-in-an-expression/)** | Push open, pop close | Nesting, brackets | `push/pop with match` |
| 11 | **[Monotonic Stack](https://www.geeksforgeeks.org/introduction-to-monotonic-stack-data-structure-and-algorithm-tutorials/)** | Maintain sorted order | Next greater/smaller | `while violates: pop` |

### Day 3 and Day 4: Advanced Patterns

| # | Pattern | Key Idea | When to Use | Template |
|---|---------|----------|-------------|----------|
| 12 | **[Binary Search](https://www.geeksforgeeks.org/binary-search/)** | Halve search space | Sorted data, answer range | `while lo<=hi: mid` |
| 13 | **[Subsets](https://www.geeksforgeeks.org/backtracking-to-find-all-subsets/)** | Include or exclude | All subsets/combinations | `take; recurse; undo` |
| 14 | **[Permutations](https://www.geeksforgeeks.org/write-a-c-program-to-print-all-permutations-of-a-given-string/)** | Choose from remaining | All orderings | `for x in remaining: try` |
| 15 | **[Tree Traversal](https://www.geeksforgeeks.org/tree-traversals-inorder-preorder-and-postorder/)** | DFS (in/pre/post) + BFS | Explore tree structure | `recurse(left); process; recurse(right)` |
| 16 | **[Recursive Tree](https://www.geeksforgeeks.org/binary-tree-data-structure/)** | Left + right, combine | Tree properties | `return f(solve(L), solve(R))` |
| 17 | **[Heap / Top-K](https://www.geeksforgeeks.org/heap-data-structure/)** | Min/max for streaming | Top-K, merge streams | `heappush/heappop` |
| 18 | **[Trie](https://www.geeksforgeeks.org/trie-insert-and-search/)** | Prefix tree for strings | Prefix match, autocomplete | `node.children[c]` |
| 19 | **[BFS](https://www.geeksforgeeks.org/breadth-first-search-or-bfs-for-a-graph/)** | Level-by-level | Shortest path, levels | `queue + visited` |
| 20 | **[DFS](https://www.geeksforgeeks.org/depth-first-search-or-dfs-for-a-graph/)** | Go deep, backtrack | All paths, cycles | `stack/recursion + visited` |
| 21 | **[Union-Find](https://www.geeksforgeeks.org/union-find/)** | Track/merge groups | Connected components | `find(x), union(a,b)` |
| 22 | **[Greedy](https://www.geeksforgeeks.org/greedy-algorithms/)** | Local optimal choice | Intervals, scheduling | `sort + greedy select` |
| 23 | **[1D DP](https://www.geeksforgeeks.org/dynamic-programming/)** | Linear optimization | Sequence problems | `dp[i] = f(dp[i-1]...)` |
| 24 | **[2D DP](https://www.geeksforgeeks.org/dynamic-programming/)** | Grid/string comparison | Two sequences, grids | `dp[i][j] = f(neighbors)` |
| 25 | **[Knapsack DP](https://www.geeksforgeeks.org/0-1-knapsack-problem-dp-10/)** | Items + capacity | Subset sum, coin change | `dp[i][w] = take/leave` |

---

## Pattern Recognition Flowchart

```
GIVEN A PROBLEM, ASK:

 1. Is the input SORTED or can I SORT it?
    -> Binary Search, Two Pointers, Merge

 2. Does it ask about SUBARRAY / SUBSTRING?
    -> Sliding Window (contiguous), Prefix Sum (sum queries)

 3. Does it ask about FREQUENCY / COUNTING / LOOKUP?
    -> HashMap / HashSet

 4. Does it ask about PALINDROMES?
    -> Expand Around Center, Two Pointers, DP

 5. Does it ask about TOP K / MIN / MAX from streaming data?
    -> Heap (Priority Queue)

 6. Does it ask about VALID SEQUENCE / MATCHING?
    -> Stack

 7. Does it ask about NEXT GREATER / SMALLER?
    -> Monotonic Stack

 8. Does it ask about TREE traversal / properties?
    -> DFS (recursive) or BFS (level-order)

 9. Does it ask about SHORTEST PATH or LEVEL-BY-LEVEL?
    -> BFS (unweighted), Dijkstra (weighted)

10. Does it ask about ALL PATHS / CONNECTIVITY / CYCLES?
    -> DFS (directed), Union-Find (undirected)

11. Does it ask about CONNECTED COMPONENTS / GROUPING?
    -> Union-Find or BFS/DFS

12. Does it ask about OPTIMIZATION (min/max) with choices?
    -> Dynamic Programming

13. Does it ask about ALL COMBINATIONS / SUBSETS / PERMUTATIONS?
    -> Backtracking

14. Does it ask about ORDERING / DEPENDENCIES?
    -> Topological Sort (DFS postorder)

15. Does it ask about SCHEDULING / INTERVALS?
    -> Sort + Greedy, or Heap

16. Does it ask about PREFIX MATCHING / AUTOCOMPLETE?
    -> Trie
```

---

## Top 30 Must-Know Problems

### Easy (10) — Absolute Must-Know

| # | Problem | Pattern | LeetCode | Solution |
|---|---------|---------|----------|----------|
| 1 | Two Sum | HashMap | [LeetCode #1](https://leetcode.com/problems/two-sum/) | [Solution](https://github.com/AlgoMaster-io/leetcode-solutions/tree/main/python) |
| 20 | Valid Parentheses | Stack | [LeetCode #20](https://leetcode.com/problems/valid-parentheses/) | [Solution](https://github.com/AlgoMaster-io/leetcode-solutions/tree/main/python) |
| 21 | Merge Two Sorted Lists | Linked List | [LeetCode #21](https://leetcode.com/problems/merge-two-sorted-lists/) | [Solution](https://github.com/AlgoMaster-io/leetcode-solutions/tree/main/python) |
| 70 | Climbing Stairs | DP | [LeetCode #70](https://leetcode.com/problems/climbing-stairs/) | [Solution](https://github.com/AlgoMaster-io/leetcode-solutions/tree/main/python) |
| 104 | Max Depth Binary Tree | Tree DFS | [LeetCode #104](https://leetcode.com/problems/maximum-depth-of-binary-tree/) | [Solution](https://github.com/AlgoMaster-io/leetcode-solutions/tree/main/python) |
| 121 | Best Time Buy/Sell | Linear Scan | [LeetCode #121](https://leetcode.com/problems/best-time-to-buy-and-sell-stock/) | [Solution](https://github.com/AlgoMaster-io/leetcode-solutions/tree/main/python) |
| 125 | Valid Palindrome | Two Pointers | [LeetCode #125](https://leetcode.com/problems/valid-palindrome/) | [Solution](https://github.com/AlgoMaster-io/leetcode-solutions/tree/main/python) |
| 136 | Single Number | Bit XOR | [LeetCode #136](https://leetcode.com/problems/single-number/) | [Solution](https://github.com/AlgoMaster-io/leetcode-solutions/tree/main/python) |
| 141 | Linked List Cycle | Slow/Fast | [LeetCode #141](https://leetcode.com/problems/linked-list-cycle/) | [Solution](https://github.com/AlgoMaster-io/leetcode-solutions/tree/main/python) |
| 206 | Reverse Linked List | Three Pointers | [LeetCode #206](https://leetcode.com/problems/reverse-linked-list/) | [Solution](https://github.com/AlgoMaster-io/leetcode-solutions/tree/main/python) |

### Medium (15) — Core Interview Questions

| # | Problem | Pattern | LeetCode | Solution |
|---|---------|---------|----------|----------|
| 3 | Longest Substring No Repeat | Sliding Window | [LeetCode #3](https://leetcode.com/problems/longest-substring-without-repeating-characters/) | [Solution](https://github.com/AlgoMaster-io/leetcode-solutions/tree/main/python) |
| 5 | Longest Palindromic Substr | Expand Center | [LeetCode #5](https://leetcode.com/problems/longest-palindromic-substring/) | [Solution](https://github.com/AlgoMaster-io/leetcode-solutions/tree/main/python) |
| 11 | Container With Most Water | Two Pointers | [LeetCode #11](https://leetcode.com/problems/container-with-most-water/) | [Solution](https://github.com/AlgoMaster-io/leetcode-solutions/tree/main/python) |
| 15 | 3Sum | Sort + Two Pointers | [LeetCode #15](https://leetcode.com/problems/3sum/) | [Solution](https://github.com/AlgoMaster-io/leetcode-solutions/tree/main/python) |
| 33 | Search Rotated Array | Binary Search | [LeetCode #33](https://leetcode.com/problems/search-in-rotated-sorted-array/) | [Solution](https://github.com/AlgoMaster-io/leetcode-solutions/tree/main/python) |
| 49 | Group Anagrams | HashMap | [LeetCode #49](https://leetcode.com/problems/group-anagrams/) | [Solution](https://github.com/AlgoMaster-io/leetcode-solutions/tree/main/python) |
| 53 | Maximum Subarray | Kadane's | [LeetCode #53](https://leetcode.com/problems/maximum-subarray/) | [Solution](https://github.com/AlgoMaster-io/leetcode-solutions/tree/main/python) |
| 55 | Jump Game | Greedy | [LeetCode #55](https://leetcode.com/problems/jump-game/) | [Solution](https://github.com/AlgoMaster-io/leetcode-solutions/tree/main/python) |
| 56 | Merge Intervals | Sort + Sweep | [LeetCode #56](https://leetcode.com/problems/merge-intervals/) | [Solution](https://github.com/AlgoMaster-io/leetcode-solutions/tree/main/python) |
| 78 | Subsets | Backtracking | [LeetCode #78](https://leetcode.com/problems/subsets/) | [Solution](https://github.com/AlgoMaster-io/leetcode-solutions/tree/main/python) |
| 102 | Level Order Traversal | Tree BFS | [LeetCode #102](https://leetcode.com/problems/binary-tree-level-order-traversal/) | [Solution](https://github.com/AlgoMaster-io/leetcode-solutions/tree/main/python) |
| 200 | Number of Islands | BFS/DFS | [LeetCode #200](https://leetcode.com/problems/number-of-islands/) | [Solution](https://github.com/AlgoMaster-io/leetcode-solutions/tree/main/python) |
| 207 | Course Schedule | Topo Sort | [LeetCode #207](https://leetcode.com/problems/course-schedule/) | [Solution](https://github.com/AlgoMaster-io/leetcode-solutions/tree/main/python) |
| 236 | Lowest Common Ancestor | Tree DFS | [LeetCode #236](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree/) | [Solution](https://github.com/AlgoMaster-io/leetcode-solutions/tree/main/python) |
| 322 | Coin Change | DP | [LeetCode #322](https://leetcode.com/problems/coin-change/) | [Solution](https://github.com/AlgoMaster-io/leetcode-solutions/tree/main/python) |

### Hard (5) — Stand Out

| # | Problem | Pattern | LeetCode | Solution |
|---|---------|---------|----------|----------|
| 23 | Merge K Sorted Lists | Heap + LL | [LeetCode #23](https://leetcode.com/problems/merge-k-sorted-lists/) | [Solution](https://github.com/AlgoMaster-io/leetcode-solutions/tree/main/python) |
| 42 | Trapping Rain Water | Two Pointers | [LeetCode #42](https://leetcode.com/problems/trapping-rain-water/) | [Solution](https://github.com/AlgoMaster-io/leetcode-solutions/tree/main/python) |
| 76 | Min Window Substring | Sliding Window | [LeetCode #76](https://leetcode.com/problems/minimum-window-substring/) | [Solution](https://github.com/AlgoMaster-io/leetcode-solutions/tree/main/python) |
| 84 | Largest Rectangle | Monotonic Stack | [LeetCode #84](https://leetcode.com/problems/largest-rectangle-in-histogram/) | [Solution](https://github.com/AlgoMaster-io/leetcode-solutions/tree/main/python) |
| 124 | Max Path Sum Tree | Tree DFS | [LeetCode #124](https://leetcode.com/problems/binary-tree-maximum-path-sum/) | [Solution](https://github.com/AlgoMaster-io/leetcode-solutions/tree/main/python) |

---

## Common Python Tricks

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

# String building (O(n) not O(n^2))
result = ''.join(char_list)  # correct
# NOT: result += char         # slow, O(n^2)

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

[Back to Course](README.md) | [Interview Playbook](interview-playbook.md)
