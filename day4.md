# 07: Day 4 — Tries, Graphs, Union-Find, Greedy, and Dynamic Programming

## Advanced Structures and Algorithm Paradigms

**What this day covers:** [Tries](https://www.geeksforgeeks.org/trie-insert-and-search/) (prefix matching), [Graphs](https://www.geeksforgeeks.org/graph-data-structure-and-algorithms/) (BFS, DFS, topological sort, Dijkstra), [Union-Find](https://www.geeksforgeeks.org/union-find/) (connected components), [Greedy Algorithms](https://www.geeksforgeeks.org/greedy-algorithms/) (intervals, scheduling), and [Dynamic Programming](https://www.geeksforgeeks.org/dynamic-programming/) (1D, 2D, knapsack).

This final day ties everything together with the most advanced material. After completing all four days, you will have covered every major topic that appears in coding interviews.

---

# Tries (Prefix Trees)

## What is a Trie?

A [Trie](https://www.geeksforgeeks.org/trie-insert-and-search/) (pronounced "try") is a tree where each node represents a character, and paths from root to nodes spell out prefixes. It's the ultimate data structure for prefix-based operations.

![Trie](https://upload.wikimedia.org/wikipedia/commons/b/be/Trie_example.svg)

```
Insert: "cat", "car", "card", "dog"

         (root)
        /      \
       c        d
       |        |
       a        o
      / \       |
     t   r      g*
     *   |
         d*

* = end of a valid word
```

### Why Use a Trie?

| Operation | HashMap | Trie |
|-----------|---------|------|
| Exact word lookup | O(L) | O(L) |
| Find all words with prefix "car" | O(n) scan all | O(L + matches) |
| Autocomplete | Expensive | Natural |
| Spell checker | Expensive | Natural |

### Trie Implementation ([LeetCode #208](https://leetcode.com/problems/implement-trie-prefix-tree/)) | [Solution](https://github.com/AlgoMaster-io/leetcode-solutions/tree/main/python)

```python
class TrieNode:
    def __init__(self):
        self.children = {}            # char -> TrieNode
        self.is_end = False           # does a word end here?

class Trie:
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, word):
        node = self.root
        for c in word:
            if c not in node.children:
                node.children[c] = TrieNode()
            node = node.children[c]
        node.is_end = True
    
    def search(self, word):
        node = self._find(word)
        return node is not None and node.is_end
    
    def startsWith(self, prefix):
        return self._find(prefix) is not None
    
    def _find(self, word):
        node = self.root
        for c in word:
            if c not in node.children: return None
            node = node.children[c]
        return node
```

### Word Search II ([LeetCode #212](https://leetcode.com/problems/word-search-ii/)) | [Solution](https://github.com/AlgoMaster-io/leetcode-solutions/tree/main/python)

**The Concept:** Build a Trie from all target words, then DFS through the grid. At each cell, follow the Trie — if the Trie has no branch for a character, prune that search path.

> **Common Pitfalls:**
> 1. Not pruning empty Trie branches after finding a word (causes TLE)
> 2. Forgetting to backtrack the visited cell marker

```python
def findWords(board, words):
    trie = {}
    for w in words:
        node = trie
        for c in w:
            node = node.setdefault(c, {})
        node['#'] = w              # mark end with the full word
    
    R, C = len(board), len(board[0])
    res = set()
    
    def dfs(r, c, node):
        if '#' in node:
            res.add(node['#'])
        if r < 0 or r >= R or c < 0 or c >= C: return
        ch = board[r][c]
        if ch not in node: return
        board[r][c] = '.'          # mark visited
        for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
            dfs(r+dr, c+dc, node[ch])
        board[r][c] = ch           # backtrack
    
    for r in range(R):
        for c in range(C):
            dfs(r, c, trie)
    return list(res)
```

---

# Graphs

## What is a Graph?

A graph models relationships between things. It's the most general data structure — trees, linked lists, and even arrays can be viewed as special cases of graphs.

### Real-World Examples

- **Social network** — people = nodes, friendships = edges
- **GPS/Maps** — intersections = nodes, roads = edges (weighted)
- **Course prerequisites** — courses = nodes, "must take before" = directed edges

### Types

```
Undirected:  A -- B    (friendship: mutual)
Directed:    A -> B    (follow: one-way)
Weighted:    A --5-- B (road with distance 5)
Cyclic:      A -> B -> C -> A  (loops exist)
Acyclic:     A -> B -> C      (no loops; a tree is acyclic)
```

### Representation — Adjacency List

```python
from collections import defaultdict
graph = defaultdict(list)
for u, v in edges:
    graph[u].append(v)
    graph[v].append(u)     # remove for directed
```

Visualize pathfinding: [pathfinding.js.org](https://qiao.github.io/PathFinding.js/visual/)

---

## Pattern 17: BFS — Shortest Path and Level-by-Level

### The Core Idea

> "Explore all neighbors at distance 1 first, then distance 2, then 3... Naturally finds shortest path."

Think of ripples in a pond — expanding outward uniformly.

![BFS Animation](https://upload.wikimedia.org/wikipedia/commons/4/46/Animated_BFS.gif)

> 🔗 **Simulate:** [Pathfinding Visualizer — see BFS/DFS/Dijkstra live](https://qiao.github.io/PathFinding.js/visual/)

```python
def bfs(graph, start):
    visited = {start}
    q = deque([start])
    while q:
        node = q.popleft()
        for nb in graph[node]:
            if nb not in visited:
                visited.add(nb)
                q.append(nb)
```

### Number of Islands ([LeetCode #200](https://leetcode.com/problems/number-of-islands/)) | [Solution](https://github.com/AlgoMaster-io/leetcode-solutions/tree/main/python)

**The Concept:** Iterate the grid; when you find a '1', increment count and BFS/DFS to mark all connected '1's as visited ('0').

> **Common Pitfalls:**
> 1. Forgetting to mark cells as visited before adding to the queue (causes duplicates)
> 2. Using DFS on very large grids (can cause stack overflow — use BFS instead)

```python
def numIslands(grid):
    R, C = len(grid), len(grid[0])
    count = 0
    for r in range(R):
        for c in range(C):
            if grid[r][c] == '1':
                count += 1
                q = deque([(r, c)])
                grid[r][c] = '0'
                while q:
                    row, col = q.popleft()
                    for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
                        nr, nc = row+dr, col+dc
                        if 0<=nr<R and 0<=nc<C and grid[nr][nc]=='1':
                            grid[nr][nc] = '0'
                            q.append((nr, nc))
    return count
```

### Rotting Oranges ([LeetCode #994](https://leetcode.com/problems/rotting-oranges/)) — Multi-source BFS | [Solution](https://github.com/AlgoMaster-io/leetcode-solutions/tree/main/python)

**The Concept:** Start BFS from all rotten oranges simultaneously. Each "level" = 1 minute.

> **Common Pitfalls:**
> 1. Not starting BFS from ALL rotten oranges simultaneously (multi-source BFS)
> 2. Forgetting to check if there are any fresh oranges remaining at the end

```python
def orangesRotting(grid):
    R, C = len(grid), len(grid[0])
    q = deque()
    fresh = 0
    for r in range(R):
        for c in range(C):
            if grid[r][c] == 2: q.append((r, c, 0))
            elif grid[r][c] == 1: fresh += 1
    time = 0
    while q:
        r, c, t = q.popleft()
        for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
            nr, nc = r+dr, c+dc
            if 0<=nr<R and 0<=nc<C and grid[nr][nc] == 1:
                grid[nr][nc] = 2
                fresh -= 1; time = t + 1
                q.append((nr, nc, t + 1))
    return time if fresh == 0 else -1
```

---

## Pattern 18: DFS — All Paths and Cycle Detection

### The Core Idea

> "Go as deep as possible, then backtrack. Use 3 states to detect cycles in directed graphs."

![DFS Animation](https://upload.wikimedia.org/wikipedia/commons/7/7f/Depth-First-Search.gif)

```
Three states:
0 = UNVISITED
1 = VISITING (currently exploring -- on current path)
2 = VISITED  (fully explored)

Reaching a node in state 1 means you have found a CYCLE.
```

### Course Schedule ([LeetCode #207](https://leetcode.com/problems/course-schedule/)) | [Solution](https://github.com/AlgoMaster-io/leetcode-solutions/tree/main/python)

**The Concept:** Model as a directed graph. If there's a cycle, you can't finish all courses. Detect cycles with 3-state DFS.

> **Common Pitfalls:**
> 1. Using only 2 states (visited/not) — need 3 states to distinguish "currently exploring" from "fully explored"
> 2. Building the graph in the wrong direction (prereq → course, not course → prereq)

```python
def canFinish(n, prereqs):
    graph = defaultdict(list)
    for c, p in prereqs: graph[p].append(c)
    state = [0] * n
    def has_cycle(node):
        if state[node] == 1: return True
        if state[node] == 2: return False
        state[node] = 1
        for nb in graph[node]:
            if has_cycle(nb): return True
        state[node] = 2
        return False
    return not any(has_cycle(i) for i in range(n))
```

### Course Schedule II ([LeetCode #210](https://leetcode.com/problems/course-schedule-ii/)) — Topological Sort | [Solution](https://github.com/AlgoMaster-io/leetcode-solutions/tree/main/python)

**The Concept:** A topological ordering = valid course order. Use DFS postorder (add to result when done), then reverse.

```python
def findOrder(n, prereqs):
    graph = defaultdict(list)
    for c, p in prereqs: graph[p].append(c)
    state = [0] * n
    order = []
    def dfs(node):
        if state[node] == 1: return False
        if state[node] == 2: return True
        state[node] = 1
        for nb in graph[node]:
            if not dfs(nb): return False
        state[node] = 2
        order.append(node)
        return True
    if not all(dfs(i) for i in range(n)): return []
    return order[::-1]
```

### Dijkstra's Algorithm — Weighted Shortest Path

**The Concept:** BFS finds shortest path in unweighted graphs. Dijkstra uses a min-heap to always process the closest unvisited node.

![Dijkstra Animation](https://upload.wikimedia.org/wikipedia/commons/5/57/Dijkstra_Animation.gif)

```python
def dijkstra(graph, start, n):
    dist = [float('inf')] * n
    dist[start] = 0
    heap = [(0, start)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]: continue
        for v, w in graph[u]:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                heapq.heappush(heap, (dist[v], v))
    return dist
# O((V + E) log V)
```

---

# Union-Find (Disjoint Set Union)

## What is Union-Find?

[Union-Find](https://www.geeksforgeeks.org/union-find/) tracks groups of connected elements. It answers two questions instantly:
- **Find:** Which group does element X belong to?
- **Union:** Merge two groups together.

Think of social groups at a party. Initially everyone is standalone. When two people become friends, their friend groups merge. Union-Find efficiently tracks who is in whose group.

> 🔗 **Visualize:** [Union-Find on USFCA](https://www.cs.usfca.edu/~galles/visualization/DisjointSets.html)

### Implementation with Path Compression + Union by Rank

```python
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))  # everyone is their own parent
        self.rank = [0] * n
        self.components = n           # track number of groups
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # path compression
        return self.parent[x]
    
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb: return False      # already connected
        if self.rank[ra] < self.rank[rb]: ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]: self.rank[ra] += 1
        self.components -= 1
        return True
    
    def connected(self, a, b):
        return self.find(a) == self.find(b)
```

### When to Use Union-Find

- "Number of connected components"
- "Are X and Y connected?"
- "Merge/connect groups"
- "Detect cycles in undirected graphs"
- Problems where relationships grow over time

### Redundant Connection ([LeetCode #684](https://leetcode.com/problems/redundant-connection/)) | [Solution](https://github.com/AlgoMaster-io/leetcode-solutions/tree/main/python)

**The Concept:** Find the edge that creates a cycle. Add edges one by one — if union returns False (already connected), that edge is redundant.

> **Common Pitfalls:**
> 1. Off-by-one: nodes are 1-indexed but Union-Find is 0-indexed
> 2. Not implementing path compression (causes TLE on large inputs)

```python
def findRedundantConnection(edges):
    uf = UnionFind(len(edges) + 1)
    for u, v in edges:
        if not uf.union(u, v):
            return [u, v]              # this edge created a cycle!
```

### Accounts Merge ([LeetCode #721](https://leetcode.com/problems/accounts-merge/)) | [Solution](https://github.com/AlgoMaster-io/leetcode-solutions/tree/main/python)

**The Concept:** Union-Find to group accounts by shared emails.

```python
def accountsMerge(accounts):
    uf = UnionFind(len(accounts))
    email_to_id = {}
    for i, acc in enumerate(accounts):
        for email in acc[1:]:
            if email in email_to_id:
                uf.union(i, email_to_id[email])
            email_to_id[email] = i
    groups = defaultdict(set)
    for email, i in email_to_id.items():
        groups[uf.find(i)].add(email)
    return [[accounts[i][0]] + sorted(emails) for i, emails in groups.items()]
```

---

# Greedy Algorithms

## What is Greedy?

A [greedy algorithm](https://www.geeksforgeeks.org/greedy-algorithms/) makes the locally optimal choice at each step, hoping it leads to a globally optimal solution. Unlike DP, greedy doesn't reconsider past choices.

> "At each step, take the best available option. Never look back."

### When Does Greedy Work?

Greedy works when the problem has optimal substructure and the greedy choice property (local best leads to global best). It's often used for:

- Interval scheduling (start/end times)
- Activity selection
- Jump/reach problems

---

### Jump Game ([LeetCode #55](https://leetcode.com/problems/jump-game/)) | [Solution](https://github.com/AlgoMaster-io/leetcode-solutions/tree/main/python)

**The Concept:** Track the farthest position you can reach. If you can ever reach the end, return True.

> **Common Pitfalls:**
> 1. Thinking you need to track the exact path (you only need max reach)
> 2. Not checking `if i > farthest` early — means you're at a position you can't reach

```python
def canJump(nums):
    farthest = 0
    for i in range(len(nums)):
        if i > farthest: return False
        farthest = max(farthest, i + nums[i])
    return True
```

### Jump Game II ([LeetCode #45](https://leetcode.com/problems/jump-game-ii/)) | [Solution](https://github.com/AlgoMaster-io/leetcode-solutions/tree/main/python)

**The Concept:** BFS-like approach — each "level" is a jump. Track the current reachable end and the farthest from that level.

```python
def jump(nums):
    jumps = curr_end = farthest = 0
    for i in range(len(nums) - 1):
        farthest = max(farthest, i + nums[i])
        if i == curr_end:
            jumps += 1
            curr_end = farthest
    return jumps
```

### Non-overlapping Intervals ([LeetCode #435](https://leetcode.com/problems/non-overlapping-intervals/)) | [Solution](https://github.com/AlgoMaster-io/leetcode-solutions/tree/main/python)

**The Concept:** Sort by end time. Greedily keep intervals that end earliest.

> **Common Pitfalls:**
> 1. Sorting by start time instead of end time
> 2. Counting intervals to keep instead of intervals to remove

```python
def eraseOverlapIntervals(intervals):
    intervals.sort(key=lambda x: x[1])
    count = 0
    prev_end = float('-inf')
    for s, e in intervals:
        if s >= prev_end:
            prev_end = e
        else:
            count += 1
    return count
```

### Meeting Rooms II ([LeetCode #253](https://leetcode.com/problems/meeting-rooms-ii/)) | [Solution](https://github.com/AlgoMaster-io/leetcode-solutions/tree/main/python)

**The Concept:** Sort meetings by start. Use a min-heap of end times to track active rooms. Reuse a room if a meeting starts after the earliest ending.

> **Common Pitfalls:**
> 1. Not sorting by start time first
> 2. Using a regular list instead of a heap for tracking end times (O(n) vs O(log n))

```python
def minMeetingRooms(intervals):
    intervals.sort()
    heap = []
    for s, e in intervals:
        if heap and heap[0] <= s:
            heapq.heappop(heap)
        heapq.heappush(heap, e)
    return len(heap)
```

### Gas Station ([LeetCode #134](https://leetcode.com/problems/gas-station/)) | [Solution](https://github.com/AlgoMaster-io/leetcode-solutions/tree/main/python)

**The Concept:** If total gas ≥ total cost, a solution exists. Track current tank; whenever it goes negative, restart from the next station.

> **Common Pitfalls:**
> 1. Not checking the global sum first (if `sum(gas) < sum(cost)`, answer is -1)
> 2. Trying to simulate the full circular trip instead of using the greedy restart trick

```python
def canCompleteCircuit(gas, cost):
    if sum(gas) < sum(cost): return -1
    tank = start = 0
    for i in range(len(gas)):
        tank += gas[i] - cost[i]
        if tank < 0:
            start = i + 1
            tank = 0
    return start
```

---

# Dynamic Programming

## What is DP?

DP is the most feared interview topic, but at its core:

> "If you're solving the same subproblem multiple times, solve it once and save the answer."

![DP Fibonacci](https://upload.wikimedia.org/wikipedia/commons/0/06/Fibonacci_dynamic_programming.svg)

### DP = Recursion + Caching

```
Without DP: fib(5) -> fib(3) computed TWICE, fib(2) THREE times -> O(2^n)
With DP:    each fib(i) computed ONCE -> O(n)
```

### The 4-Step DP Recipe

```
1. DEFINE STATE   -> What info describes a subproblem? (index, capacity, etc.)
2. RECURRENCE    -> dp[i] = f(dp[i-1], dp[i-2], ...)
3. BASE CASE     -> dp[0] = ?, dp[1] = ?
4. DIRECTION     -> Fill from base case forward (bottom-up)
```

### How to Know It's DP

```
1. Asks for OPTIMAL (min/max) or COUNT of ways
2. Making a sequence of CHOICES
3. Same subproblems solved REPEATEDLY in brute force
4. GREEDY doesn't work (local != global optimal)
```

---

## Pattern 19: 1D DP — Linear Optimization

### Climbing Stairs ([LeetCode #70](https://leetcode.com/problems/climbing-stairs/)) | [Solution](https://github.com/AlgoMaster-io/leetcode-solutions/tree/main/python)

**The Concept:** At step n, you could have come from step n-1 or n-2. So `dp[n] = dp[n-1] + dp[n-2]` — it's Fibonacci!

> **Common Pitfalls:**
> 1. Not recognizing this IS Fibonacci (many students overcomplicate it)
> 2. Using O(n) space when O(1) is possible (only need last two values)

```python
def climbStairs(n):
    if n <= 2: return n
    a, b = 1, 2
    for _ in range(3, n+1):
        a, b = b, a + b
    return b
```

### House Robber ([LeetCode #198](https://leetcode.com/problems/house-robber/)) | [Solution](https://github.com/AlgoMaster-io/leetcode-solutions/tree/main/python)

**The Concept:** At each house: rob it (value + best from 2 ago) or skip it (best from previous).

> **Common Pitfalls:**
> 1. Thinking you must skip exactly one house between robberies (you can skip multiple)
> 2. Not handling the base case for arrays of length 1 and 2

```python
def rob(nums):
    if len(nums) <= 2: return max(nums)
    a, b = nums[0], max(nums[0], nums[1])
    for i in range(2, len(nums)):
        a, b = b, max(b, a + nums[i])
    return b
```

### Coin Change ([LeetCode #322](https://leetcode.com/problems/coin-change/)) | [Solution](https://github.com/AlgoMaster-io/leetcode-solutions/tree/main/python)

**The Concept:** `dp[amount] = 1 + min(dp[amount - coin])` for each coin.

> **Common Pitfalls:**
> 1. Not initializing dp values to infinity (except dp[0] = 0)
> 2. Returning dp[amount] without checking if it's still infinity (meaning impossible)

```python
def coinChange(coins, amount):
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    for i in range(1, amount + 1):
        for c in coins:
            if c <= i:
                dp[i] = min(dp[i], dp[i-c] + 1)
    return dp[amount] if dp[amount] != float('inf') else -1
```

### Longest Increasing Subsequence ([LeetCode #300](https://leetcode.com/problems/longest-increasing-subsequence/)) | [Solution](https://github.com/AlgoMaster-io/leetcode-solutions/tree/main/python)

**The Concept:** `dp[i]` = length of LIS ending at index i. For each j < i, if `nums[j] < nums[i]`, update `dp[i] = max(dp[i], dp[j] + 1)`.

> **Common Pitfalls:**
> 1. Not initializing all dp[i] to 1 (each element is a subsequence of length 1)
> 2. Returning dp[-1] instead of max(dp) (LIS may not end at the last element)

```python
def lengthOfLIS(nums):
    dp = [1] * len(nums)
    for i in range(1, len(nums)):
        for j in range(i):
            if nums[j] < nums[i]:
                dp[i] = max(dp[i], dp[j] + 1)
    return max(dp)
# O(n^2) -- optimizable to O(n log n) with binary search
```

### Word Break ([LeetCode #139](https://leetcode.com/problems/word-break/)) | [Solution](https://github.com/AlgoMaster-io/leetcode-solutions/tree/main/python)

**The Concept:** `dp[i]` = can string `s[0:i]` be segmented using the dictionary? Check all possible last-word boundaries.

> **Common Pitfalls:**
> 1. Not converting wordDict to a set (O(1) lookup vs O(n))
> 2. Forgetting `dp[0] = True` (empty string is always segmentable)

```python
def wordBreak(s, wordDict):
    words = set(wordDict)
    dp = [False] * (len(s) + 1)
    dp[0] = True
    for i in range(1, len(s) + 1):
        for j in range(i):
            if dp[j] and s[j:i] in words:
                dp[i] = True
                break
    return dp[len(s)]
```

---

## Pattern 20: 2D DP — Grids and Two-Sequence Comparison

### Unique Paths ([LeetCode #62](https://leetcode.com/problems/unique-paths/)) | [Solution](https://github.com/AlgoMaster-io/leetcode-solutions/tree/main/python)

**The Concept:** Each cell can be reached from above or from the left: `dp[i][j] = dp[i-1][j] + dp[i][j-1]`.

```python
def uniquePaths(m, n):
    dp = [[1]*n for _ in range(m)]
    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = dp[i-1][j] + dp[i][j-1]
    return dp[m-1][n-1]
```

### Longest Common Subsequence ([LeetCode #1143](https://leetcode.com/problems/longest-common-subsequence/)) | [Solution](https://github.com/AlgoMaster-io/leetcode-solutions/tree/main/python)

**The Concept:** If characters match, take diagonal + 1. Otherwise, take max of skipping from either string.

**Walkthrough:**
```
     ""  a  b  c  d  e
  ""  0  0  0  0  0  0
  a   0  1  1  1  1  1    -- 'a' matches
  c   0  1  1  2  2  2    -- 'c' matches
  e   0  1  1  2  2  3    -- 'e' matches -> LCS = 3
```

```python
def longestCommonSubsequence(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(1, m+1):
        for j in range(1, n+1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]
```

### Edit Distance ([LeetCode #72](https://leetcode.com/problems/edit-distance/)) | [Solution](https://github.com/AlgoMaster-io/leetcode-solutions/tree/main/python)

**The Concept:** `dp[i][j]` = min operations to convert `word1[:i]` to `word2[:j]`. If characters match, no cost. Else min(insert, delete, replace) + 1.

> **Common Pitfalls:**
> 1. Forgetting to initialize the first row and column (cost of converting to/from empty string)
> 2. Off-by-one: `dp[i][j]` compares `word1[i-1]` and `word2[j-1]`

```python
def minDistance(w1, w2):
    m, n = len(w1), len(w2)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m+1): dp[i][0] = i
    for j in range(n+1): dp[0][j] = j
    for i in range(1, m+1):
        for j in range(1, n+1):
            if w1[i-1] == w2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    return dp[m][n]
```

### Partition Equal Subset Sum ([LeetCode #416](https://leetcode.com/problems/partition-equal-subset-sum/)) | [Solution](https://github.com/AlgoMaster-io/leetcode-solutions/tree/main/python)

**The Concept:** Reduce to: can we find a subset that sums to `total/2`? Use a set to track all reachable sums.

> **Common Pitfalls:**
> 1. Not checking if total is odd first (can't partition an odd sum into two equal halves)
> 2. Using a list instead of a set for tracking reachable sums (O(n) vs O(1) lookup)

```python
def canPartition(nums):
    total = sum(nums)
    if total % 2: return False
    target = total // 2
    dp = {0}
    for n in nums:
        dp = dp | {x + n for x in dp}
    return target in dp
```

### 0/1 Knapsack Pattern

The 0/1 Knapsack is a fundamental DP pattern where you have items with weights and values, and a capacity limit. For each item, you choose to **take it** (add its value, reduce remaining capacity) or **leave it** (keep current best).

> **When to recognize it:** Any problem where you select from a set of items with a constraint (weight, capacity, budget) and want to maximize/minimize a value.
>
> **Examples:** Coin Change, Partition Equal Subset Sum, Target Sum

> **Common Pitfalls:**
> 1. Confusing 0/1 Knapsack (each item used once) with Unbounded Knapsack (unlimited use)
> 2. Iterating items in the wrong order when using 1D space optimization

```python
# 0/1 Knapsack: each item can be taken at most ONCE
# State: dp[i][w] = max value using first i items with capacity w
# Choice: take item i (if it fits) or leave it
def knapsack(weights, values, capacity):
    n = len(weights)
    dp = [[0]*(capacity+1) for _ in range(n+1)]
    for i in range(1, n+1):
        for w in range(capacity+1):
            dp[i][w] = dp[i-1][w]              # LEAVE item i
            if weights[i-1] <= w:
                dp[i][w] = max(dp[i][w],
                    dp[i-1][w-weights[i-1]] + values[i-1])  # TAKE item i
    return dp[n][capacity]
```

---

## DP Decision Guide

```
Single sequence?         -> 1D DP     (House Robber, Climbing Stairs, LIS)
Two strings?             -> 2D DP     (LCS, Edit Distance)
Grid?                    -> 2D DP     (Unique Paths, Min Path Sum)
Items + capacity?        -> Knapsack  (Coin Change, Subset Sum)
Can decompose to choices?-> DP        (if greedy doesn't work)
```

---

# Day 4 Summary — All Advanced Patterns

| # | Pattern | Core Insight | Key Problem |
|---|---------|-------------|-------------|
| 17 | **BFS Graph** | Queue + visited = shortest path | [Islands #200](https://leetcode.com/problems/number-of-islands/), [Rotting Oranges #994](https://leetcode.com/problems/rotting-oranges/) |
| 18 | **DFS Graph** | 3-state cycle detection | [Course Schedule #207](https://leetcode.com/problems/course-schedule/) |
| 19 | **1D DP** | dp[i] = f(previous values) | [Coin Change #322](https://leetcode.com/problems/coin-change/), [LIS #300](https://leetcode.com/problems/longest-increasing-subsequence/) |
| 20 | **2D DP** | dp[i][j] for grids/strings | [LCS #1143](https://leetcode.com/problems/longest-common-subsequence/), [Edit Distance #72](https://leetcode.com/problems/edit-distance/) |
| 21 | **Trie** | Prefix-based string operations | [Word Search II #212](https://leetcode.com/problems/word-search-ii/) |
| 22 | **Union-Find** | Track connected components | [Redundant Connection #684](https://leetcode.com/problems/redundant-connection/) |
| 23 | **Greedy** | Local optimal = global optimal | [Jump Game #55](https://leetcode.com/problems/jump-game/), [Meeting Rooms #253](https://leetcode.com/problems/meeting-rooms-ii/) |

### Practice Problems for Day 4

```
Easy:
  #70   Climbing Stairs

Medium:
  #200  Number of Islands
  #207  Course Schedule
  #55   Jump Game
  #322  Coin Change
  #300  Longest Increasing Subsequence
  #1143 Longest Common Subsequence
  #435  Non-overlapping Intervals
  #684  Redundant Connection

Hard:
  #212  Word Search II
  #72   Edit Distance
```

---

# Complete Course — Quick Pattern Recognition

```
"Find pair with property X"          -> HashMap or Two Pointers
"Longest/shortest subarray"          -> Sliding Window
"Find in sorted data"               -> Binary Search
"Search answer range"               -> Binary Search on Answer
"All subsets/combos/perms"           -> Backtracking
"Cycle in linked list"              -> Slow/Fast Pointers
"Matching brackets/nesting"          -> Stack
"Next greater/smaller"              -> Monotonic Stack
"Level-by-level / shortest path"    -> BFS
"All paths / cycle detection"       -> DFS
"Connected components"              -> Union-Find
"Schedule/select intervals"         -> Greedy (sort by end)
"Min/max with overlapping choices"  -> Dynamic Programming
"Prefix matching / autocomplete"    -> Trie
"Top K / streaming min/max"         -> Heap
```

## Top 25 — If You Only Do These

```
Easy:
  #1    Two Sum                #70   Climbing Stairs
  #20   Valid Parentheses      #104  Max Depth Tree
  #206  Reverse Linked List    #136  Single Number

Medium:
  #3    Longest Substring      #53   Max Subarray
  #5    Longest Palindrome     #15   3Sum
  #33   Search Rotated         #56   Merge Intervals
  #78   Subsets                #102  Level Order
  #200  Number of Islands      #207  Course Schedule
  #236  LCA                    #322  Coin Change
  #300  LIS                    #1143 LCS
  #55   Jump Game

Hard:
  #42   Trapping Rain Water    #84   Largest Rectangle
  #23   Merge K Sorted         #124  Max Path Sum
  #212  Word Search II
```

---

## Keep Going

| Resource | Link | Purpose |
|----------|------|---------|
| **NeetCode Roadmap** | [neetcode.io/roadmap](https://neetcode.io/roadmap) | Structured problem list |
| **Striver A2Z Sheet** | [takeuforward.org](https://takeuforward.org/strivers-a2z-dsa-course/strivers-a2z-dsa-course-sheet-2) | 450+ problems by topic |
| **LeetCode Patterns** | [seanprashad.com/leetcode-patterns](https://seanprashad.com/leetcode-patterns/) | Pattern-based problem list |
| **Visualizations** | [visualgo.net](https://visualgo.net/) | See every algorithm animate |

Consistency beats intensity. Solve 2-3 problems daily and you will crack any interview.

---

*See also: [cheatsheet.md](cheatsheet.md) for quick-reference, [interview-playbook.md](interview-playbook.md) for interview day strategy.*
