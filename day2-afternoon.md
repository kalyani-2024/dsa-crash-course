# 🌤️ Day 2 — Afternoon Session (1:30 PM - 5:00 PM)

## Graphs → Dynamic Programming → Greedy → Tries

---

# 🗺️ Part 16: Graphs — BFS, DFS & Shortest Paths (75 min)

## What is a Graph?

A graph is **nodes (vertices)** connected by **edges**. Unlike trees, graphs can have **cycles** and nodes with **multiple parents**.

```
Undirected:          Directed:           Weighted:
  1 --- 2              1 → 2              1 --5-- 2
  |     |              ↓   ↓              |       |
  4 --- 3              4   3              3---2---4
                                            (edge weights)
```

### 🎬 **Visualize it:** [visualgo.net/graphds](https://visualgo.net/en/graphds) — Build graphs interactively
### 🎬 **Visualize it:** [pathfinding.js.org](https://qiao.github.io/PathFinding.js/visual/) — Watch BFS vs DFS vs Dijkstra on a grid

### Graph Representation

```python
# 1. Adjacency List (MOST COMMON — use this)
graph = {
    0: [1, 2],
    1: [0, 3],
    2: [0, 3],
    3: [1, 2]
}
# Or using defaultdict
from collections import defaultdict
graph = defaultdict(list)
for u, v in edges:
    graph[u].append(v)
    graph[v].append(u)  # Remove for directed graph

# 2. Adjacency Matrix (use for dense graphs)
#     0  1  2  3
# 0 [[0, 1, 1, 0],
# 1  [1, 0, 0, 1],
# 2  [1, 0, 0, 1],
# 3  [0, 1, 1, 0]]

# 3. Edge List
edges = [(0,1), (0,2), (1,3), (2,3)]
```

---

## 🧩 Pattern 19: BFS (Breadth-First Search) — Level by Level

**When to use:** Shortest path in unweighted graph, level-order traversal, minimum steps.

```
       Start: 0
       Level 0: [0]
       Level 1: [1, 2]         ← Distance 1 from start
       Level 2: [3, 4]         ← Distance 2 from start

BFS explores in "ripple" pattern — like circles in water 🌊
```

```python
from collections import deque

def bfs(graph, start):
    """BFS — explores level by level. Finds shortest path."""
    visited = {start}
    queue = deque([start])
    
    while queue:
        node = queue.popleft()
        print(node)  # Process node
        
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
# Time: O(V + E) | Space: O(V)
```

### Problem: Number of Islands (LeetCode #200)

```python
def numIslands(grid):
    """Each connected group of '1's is an island."""
    if not grid:
        return 0
    
    rows, cols = len(grid), len(grid[0])
    count = 0
    
    def bfs(r, c):
        queue = deque([(r, c)])
        grid[r][c] = '0'  # Mark visited
        while queue:
            row, col = queue.popleft()
            for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
                nr, nc = row+dr, col+dc
                if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == '1':
                    grid[nr][nc] = '0'
                    queue.append((nr, nc))
    
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == '1':
                bfs(r, c)
                count += 1
    
    return count
# Time: O(m×n) | Space: O(m×n)
```

### Problem: Rotting Oranges (LeetCode #994)
```python
def orangesRotting(grid):
    """Multi-source BFS — start from ALL rotten oranges simultaneously."""
    rows, cols = len(grid), len(grid[0])
    queue = deque()
    fresh = 0
    
    # Find all rotten oranges and count fresh ones
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 2:
                queue.append((r, c, 0))  # (row, col, time)
            elif grid[r][c] == 1:
                fresh += 1
    
    max_time = 0
    while queue:
        r, c, time = queue.popleft()
        for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 1:
                grid[nr][nc] = 2
                fresh -= 1
                max_time = time + 1
                queue.append((nr, nc, time + 1))
    
    return max_time if fresh == 0 else -1
```

---

## 🧩 Pattern 20: DFS (Depth-First Search) — Go Deep, Then Backtrack

**When to use:** Path finding, cycle detection, connected components, topological sort.

```python
def dfs(graph, start, visited=None):
    """DFS — goes as deep as possible before backtracking."""
    if visited is None:
        visited = set()
    
    visited.add(start)
    print(start)  # Process node
    
    for neighbor in graph[start]:
        if neighbor not in visited:
            dfs(graph, neighbor, visited)
# Time: O(V + E) | Space: O(V)
```

### Problem: Clone Graph (LeetCode #133)
```python
def cloneGraph(node):
    if not node:
        return None
    
    cloned = {}
    
    def dfs(original):
        if original in cloned:
            return cloned[original]
        
        copy = Node(original.val)
        cloned[original] = copy
        
        for neighbor in original.neighbors:
            copy.neighbors.append(dfs(neighbor))
        
        return copy
    
    return dfs(node)
```

### Problem: Course Schedule (LeetCode #207) — Cycle Detection / Topological Sort

```python
def canFinish(numCourses, prerequisites):
    """Detect cycle in directed graph using DFS."""
    graph = defaultdict(list)
    for course, prereq in prerequisites:
        graph[prereq].append(course)
    
    # 0=unvisited, 1=visiting (in current path), 2=visited
    state = [0] * numCourses
    
    def has_cycle(node):
        if state[node] == 1:  return True   # Cycle!
        if state[node] == 2:  return False   # Already fully processed
        
        state[node] = 1  # Mark as visiting
        for neighbor in graph[node]:
            if has_cycle(neighbor):
                return True
        state[node] = 2  # Done visiting
        return False
    
    return not any(has_cycle(i) for i in range(numCourses))
```

### Topological Sort (Kahn's Algorithm — BFS)
```python
def topological_sort(numCourses, prerequisites):
    """BFS-based topological sort using indegree."""
    graph = defaultdict(list)
    indegree = [0] * numCourses
    
    for course, prereq in prerequisites:
        graph[prereq].append(course)
        indegree[course] += 1
    
    # Start with nodes having no prerequisites
    queue = deque([i for i in range(numCourses) if indegree[i] == 0])
    order = []
    
    while queue:
        node = queue.popleft()
        order.append(node)
        for neighbor in graph[node]:
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                queue.append(neighbor)
    
    return order if len(order) == numCourses else []  # Empty if cycle
```

### Dijkstra's Algorithm — Shortest Path in Weighted Graph
```python
import heapq

def dijkstra(graph, start, n):
    """Shortest distance from start to all other nodes."""
    dist = [float('inf')] * n
    dist[start] = 0
    heap = [(0, start)]  # (distance, node)
    
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue  # Already found shorter path
        
        for v, w in graph[u]:  # (neighbor, weight)
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                heapq.heappush(heap, (dist[v], v))
    
    return dist
# Time: O((V+E) log V) | Space: O(V)
```

### 🎬 **Visualize Dijkstra:** [visualgo.net/sssp](https://visualgo.net/en/sssp) — Step through Dijkstra's algorithm

---

## Union-Find (Disjoint Set Union) — Quick Connected Components

```python
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]
    
    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return False  # Already connected
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        return True
    
    def connected(self, x, y):
        return self.find(x) == self.find(y)
# Operations: O(α(n)) ≈ O(1) amortized
```

---

# 🧮 Part 17: Dynamic Programming — The Most Important Topic (75 min)

## What is DP?

DP = **Recursion + Memoization** (or bottom-up tabulation).

It solves problems with:
1. **Overlapping subproblems** — Same calculation repeated
2. **Optimal substructure** — Optimal solution uses optimal sub-solutions

```
Fibonacci WITHOUT DP:        WITH DP:
fib(5)                       fib(5) → check cache
├── fib(4)                   ├── fib(4) → check cache
│   ├── fib(3)               │   ├── fib(3) → check cache
│   │   ├── fib(2)  ← dup!  │   │   ├── fib(2) → compute once
│   │   └── fib(1)           │   │   └── fib(1)
│   └── fib(2)    ← dup!    │   └── fib(2) → CACHED ✅
└── fib(3)        ← dup!    └── fib(3) → CACHED ✅
    ├── fib(2)
    └── fib(1)               O(n) instead of O(2^n)!
```

### 🎬 **Visualize it:** [visualgo.net/dp](https://visualgo.net/en/recursion) — Step through DP problems

## The DP Recipe (4 Steps)

```
Step 1: DEFINE STATE     → What parameters describe a subproblem?
Step 2: RECURRENCE       → How does this subproblem relate to smaller ones?
Step 3: BASE CASE        → What's the answer for the smallest subproblem?
Step 4: COMPUTATION ORDER → Bottom-up: fill table in order of dependency
```

---

## 🧩 Pattern 21: 1D DP

### Problem: Climbing Stairs (LeetCode #70)

```
Ways to climb n stairs (1 or 2 steps at a time):
  dp[n] = dp[n-1] + dp[n-2]   ← Same as Fibonacci!

  dp[1] = 1   (one way: [1])
  dp[2] = 2   (two ways: [1,1], [2])
  dp[3] = 3   (three ways: [1,1,1], [1,2], [2,1])
  dp[4] = 5
```

```python
def climbStairs(n):
    if n <= 2:
        return n
    
    # Space-optimized: only need last 2 values
    prev2, prev1 = 1, 2
    for i in range(3, n + 1):
        curr = prev1 + prev2
        prev2, prev1 = prev1, curr
    
    return prev1
# Time: O(n) | Space: O(1)
```

### Problem: House Robber (LeetCode #198)

```
Can't rob adjacent houses. Maximize stolen amount.

At each house: ROB it (skip previous) or SKIP it (keep previous best)
dp[i] = max(dp[i-1], dp[i-2] + nums[i])
```

```python
def rob(nums):
    if len(nums) <= 2:
        return max(nums)
    
    prev2, prev1 = nums[0], max(nums[0], nums[1])
    for i in range(2, len(nums)):
        curr = max(prev1, prev2 + nums[i])
        prev2, prev1 = prev1, curr
    
    return prev1
# Time: O(n) | Space: O(1)
```

### Problem: Coin Change (LeetCode #322)

```
coins = [1, 2, 5], amount = 11
dp[i] = minimum coins to make amount i

dp[0] = 0
dp[1] = 1 (use coin 1)
dp[2] = 1 (use coin 2)
dp[3] = 2 (2+1)
dp[4] = 2 (2+2)
dp[5] = 1 (use coin 5)
...
dp[11] = 3 (5+5+1) ✅
```

```python
def coinChange(coins, amount):
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    
    for i in range(1, amount + 1):
        for coin in coins:
            if coin <= i:
                dp[i] = min(dp[i], dp[i - coin] + 1)
    
    return dp[amount] if dp[amount] != float('inf') else -1
# Time: O(amount × coins) | Space: O(amount)
```

### Problem: Longest Increasing Subsequence (LeetCode #300)
```python
def lengthOfLIS(nums):
    """dp[i] = length of LIS ending at index i."""
    n = len(nums)
    dp = [1] * n
    
    for i in range(1, n):
        for j in range(i):
            if nums[j] < nums[i]:
                dp[i] = max(dp[i], dp[j] + 1)
    
    return max(dp)
# Time: O(n²) | Space: O(n)
# Can be optimized to O(n log n) with binary search + patience sorting
```

---

## 🧩 Pattern 22: 2D DP (Grid Problems)

### Problem: Unique Paths (LeetCode #62)

```
Grid m × n. Move only right or down. Count paths to bottom-right.

    1  1  1  1
    1  2  3  4
    1  3  6  10   ← dp[i][j] = dp[i-1][j] + dp[i][j-1]
```

```python
def uniquePaths(m, n):
    dp = [[1] * n for _ in range(m)]
    
    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = dp[i-1][j] + dp[i][j-1]
    
    return dp[m-1][n-1]
# Time: O(m×n) | Space: O(m×n), can be O(n)
```

### Problem: Minimum Path Sum (LeetCode #64)
```python
def minPathSum(grid):
    m, n = len(grid), len(grid[0])
    
    for i in range(m):
        for j in range(n):
            if i == 0 and j == 0: continue
            elif i == 0: grid[i][j] += grid[i][j-1]
            elif j == 0: grid[i][j] += grid[i-1][j]
            else: grid[i][j] += min(grid[i-1][j], grid[i][j-1])
    
    return grid[m-1][n-1]
```

---

## 🧩 Pattern 23: Subsequence DP (LCS, Edit Distance)

### Problem: Longest Common Subsequence (LeetCode #1143)

```
text1 = "abcde", text2 = "ace"
LCS = "ace" (length 3)

     ""  a  c  e
""  [ 0  0  0  0 ]
a   [ 0  1  1  1 ]
b   [ 0  1  1  1 ]
c   [ 0  1  2  2 ]
d   [ 0  1  2  2 ]
e   [ 0  1  2  3 ] ← Answer
```

```python
def longestCommonSubsequence(text1, text2):
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i-1] == text2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    return dp[m][n]
# Time: O(m×n) | Space: O(m×n)
```

### Problem: Edit Distance (LeetCode #72)
```python
def minDistance(word1, word2):
    """Minimum edits (insert, delete, replace) to convert word1 to word2."""
    m, n = len(word1), len(word2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1): dp[i][0] = i
    for j in range(n + 1): dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i-1] == word2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j],     # Delete
                    dp[i][j-1],     # Insert
                    dp[i-1][j-1]    # Replace
                )
    
    return dp[m][n]
# Time: O(m×n) | Space: O(m×n)
```

---

## 🧩 Pattern 24: Knapsack DP (Subset Sum, 0/1 Knapsack)

### Problem: Partition Equal Subset Sum (LeetCode #416)
```python
def canPartition(nums):
    """Can we split array into two subsets with equal sum?"""
    total = sum(nums)
    if total % 2 != 0:
        return False
    
    target = total // 2
    dp = set([0])
    
    for num in nums:
        dp = dp | {x + num for x in dp}
        if target in dp:
            return True
    
    return target in dp
# Time: O(n × sum) | Space: O(sum)
```

---

## 🧩 DP Decision Tree — "Which DP Pattern?"

```
Is it about sequences/arrays?
├── Yes → Is it about a single sequence?
│         ├── Yes → 1D DP (LIS, House Robber, Climbing Stairs)
│         └── No → Two sequences → Subsequence DP (LCS, Edit Distance)
├── No → Is it about a grid/matrix?
│         └── Yes → 2D Grid DP (Unique Paths, Min Path Sum)
├── No → Is it about selecting items with capacity?
│         └── Yes → Knapsack DP (0/1 Knapsack, Coin Change)
└── No → Is it about intervals/partitions?
          └── Yes → Interval DP (Matrix Chain, Burst Balloons)
```

---

# 💚 Part 18: Greedy Algorithms (20 min)

## What is Greedy?

Make the **locally optimal choice** at each step, hoping for a **globally optimal** result.

> **When does Greedy work?** When the problem has the "greedy choice property" — making the best local choice doesn't prevent finding the best global solution.

### Problem: Jump Game (LeetCode #55)
```python
def canJump(nums):
    """Can you reach the last index?"""
    farthest = 0
    for i in range(len(nums)):
        if i > farthest:
            return False
        farthest = max(farthest, i + nums[i])
    return True
# Time: O(n) | Space: O(1)
```

### Problem: Jump Game II (LeetCode #45)
```python
def jump(nums):
    """Minimum jumps to reach the end."""
    jumps = 0
    current_end = farthest = 0
    
    for i in range(len(nums) - 1):
        farthest = max(farthest, i + nums[i])
        if i == current_end:
            jumps += 1
            current_end = farthest
    
    return jumps
```

### Problem: Merge Intervals (LeetCode #56)
```python
def merge(intervals):
    intervals.sort()
    result = [intervals[0]]
    
    for start, end in intervals[1:]:
        if start <= result[-1][1]:
            result[-1][1] = max(result[-1][1], end)
        else:
            result.append([start, end])
    
    return result
# Time: O(n log n) | Space: O(n)
```

### Problem: Activity Selection / Meeting Rooms II (LeetCode #253)
```python
def minMeetingRooms(intervals):
    """Sort by start, use min-heap for end times."""
    if not intervals:
        return 0
    
    intervals.sort()
    heap = [intervals[0][1]]  # End time of first meeting
    
    for start, end in intervals[1:]:
        if start >= heap[0]:
            heapq.heapreplace(heap, end)  # Reuse room
        else:
            heapq.heappush(heap, end)     # New room needed
    
    return len(heap)
```

---

# 🔤 Part 19: Tries — String Search Trees (15 min)

## What is a Trie?

A tree where each path represents a word. Shared prefixes share the same path.

```
Words: apple, app, bat

          root
         /    \
        a      b
        |      |
        p      a
        |      |
        p*     t*     (* = end of word)
        |
        l
        |
        e*
```

### 🎬 **Visualize it:** [visualgo.net/trie](https://visualgo.net/en/trie)

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False

class Trie:
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True
    
    def search(self, word):
        node = self._find(word)
        return node is not None and node.is_end
    
    def startsWith(self, prefix):
        return self._find(prefix) is not None
    
    def _find(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return None
            node = node.children[char]
        return node
# All operations: O(L) where L = word/prefix length
```

---

## 🌆 Afternoon Review — Patterns 19-24

| # | Pattern | When to Use | Think |
|---|---------|-------------|-------|
| 19 | BFS | Shortest path, level-by-level | "Water ripple" 🌊 |
| 20 | DFS | Explore all paths, cycles, components | "Maze walking" 🏃 |
| 21 | 1D DP | Single sequence optimization | "Choose or skip" |
| 22 | 2D DP | Grid paths, two variables | "Build from corner" |
| 23 | Subsequence DP | Two string comparison | "Match or skip" |
| 24 | Knapsack DP | Select items with constraint | "Take or leave" |

---

*Head to [day2-practice.md](day2-practice.md) for your final practice session!*

[← Day 2 Morning](day2-morning.md) | [Back to Schedule](README.md) | [Next: Day 2 Practice →](day2-practice.md)
