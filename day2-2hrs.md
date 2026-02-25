# ⚡ Day 2 — Advanced Data Structures & Algorithms

## Recursion → Trees → Heaps → Tries → Graphs → Union-Find → Greedy → DP → Intervals

> **Goal:** Master every advanced topic asked in interviews. After Day 1 + Day 2, you're equipped to tackle 80%+ of coding interview problems.

---

## ⏱ Schedule

| Time | Topic | What You'll Learn |
|------|-------|-------------------|
| 0:00 - 0:15 | Recursion & Backtracking | Base case thinking, generate all possibilities |
| 0:15 - 0:40 | Trees & BST | Traversals, recursive properties, validation |
| 0:40 - 0:55 | Heaps / Priority Queues | Top-K, merge K sorted, median |
| 0:55 - 1:05 | Tries (Prefix Trees) | Prefix matching, autocomplete, word search |
| 1:05 - 1:25 | Graphs | BFS, DFS, topological sort, Dijkstra |
| 1:25 - 1:35 | Union-Find | Connected components, cycle detection |
| 1:35 - 1:45 | Greedy Algorithms | Local optimal = global optimal, intervals |
| 1:45 - 2:00 | Dynamic Programming | 1D, 2D, knapsack, subsequences |

---

# 🔁 0:00 — Recursion & Backtracking (15 min)

## What is Recursion?

Recursion is when a function **calls itself** to solve a smaller version of the same problem. It's not a data structure — it's a **way of thinking**.

Every recursive solution has:
1. **Base case** — when to stop (prevents infinite loops)
2. **Recursive case** — break the problem into a smaller identical problem

**Analogy:** Russian nesting dolls — open one, find another inside, open that one... until you reach the smallest (base case). Then work back up.

### How to Think Recursively

> **"Assume the recursive call works perfectly. How do I use its result to solve the current problem?"**

```python
def factorial(n):
    if n <= 1: return 1        # base case
    return n * factorial(n-1)  # trust that factorial(n-1) works
```

> 🎬 Visualize YOUR recursive code: [pythontutor.com](https://pythontutor.com/)

---

## What is Backtracking?

Backtracking is recursion with **undo**. You make a choice, explore it fully, then **undo** and try the next option. It systematically explores all possibilities.

**Analogy:** Solving a maze — at each fork, pick a path. Dead end? Walk BACK and try another.

```
Backtracking Template:
1. Make a CHOICE (add element, place queen)
2. RECURSE with the choice
3. UNDO the choice (backtrack)
4. Try the NEXT choice
```

---

## Pattern 1: Subsets — Include or Exclude

### The Core Idea

> **"For each element: include it or skip it. This creates 2ⁿ subsets."**

```
Elements: [1, 2, 3]

             []
          /      \
       [1]        []           ← include 1 or skip?
      /    \    /    \
  [1,2]  [1]  [2]   []        ← include 2 or skip?
  / \    / \   / \   / \
[123][12][13][1][23][2][3][]   ← include 3 or skip?
```

### Subsets (LeetCode #78)

```python
def subsets(nums):
    res = []
    def bt(i, curr):
        if i == len(nums):
            res.append(curr[:])
            return
        curr.append(nums[i])    # include
        bt(i + 1, curr)
        curr.pop()              # UNDO (backtrack)
        bt(i + 1, curr)         # exclude
    bt(0, [])
    return res
# O(2^n)
```

### Combination Sum (LeetCode #39)

**The Concept:** Find combinations summing to target. Can **reuse** elements. At each step, try each candidate, recurse with reduced target, backtrack.

```python
def combinationSum(candidates, target):
    res = []
    def bt(start, curr, remain):
        if remain == 0: res.append(curr[:]); return
        if remain < 0: return
        for i in range(start, len(candidates)):
            curr.append(candidates[i])
            bt(i, curr, remain - candidates[i])  # i not i+1: reuse OK
            curr.pop()
    bt(0, [], target)
    return res
```

### Permutations (LeetCode #46)

**The Concept:** Order matters. For each position, choose from remaining elements. n! total.

```python
def permute(nums):
    res = []
    def bt(curr, remaining):
        if not remaining:
            res.append(curr[:])
            return
        for i in range(len(remaining)):
            curr.append(remaining[i])
            bt(curr, remaining[:i] + remaining[i+1:])
            curr.pop()
    bt([], nums)
    return res
```

### ⭐ N-Queens (LeetCode #51)

**The Concept:** Place n queens so none attack each other. Row by row — at each row, try each column. Track attacks using sets for columns (`col`), diagonals (`row-col`), anti-diagonals (`row+col`).

```python
def solveNQueens(n):
    res = []
    cols, diag, anti = set(), set(), set()
    board = [['.']*n for _ in range(n)]
    def bt(row):
        if row == n:
            res.append([''.join(r) for r in board]); return
        for col in range(n):
            if col in cols or row-col in diag or row+col in anti: continue
            board[row][col] = 'Q'
            cols.add(col); diag.add(row-col); anti.add(row+col)
            bt(row + 1)
            board[row][col] = '.'
            cols.discard(col); diag.discard(row-col); anti.discard(row+col)
    bt(0)
    return res
```

### Word Search (LeetCode #79)

**The Concept:** DFS + backtracking on a grid. At each cell, try all 4 directions. Mark visited cells to avoid reuse; unmark on backtrack.

```python
def exist(board, word):
    R, C = len(board), len(board[0])
    def dfs(r, c, k):
        if k == len(word): return True
        if r < 0 or r >= R or c < 0 or c >= C: return False
        if board[r][c] != word[k]: return False
        tmp, board[r][c] = board[r][c], '#'  # mark visited
        found = any(dfs(r+dr, c+dc, k+1) for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)])
        board[r][c] = tmp                     # unmark (backtrack)
        return found
    return any(dfs(r, c, 0) for r in range(R) for c in range(C))
```

---

# 🌳 0:15 — Trees & BST (25 min)

## What is a Tree?

A tree is a **hierarchical** data structure — nodes connected by parent-child relationships. Think of a family tree or folder structure.

```
        1  (root)
       / \
      2   3
     / \
    4   5  (leaves)
```

### Key Terminology

| Term | Meaning |
|------|---------|
| **Root** | Topmost node (no parent) |
| **Leaf** | Node with no children |
| **Depth** | Distance from root (root = 0) |
| **Height** | Longest path from node to a leaf |
| **Binary Tree** | Each node has at most 2 children |
| **BST** | Binary tree where `left < node < right` ALWAYS |

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
```

> 🎬 Visualize: [visualgo.net/bst](https://visualgo.net/en/bst)

---

## Pattern 2: Tree Traversals — Four Ways to Visit Nodes

### DFS (Depth-First) — Go deep before going wide

```
Tree:       1             Inorder   (L, Root, R): 4,2,5,1,3  ← SORTED for BST!
           / \            Preorder  (Root, L, R): 1,2,4,5,3  ← copy/serialize
          2   3           Postorder (L, R, Root): 4,5,2,3,1  ← delete/evaluate
         / \
        4   5
```

### BFS (Breadth-First) — Go level by level

```
Level Order: [1], [2, 3], [4, 5]  ← level-based questions, shortest path
```

### When to Use Which

```
Sorted data from BST?              → Inorder
Process levels?                     → BFS
Process children before parent?     → Postorder
Process parent before children?     → Preorder
```

```python
# DFS — Recursive
def inorder(root):
    if not root: return []
    return inorder(root.left) + [root.val] + inorder(root.right)

# BFS — Level Order (LeetCode #102)
from collections import deque
def levelOrder(root):
    if not root: return []
    res, q = [], deque([root])
    while q:
        level = []
        for _ in range(len(q)):
            node = q.popleft()
            level.append(node.val)
            if node.left:  q.append(node.left)
            if node.right: q.append(node.right)
        res.append(level)
    return res
```

---

## Pattern 3: Recursive Tree Properties

### The Core Idea

> **"Almost every tree problem: solve for left, solve for right, combine."**

```python
def solve(root):
    if not root: return BASE_CASE
    left  = solve(root.left)
    right = solve(root.right)
    return COMBINE(root.val, left, right)
```

### Max Depth (LeetCode #104)

```python
def maxDepth(root):
    if not root: return 0
    return 1 + max(maxDepth(root.left), maxDepth(root.right))
```

### Diameter of Binary Tree (LeetCode #543)

**The Concept:** Longest path = the path that bends through some node. At each node, it's `left_height + right_height`. Track the global max as a side effect.

```python
def diameterOfBinaryTree(root):
    diameter = 0
    def height(node):
        nonlocal diameter
        if not node: return 0
        L, R = height(node.left), height(node.right)
        diameter = max(diameter, L + R)
        return 1 + max(L, R)
    height(root)
    return diameter
```

### Invert Binary Tree (LeetCode #226)

```python
def invertTree(root):
    if not root: return None
    root.left, root.right = invertTree(root.right), invertTree(root.left)
    return root
```

### ⭐ Lowest Common Ancestor (LeetCode #236)

**The Concept:** If both children return a result → this node is the LCA. If only one returns → pass it up.

```python
def lowestCommonAncestor(root, p, q):
    if not root or root == p or root == q: return root
    L = lowestCommonAncestor(root.left, p, q)
    R = lowestCommonAncestor(root.right, p, q)
    if L and R: return root
    return L or R
```

### Validate BST (LeetCode #98)

**The Concept:** Pass valid bounds downward. Every node must satisfy `lo < val < hi`.

```python
def isValidBST(root, lo=float('-inf'), hi=float('inf')):
    if not root: return True
    if root.val <= lo or root.val >= hi: return False
    return isValidBST(root.left, lo, root.val) and \
           isValidBST(root.right, root.val, hi)
```

### ⭐ Maximum Path Sum (LeetCode #124) — Hard

```python
def maxPathSum(root):
    best = float('-inf')
    def helper(node):
        nonlocal best
        if not node: return 0
        L = max(0, helper(node.left))
        R = max(0, helper(node.right))
        best = max(best, node.val + L + R)
        return node.val + max(L, R)
    helper(root)
    return best
```

### Serialize & Deserialize (LeetCode #297)

**The Concept:** Convert tree ↔ string. Use preorder traversal with "null" markers for missing nodes.

```python
class Codec:
    def serialize(self, root):
        if not root: return "null"
        return f"{root.val},{self.serialize(root.left)},{self.serialize(root.right)}"
    
    def deserialize(self, data):
        nodes = iter(data.split(","))
        def build():
            val = next(nodes)
            if val == "null": return None
            node = TreeNode(int(val))
            node.left = build()
            node.right = build()
            return node
        return build()
```

---

# 📊 0:40 — Heaps / Priority Queues (15 min)

## What is a Heap?

A heap is a **complete binary tree** where every parent is smaller (min-heap) or larger (max-heap) than its children. The root is always the min (or max).

```
Min-Heap:       1          → root is always the minimum
               / \
              3   2        → parent ≤ children at every level
             / \
            7   5
```

### Why Use a Heap?

| Need | Alternative | Heap |
|------|------------|------|
| Get min/max | Sort first O(n log n) | O(1) peek |
| Insert new element | Re-sort O(n log n) | O(log n) |
| Remove min/max | O(n) scan | O(log n) |

**Use heaps when you need the min/max element repeatedly as data changes.**

### Python's `heapq` — Min-Heap by Default

```python
import heapq

heap = []
heapq.heappush(heap, 5)       # insert: O(log n)
heapq.heappush(heap, 3)
heapq.heappush(heap, 7)
min_val = heapq.heappop(heap)  # remove min: O(log n) → returns 3
peek = heap[0]                 # see min without removing: O(1)

# Max-heap trick: negate values
heapq.heappush(heap, -val)     # insert negated
max_val = -heapq.heappop(heap) # negate back
```

---

## Pattern 4: Top-K Problems

### The Core Idea

> **"Maintain a heap of size K. The root gives you the Kth element."**

### Kth Largest Element (LeetCode #215)

```python
def findKthLargest(nums, k):
    # Min-heap of size k: root = kth largest
    heap = nums[:k]
    heapq.heapify(heap)
    for num in nums[k:]:
        if num > heap[0]:
            heapq.heapreplace(heap, num)
    return heap[0]
# O(n log k) — much better than sorting O(n log n)
```

### Merge K Sorted Lists (LeetCode #23) — Hard

**The Concept:** Put the head of each list in a min-heap. Pop the smallest, push its next node. The heap always gives you the globally smallest available node.

```python
def mergeKLists(lists):
    heap = []
    for i, l in enumerate(lists):
        if l: heapq.heappush(heap, (l.val, i, l))
    dummy = curr = ListNode(0)
    while heap:
        val, i, node = heapq.heappop(heap)
        curr.next = node
        curr = curr.next
        if node.next:
            heapq.heappush(heap, (node.next.val, i, node.next))
    return dummy.next
# O(N log K) where N = total nodes, K = number of lists
```

### Task Scheduler (LeetCode #621)

**The Concept:** Always execute the most frequent task first (max-heap). After executing, put it in a cooldown queue for `n` intervals.

```python
def leastInterval(tasks, n):
    freq = list(Counter(tasks).values())
    max_heap = [-f for f in freq]
    heapq.heapify(max_heap)
    cooldown = deque()  # (remaining_count, available_time)
    time = 0
    while max_heap or cooldown:
        time += 1
        if max_heap:
            remaining = heapq.heappop(max_heap) + 1  # negated
            if remaining < 0:
                cooldown.append((remaining, time + n))
        if cooldown and cooldown[0][1] == time:
            heapq.heappush(max_heap, cooldown.popleft()[0])
    return time
```

### Find Median from Data Stream (LeetCode #295)

**The Concept:** Maintain two heaps: a max-heap for the smaller half and a min-heap for the larger half. The median is at the tops.

```python
class MedianFinder:
    def __init__(self):
        self.lo = []  # max-heap (negated) — smaller half
        self.hi = []  # min-heap — larger half
    
    def addNum(self, num):
        heapq.heappush(self.lo, -num)
        heapq.heappush(self.hi, -heapq.heappop(self.lo))
        if len(self.hi) > len(self.lo):
            heapq.heappush(self.lo, -heapq.heappop(self.hi))
    
    def findMedian(self):
        if len(self.lo) > len(self.hi):
            return -self.lo[0]
        return (-self.lo[0] + self.hi[0]) / 2
```

---

# 🔤 0:55 — Tries (Prefix Trees) (10 min)

## What is a Trie?

A Trie (pronounced "try") is a tree where each node represents a **character**, and paths from root to nodes spell out **prefixes**. It's the ultimate data structure for prefix-based operations.

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
| Exact word lookup | O(L) ✅ | O(L) ✅ |
| Find all words with prefix "car" | O(n) scan all ❌ | O(L + matches) ✅ |
| Autocomplete | Expensive | Natural ✅ |
| Spell checker | Expensive | Natural ✅ |

### Trie Implementation

```python
class TrieNode:
    def __init__(self):
        self.children = {}            # char → TrieNode
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

### ⭐ Word Search II (LeetCode #212) — Hard

**The Concept:** Build a Trie from all target words, then DFS through the grid. At each cell, follow the Trie — if the Trie has no branch for a character, prune that search path.

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

# 🗺️ 1:05 — Graphs (20 min)

## What is a Graph?

A graph models **relationships** between things. It's the most general data structure — trees, linked lists, and even arrays can be viewed as special cases of graphs.

### Real-World Examples

- **Social network** — people = nodes, friendships = edges
- **GPS/Maps** — intersections = nodes, roads = edges (weighted)
- **Course prerequisites** — courses = nodes, "must take before" = directed edges

### Types

```
Undirected:  A — B    (friendship: mutual)
Directed:    A → B    (follow: one-way)
Weighted:    A —5— B  (road with distance 5)
Cyclic:      A → B → C → A  (loops exist)
Acyclic:     A → B → C      (no loops; a tree is acyclic)
```

### Representation — Adjacency List

```python
from collections import defaultdict
graph = defaultdict(list)
for u, v in edges:
    graph[u].append(v)
    graph[v].append(u)     # remove for directed
```

> 🎬 Visualize: [pathfinding.js.org](https://qiao.github.io/PathFinding.js/visual/)

---

## Pattern 5: BFS — Shortest Path & Level-by-Level

### The Core Idea

> **"Explore ALL neighbors at distance 1 first, then distance 2, then 3... Naturally finds shortest path."**

**Analogy:** Ripples in a pond — expanding outward uniformly.

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

### Number of Islands (LeetCode #200)

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

### Rotting Oranges (LeetCode #994) — Multi-source BFS

**The Concept:** Start BFS from ALL rotten oranges simultaneously. Each "level" = 1 minute.

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

## Pattern 6: DFS — All Paths & Cycle Detection

### The Core Idea

> **"Go as deep as possible, then backtrack. Use 3 states to detect cycles in directed graphs."**

```
Three states:
0 = UNVISITED
1 = VISITING (currently exploring — on current path)
2 = VISITED  (fully explored)

Reaching a node in state 1 → CYCLE!
```

### Course Schedule (LeetCode #207)

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

### Course Schedule II (LeetCode #210) — Topological Sort

**The Concept:** A topological ordering = valid course order. Use DFS postorder (add to result when DONE), then reverse.

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

**The Concept:** BFS finds shortest path in unweighted graphs. Dijkstra uses a **min-heap** to always process the closest unvisited node.

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

# 🔗 1:25 — Union-Find (Disjoint Set Union) (10 min)

## What is Union-Find?

Union-Find tracks **groups of connected elements**. It answers two questions instantly:
- **Find:** Which group does element X belong to?
- **Union:** Merge two groups together.

**Analogy:** Social groups at a party. Initially everyone is standalone. When two people become friends, their friend groups merge. Union-Find efficiently tracks who's in whose group.

### Key Operations

| Operation | Naive | With Optimizations |
|-----------|-------|-------------------|
| Find (which group?) | O(n) | O(α(n)) ≈ O(1) |
| Union (merge groups) | O(n) | O(α(n)) ≈ O(1) |

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
        # union by rank: attach smaller tree to larger
        if self.rank[ra] < self.rank[rb]: ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]: self.rank[ra] += 1
        self.components -= 1
        return True
    
    def connected(self, a, b):
        return self.find(a) == self.find(b)
```

### When to Use Union-Find

```
✅ "Number of connected components"
✅ "Are X and Y connected?"
✅ "Merge/connect groups"
✅ "Detect cycles in undirected graphs"
✅ Problems where relationships grow over time
```

### Number of Connected Components (LeetCode #323)

```python
def countComponents(n, edges):
    uf = UnionFind(n)
    for u, v in edges:
        uf.union(u, v)
    return uf.components
```

### Redundant Connection (LeetCode #684)

**The Concept:** Find the edge that creates a cycle. Add edges one by one — if union returns False (already connected), that edge is redundant.

```python
def findRedundantConnection(edges):
    uf = UnionFind(len(edges) + 1)
    for u, v in edges:
        if not uf.union(u, v):
            return [u, v]              # this edge created a cycle!
```

### Accounts Merge (LeetCode #721)

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
    # Group emails by root account
    groups = defaultdict(set)
    for email, i in email_to_id.items():
        groups[uf.find(i)].add(email)
    return [[accounts[i][0]] + sorted(emails) for i, emails in groups.items()]
```

---

# 💰 1:35 — Greedy Algorithms (10 min)

## What is Greedy?

A greedy algorithm makes the **locally optimal choice** at each step, hoping it leads to a globally optimal solution. Unlike DP, greedy doesn't reconsider past choices.

> **"At each step, take the best available option. Never look back."**

### When Does Greedy Work?

Greedy works when the problem has **optimal substructure** AND the **greedy choice property** (local best → global best). It's often used for:

```
✅ Interval scheduling (start/end times)
✅ Activity selection
✅ Minimum coins (specific cases)
✅ Huffman coding
✅ Jump/reach problems
```

### How to Verify Greedy Is Correct

1. Can you prove that the greedy choice is always part of an optimal solution?
2. Or: Does choosing the "best now" ever prevent a better future choice? If no → greedy works.

---

## Key Greedy Problems

### Jump Game (LeetCode #55)

**The Concept:** Track the farthest position you can reach. If you can ever reach the end, return True.

```python
def canJump(nums):
    farthest = 0
    for i in range(len(nums)):
        if i > farthest: return False
        farthest = max(farthest, i + nums[i])
    return True
# O(n), O(1)
```

### Jump Game II (LeetCode #45)

**The Concept:** BFS-style: track the farthest you can reach in each "jump." When you exhaust the current jump range, increment jumps.

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

### Non-overlapping Intervals (LeetCode #435)

**The Concept:** Sort by END time. Greedily keep intervals that end earliest (leave maximum room for future intervals).

```python
def eraseOverlapIntervals(intervals):
    intervals.sort(key=lambda x: x[1])  # sort by end time
    count = 0
    prev_end = float('-inf')
    for s, e in intervals:
        if s >= prev_end:
            prev_end = e               # no overlap → keep it
        else:
            count += 1                 # overlap → remove it
    return count
```

### Meeting Rooms II (LeetCode #253)

**The Concept:** Sort by start time. Use a min-heap of end times. If the earliest-ending meeting ends before the current start, reuse that room.

```python
def minMeetingRooms(intervals):
    intervals.sort()
    heap = []                          # end times of active meetings
    for s, e in intervals:
        if heap and heap[0] <= s:
            heapq.heappop(heap)        # reuse room (meeting ended)
        heapq.heappush(heap, e)
    return len(heap)                   # heap size = rooms needed
```

### Gas Station (LeetCode #134)

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

# 🧮 1:45 — Dynamic Programming (15 min)

## What is DP?

DP is the most feared interview topic, but at its core:

> **"If you're solving the same subproblem multiple times, solve it ONCE and save the answer."**

### DP = Recursion + Caching

```
Without DP: fib(5) → fib(3) computed TWICE, fib(2) THREE times → O(2ⁿ) 💀
With DP:    each fib(i) computed ONCE → O(n) ✅
```

### The 4-Step DP Recipe

```
1. DEFINE STATE   → What info describes a subproblem? (index, capacity, etc.)
2. RECURRENCE    → dp[i] = f(dp[i-1], dp[i-2], ...)
3. BASE CASE     → dp[0] = ?, dp[1] = ?
4. DIRECTION     → Fill from base case forward (bottom-up)
```

### How to Know It's DP

```
1. Asks for OPTIMAL (min/max) or COUNT of ways
2. Making a sequence of CHOICES
3. Same subproblems solved REPEATEDLY in brute force
4. GREEDY doesn't work (local ≠ global optimal)
```

---

## Pattern 7: 1D DP — Linear Optimization

### Climbing Stairs (LeetCode #70)

**The Concept:** `dp[n] = dp[n-1] + dp[n-2]` — literally Fibonacci!

```python
def climbStairs(n):
    if n <= 2: return n
    a, b = 1, 2
    for _ in range(3, n+1):
        a, b = b, a + b
    return b
```

### House Robber (LeetCode #198)

**The Concept:** At each house: rob it (value + best from 2 ago) or skip it (best from previous).

```python
def rob(nums):
    if len(nums) <= 2: return max(nums)
    a, b = nums[0], max(nums[0], nums[1])
    for i in range(2, len(nums)):
        a, b = b, max(b, a + nums[i])
    return b
```

### Coin Change (LeetCode #322)

**The Concept:** `dp[amount] = 1 + min(dp[amount - coin])` for each coin.

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

### Longest Increasing Subsequence (LeetCode #300)

```python
def lengthOfLIS(nums):
    dp = [1] * len(nums)
    for i in range(1, len(nums)):
        for j in range(i):
            if nums[j] < nums[i]:
                dp[i] = max(dp[i], dp[j] + 1)
    return max(dp)
# O(n²) — optimizable to O(n log n) with binary search
```

### Word Break (LeetCode #139)

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

## Pattern 8: 2D DP — Grids & Two-Sequence Comparison

### Unique Paths (LeetCode #62)

```python
def uniquePaths(m, n):
    dp = [[1]*n for _ in range(m)]
    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = dp[i-1][j] + dp[i][j-1]
    return dp[m-1][n-1]
```

### ⭐ Longest Common Subsequence (LeetCode #1143)

**Walkthrough:**
```
     ""  a  b  c  d  e
  ""  0  0  0  0  0  0
  a   0  1  1  1  1  1    ← 'a' matches
  c   0  1  1  2  2  2    ← 'c' matches
  e   0  1  1  2  2  3    ← 'e' matches → LCS = 3
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

### Edit Distance (LeetCode #72)

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

### Partition Equal Subset Sum (LeetCode #416)

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

**The Concept:** Choose items with weights and values to maximize value within a weight limit.

```python
def knapsack(weights, values, capacity):
    n = len(weights)
    dp = [[0]*(capacity+1) for _ in range(n+1)]
    for i in range(1, n+1):
        for w in range(capacity+1):
            dp[i][w] = dp[i-1][w]            # skip item i
            if weights[i-1] <= w:
                dp[i][w] = max(dp[i][w], dp[i-1][w-weights[i-1]] + values[i-1])
    return dp[n][capacity]
```

---

## 🧠 DP Decision Guide

```
Single sequence?         → 1D DP     (House Robber, Climbing Stairs, LIS)
Two strings?             → 2D DP     (LCS, Edit Distance)
Grid?                    → 2D DP     (Unique Paths, Min Path Sum)
Items + capacity?        → Knapsack  (Coin Change, Subset Sum)
Can decompose to choices?→ DP        (if greedy doesn't work)
```

---

# ✅ Day 2 Summary — All Advanced Patterns

| # | Pattern | Core Insight | Key Problem |
|---|---------|-------------|-------------|
| 1 | **Backtracking** | Choose → Explore → Undo | Subsets #78, N-Queens #51 |
| 2 | **Tree Traversal** | DFS (3 orders) + BFS | Level Order #102 |
| 3 | **Recursive Tree** | Solve left + right → combine | LCA #236, Max Path Sum #124 |
| 4 | **Top-K / Heap** | Min/max heap for streaming data | Kth Largest #215, Merge K #23 |
| 5 | **BFS Graph** | Queue + visited = shortest path | Islands #200, Rotting Oranges #994 |
| 6 | **DFS Graph** | 3-state cycle detection | Course Schedule #207 |
| 7 | **1D DP** | dp[i] = f(previous values) | Coin Change #322, LIS #300 |
| 8 | **2D DP** | dp[i][j] for grids/strings | LCS #1143, Edit Distance #72 |
| **Bonus** | **Trie** | Prefix-based string operations | Word Search II #212 |
| **Bonus** | **Union-Find** | Track connected components | Redundant Connection #684 |
| **Bonus** | **Greedy** | Local optimal = global optimal | Jump Game #55, Meeting Rooms #253 |

---

# 🏆 Complete Crash Course — All Topics Covered

## Quick Pattern Recognition

```
"Find pair with property X"          → HashMap or Two Pointers
"Longest/shortest subarray"          → Sliding Window
"Find in sorted data"               → Binary Search
"Search answer range"               → Binary Search on Answer
"All subsets/combos/perms"           → Backtracking
"Cycle in linked list"              → Slow/Fast Pointers
"Matching brackets/nesting"          → Stack
"Next greater/smaller"              → Monotonic Stack
"Level-by-level / shortest path"    → BFS
"All paths / cycle detection"       → DFS
"Connected components"              → Union-Find
"Schedule/select intervals"         → Greedy (sort by end)
"Min/max with overlapping choices"  → Dynamic Programming
"Prefix matching / autocomplete"    → Trie
"Top K / streaming min/max"         → Heap
```

## 🎯 Top 25 — If You Only Do These

```
🟢 #1    Two Sum              🟡 #53   Max Subarray
🟡 #3    Longest Substring    🟡 #15   3Sum
🟡 #5    Longest Palindrome   🟢 #206  Reverse Linked List
🟢 #20   Valid Parentheses    🟡 #739  Daily Temperatures
🟡 #33   Search Rotated       🟡 #56   Merge Intervals
🟡 #78   Subsets              🟡 #102  Level Order
🟢 #104  Max Depth Tree       🟡 #200  Number of Islands
🟡 #207  Course Schedule      🟡 #236  LCA
🟡 #322  Coin Change          🟡 #300  LIS
🟡 #1143 LCS                  🟡 #55   Jump Game
🔴 #42   Trapping Rain Water  🔴 #84   Largest Rectangle
🔴 #23   Merge K Sorted       🔴 #124  Max Path Sum
🔴 #212  Word Search II
```

---

## 📚 Keep Going

| Resource | Link | Purpose |
|----------|------|---------|
| **NeetCode Roadmap** | [neetcode.io/roadmap](https://neetcode.io/roadmap) | Structured problem list |
| **Striver A2Z Sheet** | [takeuforward.org](https://takeuforward.org/strivers-a2z-dsa-course/strivers-a2z-dsa-course-sheet-2) | 450+ problems by topic |
| **LeetCode Patterns** | [seanprashad.com/leetcode-patterns](https://seanprashad.com/leetcode-patterns/) | Pattern-based problem list |
| **Visualizations** | [visualgo.net](https://visualgo.net/) | See every algorithm animate |

> **Consistency beats intensity. Solve 2-3 problems daily and you'll crack any interview.** 🚀

---

*See also: [cheatsheet.md](cheatsheet.md) for quick-reference, [interview-playbook.md](interview-playbook.md) for interview day strategy.*
