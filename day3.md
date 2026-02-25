# Day 3 -- Recursion, Backtracking, Trees, BST, and Heaps

## Recursive Thinking and Hierarchical Data Structures

**What this day covers:** Recursion (base case thinking), Backtracking (subsets, permutations, combinations, constraint satisfaction), Trees and BST (traversals, recursive properties, validation), and Heaps / Priority Queues (Top-K, merge K sorted, median).

Recursion is the foundation for understanding trees and graphs. Once you learn to think recursively, tree problems become straightforward -- and heaps give you a powerful tool for streaming data problems.

---

# Recursion and Backtracking

## What is Recursion?

Recursion is when a function calls itself to solve a smaller version of the same problem. It's not a data structure -- it's a way of thinking.

Every recursive solution has:
1. **Base case** -- when to stop (prevents infinite loops)
2. **Recursive case** -- break the problem into a smaller identical problem

Think of Russian nesting dolls -- open one, find another inside, open that one... until you reach the smallest one (base case). Then work back up.

### How to Think Recursively

> "Assume the recursive call works perfectly. How do I use its result to solve the current problem?"

```python
def factorial(n):
    if n <= 1: return 1        # base case
    return n * factorial(n-1)  # trust that factorial(n-1) works
```

Visualize your recursive code: [pythontutor.com](https://pythontutor.com/)

---

## What is Backtracking?

Backtracking is recursion with undo. You make a choice, explore it fully, then undo and try the next option. It systematically explores all possibilities.

Think of solving a maze -- at each fork, pick a path. Dead end? Walk back and try another.

```
Backtracking Template:
1. Make a CHOICE (add element, place queen)
2. RECURSE with the choice
3. UNDO the choice (backtrack)
4. Try the NEXT choice
```

---

## Pattern 13: Subsets -- Include or Exclude

### The Core Idea

> "For each element: include it or skip it. This creates 2^n subsets."

```
Elements: [1, 2, 3]

             []
          /      \
       [1]        []           -- include 1 or skip?
      /    \    /    \
  [1,2]  [1]  [2]   []        -- include 2 or skip?
  / \    / \   / \   / \
[123][12][13][1][23][2][3][]   -- include 3 or skip?
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

**The Concept:** Find combinations summing to target. Can reuse elements. At each step, try each candidate, recurse with reduced target, backtrack.

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

### N-Queens (LeetCode #51)

**The Concept:** Place n queens so none attack each other. Row by row, at each row, try each column. Track attacks using sets for columns (`col`), diagonals (`row-col`), anti-diagonals (`row+col`).

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

# Trees and BST

## What is a Tree?

A tree is a hierarchical data structure -- nodes connected by parent-child relationships. Think of a family tree or a folder structure.

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
| **BST** | Binary tree where `left < node < right` always |

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
```

Visualize tree operations: [visualgo.net/bst](https://visualgo.net/en/bst)

---

## Pattern 14: Tree Traversals -- Four Ways to Visit Nodes

### DFS (Depth-First) -- Go deep before going wide

```
Tree:       1             Inorder   (L, Root, R): 4,2,5,1,3  -- SORTED for BST!
           / \            Preorder  (Root, L, R): 1,2,4,5,3  -- copy/serialize
          2   3           Postorder (L, R, Root): 4,5,2,3,1  -- delete/evaluate
         / \
        4   5
```

### BFS (Breadth-First) -- Go level by level

```
Level Order: [1], [2, 3], [4, 5]  -- level-based questions, shortest path
```

### When to Use Which

```
Sorted data from BST?              -> Inorder
Process levels?                     -> BFS
Process children before parent?     -> Postorder
Process parent before children?     -> Preorder
```

```python
# DFS -- Recursive
def inorder(root):
    if not root: return []
    return inorder(root.left) + [root.val] + inorder(root.right)

# BFS -- Level Order (LeetCode #102)
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

## Pattern 15: Recursive Tree Properties

### The Core Idea

> "Almost every tree problem: solve for left, solve for right, combine."

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

### Lowest Common Ancestor (LeetCode #236)

**The Concept:** If both children return a result, this node is the LCA. If only one returns, pass it up.

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

### Maximum Path Sum (LeetCode #124) -- Hard

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

### Serialize and Deserialize (LeetCode #297)

**The Concept:** Convert tree to/from a string. Use preorder traversal with "null" markers for missing nodes.

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

# Heaps and Priority Queues

## What is a Heap?

A heap is a complete binary tree where every parent is smaller (min-heap) or larger (max-heap) than its children. The root is always the min (or max).

```
Min-Heap:       1          -> root is always the minimum
               / \
              3   2        -> parent <= children at every level
             / \
            7   5
```

### Why Use a Heap?

| Need | Alternative | Heap |
|------|------------|------|
| Get min/max | Sort first O(n log n) | O(1) peek |
| Insert new element | Re-sort O(n log n) | O(log n) |
| Remove min/max | O(n) scan | O(log n) |

Use heaps when you need the min or max element repeatedly as data changes.

### Python's heapq -- Min-Heap by Default

```python
import heapq

heap = []
heapq.heappush(heap, 5)       # insert: O(log n)
heapq.heappush(heap, 3)
heapq.heappush(heap, 7)
min_val = heapq.heappop(heap)  # remove min: O(log n) -> returns 3
peek = heap[0]                 # see min without removing: O(1)

# Max-heap trick: negate values
heapq.heappush(heap, -val)     # insert negated
max_val = -heapq.heappop(heap) # negate back
```

---

## Pattern 16: Top-K Problems

### The Core Idea

> "Maintain a heap of size K. The root gives you the Kth element."

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
# O(n log k) -- much better than sorting O(n log n)
```

### Merge K Sorted Lists (LeetCode #23) -- Hard

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
        self.lo = []  # max-heap (negated) -- smaller half
        self.hi = []  # min-heap -- larger half
    
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

# Day 3 Summary -- 4 Patterns

| # | Pattern | Core Insight | Key Problem |
|---|---------|-------------|-------------|
| 13 | **Backtracking** | Choose, Explore, Undo | Subsets #78, N-Queens #51 |
| 14 | **Tree Traversal** | DFS (3 orders) + BFS | Level Order #102 |
| 15 | **Recursive Tree** | Solve left + right, combine | LCA #236, Max Path Sum #124 |
| 16 | **Top-K / Heap** | Min/max heap for streaming data | Kth Largest #215, Merge K #23 |

### Practice Problems for Day 3

```
Easy:
  #70   Climbing Stairs
  #104  Maximum Depth of Binary Tree
  #226  Invert Binary Tree
  #78   Subsets

Medium:
  #46   Permutations
  #39   Combination Sum
  #79   Word Search
  #102  Binary Tree Level Order Traversal
  #98   Validate BST
  #236  Lowest Common Ancestor
  #543  Diameter of Binary Tree
  #215  Kth Largest Element

Hard:
  #51   N-Queens
  #23   Merge K Sorted Lists
  #124  Binary Tree Maximum Path Sum
```

---

*Next: Tries, Graphs, Greedy, and Dynamic Programming -- [day4.md](day4.md)*
