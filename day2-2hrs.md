# ⚡ Day 2 — DSA Crash Course (2 Hours)

## Linked Lists → Stacks → Trees → Graphs → Dynamic Programming

> **Goal:** After this session + Day 1, you can tackle 80% of Easy, 50% of Medium, and understand every major DSA topic asked in interviews.

---

## ⏱ Schedule

| Time | Topic | Key Pattern |
|------|-------|-------------|
| 0:00 - 0:20 | Linked Lists | Slow/fast pointers, reversal |
| 0:20 - 0:40 | Stacks & Queues | Matching, monotonic stack |
| 0:40 - 1:10 | Trees & BST | DFS, BFS, recursive properties |
| 1:10 - 1:30 | Graphs | BFS, DFS, topological sort |
| 1:30 - 2:00 | Dynamic Programming | 1D, 2D, subsequences |

---

# 🔗 0:00 — Linked Lists (20 min)

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
```

> 🎬 Visualize: [visualgo.net/list](https://visualgo.net/en/list)

## Pattern 8: Slow & Fast Pointers (Floyd's)

Slow moves 1 step, fast moves 2 steps. When fast reaches end, **slow is at the middle**.

### Middle of Linked List (LeetCode #876)
```python
def middleNode(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    return slow
```

### Linked List Cycle (LeetCode #141)
```python
def hasCycle(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast: return True
    return False
```

## Pattern 9: Reverse Linked List — 3-Pointer Swap

### ⭐ Reverse Linked List (LeetCode #206) — Top 5 Interview Q

```
prev=None  curr=1→2→3→null
Step 1: save next=2, point 1→None, prev=1, curr=2
Step 2: save next=3, point 2→1,    prev=2, curr=3
Step 3: save next=None, point 3→2, prev=3, curr=None
Result: 3→2→1→None ✅
```

```python
def reverseList(head):
    prev = None
    curr = head
    while curr:
        nxt = curr.next     # save
        curr.next = prev    # reverse
        prev = curr         # advance
        curr = nxt
    return prev
# O(n), O(1) — MEMORIZE THIS
```

### Merge Two Sorted Lists (LeetCode #21)
```python
def mergeTwoLists(l1, l2):
    dummy = curr = ListNode(0)
    while l1 and l2:
        if l1.val <= l2.val:
            curr.next = l1; l1 = l1.next
        else:
            curr.next = l2; l2 = l2.next
        curr = curr.next
    curr.next = l1 or l2
    return dummy.next
```

### Reorder List (LeetCode #143)
```python
def reorderList(head):
    # 1. Find middle
    slow = fast = head
    while fast.next and fast.next.next:
        slow = slow.next
        fast = fast.next.next
    
    # 2. Reverse second half
    prev, curr = None, slow.next
    slow.next = None
    while curr:
        nxt = curr.next
        curr.next = prev
        prev = curr
        curr = nxt
    
    # 3. Interleave
    first, second = head, prev
    while second:
        tmp1, tmp2 = first.next, second.next
        first.next = second
        second.next = tmp1
        first, second = tmp1, tmp2
```

> **💡 Linked list recipe:** Find middle → Reverse half → Merge/Compare

---

# 📚 0:20 — Stacks & Queues (20 min)

```python
# Stack = LIFO (use list)
stack = []
stack.append(x)   # push O(1)
stack.pop()       # pop O(1)
stack[-1]         # peek O(1)

# Queue = FIFO (use deque)
from collections import deque
q = deque()
q.append(x)       # enqueue O(1)
q.popleft()       # dequeue O(1)
```

## Pattern 10: Stack for Matching

### Valid Parentheses (LeetCode #20)
```python
def isValid(s):
    stack = []
    match = {')':'(', '}':'{', ']':'['}
    for c in s:
        if c in '({[':
            stack.append(c)
        elif not stack or stack.pop() != match[c]:
            return False
    return not stack
```

## Pattern 11: Monotonic Stack — "Next Greater Element"

Maintain a stack in decreasing order. When a bigger element arrives, pop and record.

### Daily Temperatures (LeetCode #739)
```python
def dailyTemperatures(temps):
    n = len(temps)
    res = [0] * n
    stack = []                     # indices of decreasing temps
    for i in range(n):
        while stack and temps[i] > temps[stack[-1]]:
            j = stack.pop()
            res[j] = i - j        # days until warmer
        stack.append(i)
    return res
# O(n) — each element pushed and popped at most once
```

### ⭐ Largest Rectangle in Histogram (LeetCode #84) — Hard
```python
def largestRectangleArea(heights):
    stack = []
    best = 0
    for i in range(len(heights) + 1):
        h = heights[i] if i < len(heights) else 0
        while stack and heights[stack[-1]] > h:
            height = heights[stack.pop()]
            width = i if not stack else i - stack[-1] - 1
            best = max(best, height * width)
        stack.append(i)
    return best
# O(n)
```

### Min Stack (LeetCode #155)
```python
class MinStack:
    def __init__(self):
        self.stack = []           # (value, current_min)
    def push(self, val):
        mn = min(val, self.stack[-1][1]) if self.stack else val
        self.stack.append((val, mn))
    def pop(self):
        self.stack.pop()
    def top(self):
        return self.stack[-1][0]
    def getMin(self):
        return self.stack[-1][1]
# All operations O(1)
```

---

# 🌳 0:40 — Trees & BST (30 min)

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
```

> 🎬 Visualize: [visualgo.net/bst](https://visualgo.net/en/bst) — Build trees and watch traversals

## Pattern 12: Tree Traversals — Know All 4

```
Tree:     1          Inorder   (L,Root,R): 4,2,5,1,3  ← Sorted for BST!
         / \         Preorder  (Root,L,R): 1,2,4,5,3
        2   3        Postorder (L,R,Root): 4,5,2,3,1
       / \           Level Order (BFS):    [1],[2,3],[4,5]
      4   5
```

```python
# DFS — Recursive
def inorder(root):
    if not root: return []
    return inorder(root.left) + [root.val] + inorder(root.right)

# BFS — Level Order (LeetCode #102)
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

## Pattern 13: Recursive Tree Properties

Almost every tree problem uses this template:
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
```python
def diameterOfBinaryTree(root):
    diameter = 0
    def height(node):
        nonlocal diameter
        if not node: return 0
        L = height(node.left)
        R = height(node.right)
        diameter = max(diameter, L + R)  # path through this node
        return 1 + max(L, R)
    height(root)
    return diameter
```

### ⭐ Lowest Common Ancestor (LeetCode #236)
```python
def lowestCommonAncestor(root, p, q):
    if not root or root == p or root == q:
        return root
    L = lowestCommonAncestor(root.left, p, q)
    R = lowestCommonAncestor(root.right, p, q)
    if L and R: return root     # both sides found → this is LCA
    return L or R
```

### Validate BST (LeetCode #98)
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
        L = max(0, helper(node.left))    # ignore negative
        R = max(0, helper(node.right))
        best = max(best, node.val + L + R)
        return node.val + max(L, R)      # can only go one direction
    helper(root)
    return best
```

### Top-K with Heaps (LeetCode #215)
```python
import heapq
def findKthLargest(nums, k):
    return heapq.nlargest(k, nums)[-1]
# Or: min-heap of size k for O(n log k)
```

---

# 🗺️ 1:10 — Graphs (20 min)

```python
# Graph as adjacency list
from collections import defaultdict
graph = defaultdict(list)
for u, v in edges:
    graph[u].append(v)
    graph[v].append(u)     # remove for directed
```

> 🎬 Visualize: [pathfinding.js.org](https://qiao.github.io/PathFinding.js/visual/) — Watch BFS vs DFS on grids

## Pattern 14: BFS — Shortest Path / Level-by-Level

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
    rows, cols = len(grid), len(grid[0])
    count = 0
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == '1':
                count += 1
                q = deque([(r, c)])
                grid[r][c] = '0'
                while q:
                    row, col = q.popleft()
                    for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
                        nr, nc = row+dr, col+dc
                        if 0<=nr<rows and 0<=nc<cols and grid[nr][nc]=='1':
                            grid[nr][nc] = '0'
                            q.append((nr, nc))
    return count
```

### Rotting Oranges (LeetCode #994) — Multi-source BFS
```python
def orangesRotting(grid):
    R, C = len(grid), len(grid[0])
    q = deque()
    fresh = 0
    for r in range(R):
        for c in range(C):
            if grid[r][c] == 2: q.append((r,c,0))
            elif grid[r][c] == 1: fresh += 1
    time = 0
    while q:
        r,c,t = q.popleft()
        for dr,dc in [(0,1),(0,-1),(1,0),(-1,0)]:
            nr,nc = r+dr,c+dc
            if 0<=nr<R and 0<=nc<C and grid[nr][nc]==1:
                grid[nr][nc] = 2
                fresh -= 1
                time = t+1
                q.append((nr,nc,t+1))
    return time if fresh == 0 else -1
```

## Pattern 15: DFS — Cycle Detection / Topological Sort

### Course Schedule (LeetCode #207)
```python
def canFinish(n, prereqs):
    graph = defaultdict(list)
    for c, p in prereqs:
        graph[p].append(c)
    state = [0]*n   # 0=unvisited, 1=visiting, 2=done
    
    def has_cycle(node):
        if state[node] == 1: return True     # cycle!
        if state[node] == 2: return False
        state[node] = 1
        for nb in graph[node]:
            if has_cycle(nb): return True
        state[node] = 2
        return False
    
    return not any(has_cycle(i) for i in range(n))
```

### Dijkstra's — Shortest Path in Weighted Graph
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
# O((V+E) log V)
```

---

# 🧮 1:30 — Dynamic Programming (30 min)

## DP = Recursion + Cache = No Repeated Work

```
Without DP:  fib(5) calls fib(3) TWICE, fib(2) THREE times → O(2ⁿ)
With DP:     fib(5) computes each fib(i) ONCE → O(n)
```

## The DP Recipe
```
1. DEFINE STATE  → What describes a subproblem? (usually index, remaining capacity, etc.)
2. RECURRENCE   → dp[i] = f(dp[i-1], dp[i-2], ...)
3. BASE CASE    → dp[0] = ?, dp[1] = ?
4. DIRECTION    → Bottom-up: fill from base case forward
```

> 🎬 Visualize: [visualgo.net/dp](https://visualgo.net/en/recursion)

## Pattern 16: 1D DP

### Climbing Stairs (LeetCode #70)
```python
def climbStairs(n):
    if n <= 2: return n
    a, b = 1, 2
    for _ in range(3, n+1):
        a, b = b, a + b
    return b
# dp[n] = dp[n-1] + dp[n-2] → Fibonacci!
```

### House Robber (LeetCode #198)
```python
def rob(nums):
    if len(nums) <= 2: return max(nums)
    a, b = nums[0], max(nums[0], nums[1])
    for i in range(2, len(nums)):
        a, b = b, max(b, a + nums[i])   # skip or rob
    return b
# dp[i] = max(dp[i-1], dp[i-2] + nums[i])
```

### Coin Change (LeetCode #322)
```python
def coinChange(coins, amount):
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    for i in range(1, amount+1):
        for c in coins:
            if c <= i:
                dp[i] = min(dp[i], dp[i-c] + 1)
    return dp[amount] if dp[amount] != float('inf') else -1
# O(amount × coins)
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
# O(n²) — can be O(n log n) with binary search
```

---

## Pattern 17: 2D DP — Grids & Two Sequences

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
# O(m×n)
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

---

## 🧠 DP Decision Guide
```
Single sequence?         → 1D DP   (House Robber, Climbing Stairs)
Two sequences?           → 2D DP   (LCS, Edit Distance)
Grid?                    → 2D DP   (Unique Paths, Min Path Sum)
Items + capacity?        → Knapsack (Coin Change, Subset Sum)
Optimization + choices?  → DP      (if greedy doesn't work)
```

---

# ✅ Day 2 Summary — Patterns 8-17

| # | Pattern | Key Technique | Top Problem |
|---|---------|---------------|-------------|
| 8 | **Slow/Fast Pointers** | Cycle, middle | Linked List Cycle #141 |
| 9 | **Reverse LL** | 3-pointer swap | Reverse LL #206 |
| 10 | **Stack Matching** | Push/pop pairs | Valid Parentheses #20 |
| 11 | **Monotonic Stack** | Next greater/smaller | Daily Temps #739, Histogram #84 |
| 12 | **Tree Traversal** | DFS + BFS | Level Order #102 |
| 13 | **Recursive Tree** | Base + left + right | LCA #236, Max Path Sum #124 |
| 14 | **BFS Graph** | Queue + visited | Islands #200, Rotting Oranges #994 |
| 15 | **DFS Graph** | Recursion + state | Course Schedule #207 |
| 16 | **1D DP** | dp[i] = f(dp[i-1]...) | Coin Change #322, LIS #300 |
| 17 | **2D DP** | dp[i][j] = f(neighbors) | LCS #1143, Edit Distance #72 |

---

# 🏆 Complete Crash Course Summary — 17 Patterns in 4 Hours

## Quick Pattern Recognition Cheat
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
"Optimize with overlapping sub"     → DP
```

## 🎯 Top 20 — If You Only Do These Problems

```
🟢 #1    Two Sum              🟡 #53   Max Subarray (Kadane's)
🟡 #3    Longest Substring    🟡 #15   3Sum
🟡 #33   Search Rotated       🟡 #56   Merge Intervals
🟡 #78   Subsets              🟢 #206  Reverse Linked List
🟢 #20   Valid Parentheses    🟡 #739  Daily Temperatures
🟡 #102  Level Order          🟢 #104  Max Depth Tree
🟡 #236  LCA                  🟡 #200  Number of Islands
🟡 #207  Course Schedule      🟡 #322  Coin Change
🟡 #300  LIS                  🟡 #1143 LCS
🔴 #42   Trapping Rain Water  🔴 #84   Largest Rectangle
```

---

## 📚 Keep Going

| Resource | Link | Purpose |
|----------|------|---------|
| **NeetCode Roadmap** | [neetcode.io/roadmap](https://neetcode.io/roadmap) | Structured problem list |
| **Striver A2Z Sheet** | [takeuforward.org](https://takeuforward.org/strivers-a2z-dsa-course/strivers-a2z-dsa-course-sheet-2) | 450+ problems by topic |
| **LeetCode Patterns** | [seanprashad.com/leetcode-patterns](https://seanprashad.com/leetcode-patterns/) | Pattern-based problem list |
| **Visualizations** | [visualgo.net](https://visualgo.net/) | See every algorithm animate |
| **Full Workshop** | [../README.md](../README.md) | Deep-dive 18-chapter version |

> **Consistency beats intensity. Solve 2-3 problems daily and you'll crack any interview.** 🚀

---

*See also: [cheatsheet.md](cheatsheet.md) for a printable quick-reference, [interview-playbook.md](interview-playbook.md) for interview day strategy.*
