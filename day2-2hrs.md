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

## What is a Linked List?

Unlike arrays, where elements sit next to each other in memory (like houses on a street with consecutive addresses), a linked list stores elements **scattered across memory**, connected by pointers (like a treasure hunt where each clue tells you where the next one is).

```
Array:   [10][20][30][40]  ← elements are contiguous, accessed by index
                           ← arr[2] is instant (O(1)) because you calculate the address

Linked:  10→ 20→ 30→ 40→ None  ← elements are scattered, connected by "next" pointers
                                ← to reach element 3, you MUST walk through 1 and 2 (O(n))
```

### Why Use Linked Lists?

- **Insertion/deletion at known position is O(1)** — just re-wire pointers, no shifting needed
- **No fixed size** — grows dynamically, no pre-allocation
- **Used extensively in interview problems** because pointer manipulation tests your understanding of references

### The Node Definition

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val    # the data
        self.next = next  # pointer to the next node (or None if last)
```

> 🎬 Visualize: [visualgo.net/list](https://visualgo.net/en/list) — watch insertions, deletions, and reversal step by step

---

## Pattern 8: Slow & Fast Pointers (Floyd's Tortoise and Hare)

### The Core Idea

> **"Use two pointers moving at different speeds. The fast pointer reaches the end in half the time, which means when fast is done, slow is at the middle."**

**Analogy:** Imagine two runners on a circular track. If one runs at double speed, the fast runner will eventually lap the slow runner — proving the track is circular (has a cycle). On a straight track, the fast runner reaches the end while the slow runner is at the midpoint.

### This Pattern Solves Three Types of Problems:

1. **Finding the middle** of a linked list
2. **Detecting cycles** in a linked list
3. **Finding the start of a cycle** (Floyd's full algorithm)

### Middle of Linked List (LeetCode #876)

**The Concept:** Slow moves 1 step, fast moves 2 steps. When fast reaches the end, slow is exactly at the middle.

```
1 → 2 → 3 → 4 → 5 → None

Step 0: slow=1, fast=1
Step 1: slow=2, fast=3
Step 2: slow=3, fast=5 (fast.next is None → stop)
Answer: slow = 3 ✅ (the middle)
```

```python
def middleNode(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next           # 1 step
        fast = fast.next.next      # 2 steps
    return slow                    # slow is now at the middle
```

### Linked List Cycle (LeetCode #141)

**The Concept:** If there's a cycle, the fast pointer will eventually "lap" the slow pointer (they meet). If there's no cycle, the fast pointer hits `None` (reaches the end).

```python
def hasCycle(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast: return True   # they met → cycle exists!
    return False                       # fast reached end → no cycle
```

---

## Pattern 9: Reverse a Linked List — The 3-Pointer Technique

### The Core Idea

> **"Walk through the list, and at each node, reverse its arrow to point backwards instead of forwards."**

You need three pointers because to reverse a link, you need to know:
- **Where you came from** (`prev`) — the new target for the reversed arrow
- **Where you are** (`curr`) — the node whose arrow you're reversing
- **Where you're going** (`nxt`) — saved before you reverse, so you don't lose it

### ⭐ Reverse Linked List (LeetCode #206) — Top 5 Interview Question

**Step-by-step visualization:**
```
Starting:  prev=None   curr=1 → 2 → 3 → None

Step 1:  Save nxt=2.  Point 1→None.   Move prev=1, curr=2
         None ← 1   2 → 3 → None

Step 2:  Save nxt=3.  Point 2→1.      Move prev=2, curr=3
         None ← 1 ← 2   3 → None

Step 3:  Save nxt=None. Point 3→2.    Move prev=3, curr=None
         None ← 1 ← 2 ← 3

Result: 3 → 2 → 1 → None ✅
```

```python
def reverseList(head):
    prev = None
    curr = head
    while curr:
        nxt = curr.next     # 1. SAVE the next node
        curr.next = prev    # 2. REVERSE the arrow
        prev = curr         # 3. ADVANCE prev
        curr = nxt          # 4. ADVANCE curr
    return prev             # prev is the new head
# O(n) time, O(1) space — MEMORIZE THIS
```

### Merge Two Sorted Lists (LeetCode #21)

**The Concept:** Compare the heads of both lists. Take the smaller one, attach it to your result, and advance that list's pointer. Repeat. When one list is empty, attach the remainder of the other.

**Key technique — the Dummy Node:** Create a fake "dummy" node at the start so you don't need special logic for the first element.

```python
def mergeTwoLists(l1, l2):
    dummy = curr = ListNode(0)     # dummy simplifies edge cases
    while l1 and l2:
        if l1.val <= l2.val:
            curr.next = l1; l1 = l1.next
        else:
            curr.next = l2; l2 = l2.next
        curr = curr.next
    curr.next = l1 or l2           # attach whatever's left
    return dummy.next              # skip the dummy
```

### Reorder List (LeetCode #143)

**The Concept:** This beautifully combines THREE linked list techniques:

1. **Find the middle** (slow/fast pointers)
2. **Reverse the second half** (3-pointer reversal)
3. **Interleave the two halves** (merge)

This is why interviewers love it — it tests three patterns in one problem.

```python
def reorderList(head):
    # Step 1: Find middle using slow/fast
    slow = fast = head
    while fast.next and fast.next.next:
        slow = slow.next
        fast = fast.next.next
    
    # Step 2: Reverse the second half
    prev, curr = None, slow.next
    slow.next = None               # cut the list in half
    while curr:
        nxt = curr.next
        curr.next = prev
        prev = curr
        curr = nxt
    
    # Step 3: Interleave first and reversed-second
    first, second = head, prev
    while second:
        tmp1, tmp2 = first.next, second.next
        first.next = second
        second.next = tmp1
        first, second = tmp1, tmp2
```

> **💡 Linked List Recipe:** Find middle → Reverse half → Merge/Compare

---

# 📚 0:20 — Stacks & Queues (20 min)

## Stacks vs Queues — Two Ways to Organize Data

Think of these as two types of containers:

### Stack = LIFO (Last In, First Out)

**Analogy:** A stack of plates. You can only add/remove from the **top**. The last plate you put on is the first one you take off.

**Used for:** Undo operations, matching brackets, DFS, evaluating expressions, function call stack.

### Queue = FIFO (First In, First Out)

**Analogy:** A line at a movie theater. The first person in line is the first one served.

**Used for:** BFS, task scheduling, level-by-level processing.

```python
# Stack — use a Python list
stack = []
stack.append(x)   # push: add to top     O(1)
stack.pop()       # pop: remove from top  O(1)
stack[-1]         # peek: look at top     O(1)

# Queue — use collections.deque (NOT a list! list.pop(0) is O(n))
from collections import deque
q = deque()
q.append(x)       # enqueue: add to back  O(1)
q.popleft()       # dequeue: remove front  O(1)
```

---

## Pattern 10: Stack for Matching

### The Core Idea

> **"When you see an 'opening' element, push it. When you see a 'closing' element, pop and check if they match."**

This works because stacks naturally handle **nesting** — the most recently opened bracket must be the first to close.

### Valid Parentheses (LeetCode #20)

**The Concept:** Push every opening bracket. When you see a closing bracket, pop and check if it matches. If the stack is empty when you try to pop (nothing to match), or if it's not empty at the end (unclosed brackets), the answer is false.

```python
def isValid(s):
    stack = []
    match = {')':'(', '}':'{', ']':'['}
    for c in s:
        if c in '({[':
            stack.append(c)            # opening: push
        elif not stack or stack.pop() != match[c]:
            return False               # nothing to match, or wrong match
    return not stack                    # stack should be empty (all matched)
```

---

## Pattern 11: Monotonic Stack — "Next Greater Element"

### The Core Idea

> **"Maintain a stack where elements are always in increasing (or decreasing) order. When a new element violates this order, pop — the popped element just found its 'answer'."**

**Analogy:** People standing in line for a rollercoaster, arranged by height. When a taller person joins, all shorter people in front "see" the taller person as their "next greater element" and step out.

### Why is this O(n)?

Even though there's a while loop inside a for loop, each element is pushed **at most once** and popped **at most once**. So the total work is O(2n) = O(n). This is a common amortized analysis pattern.

### Daily Temperatures (LeetCode #739)

**The Concept:** For each day, find how many days until a warmer temperature. Maintain a stack of indices with decreasing temperatures. When a warmer day arrives, pop all cooler days — they've found their answer.

```python
def dailyTemperatures(temps):
    n = len(temps)
    res = [0] * n
    stack = []                         # indices of decreasing temps
    for i in range(n):
        while stack and temps[i] > temps[stack[-1]]:
            j = stack.pop()            # day j just found a warmer day
            res[j] = i - j            # how many days apart
        stack.append(i)
    return res
# O(n) — each element pushed and popped at most once
```

### ⭐ Largest Rectangle in Histogram (LeetCode #84) — Hard

**The Concept:** For each bar, the largest rectangle using that bar extends left and right until a shorter bar is hit. A monotonic increasing stack efficiently finds these boundaries.

```python
def largestRectangleArea(heights):
    stack = []
    best = 0
    for i in range(len(heights) + 1):
        h = heights[i] if i < len(heights) else 0  # sentinel: force pop at end
        while stack and heights[stack[-1]] > h:
            height = heights[stack.pop()]
            width = i if not stack else i - stack[-1] - 1
            best = max(best, height * width)
        stack.append(i)
    return best
# O(n) — elegant and efficient
```

### Min Stack (LeetCode #155)

**The Concept:** Support push, pop, top, AND getMin — all in O(1). The trick: at each level of the stack, store both the value AND the minimum value at that point. When you pop, the minimum is automatically correct.

```python
class MinStack:
    def __init__(self):
        self.stack = []               # stores (value, current_min) pairs
    def push(self, val):
        mn = min(val, self.stack[-1][1]) if self.stack else val
        self.stack.append((val, mn))  # snapshot the min at this level
    def pop(self):
        self.stack.pop()
    def top(self):
        return self.stack[-1][0]
    def getMin(self):
        return self.stack[-1][1]      # min is always up-to-date
# All operations O(1) — the key insight is storing the min as metadata
```

---

# 🌳 0:40 — Trees & BST (30 min)

## What is a Tree?

A tree is a **hierarchical** data structure where each node has zero or more children. Think of it like a family tree or an organizational chart:

```
        CEO (root)
       /    \
     CTO     CFO
    / \       \
  Dev1 Dev2   Accountant
```

### Key Terminology (You MUST Know These)

- **Root:** The topmost node (no parent)
- **Leaf:** A node with no children (the bottom)
- **Depth:** How far a node is from the root (root = depth 0)
- **Height:** The longest path from a node down to a leaf
- **Binary Tree:** Each node has at most 2 children (left and right)
- **Binary Search Tree (BST):** A binary tree where `left < node < right` for every node

### The Node Definition

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
```

> 🎬 Visualize: [visualgo.net/bst](https://visualgo.net/en/bst) — build trees and watch traversals live

---

## Pattern 12: Tree Traversals — The Four Ways to Visit Nodes

There are four fundamental ways to visit every node in a tree. Understanding when to use each is crucial:

### DFS (Depth-First Search) — Go Deep First

DFS explores as far down as possible before backtracking. Three variations based on *when* you process the current node:

```
Example Tree:    1
                / \
               2   3
              / \
             4   5

Inorder   (Left, Root, Right):  4, 2, 5, 1, 3  ← Gives SORTED order for BST!
Preorder  (Root, Left, Right):  1, 2, 4, 5, 3  ← Useful for copying/serializing
Postorder (Left, Right, Root):  4, 5, 2, 3, 1  ← Useful for deletion/evaluation
```

### BFS (Breadth-First Search) — Go Level by Level

BFS visits all nodes at depth 0, then depth 1, then depth 2, etc.

```
Level Order:    [1], [2, 3], [4, 5]  ← Useful for level-based questions
```

### When to Use Which?

```
Need SORTED data from BST?          → Inorder DFS
Need to process LEVELS?             → BFS
Need to process CHILDREN before PARENT? → Postorder DFS
Need to process PARENT before CHILDREN? → Preorder DFS
Need SHORTEST PATH in a tree?       → BFS
```

### Implementation

```python
# DFS — Recursive (most natural for trees)
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
        for _ in range(len(q)):        # process entire level at once
            node = q.popleft()
            level.append(node.val)
            if node.left:  q.append(node.left)
            if node.right: q.append(node.right)
        res.append(level)
    return res
```

---

## Pattern 13: Recursive Tree Properties — The Template That Solves Everything

### The Core Idea

> **"Almost every tree problem follows the same template: solve it for the left subtree, solve it for the right subtree, then combine the results."**

This is the **divide-and-conquer** approach applied to trees. The recursion naturally follows the tree structure.

### The Universal Template

```python
def solve(root):
    if not root: return BASE_CASE      # leaf boundary
    left_result  = solve(root.left)    # trust that this works
    right_result = solve(root.right)   # trust that this works
    return COMBINE(root.val, left_result, right_result)
```

### Max Depth (LeetCode #104)

**The Concept:** The depth of a tree = 1 + max(depth of left, depth of right). Base case: empty tree has depth 0.

```python
def maxDepth(root):
    if not root: return 0
    return 1 + max(maxDepth(root.left), maxDepth(root.right))
```

### Diameter of Binary Tree (LeetCode #543)

**The Concept:** The longest path may or may not pass through the root. At each node, the longest path *through* that node is `left_height + right_height`. Track the global maximum.

**Why this is tricky:** The answer isn't the height — it's the longest path, which could bend through any node. So we track the answer as a side effect while computing heights.

```python
def diameterOfBinaryTree(root):
    diameter = 0
    def height(node):
        nonlocal diameter
        if not node: return 0
        L = height(node.left)
        R = height(node.right)
        diameter = max(diameter, L + R)  # path through this node
        return 1 + max(L, R)             # height (for parent's use)
    height(root)
    return diameter
```

### ⭐ Lowest Common Ancestor — LCA (LeetCode #236)

**The Concept:** The LCA of two nodes `p` and `q` is the deepest node that has both `p` and `q` in its subtree. The elegant recursive insight:

- If `root` is `p` or `q`, return it (found one!)
- Recurse left and right
- If both sides found something → this root is the LCA
- If only one side found something → pass it upward

```python
def lowestCommonAncestor(root, p, q):
    if not root or root == p or root == q:
        return root                    # base: null or found target
    L = lowestCommonAncestor(root.left, p, q)
    R = lowestCommonAncestor(root.right, p, q)
    if L and R: return root            # p and q on different sides → LCA!
    return L or R                      # both on same side → pass upward
```

### Validate BST (LeetCode #98)

**The Concept:** A BST requires `left < node < right`, but this must hold for the ENTIRE subtree, not just immediate children. Solution: pass down valid bounds.

```
        5
       / \
      3   7      ← 3 < 5 < 7 ✅
     / \
    1   4        ← 1 < 3 < 4 ✅, AND 1 > -∞ ✅, AND 4 < 5 ✅
```

```python
def isValidBST(root, lo=float('-inf'), hi=float('inf')):
    if not root: return True
    if root.val <= lo or root.val >= hi:
        return False                   # violates bounds
    return isValidBST(root.left, lo, root.val) and \
           isValidBST(root.right, root.val, hi)
```

### ⭐ Maximum Path Sum (LeetCode #124) — Hard

**The Concept:** A "path" can start and end at any node. At each node, the maximum path *through* that node uses `node.val + best_from_left + best_from_right`. But when passing the result upward, you can only go in ONE direction (a path can't fork).

```python
def maxPathSum(root):
    best = float('-inf')
    def helper(node):
        nonlocal best
        if not node: return 0
        L = max(0, helper(node.left))    # ignore negative branches
        R = max(0, helper(node.right))
        best = max(best, node.val + L + R)  # path through this node
        return node.val + max(L, R)      # pass ONE direction upward
    helper(root)
    return best
```

### Top-K with Heaps (LeetCode #215)

**The Concept:** A heap (priority queue) efficiently gives you the minimum (or maximum) element. For "Kth largest," maintain a min-heap of size k. The root is always the kth largest.

**Why a heap?** Keeping a sorted structure where you can efficiently insert and remove elements. A heap does both in O(log n).

```python
import heapq
def findKthLargest(nums, k):
    return heapq.nlargest(k, nums)[-1]
# Under the hood: maintains a min-heap of size k → O(n log k)
```

---

# 🗺️ 1:10 — Graphs (20 min)

## What is a Graph?

A graph is the most general data structure — it models **relationships between things**. Trees are actually a special case of graphs (connected, acyclic graphs).

### Real-World Examples

- **Social network** — people are nodes, friendships are edges
- **Map/GPS** — intersections are nodes, roads are edges (weighted by distance)
- **Course prerequisites** — courses are nodes, "must take before" are directed edges
- **Internet** — web pages are nodes, hyperlinks are edges

### Types of Graphs

```
Undirected:  A — B   (friendship: mutual)
Directed:    A → B   (Twitter follow: one-way)
Weighted:    A —5— B  (road with distance 5)
Unweighted:  A — B    (all edges same cost)
Cyclic:      A → B → C → A  (can go in circles)
Acyclic:     A → B → C      (no circles; trees are acyclic)
```

### How to Represent a Graph

The most common representation is an **adjacency list** — a dictionary where each node maps to its neighbors:

```python
# Build adjacency list from edge list
from collections import defaultdict
graph = defaultdict(list)
for u, v in edges:
    graph[u].append(v)
    graph[v].append(u)     # remove this line for directed graphs
```

> 🎬 Visualize: [pathfinding.js.org](https://qiao.github.io/PathFinding.js/visual/) — watch BFS vs DFS explore grids in real-time

---

## Pattern 14: BFS — Shortest Path & Level-by-Level

### The Core Idea

> **"Explore all neighbors at distance 1 first, then distance 2, then distance 3... This naturally finds the shortest path."**

**Analogy:** Drop a stone in a pond. Ripples expand outward in circles, reaching closer points first. BFS explores exactly like these ripples.

### The BFS Template

```python
from collections import deque
def bfs(graph, start):
    visited = {start}
    q = deque([start])
    while q:
        node = q.popleft()            # process nearest unvisited node
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)  # mark BEFORE adding to queue
                q.append(neighbor)
```

### When to Use BFS

```
✅ Shortest path in UNWEIGHTED graphs
✅ Level-by-level processing
✅ Nearest/closest queries
✅ Multi-source: start BFS from MULTIPLE points simultaneously
```

### Number of Islands (LeetCode #200)

**The Concept:** Each '1' cell is land, each '0' is water. An "island" is a group of connected '1's. Scan the grid; when you find an unvisited '1', BFS outward to mark the entire island, and increment your counter.

```python
def numIslands(grid):
    rows, cols = len(grid), len(grid[0])
    count = 0
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == '1':
                count += 1             # found a new island!
                q = deque([(r, c)])
                grid[r][c] = '0'       # mark as visited
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

**The Concept:** Multiple rotten oranges spread rot simultaneously. This is **multi-source BFS**: start with ALL rotten oranges in the queue at once. Each "level" of BFS represents one minute of rotting.

```python
def orangesRotting(grid):
    R, C = len(grid), len(grid[0])
    q = deque()
    fresh = 0
    for r in range(R):
        for c in range(C):
            if grid[r][c] == 2: q.append((r, c, 0))   # all rotten → queue
            elif grid[r][c] == 1: fresh += 1
    time = 0
    while q:
        r, c, t = q.popleft()
        for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
            nr, nc = r+dr, c+dc
            if 0<=nr<R and 0<=nc<C and grid[nr][nc] == 1:
                grid[nr][nc] = 2
                fresh -= 1
                time = t + 1
                q.append((nr, nc, t + 1))
    return time if fresh == 0 else -1
```

---

## Pattern 15: DFS — All Paths & Cycle Detection

### The Core Idea

> **"Go as deep as possible down one path before backtracking and trying another. Keep track of the state of each node to detect cycles."**

### Three States for Cycle Detection

This is a crucial concept for directed graphs:

```
0 = UNVISITED   — haven't touched this node yet
1 = VISITING    — currently exploring this node's subtree (on the current path)
2 = VISITED     — fully processed, all descendants explored

If you reach a node in state 1 → CYCLE FOUND! (you've come back to where you currently are)
```

### Course Schedule (LeetCode #207)

**The Concept:** Can you finish all courses given prerequisites? This is really asking: **does the prerequisite graph have a cycle?** If courses form a cycle (A requires B, B requires C, C requires A), it's impossible.

```python
def canFinish(n, prereqs):
    graph = defaultdict(list)
    for course, prereq in prereqs:
        graph[prereq].append(course)
    state = [0] * n                    # 0=unvisited, 1=visiting, 2=done
    
    def has_cycle(node):
        if state[node] == 1: return True     # back edge → CYCLE!
        if state[node] == 2: return False    # already fully explored
        state[node] = 1                      # mark as "currently exploring"
        for neighbor in graph[node]:
            if has_cycle(neighbor): return True
        state[node] = 2                      # mark as "fully done"
        return False
    
    return not any(has_cycle(i) for i in range(n))
```

### Dijkstra's Algorithm — Shortest Path in Weighted Graphs

**The Concept:** BFS finds shortest paths in unweighted graphs, but what about weighted edges? Dijkstra's algorithm uses a **priority queue (min-heap)** to always process the node with the smallest known distance first.

**Why a heap?** You always want to expand the closest unprocessed node — a heap gives you the minimum in O(log n).

```python
import heapq
def dijkstra(graph, start, n):
    dist = [float('inf')] * n
    dist[start] = 0
    heap = [(0, start)]                # (distance, node)
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]: continue       # stale entry, skip
        for v, weight in graph[u]:
            if dist[u] + weight < dist[v]:
                dist[v] = dist[u] + weight
                heapq.heappush(heap, (dist[v], v))
    return dist
# O((V + E) log V)
```

---

# 🧮 1:30 — Dynamic Programming (30 min)

## What is Dynamic Programming? (The Hardest Concept to Learn)

DP is arguably the most feared interview topic, but at its core, it's a simple idea:

> **"If you're solving the same subproblem multiple times, solve it ONCE and save the answer for later."**

### DP = Recursion + Caching (Memoization)

**Without DP:**
```
fib(5) calls fib(4) and fib(3)
fib(4) calls fib(3) and fib(2)      ← fib(3) computed TWICE!
fib(3) calls fib(2) and fib(1)      ← fib(2) computed THREE times!
Total calls: exponential O(2ⁿ) 💀
```

**With DP:**
```
fib(5): compute fib(1)=1, fib(2)=1, fib(3)=2, fib(4)=3, fib(5)=5
Each fib(i) computed ONCE → O(n) ✅
```

### Two Approaches to DP

1. **Top-down (Memoization):** Write the natural recursive solution, then add a cache
2. **Bottom-up (Tabulation):** Fill a table from the base case forward (usually preferred in interviews — no recursion depth issues)

### The 4-Step DP Recipe

This works for ANY DP problem:

```
1. DEFINE STATE  → What info uniquely describes a subproblem?
                   (usually: index, remaining capacity, length so far)
2. RECURRENCE  → How does dp[i] relate to previous values?
                   dp[i] = some function of dp[i-1], dp[i-2], etc.
3. BASE CASE   → What are the trivially known answers?
                   dp[0] = ?, dp[1] = ?
4. DIRECTION   → Fill from base case forward (bottom-up)
                   OR recurse from the answer backward (top-down)
```

> 🎬 Visualize: [visualgo.net/dp](https://visualgo.net/en/recursion) — watch DP table filling step by step

---

## Pattern 16: 1D DP — Linear Sequence Optimization

### The Core Idea

> **"The answer for position `i` depends on the answers for a few previous positions."**

### Climbing Stairs (LeetCode #70) — The Gateway to DP

**The Concept:** You can climb 1 or 2 stairs at a time. How many ways to reach step `n`? At step `n`, you either came from step `n-1` (took 1 step) or step `n-2` (took 2 steps). So:

```
dp[n] = dp[n-1] + dp[n-2]   ← This is literally Fibonacci!
```

**Walkthrough:**
```
Stairs: 1  2  3  4  5
Ways:   1  2  3  5  8

Step 3: could come from step 2 (2 ways) or step 1 (1 way) = 3 ways
Step 4: could come from step 3 (3 ways) or step 2 (2 ways) = 5 ways
```

```python
def climbStairs(n):
    if n <= 2: return n
    a, b = 1, 2                        # dp[1]=1, dp[2]=2
    for _ in range(3, n+1):
        a, b = b, a + b               # dp[i] = dp[i-1] + dp[i-2]
    return b
# O(n) time, O(1) space — only need the last two values!
```

### House Robber (LeetCode #198)

**The Concept:** You can't rob two adjacent houses. At each house, you choose: **rob it** (get its value + best from two houses ago) or **skip it** (keep the best from the previous house).

```
dp[i] = max(dp[i-1],          ← skip this house
             dp[i-2] + nums[i])  ← rob this house + best before adjacent
```

```python
def rob(nums):
    if len(nums) <= 2: return max(nums)
    a, b = nums[0], max(nums[0], nums[1])
    for i in range(2, len(nums)):
        a, b = b, max(b, a + nums[i])   # skip or rob?
    return b
```

### Coin Change (LeetCode #322)

**The Concept:** Find the minimum number of coins to make the target amount. For each amount `i`, try every coin denomination — `dp[i] = min(dp[i - coin] + 1)` across all coins.

**Analogy:** Making change. To make \$11 with coins [1, 5, 6]: the answer for \$11 = 1 + min(answer for \$10, answer for \$6, answer for \$5). Try each coin and take the best.

```python
def coinChange(coins, amount):
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0                          # base case: 0 coins for amount 0
    for i in range(1, amount + 1):
        for c in coins:
            if c <= i:
                dp[i] = min(dp[i], dp[i - c] + 1)
    return dp[amount] if dp[amount] != float('inf') else -1
# O(amount × number_of_coins)
```

### Longest Increasing Subsequence — LIS (LeetCode #300)

**The Concept:** `dp[i]` = length of the longest increasing subsequence *ending at index i*. For each `i`, check all previous `j < i` — if `nums[j] < nums[i]`, you can extend that subsequence.

```python
def lengthOfLIS(nums):
    dp = [1] * len(nums)               # every element is a subsequence of length 1
    for i in range(1, len(nums)):
        for j in range(i):
            if nums[j] < nums[i]:
                dp[i] = max(dp[i], dp[j] + 1)  # extend j's subsequence
    return max(dp)
# O(n²) — can be optimized to O(n log n) with binary search (patience sorting)
```

---

## Pattern 17: 2D DP — Grids & Two-Sequence Comparison

### The Core Idea

> **"When the state needs two variables (two indices, grid position, etc.), use a 2D table where `dp[i][j]` represents the answer for the subproblem defined by `i` and `j`."**

### Unique Paths (LeetCode #62)

**The Concept:** On a grid, you can only move right or down. The number of ways to reach cell (i,j) = ways to reach from above + ways to reach from the left.

```
dp[i][j] = dp[i-1][j] + dp[i][j-1]
```

```python
def uniquePaths(m, n):
    dp = [[1]*n for _ in range(m)]     # first row and column are all 1s
    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = dp[i-1][j] + dp[i][j-1]
    return dp[m-1][n-1]
```

### ⭐ Longest Common Subsequence — LCS (LeetCode #1143)

**The Concept:** Compare two strings character by character. If characters match, extend the sequence from the diagonal. If not, take the best from either dropping a character from string 1 or string 2.

**Walkthrough for "ace" and "abcde":**
```
     ""  a  b  c  d  e
  ""  0  0  0  0  0  0
  a   0  1  1  1  1  1    ← 'a' matches 'a', count goes to 1
  c   0  1  1  2  2  2    ← 'c' matches 'c', count goes to 2
  e   0  1  1  2  2  3    ← 'e' matches 'e', count goes to 3
```

```python
def longestCommonSubsequence(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(1, m+1):
        for j in range(1, n+1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1      # characters match!
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])  # skip one char
    return dp[m][n]
# O(m × n)
```

### Edit Distance (LeetCode #72)

**The Concept:** Minimum operations (insert, delete, replace) to turn one word into another. This is the classic 2D DP problem that powers spell checkers, diffs, and DNA comparison.

- If characters match: no operation needed, take diagonal
- If not: try all three operations, take the minimum + 1

```python
def minDistance(w1, w2):
    m, n = len(w1), len(w2)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m+1): dp[i][0] = i   # deleting all of w1
    for j in range(n+1): dp[0][j] = j   # inserting all of w2
    for i in range(1, m+1):
        for j in range(1, n+1):
            if w1[i-1] == w2[j-1]:
                dp[i][j] = dp[i-1][j-1]           # match: no cost
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j],        # delete from w1
                    dp[i][j-1],        # insert into w1
                    dp[i-1][j-1]       # replace
                )
    return dp[m][n]
```

### Partition Equal Subset Sum (LeetCode #416)

**The Concept:** Can you divide the array into two subsets with equal sum? This reduces to: can you find a subset that sums to `total/2`? (A variant of the 0/1 Knapsack problem.)

```python
def canPartition(nums):
    total = sum(nums)
    if total % 2: return False         # odd total → impossible
    target = total // 2
    achievable = {0}                   # set of achievable sums
    for n in nums:
        achievable = achievable | {x + n for x in achievable}
    return target in achievable
```

---

## 🧠 DP Decision Guide

When you see a DP problem, identify the structure:

```
Single sequence / linear choices?    → 1D DP   (House Robber, Climbing Stairs, LIS)
Two strings to compare?             → 2D DP   (LCS, Edit Distance)
Grid movement?                      → 2D DP   (Unique Paths, Min Path Sum)
Items with weight/capacity?         → Knapsack (Coin Change, Subset Sum)
Optimization + overlapping subs?    → DP       (if greedy doesn't work, try DP)
Can decompose into choices?         → DP       (at each step, what are my options?)
```

### How to Know It's DP

1. The problem asks for **optimal** (min/max) or **count** of ways
2. Making a sequence of **choices** leads to the answer
3. The same subproblems are solved **repeatedly** in the brute force
4. **Greedy doesn't work** (local optimal ≠ global optimal)

---

# ✅ Day 2 Summary — Patterns 8-17

| # | Pattern | Core Insight | When to Use | Top Problem |
|---|---------|-------------|-------------|-------------|
| 8 | **Slow/Fast Pointers** | Different speeds reveal structure | Cycle, middle of list | Linked List Cycle #141 |
| 9 | **Reverse LL** | Save → Reverse → Advance | Restructuring linked lists | Reverse LL #206 |
| 10 | **Stack Matching** | Push open, pop close | Matching/nesting | Valid Parentheses #20 |
| 11 | **Monotonic Stack** | Maintain order, pop violations | Next greater/smaller | Daily Temps #739, Histogram #84 |
| 12 | **Tree Traversal** | DFS (3 orders) + BFS | Explore tree structure | Level Order #102 |
| 13 | **Recursive Tree** | Solve left + right → combine | Tree properties | LCA #236, Max Path Sum #124 |
| 14 | **BFS Graph** | Queue + visited = shortest path | Shortest path, levels | Islands #200, Rotting Oranges #994 |
| 15 | **DFS Graph** | Go deep, 3-state cycle detect | All paths, cycles | Course Schedule #207 |
| 16 | **1D DP** | dp[i] = f(previous values) | Linear optimization | Coin Change #322, LIS #300 |
| 17 | **2D DP** | dp[i][j] for two-variable state | Grids, string comparison | LCS #1143, Edit Distance #72 |

---

# 🏆 Complete Crash Course Summary — 17 Patterns in 4 Hours

## Quick Pattern Recognition Cheat

Ask yourself these questions to identify the pattern:

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
