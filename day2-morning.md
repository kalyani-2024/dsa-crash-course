# 🌅 Day 2 — Morning Session (9:00 AM - 12:30 PM)

## Linked Lists → Stacks & Queues → Trees → BST → Heaps

---

# 🔗 Part 12: Linked Lists — Pointer Manipulation (45 min)

## What is a Linked List?

Unlike arrays (contiguous memory), a linked list uses **nodes connected by pointers**.

```
Array:    [10][20][30][40][50]     ← Contiguous memory, O(1) access

Linked List:
Head → [10|→] → [20|→] → [30|→] → [40|→] → null
       (data|next)                  ← Non-contiguous, O(n) access
```

### 🎬 **Visualize it:** [visualgo.net/list](https://visualgo.net/en/list) — Watch insertions, deletions, reversals

### Node Definition

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
```

### Array vs Linked List

| Operation | Array | Linked List |
|-----------|-------|-------------|
| Access by index | O(1) ✅ | O(n) ❌ |
| Insert at beginning | O(n) ❌ | O(1) ✅ |
| Insert at end | O(1)* | O(n) or O(1) with tail |
| Delete by value | O(n) | O(n) |
| Memory | Contiguous | Scattered |

---

## 🧩 Pattern 12: Slow & Fast Pointers (Floyd's Tortoise and Hare)

**This is THE most important linked list technique.**

```
Slow moves 1 step, Fast moves 2 steps:

Start:  S,F → [1] → [2] → [3] → [4] → [5] → null
Step 1:        S      F
Step 2:               S           F
Step 3:                      S              F (at null)

When fast reaches end, slow is at the MIDDLE!
```

### Problem: Middle of Linked List (LeetCode #876)
```python
def middleNode(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    return slow  # slow is at the middle
# Time: O(n) | Space: O(1)
```

### Problem: Detect Cycle (LeetCode #141)
```python
def hasCycle(head):
    """Fast pointer will meet slow pointer if cycle exists."""
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True
    return False
# Time: O(n) | Space: O(1)
```

### Problem: Find Cycle Start (LeetCode #142)
```python
def detectCycle(head):
    """After detecting cycle, reset one pointer to head.
    Both move at speed 1 — they meet at cycle start."""
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            # Reset one pointer to head
            slow = head
            while slow != fast:
                slow = slow.next
                fast = fast.next
            return slow
    return None
```

## 🧩 Pattern 13: Reverse a Linked List

### Problem: Reverse Linked List (LeetCode #206) — Top 5 Interview Question

```
Before: 1 → 2 → 3 → 4 → 5 → null
After:  5 → 4 → 3 → 2 → 1 → null

Step by step:
prev=null, curr=1
  next=2, 1→null, prev=1, curr=2
prev=1,    curr=2
  next=3, 2→1, prev=2, curr=3
prev=2,    curr=3
  next=4, 3→2, prev=3, curr=4
...and so on
```

```python
def reverseList(head):
    """Three-pointer reversal. MUST memorize this."""
    prev = None
    curr = head
    
    while curr:
        next_node = curr.next   # Save next
        curr.next = prev        # Reverse pointer
        prev = curr             # Advance prev
        curr = next_node        # Advance curr
    
    return prev  # New head
# Time: O(n) | Space: O(1)
```

### Problem: Reverse Nodes in K-Group (LeetCode #25) — ⭐ Hard
```python
def reverseKGroup(head, k):
    """Reverse every k nodes. Use dummy node for cleaner code."""
    # Count total nodes
    count = 0
    node = head
    while node:
        count += 1
        node = node.next
    
    dummy = ListNode(0)
    dummy.next = head
    group_prev = dummy
    
    while count >= k:
        tail = group_prev.next  # Will become tail after reversal
        prev = None
        curr = group_prev.next
        
        for _ in range(k):
            next_node = curr.next
            curr.next = prev
            prev = curr
            curr = next_node
        
        group_prev.next = prev     # Connect to reversed head
        tail.next = curr           # Connect tail to next group
        group_prev = tail          # Move for next iteration
        count -= k
    
    return dummy.next
```

### Problem: Merge Two Sorted Lists (LeetCode #21)
```python
def mergeTwoLists(l1, l2):
    dummy = ListNode(0)
    curr = dummy
    
    while l1 and l2:
        if l1.val <= l2.val:
            curr.next = l1
            l1 = l1.next
        else:
            curr.next = l2
            l2 = l2.next
        curr = curr.next
    
    curr.next = l1 or l2  # Append remaining
    return dummy.next
```

> **🎯 Pro Tip:** When manipulating linked lists, always draw it out on paper first! It saves hours of debugging.

---

# 📚 Part 13: Stacks & Queues — LIFO vs FIFO (45 min)

## Stack (LIFO — Last In, First Out)

```
Think: Stack of plates 🍽️
Push → [5]  [3]  [7]  [2]  ← Top
Pop ← removes 2 (last added)

Operations (all O(1)):
  push(x)   — add to top
  pop()     — remove from top
  peek()    — see top without removing
  isEmpty() — check if empty
```

### 🎬 **Visualize it:** [cs.usfca.edu - Stack](https://www.cs.usfca.edu/~galles/visualization/StackArray.html)

```python
# Python: use list as stack
stack = []
stack.append(5)  # push
stack.append(3)  # push
stack.pop()      # returns 3
stack[-1]        # peek → 5
```

## Queue (FIFO — First In, First Out)

```
Think: Line at a restaurant 🧑‍🤝‍🧑
Enqueue → [A][B][C][D] → Dequeue
First person in line gets served first.
```

```python
from collections import deque
queue = deque()
queue.append(1)    # enqueue (right end)
queue.popleft()    # dequeue (left end) — O(1)!
# Don't use list.pop(0) — it's O(n)!
```

## 🧩 Pattern 14: Stack for Matching/Nesting

### Problem: Valid Parentheses (LeetCode #20)
```python
def isValid(s):
    """Push opening brackets, pop for closing, check match."""
    stack = []
    matching = {')': '(', '}': '{', ']': '['}
    
    for char in s:
        if char in '({[':
            stack.append(char)
        elif char in ')}]':
            if not stack or stack[-1] != matching[char]:
                return False
            stack.pop()
    
    return len(stack) == 0
# Time: O(n) | Space: O(n)
```

## 🧩 Pattern 15: Monotonic Stack — Next Greater/Smaller Element

**The most powerful stack pattern.** When you need "next greater/smaller element" for each position.

```
arr = [2, 1, 2, 4, 3]
Next Greater Element for each:

For 2: next greater = 4
For 1: next greater = 2
For 2: next greater = 4
For 4: next greater = -1 (none)
For 3: next greater = -1

Answer: [4, 2, 4, -1, -1]
```

```python
def nextGreaterElements(nums):
    """Monotonic DECREASING stack (stores candidates)."""
    n = len(nums)
    result = [-1] * n
    stack = []  # Stores INDICES
    
    for i in range(n):
        # Pop all elements smaller than current
        while stack and nums[stack[-1]] < nums[i]:
            idx = stack.pop()
            result[idx] = nums[i]  # nums[i] is the next greater
        stack.append(i)
    
    return result
# Time: O(n) | Space: O(n)
```

### Problem: Largest Rectangle in Histogram (LeetCode #84) — ⭐ Hard

```python
def largestRectangleArea(heights):
    """For each bar as the shortest, find max width it spans."""
    stack = []  # Stores indices
    max_area = 0
    
    for i in range(len(heights) + 1):
        h = heights[i] if i < len(heights) else 0  # Sentinel
        
        while stack and heights[stack[-1]] > h:
            height = heights[stack.pop()]
            width = i if not stack else i - stack[-1] - 1
            max_area = max(max_area, height * width)
        
        stack.append(i)
    
    return max_area
# Time: O(n) | Space: O(n)
```

### Problem: Daily Temperatures (LeetCode #739)
```python
def dailyTemperatures(temperatures):
    """How many days until a warmer temperature?"""
    n = len(temperatures)
    result = [0] * n
    stack = []  # Stores indices of decreasing temperatures
    
    for i in range(n):
        while stack and temperatures[i] > temperatures[stack[-1]]:
            idx = stack.pop()
            result[idx] = i - idx
        stack.append(i)
    
    return result
```

---

# 🌳 Part 14: Binary Trees — The Heart of DSA (60 min)

## What is a Binary Tree?

Each node has at most **two children** (left and right).

```
        1           ← Root
       / \
      2    3        ← Level 1
     / \    \
    4   5    6      ← Level 2 (Leaves: 4, 5, 6)
```

### 🎬 **Visualize it:** [visualgo.net/bst](https://visualgo.net/en/bst) — Build and traverse trees
### 🎬 **Visualize it:** [cs.usfca.edu - BST](https://www.cs.usfca.edu/~galles/visualization/BST.html)

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
```

## 🧩 Pattern 16: Tree Traversals (MUST KNOW ALL 4)

```
Tree:     1
         / \
        2    3
       / \
      4    5

Inorder   (Left, Root, Right): 4, 2, 5, 1, 3  ← Sorted for BST!
Preorder  (Root, Left, Right): 1, 2, 4, 5, 3  ← Copy/serialize tree
Postorder (Left, Right, Root): 4, 5, 2, 3, 1  ← Delete tree
Level Order (BFS):             1 | 2, 3 | 4, 5  ← Level by level
```

```python
# Recursive Traversals
def inorder(root):
    if not root: return []
    return inorder(root.left) + [root.val] + inorder(root.right)

def preorder(root):
    if not root: return []
    return [root.val] + preorder(root.left) + preorder(root.right)

def postorder(root):
    if not root: return []
    return postorder(root.left) + postorder(root.right) + [root.val]

# Level Order Traversal (BFS) — LeetCode #102
from collections import deque

def levelOrder(root):
    if not root: return []
    result = []
    queue = deque([root])
    
    while queue:
        level = []
        for _ in range(len(queue)):  # Process entire level
            node = queue.popleft()
            level.append(node.val)
            if node.left:  queue.append(node.left)
            if node.right: queue.append(node.right)
        result.append(level)
    
    return result
```

## 🧩 Pattern 17: Recursive Tree Properties

### Problem: Maximum Depth (LeetCode #104)
```python
def maxDepth(root):
    if not root:
        return 0
    return 1 + max(maxDepth(root.left), maxDepth(root.right))
# Time: O(n) | Space: O(h) where h = height
```

### Problem: Diameter of Binary Tree (LeetCode #543)
```python
def diameterOfBinaryTree(root):
    """Diameter = longest path between any two nodes."""
    diameter = 0
    
    def height(node):
        nonlocal diameter
        if not node:
            return 0
        left_h = height(node.left)
        right_h = height(node.right)
        diameter = max(diameter, left_h + right_h)  # Update!
        return 1 + max(left_h, right_h)
    
    height(root)
    return diameter
```

### Problem: Lowest Common Ancestor (LeetCode #236)
```python
def lowestCommonAncestor(root, p, q):
    """If I find p or q, return it. If both sides return,
    current node is the LCA."""
    if not root or root == p or root == q:
        return root
    
    left = lowestCommonAncestor(root.left, p, q)
    right = lowestCommonAncestor(root.right, p, q)
    
    if left and right:
        return root   # Both found → current is LCA
    return left or right
# Time: O(n) | Space: O(h)
```

### Problem: Maximum Path Sum (LeetCode #124) — ⭐ Hard
```python
def maxPathSum(root):
    max_sum = float('-inf')
    
    def helper(node):
        nonlocal max_sum
        if not node:
            return 0
        
        left = max(0, helper(node.left))    # Ignore negative paths
        right = max(0, helper(node.right))
        
        # Path through this node
        max_sum = max(max_sum, node.val + left + right)
        
        # Return single-direction max for parent to use
        return node.val + max(left, right)
    
    helper(root)
    return max_sum
```

---

## 🔍 Binary Search Tree (BST) — Trees with Ordering

**BST Property:** Left subtree < Node < Right subtree

```
        8
       / \
      3   10
     / \    \
    1   6   14

Inorder traversal = [1, 3, 6, 8, 10, 14]  ← SORTED!
```

### Problem: Validate BST (LeetCode #98)
```python
def isValidBST(root, lo=float('-inf'), hi=float('inf')):
    if not root:
        return True
    if root.val <= lo or root.val >= hi:
        return False
    return (isValidBST(root.left, lo, root.val) and
            isValidBST(root.right, root.val, hi))
```

### Problem: Kth Smallest Element in BST (LeetCode #230)
```python
def kthSmallest(root, k):
    """Inorder traversal gives sorted order. Return kth element."""
    stack = []
    curr = root
    
    while stack or curr:
        while curr:
            stack.append(curr)
            curr = curr.left
        
        curr = stack.pop()
        k -= 1
        if k == 0:
            return curr.val
        curr = curr.right
```

---

# ⛰️ Part 15: Heaps / Priority Queues — Top-K Problems (30 min)

## What is a Heap?

A **complete binary tree** where parent ≤ children (min-heap) or parent ≥ children (max-heap).

```
Min-Heap:        Max-Heap:
     1                9
    / \              / \
   3   5            7   8
  / \              / \
 7   9            3   5
```

### 🎬 **Visualize it:** [cs.usfca.edu - Heap](https://www.cs.usfca.edu/~galles/visualization/Heap.html) — Watch heap operations

```python
import heapq

# Python heapq = MIN HEAP
heap = []
heapq.heappush(heap, 5)
heapq.heappush(heap, 3)
heapq.heappush(heap, 7)
print(heapq.heappop(heap))  # 3 (smallest)

# For MAX HEAP: negate values
heapq.heappush(heap, -5)
max_val = -heapq.heappop(heap)  # 5
```

## 🧩 Pattern 18: Top-K / Kth Largest

### Problem: Kth Largest Element (LeetCode #215)
```python
def findKthLargest(nums, k):
    """Min-heap of size k. Top of heap = kth largest."""
    heap = nums[:k]
    heapq.heapify(heap)
    
    for num in nums[k:]:
        if num > heap[0]:
            heapq.heapreplace(heap, num)
    
    return heap[0]
# Time: O(n log k) | Space: O(k)
```

### Problem: Top K Frequent Elements (LeetCode #347)
```python
def topKFrequent(nums, k):
    freq = Counter(nums)
    return heapq.nlargest(k, freq.keys(), key=freq.get)
# Time: O(n log k) | Space: O(n)
```

### Problem: Merge K Sorted Lists (LeetCode #23) — ⭐ Hard
```python
def mergeKLists(lists):
    """Min-heap with one element from each list."""
    heap = []
    for i, lst in enumerate(lists):
        if lst:
            heapq.heappush(heap, (lst.val, i, lst))
    
    dummy = curr = ListNode(0)
    
    while heap:
        val, i, node = heapq.heappop(heap)
        curr.next = node
        curr = curr.next
        if node.next:
            heapq.heappush(heap, (node.next.val, i, node.next))
    
    return dummy.next
# Time: O(N log k) where N = total nodes, k = number of lists
```

### Problem: Find Median from Data Stream (LeetCode #295)
```python
class MedianFinder:
    """Two heaps: max-heap (left half) + min-heap (right half)."""
    def __init__(self):
        self.left = []   # Max-heap (negate values)
        self.right = []  # Min-heap
    
    def addNum(self, num):
        heapq.heappush(self.left, -num)
        heapq.heappush(self.right, -heapq.heappop(self.left))
        
        if len(self.right) > len(self.left):
            heapq.heappush(self.left, -heapq.heappop(self.right))
    
    def findMedian(self):
        if len(self.left) > len(self.right):
            return -self.left[0]
        return (-self.left[0] + self.right[0]) / 2
# addNum: O(log n) | findMedian: O(1)
```

---

## ☕ Morning Review — Patterns 12-18

| # | Pattern | Key Technique | Problems It Solves |
|---|---------|---------------|-------------------|
| 12 | Slow/Fast Pointer | Two speeds | Cycle detection, find middle |
| 13 | Reverse Linked List | 3-pointer swap | Reverse, palindrome check |
| 14 | Stack Matching | Push/pop pairs | Parentheses, expression eval |
| 15 | Monotonic Stack | Maintain order | Next greater, histogram area |
| 16 | Tree Traversal | DFS (3 types) + BFS | Almost every tree problem |
| 17 | Recursive Tree | Base + recurse | Height, diameter, LCA |
| 18 | Heap / Top-K | Min/max heap | Kth element, merge sorted |

---

*Lunch break! 🍕 Continue with [day2-afternoon.md](day2-afternoon.md) for Graphs, DP, and Greedy!*

[← Day 1 Practice](day1-practice.md) | [Back to Schedule](README.md) | [Next: Day 2 Afternoon →](day2-afternoon.md)
