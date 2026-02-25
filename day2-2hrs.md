# Day 2 -- HashMaps, Linked Lists, Stacks, Queues, Sorting, and Binary Search

## Core Data Structures and Searching Techniques

**What this day covers:** Hash Maps and Sets (O(1) lookup, frequency counting, grouping), Linked Lists (traversal, reversal, slow/fast pointers, merge), Stacks (matching, monotonic stack), Queues (BFS, level-by-level processing), Sorting as a preprocessing step, Binary Search (standard and on-answer), and Bit Manipulation tricks.

These data structures and techniques come up constantly in interviews. Understanding when and why to reach for each one is just as important as knowing how they work.

---

# Hash Maps and Sets

## What is Hashing?

Hashing is arguably the single most important concept in DSA interviews. A HashMap (dictionary) takes any data and computes a "shelf code" (hash) that maps directly to a memory location.

Think of it like a library where every book has a shelf code. Instead of searching sequentially, you go directly to the right shelf.

### Key Operations -- All O(1) Average

| Operation | HashMap (dict) | HashSet (set) |
|-----------|----------------|---------------|
| Insert | `d[key] = val` | `s.add(val)` |
| Lookup | `key in d` / `d[key]` | `val in s` |
| Delete | `del d[key]` | `s.remove(val)` |

### When to Use

```
HashMap: Need to store KEY -> VALUE pairs     (count occurrences, index lookup)
HashSet: Need to check MEMBERSHIP only       (seen before? duplicate? exists?)
```

### The Universal HashMap Pattern

> "Am I checking `if X exists` inside a loop? Then use a HashMap to make it O(1)."

```
Without HashMap: For each element, scan the rest  -> O(n^2)
With HashMap:    For each element, check the map   -> O(n)
```

---

## Pattern 7: Frequency Counting and Lookup

### Contains Duplicate (LeetCode #217)

```python
def containsDuplicate(nums):
    return len(nums) != len(set(nums))
```

### Longest Consecutive Sequence (LeetCode #128)

**The Concept:** Only start counting from the beginning of a sequence (where `n-1` is NOT in the set).

```python
def longestConsecutive(nums):
    s = set(nums)
    best = 0
    for n in s:
        if n - 1 not in s:            # START of a sequence
            length = 0
            while n + length in s:
                length += 1
            best = max(best, length)
    return best
# O(n) -- each number visited at most twice
```

### Top K Frequent Elements (LeetCode #347)

**The Concept:** Count frequencies with a HashMap, then find the top K. Bucket sort avoids the O(n log n) of a regular sort.

```python
def topKFrequent(nums, k):
    count = Counter(nums)
    # Bucket sort: index = frequency, value = list of numbers
    buckets = [[] for _ in range(len(nums) + 1)]
    for num, freq in count.items():
        buckets[freq].append(num)
    res = []
    for freq in range(len(buckets) - 1, -1, -1):
        for num in buckets[freq]:
            res.append(num)
            if len(res) == k: return res
# O(n) -- no sorting needed!
```

---

# Linked Lists

## What is a Linked List?

Unlike arrays (contiguous memory), a linked list stores elements scattered across memory, connected by pointers -- each node knows where the next one is.

```
Array:   [10][20][30][40]     -> accessed by index (O(1))
Linked:  10 -> 20 -> 30 -> 40 -> None   -> accessed by walking (O(n))
```

### Why Use Linked Lists?

| | Array | Linked List |
|---|-------|-------------|
| Access by index | O(1) | O(n) |
| Insert/delete at head | O(n) | O(1) |
| Insert/delete at known position | O(n) | O(1) |
| Memory | Fixed block | Scattered, flexible |

### The Node

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
```

Visualize linked list operations: [visualgo.net/list](https://visualgo.net/en/list)

---

## Pattern 8: Slow and Fast Pointers (Floyd's Tortoise and Hare)

### The Core Idea

> "Two pointers at different speeds: fast reaches the end in half the time, so slow is at the middle. On a cycle, fast laps slow, so a cycle is detected."

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

---

## Pattern 9: Reverse a Linked List -- The 3-Pointer Technique

### The Core Idea

> "Walk through the list, reversing each arrow to point backwards. Need three pointers: where you came from (prev), where you are (curr), where you're going (nxt)."

### Reverse Linked List (LeetCode #206) -- Top 5 Interview Question

```
prev=None  curr=1->2->3->None
Step 1: save nxt=2, point 1->None,  prev=1, curr=2
Step 2: save nxt=3, point 2->1,     prev=2, curr=3
Step 3: save nxt=None, point 3->2,  prev=3, curr=None -> done!
Result: 3->2->1->None
```

```python
def reverseList(head):
    prev = None
    curr = head
    while curr:
        nxt = curr.next     # 1. SAVE
        curr.next = prev    # 2. REVERSE
        prev = curr         # 3. ADVANCE
        curr = nxt
    return prev
# O(n) time, O(1) space -- MEMORIZE THIS
```

### Merge Two Sorted Lists (LeetCode #21)

**Key technique -- Dummy Node:** Creates a fake start node to simplify edge cases.

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

**The Concept:** Combines three linked list patterns in one problem:
1. Find middle (slow/fast)
2. Reverse second half (3-pointer)
3. Interleave the two halves

```python
def reorderList(head):
    # 1. Find middle
    slow = fast = head
    while fast.next and fast.next.next:
        slow = slow.next; fast = fast.next.next
    # 2. Reverse second half
    prev, curr = None, slow.next
    slow.next = None
    while curr:
        nxt = curr.next; curr.next = prev; prev = curr; curr = nxt
    # 3. Interleave
    first, second = head, prev
    while second:
        tmp1, tmp2 = first.next, second.next
        first.next = second; second.next = tmp1
        first, second = tmp1, tmp2
```

**Linked List Recipe:** Find middle, reverse half, merge or compare.

---

# Stacks and Queues

## What Are Stacks and Queues?

Both are restricted access data structures -- you can only add or remove from specific ends.

### Stack = LIFO (Last In, First Out)

**Analogy:** Stack of plates -- add and remove from the top only.

**Used for:** Undo operations, matching brackets, DFS, expression evaluation, function call stack.

### Queue = FIFO (First In, First Out)

**Analogy:** Line at a theater -- first person in line is served first.

**Used for:** BFS traversal, task scheduling, level-by-level processing.

```python
# Stack -- Python list
stack = []
stack.append(x)   # push O(1)
stack.pop()       # pop O(1)
stack[-1]         # peek O(1)

# Queue -- ALWAYS use deque (list.pop(0) is O(n)!)
from collections import deque
q = deque()
q.append(x)       # enqueue O(1)
q.popleft()       # dequeue O(1)
```

---

## Pattern 10: Stack for Matching and Nesting

### The Core Idea

> "When you see an opening element, push it. When you see a closing element, pop and check if they match. Stacks naturally handle nesting."

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

---

## Pattern 11: Monotonic Stack -- "Next Greater Element"

### The Core Idea

> "Maintain a stack in decreasing order. When a bigger element arrives, pop -- the popped element just found its answer."

**Why O(n)?** Each element is pushed at most once and popped at most once, giving O(2n) = O(n) total.

### Daily Temperatures (LeetCode #739)

```python
def dailyTemperatures(temps):
    n = len(temps)
    res = [0] * n
    stack = []
    for i in range(n):
        while stack and temps[i] > temps[stack[-1]]:
            j = stack.pop()
            res[j] = i - j
        stack.append(i)
    return res
# O(n)
```

### Largest Rectangle in Histogram (LeetCode #84)

```python
def largestRectangleArea(heights):
    stack, best = [], 0
    for i in range(len(heights) + 1):
        h = heights[i] if i < len(heights) else 0
        while stack and heights[stack[-1]] > h:
            height = heights[stack.pop()]
            width = i if not stack else i - stack[-1] - 1
            best = max(best, height * width)
        stack.append(i)
    return best
```

### Min Stack (LeetCode #155)

**The Concept:** Store `(value, current_min)` pairs so getMin is always O(1).

```python
class MinStack:
    def __init__(self):
        self.stack = []
    def push(self, val):
        mn = min(val, self.stack[-1][1]) if self.stack else val
        self.stack.append((val, mn))
    def pop(self): self.stack.pop()
    def top(self): return self.stack[-1][0]
    def getMin(self): return self.stack[-1][1]
```

---

# Sorting and Binary Search

## Why Sorting Matters

You rarely implement sorts yourself, but sorting as a preprocessing step unlocks many techniques:

```
Sorted -> Binary Search      O(n log n + log n)
Sorted -> Two Pointers       O(n log n + n)
Sorted -> Merge Intervals    O(n log n + n)
Sorted -> Greedy decisions   O(n log n + n)
Sorted -> Duplicates adjacent
```

### Dutch National Flag (LeetCode #75) -- Sort 0s, 1s, 2s in One Pass

```python
def sortColors(nums):
    lo, mid, hi = 0, 0, len(nums) - 1
    while mid <= hi:
        if   nums[mid] == 0: nums[lo], nums[mid] = nums[mid], nums[lo]; lo += 1; mid += 1
        elif nums[mid] == 1: mid += 1
        else:                nums[mid], nums[hi] = nums[hi], nums[mid]; hi -= 1
```

### Merge Intervals (LeetCode #56)

```python
def merge(intervals):
    intervals.sort()
    res = [intervals[0]]
    for s, e in intervals[1:]:
        if s <= res[-1][1]: res[-1][1] = max(res[-1][1], e)
        else: res.append([s, e])
    return res
```

---

## Pattern 12: Binary Search -- Halve the Search Space

### The Core Idea

> "If you can determine which half contains the answer, throw away the other half. Repeat. O(log n)."

### Standard Binary Search

```python
def binary_search(arr, target):
    lo, hi = 0, len(arr) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if   arr[mid] == target: return mid
        elif arr[mid] < target:  lo = mid + 1
        else:                    hi = mid - 1
    return -1
```

### Search in Rotated Sorted Array (LeetCode #33)

**The Concept:** At any mid, one half is always sorted. Check if target is in the sorted half.

```python
def search(nums, target):
    lo, hi = 0, len(nums) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if nums[mid] == target: return mid
        if nums[lo] <= nums[mid]:
            if nums[lo] <= target < nums[mid]: hi = mid - 1
            else: lo = mid + 1
        else:
            if nums[mid] < target <= nums[hi]: lo = mid + 1
            else: hi = mid - 1
    return -1
```

### Binary Search on Answer

> "Instead of searching an array, search the range of possible answers."

### Koko Eating Bananas (LeetCode #875)

```python
import math
def minEatingSpeed(piles, h):
    lo, hi = 1, max(piles)
    while lo < hi:
        mid = (lo + hi) // 2
        if sum(math.ceil(p / mid) for p in piles) <= h:
            hi = mid
        else:
            lo = mid + 1
    return lo
```

---

# Bit Manipulation

## Four Tricks Worth Knowing

### 1. XOR Cancels Pairs -- Single Number (LeetCode #136)

`a ^ a = 0` and `a ^ 0 = a`. XOR all numbers -- pairs cancel, the unique one remains.

```python
def singleNumber(nums):
    result = 0
    for n in nums: result ^= n
    return result
```

### 2. Check Power of 2

Powers of 2 have exactly one bit set. `n & (n-1)` clears the lowest bit.

```python
def isPowerOfTwo(n):
    return n > 0 and (n & (n-1)) == 0
```

### 3. Count Set Bits (Brian Kernighan)

```python
def countBits(n):
    count = 0
    while n:
        n &= n - 1
        count += 1
    return count
```

### 4. Even/Odd

```python
n & 1 == 0  # even
n & 1 == 1  # odd
```

---

# Day 2 Summary -- 6 Patterns + Key Techniques

| # | Pattern | Core Insight | Key Problem |
|---|---------|-------------|-------------|
| 7 | **HashMap Lookup** | O(1) existence check | Two Sum #1, Consecutive #128 |
| 8 | **Slow/Fast Pointers** | Cycle, middle detection | Cycle #141, Middle #876 |
| 9 | **Reverse LL** | Save, Reverse, Advance | Reverse LL #206 |
| 10 | **Stack Matching** | Push open, pop close | Valid Parentheses #20 |
| 11 | **Monotonic Stack** | Next greater/smaller | Daily Temps #739 |
| 12 | **Binary Search** | Halve the search space | Rotated Array #33, Koko #875 |

### Practice Problems for Day 2

```
Easy:
  #206  Reverse Linked List
  #20   Valid Parentheses
  #217  Contains Duplicate
  #141  Linked List Cycle
  #704  Binary Search
  #136  Single Number

Medium:
  #128  Longest Consecutive Sequence
  #347  Top K Frequent Elements
  #739  Daily Temperatures
  #143  Reorder List
  #21   Merge Two Sorted Lists
  #33   Search in Rotated Sorted Array
  #56   Merge Intervals
  #75   Sort Colors
  #875  Koko Eating Bananas

Hard:
  #84   Largest Rectangle in Histogram
```

---

*Next: Recursion, Backtracking, Trees, and Heaps -- [day3.md](day3.md)*
