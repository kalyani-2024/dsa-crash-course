# Day 3 -- Sorting, Binary Search, Bit Manipulation, Recursion, and Backtracking

## From Searching Efficiently to Generating All Possibilities

**What this day covers:** Sorting as a preprocessing step, Binary Search (standard and on-answer), Bit Manipulation tricks, Recursion (base case thinking), and Backtracking (subsets, permutations, combinations, constraint satisfaction).

This is the bridge between fundamental data structures and advanced algorithms. Sorting and binary search show up everywhere as enabling techniques. Recursion and backtracking give you the tools to explore solution spaces systematically.

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

# Day 3 Summary -- 2 Patterns + Key Techniques

| # | Pattern | Core Insight | Key Problem |
|---|---------|-------------|-------------|
| 12 | **Binary Search** | Halve the search space | Rotated Array #33, Koko #875 |
| 13 | **Backtracking** | Choose, Explore, Undo | Subsets #78, N-Queens #51 |
| -- | **Sorting** | Preprocessing that unlocks patterns | Merge Intervals #56 |
| -- | **Bit Manipulation** | XOR tricks, power of 2, set bits | Single Number #136 |
| -- | **Recursion** | Base case + recursive case | Foundation for trees and graphs |

### Practice Problems for Day 3

```
Easy:
  #704  Binary Search
  #136  Single Number

Medium:
  #33   Search in Rotated Sorted Array
  #56   Merge Intervals
  #75   Sort Colors
  #78   Subsets
  #46   Permutations
  #39   Combination Sum
  #79   Word Search
  #875  Koko Eating Bananas

Hard:
  #51   N-Queens
```

---

*Next: Trees, Heaps, Tries, Graphs, Greedy, and Dynamic Programming -- [day4.md](day4.md)*
