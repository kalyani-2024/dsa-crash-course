# 🌤️ Day 1 — Afternoon Session (1:30 PM - 5:00 PM)

## Sorting → Binary Search → Strings → Recursion & Backtracking → Bit Manipulation

---

# 🔄 Part 6: Sorting — Know When & Why (30 min)

## Why Sorting Matters
Sorting isn't just about ordering data — it **unlocks** other techniques:
- Binary Search requires sorted data
- Two Pointers often need sorted arrays
- Greedy algorithms frequently need sorted intervals
- Many problems become trivially easier after sorting

### 🎬 **Visualize it:** [toptal.com/sorting](https://www.toptal.com/developers/sorting-algorithms) — See ALL sorting algorithms race!
### 🎬 **Visualize it:** [visualgo.net/sorting](https://visualgo.net/en/sorting) — Step through each algorithm

## The Only Sorts You Need to Know

| Algorithm | Time (Avg) | Time (Worst) | Space | Stable | Use When |
|-----------|-----------|-------------|-------|--------|----------|
| **Merge Sort** | O(n log n) | O(n log n) | O(n) | ✅ | Need stability, linked lists |
| **Quick Sort** | O(n log n) | O(n²) | O(log n) | ❌ | General purpose (fastest in practice) |
| **Counting/Bucket** | O(n + k) | O(n + k) | O(k) | ✅ | Known small range of values |

> **💡 Interview Reality:** You'll almost never implement sorting from scratch. But you MUST understand how Merge Sort works because it's used in divide-and-conquer problems.

### Merge Sort — Understand This One Deeply

```
[38, 27, 43, 3, 9, 82, 10]
         ↓ DIVIDE
  [38, 27, 43, 3]    [9, 82, 10]
         ↓                ↓
  [38,27] [43,3]    [9,82] [10]
     ↓       ↓        ↓
  [27,38] [3,43]   [9,82] [10]
         ↓                ↓
  [3, 27, 38, 43]   [9, 10, 82]
              ↓ MERGE
  [3, 9, 10, 27, 38, 43, 82]
```

```python
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    
    return merge(left, right)

def merge(left, right):
    """Merge two sorted arrays into one sorted array."""
    result = []
    i = j = 0
    
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    result.extend(left[i:])
    result.extend(right[j:])
    return result
```

### Dutch National Flag — Sort 0s, 1s, 2s (LeetCode #75)

Special sorting for 3 values only. **Super common interview question.**

```python
def sortColors(nums):
    """Sort array containing only 0, 1, 2 in ONE PASS."""
    low, mid, high = 0, 0, len(nums) - 1
    
    while mid <= high:
        if nums[mid] == 0:
            nums[low], nums[mid] = nums[mid], nums[low]
            low += 1
            mid += 1
        elif nums[mid] == 1:
            mid += 1
        else:  # nums[mid] == 2
            nums[mid], nums[high] = nums[high], nums[mid]
            high -= 1
    # Time: O(n) | Space: O(1) | Single pass!
```

---

# 🔍 Part 7: Binary Search — The Most Powerful O(log n) Tool (60 min)

## The Core Idea

Binary Search cuts the search space **in half** every step. It works on any **monotonic** (sorted or ordered) property.

```
Find 7 in sorted array:

[1, 3, 5, 7, 9, 11, 13]
 L           M          R     M=7? Yes! ✅

But binary search is SO much more than just finding an element...
```

### 🎬 **Visualize it:** [algorithm-visualizer.org/binary-search](https://algorithm-visualizer.org/brute-force/binary-search)
### 🎬 **Interactive:** [LeetCode Binary Search Card](https://leetcode.com/explore/learn/card/binary-search/)

## 🧩 Pattern 7: Standard Binary Search

```python
def binary_search(arr, target):
    """Find target in sorted array. Return index or -1."""
    lo, hi = 0, len(arr) - 1
    
    while lo <= hi:
        mid = lo + (hi - lo) // 2  # Avoids integer overflow
        
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    
    return -1
# Time: O(log n) | Space: O(1)
```

### Lower Bound & Upper Bound

```python
def lower_bound(arr, target):
    """First index where arr[i] >= target (insertion point)."""
    lo, hi = 0, len(arr)
    while lo < hi:
        mid = (lo + hi) // 2
        if arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid
    return lo

def upper_bound(arr, target):
    """First index where arr[i] > target."""
    lo, hi = 0, len(arr)
    while lo < hi:
        mid = (lo + hi) // 2
        if arr[mid] <= target:
            lo = mid + 1
        else:
            hi = mid
    return lo

# Count occurrences of target = upper_bound - lower_bound
```

### Problem: Search in Rotated Sorted Array (LeetCode #33)

```
[4, 5, 6, 7, 0, 1, 2]   target = 0

Key insight: At least ONE half is always sorted.
  mid = 3, arr[mid] = 7
  Left half [4,5,6,7] is sorted (arr[lo] <= arr[mid])
  Target 0 is NOT in [4,7], so go right → lo = mid+1
  
  Now [0, 1, 2], find 0 normally.
```

```python
def search(nums, target):
    """Binary search in a rotated sorted array."""
    lo, hi = 0, len(nums) - 1
    
    while lo <= hi:
        mid = (lo + hi) // 2
        
        if nums[mid] == target:
            return mid
        
        # Left half is sorted
        if nums[lo] <= nums[mid]:
            if nums[lo] <= target < nums[mid]:
                hi = mid - 1  # Target in left half
            else:
                lo = mid + 1  # Target in right half
        # Right half is sorted
        else:
            if nums[mid] < target <= nums[hi]:
                lo = mid + 1  # Target in right half
            else:
                hi = mid - 1  # Target in left half
    
    return -1
# Time: O(log n) | Space: O(1)
```

## 🧩 Pattern 8: Binary Search on Answer Space

**This is the MOST POWERFUL binary search pattern.** Instead of searching in an array, you search the **range of possible answers**.

> **Template:** "Find the minimum/maximum X such that condition(X) is true."

```
          Answers: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
Condition(X):       F  F  F  F  T  T  T  T  T  T
                              ↑
                    Binary search finds this boundary!
```

### Problem: Koko Eating Bananas (LeetCode #875)

```
piles = [3, 6, 7, 11], hours = 8
Koko eats at speed k bananas/hour. Find minimum k.

Speed 1: ceil(3/1)+ceil(6/1)+ceil(7/1)+ceil(11/1) = 27 hours ❌
Speed 4: ceil(3/4)+ceil(6/4)+ceil(7/4)+ceil(11/4) = 1+2+2+3 = 8 ✅
Speed 3: ceil(3/3)+ceil(6/3)+ceil(7/3)+ceil(11/3) = 1+2+3+4 = 10 ❌
→ Answer: 4
```

```python
import math

def minEatingSpeed(piles, h):
    """Binary search on the answer (eating speed)."""
    lo, hi = 1, max(piles)
    
    while lo < hi:
        mid = (lo + hi) // 2
        # Calculate hours needed at speed 'mid'
        hours = sum(math.ceil(p / mid) for p in piles)
        
        if hours <= h:
            hi = mid       # Can eat slower, try lower speed
        else:
            lo = mid + 1   # Too slow, need faster speed
    
    return lo
# Time: O(n × log(max)) | Space: O(1)
```

### Problem: Median of Two Sorted Arrays (LeetCode #4) — ⭐ Hard

```python
def findMedianSortedArrays(nums1, nums2):
    """Binary search on partition of shorter array."""
    if len(nums1) > len(nums2):
        nums1, nums2 = nums2, nums1  # Ensure nums1 is shorter
    
    m, n = len(nums1), len(nums2)
    lo, hi = 0, m
    
    while lo <= hi:
        i = (lo + hi) // 2           # Partition in nums1
        j = (m + n + 1) // 2 - i     # Partition in nums2
        
        left1 = nums1[i-1] if i > 0 else float('-inf')
        right1 = nums1[i] if i < m else float('inf')
        left2 = nums2[j-1] if j > 0 else float('-inf')
        right2 = nums2[j] if j < n else float('inf')
        
        if left1 <= right2 and left2 <= right1:
            if (m + n) % 2 == 1:
                return max(left1, left2)
            return (max(left1, left2) + min(right1, right2)) / 2
        elif left1 > right2:
            hi = i - 1
        else:
            lo = i + 1
# Time: O(log(min(m,n))) | Space: O(1)
```

---

# 🔤 Part 8: Strings — Essential Patterns (30 min)

## String Basics

```python
# Python strings are immutable
s = "hello"
# s[0] = 'H'  ← ERROR! Can't modify

# Common operations
s.lower()           # "hello"
s.upper()           # "HELLO"
s[::-1]             # "olleh" (reverse)
s.split(" ")        # Split by space
"".join(["a","b"])  # "ab"
ord('a')            # 97 (ASCII value)
chr(97)             # 'a'
```

### Problem: Valid Palindrome (LeetCode #125)
```python
def isPalindrome(s):
    """Two pointers from both ends."""
    s = ''.join(c.lower() for c in s if c.isalnum())
    return s == s[::-1]
# Time: O(n) | Space: O(n)
```

### Problem: Longest Palindromic Substring (LeetCode #5)
```python
def longestPalindrome(s):
    """Expand Around Center — try each possible center."""
    result = ""
    
    def expand(left, right):
        while left >= 0 and right < len(s) and s[left] == s[right]:
            left -= 1
            right += 1
        return s[left+1:right]
    
    for i in range(len(s)):
        # Odd length palindrome (single center)
        odd = expand(i, i)
        # Even length palindrome (two centers)
        even = expand(i, i + 1)
        
        result = max(result, odd, even, key=len)
    
    return result
# Time: O(n²) | Space: O(1)
```

### Problem: Valid Anagram (LeetCode #242)
```python
from collections import Counter

def isAnagram(s, t):
    return Counter(s) == Counter(t)
# Time: O(n) | Space: O(1) — at most 26 keys
```

---

# 🔁 Part 9: Recursion & Backtracking — Think Recursively (60 min)

## What is Recursion?

A function that **calls itself** with a **smaller problem** until it reaches a **base case**.

```
Factorial(5) = 5 × Factorial(4)
             = 5 × 4 × Factorial(3)
             = 5 × 4 × 3 × Factorial(2)
             = 5 × 4 × 3 × 2 × Factorial(1)
             = 5 × 4 × 3 × 2 × 1    ← Base case!
             = 120
```

### 🎬 **Visualize it:** [pythontutor.com](https://pythontutor.com/) — Watch the call stack grow and shrink
### 🎬 **Visualize it:** [recursion.vercel.app](https://recursion-visualizer.vercel.app/) — Tree visualization of recursive calls

### The 3 Rules of Recursion
```
1. BASE CASE    — When to STOP (without this → infinite loop → crash)
2. RECURSIVE CASE — Call yourself with a SMALLER problem
3. TRUST         — Assume the recursive call returns the correct answer
```

```python
def factorial(n):
    # 1. Base case
    if n <= 1:
        return 1
    # 2. Recursive case (smaller problem)
    return n * factorial(n - 1)

def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)
    # ⚠️ This is O(2^n) — we'll fix it with DP tomorrow!
```

## 🧩 Pattern 9: Subsets / Subsequences (Include or Exclude)

### How to Generate All Subsets (LeetCode #78)

At each element, you have exactly TWO choices: **include it** or **exclude it**.

```
Elements: [1, 2, 3]

                    []
              /          \
          [1]              []
        /     \          /    \
     [1,2]   [1]      [2]    []
     / \     / \      / \    / \
 [1,2,3][1,2][1,3][1][2,3][2][3][]

All subsets: [],[1],[2],[3],[1,2],[1,3],[2,3],[1,2,3]
```

```python
def subsets(nums):
    """Generate all subsets using include/exclude pattern."""
    result = []
    
    def backtrack(index, current):
        if index == len(nums):
            result.append(current[:])  # Make a copy!
            return
        
        # Choice 1: Include nums[index]
        current.append(nums[index])
        backtrack(index + 1, current)
        
        # Choice 2: Exclude nums[index]
        current.pop()              # Undo the choice (BACKTRACK)
        backtrack(index + 1, current)
    
    backtrack(0, [])
    return result
# Time: O(2^n) | Space: O(n) call stack
```

## 🧩 Pattern 10: Permutations

```python
def permute(nums):
    """Generate all permutations."""
    result = []
    
    def backtrack(current, remaining):
        if not remaining:
            result.append(current[:])
            return
        
        for i in range(len(remaining)):
            current.append(remaining[i])
            backtrack(current, remaining[:i] + remaining[i+1:])
            current.pop()  # BACKTRACK
    
    backtrack([], nums)
    return result
# Time: O(n!) | Space: O(n)
```

## 🧩 Pattern 11: Constraint Satisfaction (N-Queens, Sudoku)

### Problem: N-Queens (LeetCode #51) — ⭐ Classic Backtracking

```python
def solveNQueens(n):
    """Place n queens so no two attack each other."""
    result = []
    cols = set()      # Columns under attack
    diag1 = set()     # / diagonals (row - col = constant)
    diag2 = set()     # \ diagonals (row + col = constant)
    board = [['.' for _ in range(n)] for _ in range(n)]
    
    def backtrack(row):
        if row == n:
            result.append([''.join(r) for r in board])
            return
        
        for col in range(n):
            if col in cols or (row-col) in diag1 or (row+col) in diag2:
                continue
            
            # Place queen
            board[row][col] = 'Q'
            cols.add(col)
            diag1.add(row - col)
            diag2.add(row + col)
            
            backtrack(row + 1)
            
            # Remove queen (BACKTRACK)
            board[row][col] = '.'
            cols.remove(col)
            diag1.discard(row - col)
            diag2.discard(row + col)
    
    backtrack(0)
    return result
# Time: O(n!) | Space: O(n²)
```

### Problem: Combination Sum (LeetCode #39)
```python
def combinationSum(candidates, target):
    """Find all combos summing to target. Can reuse elements."""
    result = []
    
    def backtrack(start, current, remaining):
        if remaining == 0:
            result.append(current[:])
            return
        if remaining < 0:
            return
        
        for i in range(start, len(candidates)):
            current.append(candidates[i])
            backtrack(i, current, remaining - candidates[i])  # i, not i+1 (reuse allowed)
            current.pop()
    
    backtrack(0, [], target)
    return result
```

---

# 🔢 Part 10: Bit Manipulation — Secret O(1) Tricks (20 min)

## Essential Bit Operations

```
AND (&):  1010 & 1100 = 1000    "Both bits must be 1"
OR  (|):  1010 | 1100 = 1110    "Either bit is 1"
XOR (^):  1010 ^ 1100 = 0110    "Bits are different"
NOT (~):  ~1010 = ...0101        "Flip all bits"
LEFT SHIFT (<<):  0001 << 2 = 0100   "Multiply by 2^k"
RIGHT SHIFT (>>): 1000 >> 2 = 0010   "Divide by 2^k"
```

### Must-Know Bit Tricks

```python
# 1. Check if number is even/odd
n & 1 == 0  # Even
n & 1 == 1  # Odd

# 2. Check if power of 2
n > 0 and (n & (n-1)) == 0

# 3. Count set bits (Brian Kernighan)
count = 0
while n:
    n &= (n - 1)  # Removes lowest set bit
    count += 1

# 4. XOR properties
a ^ a = 0      # Self-cancel
a ^ 0 = a      # Identity
a ^ b ^ a = b  # Find the odd one out
```

### Problem: Single Number (LeetCode #136)
```python
def singleNumber(nums):
    """Every number appears twice except one. Find it."""
    result = 0
    for num in nums:
        result ^= num  # Pairs cancel out!
    return result
# Time: O(n) | Space: O(1)
```

### Problem: Number of 1 Bits (LeetCode #191)
```python
def hammingWeight(n):
    count = 0
    while n:
        n &= (n - 1)  # Remove rightmost set bit
        count += 1
    return count
```

### Problem: Power Set using Bits (Alternative to Recursion)
```python
def subsets_bitwise(nums):
    """Generate all subsets using bit manipulation."""
    n = len(nums)
    result = []
    
    for mask in range(1 << n):  # 0 to 2^n - 1
        subset = []
        for i in range(n):
            if mask & (1 << i):  # Is bit i set?
                subset.append(nums[i])
        result.append(subset)
    
    return result
```

---

## 🌆 Afternoon Review — Patterns 6-11

| # | Pattern | Template | Key Insight |
|---|---------|----------|-------------|
| 6 | Sorting | `arr.sort()` unlock | Sorting enables binary search & two pointers |
| 7 | Binary Search | `lo, hi, mid` | Works on ANY monotonic property |
| 8 | BS on Answer | `lo=min, hi=max` | Search the answer range, not the array |
| 9 | Subsets | Include / Exclude | 2 choices per element → 2^n total |
| 10 | Permutations | Choose / Backtrack | n choices, n-1 choices, ... → n! total |
| 11 | Constraint Satisfaction | Try / Validate / Undo | Place, check, backtrack |

> **🎯 Before coding, always ask:** "Can I sort first?" "Can I binary search the answer?" "Is this a choose/explore/undo problem?"

---

*Done for today? Head to [day1-practice.md](day1-practice.md) for 20 must-do problems!*

[← Day 1 Morning](day1-morning.md) | [Back to Schedule](README.md) | [Next: Day 1 Practice →](day1-practice.md)
