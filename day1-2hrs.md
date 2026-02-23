# ⚡ Day 1 — DSA Crash Course (2 Hours)

## Arrays → Hashing → Two Pointers → Sliding Window → Sorting → Binary Search

> **Format:** 2 hours, pure essentials. Every minute counts.
> **Goal:** After this session you can solve 70% of Easy and 40% of Medium LeetCode problems.

---

## ⏱ Schedule

| Time | Topic | Key Pattern |
|------|-------|-------------|
| 0:00 - 0:10 | Big-O & Thinking Framework | How to judge your solution |
| 0:10 - 0:30 | Arrays + Hashing | Frequency counting, prefix sum |
| 0:30 - 0:55 | Two Pointers + Sliding Window | Shrink O(n²) to O(n) |
| 0:55 - 1:15 | Sorting + Binary Search | O(n log n) power tools |
| 1:15 - 1:35 | Strings + Bit Manipulation | Quick wins |
| 1:35 - 2:00 | Recursion & Backtracking | Generate all possibilities |

---

# 🧠 0:00 — Big-O in 5 Minutes

**Big-O = how your code scales as input grows.**

```
🟢 O(1)        Hash lookup, array[i]              instant
🟢 O(log n)    Binary search                      10 steps for 1000 items
🟢 O(n)        Single loop                        1000 steps for 1000 items
🟡 O(n log n)  Sorting                            10,000 steps
🟠 O(n²)       Nested loops                       1,000,000 steps
🔴 O(2ⁿ)       Subsets without memo               💀
```

**The golden rule — look at the constraint `n`:**
```
n ≤ 1,000    → O(n²) is fine
n ≤ 100,000  → Need O(n log n) or O(n)
n ≤ 10⁷      → Need O(n)
```

**5-Step Framework (use for EVERY problem):**
```
1. UNDERSTAND — Re-read, walk through examples
2. BRUTE FORCE — What's the dumb O(n²) way?
3. OPTIMIZE — What data structure makes it faster?
4. CODE — Clean, with edge cases first
5. TEST — Dry run with example + edge case
```

> 🎬 Bookmark: [bigocheatsheet.com](https://www.bigocheatsheet.com/)

---

# 📦 0:10 — Arrays + Hashing (20 min)

## Pattern 1: HashMap for O(1) Lookup

A HashMap lets you answer **"have I seen X?"** in O(1).

### ⭐ Two Sum (LeetCode #1) — Most Asked Question Ever

```
nums = [2, 7, 11, 15], target = 9
For each num: does (target - num) exist in my map?

num=2 → need 7 → not in map → store {2: 0}
num=7 → need 2 → YES at index 0! → return [0, 1] ✅
```

```python
def twoSum(nums, target):
    seen = {}                           # value → index
    for i, num in enumerate(nums):
        if target - num in seen:
            return [seen[target - num], i]
        seen[num] = i
# Time: O(n) | Space: O(n)
```

### Group Anagrams (LeetCode #49)
```python
from collections import defaultdict
def groupAnagrams(strs):
    groups = defaultdict(list)
    for s in strs:
        groups[tuple(sorted(s))].append(s)  # sorted letters = key
    return list(groups.values())
# O(n × k log k)
```

### Longest Consecutive Sequence (LeetCode #128)
```python
def longestConsecutive(nums):
    s = set(nums)
    best = 0
    for n in s:
        if n - 1 not in s:            # start of a sequence
            length = 0
            while n + length in s:
                length += 1
            best = max(best, length)
    return best
# O(n)
```

---

## Pattern 2: Prefix Sum — Subarray Sum in O(1)

```
arr =        [1,  2,  3,  4,  5]
prefix =  [0, 1,  3,  6, 10, 15]
Sum(i..j) = prefix[j+1] - prefix[i]
```

### Subarray Sum Equals K (LeetCode #560)
```python
def subarraySum(nums, k):
    count = prefix = 0
    seen = {0: 1}                      # prefix_sum → frequency
    for num in nums:
        prefix += num
        count += seen.get(prefix - k, 0)
        seen[prefix] = seen.get(prefix, 0) + 1
    return count
# O(n)
```

---

## Pattern 3: Kadane's Algorithm — Max Subarray Sum

### Maximum Subarray (LeetCode #53) — Top 5 Interview Question

At each index: **extend** current subarray or **start fresh?**

```python
def maxSubArray(nums):
    curr = best = nums[0]
    for num in nums[1:]:
        curr = max(num, curr + num)    # extend or restart
        best = max(best, curr)
    return best
# O(n), O(1)
```

---

# 👆 0:30 — Two Pointers + Sliding Window (25 min)

## Pattern 4: Two Pointers — Opposite Ends

**When:** Sorted array, pair finding, converging search.

### Container With Most Water (LeetCode #11)
```python
def maxArea(height):
    lo, hi = 0, len(height) - 1
    best = 0
    while lo < hi:
        best = max(best, (hi - lo) * min(height[lo], height[hi]))
        if height[lo] < height[hi]:
            lo += 1                     # move the shorter wall
        else:
            hi -= 1
    return best
# O(n), O(1)
```

### 3Sum (LeetCode #15)
```python
def threeSum(nums):
    nums.sort()
    res = []
    for i in range(len(nums) - 2):
        if i > 0 and nums[i] == nums[i-1]: continue   # skip dupes
        lo, hi = i + 1, len(nums) - 1
        while lo < hi:
            s = nums[i] + nums[lo] + nums[hi]
            if s == 0:
                res.append([nums[i], nums[lo], nums[hi]])
                while lo < hi and nums[lo] == nums[lo+1]: lo += 1
                while lo < hi and nums[hi] == nums[hi-1]: hi -= 1
                lo += 1; hi -= 1
            elif s < 0: lo += 1
            else:       hi -= 1
    return res
# O(n²)
```

### ⭐ Trapping Rain Water (LeetCode #42) — Classic Hard

```python
def trap(height):
    lo, hi = 0, len(height) - 1
    lo_max = hi_max = water = 0
    while lo < hi:
        if height[lo] <= height[hi]:
            lo_max = max(lo_max, height[lo])
            water += lo_max - height[lo]
            lo += 1
        else:
            hi_max = max(hi_max, height[hi])
            water += hi_max - height[hi]
            hi -= 1
    return water
# O(n), O(1)
```

---

## Pattern 5: Sliding Window — Subarray / Substring Optimization

**Template:** Expand right, shrink left while condition breaks.

```python
# UNIVERSAL TEMPLATE
def sliding_window(arr):
    left = 0
    window = ...  # state (set, map, sum, count)
    best = ...
    for right in range(len(arr)):
        # EXPAND: add arr[right] to window
        while WINDOW_INVALID:
            # SHRINK: remove arr[left], left += 1
            pass
        # UPDATE: record best
    return best
```

### Longest Substring Without Repeating Characters (LeetCode #3)
```python
def lengthOfLongestSubstring(s):
    seen = set()
    left = best = 0
    for right in range(len(s)):
        while s[right] in seen:
            seen.remove(s[left])
            left += 1
        seen.add(s[right])
        best = max(best, right - left + 1)
    return best
# O(n)
```

### ⭐ Minimum Window Substring (LeetCode #76) — Hard
```python
from collections import Counter, defaultdict
def minWindow(s, t):
    need = Counter(t)
    have = defaultdict(int)
    required = len(need)
    formed = 0
    left = 0
    res = ""
    for right in range(len(s)):
        c = s[right]
        have[c] += 1
        if c in need and have[c] == need[c]:
            formed += 1
        while formed == required:
            if not res or right - left + 1 < len(res):
                res = s[left:right+1]
            have[s[left]] -= 1
            if s[left] in need and have[s[left]] < need[s[left]]:
                formed -= 1
            left += 1
    return res
# O(|s| + |t|)
```

### Maximum Consecutive Ones III (LeetCode #1004)
```python
def longestOnes(nums, k):
    left = zeros = best = 0
    for right in range(len(nums)):
        if nums[right] == 0: zeros += 1
        while zeros > k:
            if nums[left] == 0: zeros -= 1
            left += 1
        best = max(best, right - left + 1)
    return best
```

---

# 🔍 0:55 — Sorting + Binary Search (20 min)

## Sorting — When & Why

You rarely implement sorts, but sorting **unlocks** other techniques:
```
Sorted → enables Binary Search     O(n log n + log n)
Sorted → enables Two Pointers      O(n log n + n)
Sorted → enables Merge Intervals   O(n log n + n)
```

### Dutch National Flag — Sort 0s, 1s, 2s (LeetCode #75)
```python
def sortColors(nums):
    lo, mid, hi = 0, 0, len(nums) - 1
    while mid <= hi:
        if   nums[mid] == 0: nums[lo], nums[mid] = nums[mid], nums[lo]; lo += 1; mid += 1
        elif nums[mid] == 1: mid += 1
        else:                nums[mid], nums[hi] = nums[hi], nums[mid]; hi -= 1
# Single pass O(n), O(1)
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

## Pattern 6: Binary Search

**Core idea:** Halve the search space every step → O(log n).

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
```python
def search(nums, target):
    lo, hi = 0, len(nums) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if nums[mid] == target: return mid
        if nums[lo] <= nums[mid]:          # left half sorted
            if nums[lo] <= target < nums[mid]: hi = mid - 1
            else: lo = mid + 1
        else:                               # right half sorted
            if nums[mid] < target <= nums[hi]: lo = mid + 1
            else: hi = mid - 1
    return -1
```

### ⭐ Binary Search on Answer — Most Powerful Pattern

**Instead of searching an array, search the range of possible answers.**

### Koko Eating Bananas (LeetCode #875)
```python
import math
def minEatingSpeed(piles, h):
    lo, hi = 1, max(piles)
    while lo < hi:
        mid = (lo + hi) // 2
        if sum(math.ceil(p / mid) for p in piles) <= h:
            hi = mid           # can eat slower
        else:
            lo = mid + 1       # too slow
    return lo
# O(n log(max))
```

> 🎬 Visualize: [visualgo.net/bst](https://visualgo.net/en/bst)

---

# 🔤 1:15 — Strings + Bits (20 min)

### Longest Palindromic Substring (LeetCode #5) — Expand Around Center
```python
def longestPalindrome(s):
    res = ""
    def expand(l, r):
        while l >= 0 and r < len(s) and s[l] == s[r]:
            l -= 1; r += 1
        return s[l+1:r]
    for i in range(len(s)):
        for pal in (expand(i, i), expand(i, i+1)):
            if len(pal) > len(res): res = pal
    return res
# O(n²)
```

### Valid Anagram (LeetCode #242)
```python
from collections import Counter
def isAnagram(s, t):
    return Counter(s) == Counter(t)
```

### Bit Tricks — 4 Must-Know Operations
```python
# 1. Single Number (LeetCode #136) — XOR cancels pairs
def singleNumber(nums):
    r = 0
    for n in nums: r ^= n
    return r

# 2. Check power of 2
def isPowerOfTwo(n):
    return n > 0 and (n & (n-1)) == 0

# 3. Count set bits (Brian Kernighan)
def countBits(n):
    c = 0
    while n:
        n &= n - 1   # removes lowest set bit
        c += 1
    return c

# 4. Even/Odd check
n & 1 == 0  # even
n & 1 == 1  # odd
```

---

# 🔁 1:35 — Recursion & Backtracking (25 min)

## Recursion = Base Case + Smaller Problem

```python
def factorial(n):
    if n <= 1: return 1        # base case
    return n * factorial(n-1)  # smaller problem
```

> 🎬 Visualize YOUR code: [pythontutor.com](https://pythontutor.com/)

## Pattern 7: Subsets — Include or Exclude

At each element: **take it** or **skip it** → 2ⁿ total subsets.

### Subsets (LeetCode #78)
```python
def subsets(nums):
    res = []
    def bt(i, curr):
        if i == len(nums):
            res.append(curr[:])
            return
        curr.append(nums[i])  # include
        bt(i + 1, curr)
        curr.pop()            # exclude (BACKTRACK)
        bt(i + 1, curr)
    bt(0, [])
    return res
# O(2^n)
```

### Combination Sum (LeetCode #39)
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
# O(n!)
```

### ⭐ N-Queens (LeetCode #51) — Classic Hard
```python
def solveNQueens(n):
    res = []
    cols, d1, d2 = set(), set(), set()
    board = [['.']*n for _ in range(n)]
    
    def bt(row):
        if row == n:
            res.append([''.join(r) for r in board])
            return
        for col in range(n):
            if col in cols or row-col in d1 or row+col in d2:
                continue
            board[row][col] = 'Q'
            cols.add(col); d1.add(row-col); d2.add(row+col)
            bt(row + 1)
            board[row][col] = '.'
            cols.discard(col); d1.discard(row-col); d2.discard(row+col)
    bt(0)
    return res
```

---

# ✅ Day 1 Summary — 7 Patterns

| # | Pattern | Turns O(?) into | Key Problem |
|---|---------|----------------|-------------|
| 1 | **HashMap** | O(n²) → O(n) | Two Sum #1 |
| 2 | **Prefix Sum** | O(n²) → O(n) | Subarray Sum K #560 |
| 3 | **Kadane's** | O(n²) → O(n) | Max Subarray #53 |
| 4 | **Two Pointers** | O(n²) → O(n) | 3Sum #15, Trapping Rain Water #42 |
| 5 | **Sliding Window** | O(n²) → O(n) | Longest Substring #3, Min Window #76 |
| 6 | **Binary Search** | O(n) → O(log n) | Rotated Array #33, Koko Bananas #875 |
| 7 | **Backtracking** | Generate all | Subsets #78, N-Queens #51 |

### 🏋️ Tonight's Homework (Pick 5)
```
🟢 #1    Two Sum                    🟡 #53  Max Subarray
🟡 #3    Longest Substring          🟡 #15  3Sum
🟡 #33   Search Rotated Array       🟡 #560 Subarray Sum K
🟡 #78   Subsets                    🔴 #42  Trapping Rain Water
```

---

*Tomorrow: Linked Lists, Stacks, Trees, Graphs, DP → [day2-2hrs.md](day2-2hrs.md)*
