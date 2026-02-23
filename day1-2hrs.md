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

# 🧠 0:00 — Big-O: How to Think About Efficiency

## What is Big-O?

Big-O notation is a way to describe **how your algorithm scales as the input grows**. It answers this question: *"If I double the input size, how much slower does my code get?"*

Think of it like measuring distance — you don't say "it's 4,817 steps to the store," you say "it's a 10-minute walk." Big-O similarly gives you the **shape** of your algorithm's growth, ignoring constants and small terms.

### The Common Growth Rates

```
🟢 O(1)        → Constant     → Hash lookup, array[i]        → instant, no matter the size
🟢 O(log n)    → Logarithmic  → Binary search                → 10 steps for 1000 items, 20 for 1,000,000
🟢 O(n)        → Linear       → Single loop                  → 1000 steps for 1000 items
🟡 O(n log n)  → Linearithmic → Sorting (merge sort, etc.)   → 10,000 steps for 1000 items
🟠 O(n²)       → Quadratic    → Nested loops                 → 1,000,000 steps for 1000 items
🔴 O(2ⁿ)       → Exponential  → Subsets without memoization  → 💀 (unusable for large n)
```

### Why Does This Matter?

A computer can do roughly **10⁸ (100 million) simple operations per second**. So:

| Constraint (n) | Max Acceptable Complexity | Why |
|----------------|--------------------------|-----|
| n ≤ 10 | O(n!) | Tiny input — anything works |
| n ≤ 20 | O(2ⁿ) | Backtracking/bitmask OK |
| n ≤ 1,000 | O(n²) | Nested loops are fine |
| n ≤ 100,000 | O(n log n) or O(n) | Need sorting, hashing, or clever traversal |
| n ≤ 10⁷ | O(n) | Must be single-pass |
| n > 10⁷ | O(log n) or O(1) | Only math or binary search |

> **💡 The first thing to do with ANY problem: check the constraint `n`. It tells you which complexity you need, which tells you which patterns to try.**

### The 5-Step Framework — Use This for EVERY Problem

```
1. UNDERSTAND  — Re-read the problem, walk through examples by hand
2. BRUTE FORCE — What's the "dumb" O(n²) or O(n³) way? Can you at least solve it slowly?
3. OPTIMIZE    — What data structure or pattern could make it faster?
4. CODE        — Write clean code, handle edge cases first
5. TEST        — Dry run with the example + at least one edge case
```

> 🎬 Bookmark: [bigocheatsheet.com](https://www.bigocheatsheet.com/) — see every data structure's complexity at a glance.

---

# 📦 0:10 — Arrays + Hashing (20 min)

## What is Hashing? (The Most Important Concept in DSA)

Imagine you have a library with 1 million books. If they're piled randomly, finding a book means checking one by one — **O(n)**. But if each book has a **shelf code** (a hash) that tells you *exactly* where to look, you find any book in **O(1)**, instantly.

That's what a **HashMap (dictionary)** does. It takes any data (a number, a string, etc.) and computes a "shelf code" (a hash) that maps directly to a memory location. This gives you:

- **O(1) lookup** — "Have I seen this value before?"
- **O(1) insertion** — "Remember this value"
- **O(1) deletion** — "Forget this value"

### When to Use a HashMap

Ask yourself: **"Am I checking `if X exists` inside a loop?"** If yes → HashMap.

```
Without HashMap: For each element, scan the rest  → O(n) per check → O(n²) total
With HashMap:    For each element, check the map   → O(1) per check → O(n) total
```

This single idea — **trading space for time using a HashMap** — solves a huge percentage of interview problems.

---

## Pattern 1: HashMap for O(1) Lookup

### The Core Idea

> **"Instead of searching for something, pre-store it so you can look it up instantly."**

### ⭐ Two Sum (LeetCode #1) — Most Asked Question Ever

**The Concept:** For each number, you need to find if its *complement* (target - number) exists somewhere in the array. The brute force checks every pair — O(n²). But if you **store each number in a map as you go**, then for each new number, you can instantly check if its complement was already seen.

```
Think of it like this:
You're at a party looking for your "dance partner" (the complement).
Instead of asking every person, you put your name on a sign-up board.
When your partner arrives, they check the board and find you instantly!
```

**Walkthrough:**
```
nums = [2, 7, 11, 15], target = 9

num=2 → I need 7 to make 9 → Is 7 in my map? NO  → Store {2: index 0}
num=7 → I need 2 to make 9 → Is 2 in my map? YES → Found it at index 0! Return [0, 1] ✅
```

```python
def twoSum(nums, target):
    seen = {}                           # value → index
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:          # O(1) lookup!
            return [seen[complement], i]
        seen[num] = i                   # store for future lookups
# Time: O(n) | Space: O(n) — we trade extra space for faster lookup
```

### Group Anagrams (LeetCode #49)

**The Concept:** Two words are anagrams if they have the same letters, just rearranged ("eat" ↔ "tea" ↔ "ate"). The key insight is: **if you sort the letters of an anagram, all anagrams produce the same sorted key**. So you can group them using a HashMap where the sorted letters are the key.

```python
from collections import defaultdict
def groupAnagrams(strs):
    groups = defaultdict(list)
    for s in strs:
        key = tuple(sorted(s))         # "eat" → ('a','e','t'), same as "tea"
        groups[key].append(s)
    return list(groups.values())
# O(n × k log k) where k = max string length
```

### Longest Consecutive Sequence (LeetCode #128)

**The Concept:** You want the longest run of consecutive numbers (e.g., [1,2,3,4] in [100,4,200,1,3,2]). The trick: **only start counting from the beginning of a sequence**. A number is the start of a sequence if `n-1` is NOT in the set. This avoids counting from the middle.

```python
def longestConsecutive(nums):
    s = set(nums)                      # O(1) lookups
    best = 0
    for n in s:
        if n - 1 not in s:            # This is the START of a sequence
            length = 0
            while n + length in s:    # Keep extending
                length += 1
            best = max(best, length)
    return best
# O(n) — each number is visited at most twice (once in loop, once in while)
```

---

## Pattern 2: Prefix Sum — Answer Range Queries in O(1)

### The Core Idea

> **"Pre-compute cumulative sums so that any subarray sum becomes a single subtraction."**

Imagine a running odometer in a car. If the odometer read 150 km at point A and 200 km at point B, the distance from A to B is simply 200 - 150 = 50. You don't need to re-drive the road.

Prefix sums work the same way:

```
arr =        [1,  2,  3,  4,  5]
prefix =  [0, 1,  3,  6, 10, 15]

Sum from index 1 to index 3 = prefix[4] - prefix[1] = 10 - 1 = 9
Which is: 2 + 3 + 4 = 9 ✅
```

### Subarray Sum Equals K (LeetCode #560)

**The Concept:** You want to count how many subarrays add up to `k`. The key insight: if at position `j` the prefix sum is `P`, and at some earlier position `i` the prefix sum was `P - k`, then the subarray from `i+1` to `j` sums to exactly `k`. Use a HashMap to count how many times each prefix sum has occurred.

```python
def subarraySum(nums, k):
    count = prefix = 0
    seen = {0: 1}                      # prefix_sum → how many times we've seen it
    for num in nums:
        prefix += num
        count += seen.get(prefix - k, 0)  # how many earlier positions give sum k?
        seen[prefix] = seen.get(prefix, 0) + 1
    return count
# O(n) time, O(n) space
```

---

## Pattern 3: Kadane's Algorithm — Maximum Subarray Sum

### The Core Idea

> **"At each step, decide: should I extend the current subarray, or start fresh?"**

Think about it intuitively: if the running sum goes negative, there's no point carrying it forward — starting fresh from the current number is always better.

**Analogy:** You're collecting coins while walking. Some coins are negative (they cost you money). If your wallet hits negative, drop everything and start collecting from scratch.

### Maximum Subarray (LeetCode #53) — Top 5 Interview Question

```
nums = [-2, 1, -3, 4, -1, 2, 1, -5, 4]

curr = -2 → start fresh with 1 → extend to -2 → start fresh with 4 →
extend to 3 → extend to 5 → extend to 6 → extend to 1 → extend to 5

Best ever seen = 6, from subarray [4, -1, 2, 1]
```

```python
def maxSubArray(nums):
    curr = best = nums[0]
    for num in nums[1:]:
        curr = max(num, curr + num)    # extend or restart?
        best = max(best, curr)         # track the overall best
    return best
# O(n) time, O(1) space — can't do better than this!
```

---

# 👆 0:30 — Two Pointers + Sliding Window (25 min)

## What Are Two Pointers?

Two Pointers is a technique where you use **two indices** that move through the data, usually to avoid nested loops. Instead of checking every pair (O(n²)), you set up two pointers that intelligently skip unnecessary comparisons.

### When to Use Two Pointers

```
✅ Input is SORTED (or you can sort it)
✅ You need to find PAIRS/TRIPLETS with some condition
✅ You need to compare elements from BOTH ENDS
✅ You need to partition or rearrange elements in-place
```

### There Are Two Main Flavors:

1. **Opposite-end pointers** — Start from both ends, move inward
2. **Same-direction pointers (fast/slow)** — Both start at the beginning, one moves faster

---

## Pattern 4: Two Pointers — Opposite Ends

### The Core Idea

> **"Start from both extremes and converge inward using logic about which pointer to move."**

This works because in a sorted array, the smallest values are on the left and largest on the right. Moving the left pointer increases the sum; moving the right pointer decreases it. This gives you **precise control** over the direction of your search.

### Container With Most Water (LeetCode #11)

**The Concept:** You have vertical bars and want to find two bars that hold the most water. The area = width × height (limited by the shorter bar). Starting from the widest possible container (both ends), you shrink the width by moving the pointer at the **shorter** bar — because the short bar is the bottleneck, and there's no point keeping it.

```python
def maxArea(height):
    lo, hi = 0, len(height) - 1
    best = 0
    while lo < hi:
        area = (hi - lo) * min(height[lo], height[hi])
        best = max(best, area)
        if height[lo] < height[hi]:
            lo += 1                     # short bar is the bottleneck → move it
        else:
            hi -= 1
    return best
# O(n) time, O(1) space
```

### 3Sum (LeetCode #15)

**The Concept:** Find all unique triplets that sum to zero. The trick: **sort first**, fix one number, then use two pointers on the rest (which is now a sorted two-sum problem). Skip duplicate values to avoid repeats.

**Why sorting helps:** Once sorted, you can precisely control the sum direction. Sum too small → move left pointer right. Sum too big → move right pointer left.

```python
def threeSum(nums):
    nums.sort()
    res = []
    for i in range(len(nums) - 2):
        if i > 0 and nums[i] == nums[i-1]: continue   # skip duplicate fixed values
        lo, hi = i + 1, len(nums) - 1
        while lo < hi:
            s = nums[i] + nums[lo] + nums[hi]
            if s == 0:
                res.append([nums[i], nums[lo], nums[hi]])
                while lo < hi and nums[lo] == nums[lo+1]: lo += 1  # skip dupes
                while lo < hi and nums[hi] == nums[hi-1]: hi -= 1
                lo += 1; hi -= 1
            elif s < 0: lo += 1         # sum too small → need bigger
            else:       hi -= 1         # sum too big → need smaller
    return res
# O(n²) — much better than the O(n³) brute force
```

### ⭐ Trapping Rain Water (LeetCode #42) — Classic Hard

**The Concept:** At any position, the water it can hold = `min(max_height_to_left, max_height_to_right) - current_height`. Rather than computing left-max and right-max for every position (O(n) space), use two pointers: the shorter side is the **bottleneck**, so process that side and track its running max.

**Why it works:** Water is always limited by the shorter wall. If `left_max < right_max`, then no matter what's on the right, the water at the left pointer is determined by `left_max` alone.

```python
def trap(height):
    lo, hi = 0, len(height) - 1
    lo_max = hi_max = water = 0
    while lo < hi:
        if height[lo] <= height[hi]:   # left is bottleneck
            lo_max = max(lo_max, height[lo])
            water += lo_max - height[lo]
            lo += 1
        else:                          # right is bottleneck
            hi_max = max(hi_max, height[hi])
            water += hi_max - height[hi]
            hi -= 1
    return water
# O(n) time, O(1) space — brilliant!
```

---

## Pattern 5: Sliding Window — Subarray / Substring Optimization

### The Core Idea

> **"Maintain a window [left, right] that expands and shrinks to track the best valid subarray."**

**Analogy:** Imagine looking through a telescoping window at a street. You widen it (move right boundary) to see more. When you see something you don't want (a constraint violation), you narrow it from the left until it's valid again. You're always tracking the best view you've had.

### When to Use a Sliding Window

```
✅ Problem asks about CONTIGUOUS subarrays or substrings
✅ Keywords: "longest," "shortest," "maximum sum of subarray of size k"
✅ There's a CONDITION that defines valid vs invalid windows
```

### The Universal Template

```python
def sliding_window(arr):
    left = 0
    window_state = ...  # could be a set, counter, sum, etc.
    best = ...
    for right in range(len(arr)):
        # 1. EXPAND: add arr[right] to window state
        while WINDOW_IS_INVALID:
            # 2. SHRINK: remove arr[left] from state, then left += 1
            pass
        # 3. UPDATE: check if current window is the best so far
    return best
```

### Longest Substring Without Repeating Characters (LeetCode #3)

**The Concept:** You want the longest substring where no character repeats. Expand right to include new characters. When you hit a duplicate, shrink from the left until the duplicate is gone.

**Why sliding window works here:** Any valid substring is a contiguous range. Once you find a violation, you don't restart from scratch — you slide the left boundary forward just enough to fix it.

```python
def lengthOfLongestSubstring(s):
    seen = set()                       # characters in current window
    left = best = 0
    for right in range(len(s)):
        while s[right] in seen:        # violation: duplicate found
            seen.remove(s[left])       # shrink from left
            left += 1
        seen.add(s[right])            # expand right
        best = max(best, right - left + 1)
    return best
# O(n) — each character is added and removed from the set at most once
```

### ⭐ Minimum Window Substring (LeetCode #76) — Hard

**The Concept:** Find the smallest substring of `s` that contains ALL characters of `t`. Expand right until you have all needed characters, then shrink left to find the minimum valid window.

```python
from collections import Counter, defaultdict
def minWindow(s, t):
    need = Counter(t)                   # what we need
    have = defaultdict(int)            # what we have in window
    required = len(need)               # number of unique chars to satisfy
    formed = 0                         # how many unique chars are fully satisfied
    left = 0
    res = ""
    for right in range(len(s)):
        c = s[right]
        have[c] += 1
        if c in need and have[c] == need[c]:
            formed += 1               # one more character fully satisfied
        while formed == required:      # window has everything → try shrinking
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

**The Concept:** You can flip at most `k` zeros to ones. Find the longest subarray of all ones. Track the number of zeros in your window — if it exceeds `k`, shrink.

```python
def longestOnes(nums, k):
    left = zeros = best = 0
    for right in range(len(nums)):
        if nums[right] == 0: zeros += 1
        while zeros > k:               # too many zeros → shrink
            if nums[left] == 0: zeros -= 1
            left += 1
        best = max(best, right - left + 1)
    return best
```

---

# 🔍 0:55 — Sorting + Binary Search (20 min)

## Why Sorting Matters

You rarely need to *implement* a sorting algorithm in interviews, but **sorting as a preprocessing step** is incredibly powerful because it unlocks other techniques:

```
Sorted data → enables Binary Search         O(n log n + log n)
Sorted data → enables Two Pointers          O(n log n + n)
Sorted data → enables Merge Intervals       O(n log n + n)
Sorted data → enables Greedy decisions      O(n log n + n)
Sorted data → makes duplicates adjacent     O(n log n + n)
```

> **Key interview question:** "Is the input sorted?" If yes, think Binary Search or Two Pointers immediately.

### Dutch National Flag — Sort 0s, 1s, 2s (LeetCode #75)

**The Concept:** Partition an array into three sections in a single pass using three pointers. Everything before `lo` is 0, everything between `lo` and `mid` is 1, everything after `hi` is 2.

```python
def sortColors(nums):
    lo, mid, hi = 0, 0, len(nums) - 1
    while mid <= hi:
        if   nums[mid] == 0: nums[lo], nums[mid] = nums[mid], nums[lo]; lo += 1; mid += 1
        elif nums[mid] == 1: mid += 1
        else:                nums[mid], nums[hi] = nums[hi], nums[mid]; hi -= 1
# Single pass O(n), O(1) space
```

### Merge Intervals (LeetCode #56)

**The Concept:** Sort intervals by start time. Then walk through them — if the current interval overlaps with the last merged one, extend it; otherwise, start a new merged interval.

```python
def merge(intervals):
    intervals.sort()                    # sort by start time
    res = [intervals[0]]
    for s, e in intervals[1:]:
        if s <= res[-1][1]:            # overlap detected
            res[-1][1] = max(res[-1][1], e)  # extend
        else:
            res.append([s, e])         # no overlap → new interval
    return res
```

---

## Pattern 6: Binary Search — The Art of Halving

### The Core Idea

> **"If you can determine which half of your search space the answer lives in, throw away the other half. Repeat until you find it."**

**Analogy:** The classic "number guessing game." You guess a number between 1-100. Someone says "higher" or "lower." You always guess the middle — worst case, 7 guesses for 100 numbers. That's O(log n).

### The Key Requirements

1. The search space must be **logically sortable / ordered**
2. You must have a **condition** that tells you which half to keep

### Standard Binary Search

```python
def binary_search(arr, target):
    lo, hi = 0, len(arr) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if   arr[mid] == target: return mid
        elif arr[mid] < target:  lo = mid + 1   # answer is in right half
        else:                    hi = mid - 1   # answer is in left half
    return -1
```

### Search in Rotated Sorted Array (LeetCode #33)

**The Concept:** A sorted array has been rotated (e.g., [4,5,6,7,0,1,2]). The trick: at any `mid`, **one half is always still sorted**. Check if the target falls within the sorted half → search there. Otherwise → search the other half.

```python
def search(nums, target):
    lo, hi = 0, len(nums) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if nums[mid] == target: return mid
        if nums[lo] <= nums[mid]:          # left half is sorted
            if nums[lo] <= target < nums[mid]:
                hi = mid - 1               # target in sorted left half
            else:
                lo = mid + 1
        else:                               # right half is sorted
            if nums[mid] < target <= nums[hi]:
                lo = mid + 1               # target in sorted right half
            else:
                hi = mid - 1
    return -1
```

---

### ⭐ Binary Search on Answer — The Most Powerful Pattern You'll Learn

**The Concept:** Instead of searching an array, **search the range of possible answers** themselves. This works when:

1. The answer lies within a known range [lo, hi]
2. You can write a function `is_feasible(mid)` to check if `mid` works
3. The feasibility is **monotonic** — once it becomes feasible, it stays feasible (or vice versa)

**Analogy:** You're trying to find the minimum speed at which you can eat all bananas in `h` hours. Instead of trying every speed from 1 to max, binary search: "Can I finish at speed 5? Yes. Speed 3? No. Speed 4? Yes!" → Answer is 4.

### Koko Eating Bananas (LeetCode #875)

```python
import math
def minEatingSpeed(piles, h):
    lo, hi = 1, max(piles)            # search range: speed 1 to max pile
    while lo < hi:
        mid = (lo + hi) // 2
        hours_needed = sum(math.ceil(p / mid) for p in piles)
        if hours_needed <= h:
            hi = mid                   # mid is feasible → try slower (smaller)
        else:
            lo = mid + 1              # mid is too slow → need faster (bigger)
    return lo
# O(n × log(max_pile)) — massively faster than trying every speed
```

> 🎬 Visualize: [visualgo.net/bst](https://visualgo.net/en/bst)

---

# 🔤 1:15 — Strings + Bits (20 min)

## String Problems — Mostly Patterns You Already Know

Most string problems are really **array problems**, because a string is just an array of characters. The patterns you already learned (HashMap, Two Pointers, Sliding Window) apply directly. A few string-specific techniques:

### Longest Palindromic Substring (LeetCode #5) — Expand Around Center

**The Concept:** A palindrome reads the same forwards and backwards. Instead of checking every possible substring (O(n³)), start at each character and **expand outward** as long as the characters match. Check both odd-length (expand from single char) and even-length (expand from pair) palindromes.

```python
def longestPalindrome(s):
    res = ""
    def expand(l, r):
        while l >= 0 and r < len(s) and s[l] == s[r]:
            l -= 1; r += 1
        return s[l+1:r]               # the palindrome found
    for i in range(len(s)):
        odd = expand(i, i)             # odd-length: "aba"
        even = expand(i, i+1)          # even-length: "abba"
        for pal in (odd, even):
            if len(pal) > len(res): res = pal
    return res
# O(n²) — much better than O(n³) brute force
```

### Valid Anagram (LeetCode #242)

**The Concept:** Two strings are anagrams if they have the exact same character frequencies. The most Pythonic way:

```python
from collections import Counter
def isAnagram(s, t):
    return Counter(s) == Counter(t)
```

---

## Bit Manipulation — Four Tricks Worth Knowing

Bit manipulation operates directly on the binary representation of numbers. You don't need deep knowledge, but these four tricks show up repeatedly:

### 1. XOR Cancels Pairs (LeetCode #136 — Single Number)

**The Concept:** XOR (^) has a special property: `a ^ a = 0` and `a ^ 0 = a`. So if you XOR all numbers together, every pair cancels out, leaving only the unique number.

```python
def singleNumber(nums):
    result = 0
    for n in nums: result ^= n
    return result
# [4, 1, 2, 1, 2] → 4^1^2^1^2 → (1^1)^(2^2)^4 → 0^0^4 → 4
```

### 2. Check Power of 2

**The Concept:** Powers of 2 have exactly one bit set (e.g., 8 = `1000`). `n & (n-1)` clears the lowest set bit. If the result is 0, there was only one bit → power of 2.

```python
def isPowerOfTwo(n):
    return n > 0 and (n & (n-1)) == 0
```

### 3. Count Set Bits (Brian Kernighan's)

**The Concept:** Each `n & (n-1)` operation removes exactly one set bit. Count how many times until `n` becomes 0.

```python
def countBits(n):
    count = 0
    while n:
        n &= n - 1   # removes lowest set bit
        count += 1
    return count
```

### 4. Even/Odd Check

```python
n & 1 == 0  # even (last bit is 0)
n & 1 == 1  # odd  (last bit is 1)
```

---

# 🔁 1:35 — Recursion & Backtracking (25 min)

## What is Recursion? (The Most Important Concept After Hashing)

Recursion is when a function **calls itself** to solve a smaller version of the same problem. Every recursive solution needs:

1. **Base case** — when to stop (otherwise infinite loop!)
2. **Recursive case** — break the problem into a smaller identical problem

**Analogy:** Russian nesting dolls (Matryoshka). To find the smallest doll, you open one, find another inside, open that one, find another... until you reach the smallest (base case). Then you work your way back up.

```python
def factorial(n):
    if n <= 1: return 1        # base case: smallest doll
    return n * factorial(n-1)  # smaller problem: open the next doll
```

### How to Think Recursively

This is the hardest mindset shift for beginners. The trick:

> **"Assume the recursive call works perfectly. Now, how do I use its result to solve the current problem?"**

For example, to reverse a linked list recursively:
- Assume `reverse(head.next)` perfectly reverses everything after the head
- Now just attach the head at the end

> 🎬 Visualize YOUR recursive code: [pythontutor.com](https://pythontutor.com/) — watch the call stack grow and shrink.

---

## What is Backtracking?

Backtracking is recursion with **undo**. You make a choice, explore it fully, then **undo the choice** and try the next option. It's how you systematically explore all possibilities.

**Analogy:** You're in a maze. At each fork, you pick a path. If it's a dead end, you walk BACK to the fork and try the other path. That's backtracking.

```
The Backtracking Template:
1. Make a CHOICE (add element, place queen, etc.)
2. RECURSE with the choice
3. UNDO the choice (pop element, remove queen, etc.)
4. Try the NEXT choice
```

---

## Pattern 7: Subsets — The "Include or Exclude" Decision

### The Core Idea

> **"For each element, you have exactly two choices: include it or skip it. This creates a binary decision tree with 2ⁿ leaves (one per subset)."**

```
Elements: [1, 2, 3]

                    []
                 /      \
           [1]            []           ← include 1 or skip 1?
          /    \        /    \
      [1,2]  [1]     [2]    []        ← include 2 or skip 2?
      / \    / \     / \    / \
  [123][12][13][1] [23][2] [3] []     ← include 3 or skip 3?

Result: [1,2,3], [1,2], [1,3], [1], [2,3], [2], [3], []
```

### Subsets (LeetCode #78)

```python
def subsets(nums):
    res = []
    def bt(i, curr):
        if i == len(nums):             # base case: considered all elements
            res.append(curr[:])        # save a copy of current subset
            return
        curr.append(nums[i])           # CHOICE: include nums[i]
        bt(i + 1, curr)               # explore with this choice
        curr.pop()                     # UNDO: backtrack
        bt(i + 1, curr)               # CHOICE: skip nums[i]
    bt(0, [])
    return res
# O(2^n) — there are exactly 2^n subsets, and we must enumerate all
```

### Combination Sum (LeetCode #39)

**The Concept:** Find all combinations that sum to a target. Unlike subsets, you can reuse elements. At each step, try adding each candidate (starting from the current index to avoid duplicates), recurse with reduced target, then backtrack.

```python
def combinationSum(candidates, target):
    res = []
    def bt(start, curr, remain):
        if remain == 0:                # found a valid combination!
            res.append(curr[:])
            return
        if remain < 0: return          # overshot → prune this branch
        for i in range(start, len(candidates)):
            curr.append(candidates[i])
            bt(i, curr, remain - candidates[i])  # i, not i+1: reuse allowed
            curr.pop()                 # backtrack
    bt(0, [], target)
    return res
```

### Permutations (LeetCode #46)

**The Concept:** Order matters here (unlike subsets). For each position, choose from the remaining elements. There are n! permutations total.

```python
def permute(nums):
    res = []
    def bt(curr, remaining):
        if not remaining:              # used all elements
            res.append(curr[:])
            return
        for i in range(len(remaining)):
            curr.append(remaining[i])
            bt(curr, remaining[:i] + remaining[i+1:])  # exclude chosen
            curr.pop()                 # backtrack
    bt([], nums)
    return res
# O(n!) — n choices for first, n-1 for second, etc.
```

### ⭐ N-Queens (LeetCode #51) — Classic Hard

**The Concept:** Place n queens on an n×n board so no two attack each other. Place row by row. At each row, try each column. Use sets to track which columns, diagonals (row-col), and anti-diagonals (row+col) are under attack.

**Why sets for diagonals:** All cells on the same diagonal share the same `row - col` value. All cells on the same anti-diagonal share the same `row + col` value.

```python
def solveNQueens(n):
    res = []
    cols, diag, anti_diag = set(), set(), set()
    board = [['.']*n for _ in range(n)]
    
    def bt(row):
        if row == n:                   # placed all queens successfully
            res.append([''.join(r) for r in board])
            return
        for col in range(n):
            if col in cols or row-col in diag or row+col in anti_diag:
                continue               # this square is under attack → skip
            # PLACE queen
            board[row][col] = 'Q'
            cols.add(col); diag.add(row-col); anti_diag.add(row+col)
            bt(row + 1)
            # UNDO (backtrack)
            board[row][col] = '.'
            cols.discard(col); diag.discard(row-col); anti_diag.discard(row+col)
    bt(0)
    return res
```

---

# ✅ Day 1 Summary — 7 Patterns

| # | Pattern | Core Insight | When to Use | Key Problem |
|---|---------|-------------|-------------|-------------|
| 1 | **HashMap** | Trade space for O(1) lookup | "Have I seen X before?" | Two Sum #1 |
| 2 | **Prefix Sum** | Pre-compute cumulative sums | Subarray sum queries | Subarray Sum K #560 |
| 3 | **Kadane's** | Extend or restart at each step | Maximum subarray sum | Max Subarray #53 |
| 4 | **Two Pointers** | Converge from both ends | Sorted data, pairs | 3Sum #15, Trapping Rain Water #42 |
| 5 | **Sliding Window** | Expand right, shrink left | Contiguous subarray/substring | Longest Substring #3, Min Window #76 |
| 6 | **Binary Search** | Halve the search space | Sorted data, search on answer | Rotated Array #33, Koko Bananas #875 |
| 7 | **Backtracking** | Choose → Explore → Undo | Generate all possibilities | Subsets #78, N-Queens #51 |

### 🏋️ Tonight's Homework (Pick 5)
```
🟢 #1    Two Sum                    🟡 #53  Max Subarray
🟡 #3    Longest Substring          🟡 #15  3Sum
🟡 #33   Search Rotated Array       🟡 #560 Subarray Sum K
🟡 #78   Subsets                    🔴 #42  Trapping Rain Water
```

---

*Tomorrow: Linked Lists, Stacks, Trees, Graphs, DP → [day2-2hrs.md](day2-2hrs.md)*
