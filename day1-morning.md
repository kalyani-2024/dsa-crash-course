# 🌅 Day 1 — Morning Session (9:00 AM - 12:30 PM)

## Big-O Notation → Arrays → Hashing → Two Pointers → Sliding Window

---

# 📐 Part 1: Big-O Notation — How to Measure Your Code (30 min)

## What is Big-O?

Big-O tells you **how your code's speed changes as input grows**. It's the FIRST thing interviewers evaluate.

> **Think of it like this:** If you have a list of 10 items and your code takes 1 second, what happens with 1,000,000 items?

### The Complexity Chart

```
SPEED           ║  NAME              ║  EXAMPLE                    ║ n=1000
════════════════╬════════════════════╬═════════════════════════════╬═══════════
🟢 O(1)        ║  Constant          ║  Array access, HashMap get  ║ 1
🟢 O(log n)    ║  Logarithmic       ║  Binary Search              ║ 10
🟢 O(n)        ║  Linear            ║  Simple loop                ║ 1,000
🟡 O(n log n)  ║  Log-linear        ║  Merge Sort, Quick Sort     ║ 10,000
🟠 O(n²)       ║  Quadratic         ║  Nested loops               ║ 1,000,000
🔴 O(2ⁿ)       ║  Exponential       ║  Recursion without memo     ║ 🔥 TOO SLOW
🔴 O(n!)       ║  Factorial         ║  Permutations               ║ 💀 IMPOSSIBLE
```

### 🎬 **Visualize it:** [bigocheatsheet.com](https://www.bigocheatsheet.com/) — See the growth curves

### Rules for Calculating Big-O

```python
# Rule 1: Drop constants
O(2n) → O(n)
O(500) → O(1)

# Rule 2: Drop lower-order terms
O(n² + n) → O(n²)
O(n + log n) → O(n)

# Rule 3: Different inputs = different variables
def func(arr1, arr2):    # arr1 has length m, arr2 has length n
    for x in arr1:       # O(m)
        ...
    for y in arr2:       # O(n)
        ...
# Total: O(m + n), NOT O(n)

# Rule 4: Nested loops multiply
for i in range(n):       # O(n)
    for j in range(n):   # × O(n)
        ...              # = O(n²)
```

### Space Complexity
Same idea, but for **memory usage**:
```python
# O(1) space — only variables
total = 0

# O(n) space — array of size n
arr = [0] * n

# O(n) space — recursion with depth n
def recursive(n):
    if n == 0: return
    recursive(n - 1)    # Each call uses stack space
```

### ⏱ Interview Constraint Cheat Sheet

| **n (input size)** | **Maximum Acceptable** | **Typical Approach** |
|---------------------|----------------------|----------------------|
| n ≤ 10 | O(n!) or O(2ⁿ) | Brute force / backtracking |
| n ≤ 20 | O(2ⁿ) | Bitmask DP |
| n ≤ 100 | O(n³) | Triple nested loops |
| n ≤ 1,000 | O(n²) | Double nested loops |
| n ≤ 100,000 | O(n log n) | Sorting / Binary Search |
| n ≤ 10,000,000 | O(n) | Single pass / Hashing |
| n > 10,000,000 | O(log n) or O(1) | Math / Binary Search |

> **💡 Pro Tip:** Look at the constraints in LeetCode! If n ≤ 10⁵, you need O(n log n) or better.

---

# 📦 Part 2: Arrays — The Foundation of Everything (90 min)

## What is an Array?

An array is a **contiguous block of memory** storing elements of the same type, accessed by index.

```
Index:    0    1    2    3    4
        ┌────┬────┬────┬────┬────┐
Array:  │ 10 │ 20 │ 30 │ 40 │ 50 │
        └────┴────┴────┴────┴────┘
```

### 🎬 **Visualize it:** [pythontutor.com](https://pythontutor.com/) — Paste any array code and watch memory

### Array Operations Cheat Sheet

| Operation | Time | Code |
|-----------|------|------|
| Access by index | O(1) | `arr[i]` |
| Search (unsorted) | O(n) | `x in arr` |
| Insert at end | O(1)* | `arr.append(x)` |
| Insert at beginning | O(n) | `arr.insert(0, x)` |
| Delete by index | O(n) | `arr.pop(i)` |
| Delete from end | O(1) | `arr.pop()` |
| Sort | O(n log n) | `arr.sort()` |

*Amortized — occasionally O(n) when resizing

---

## 🧩 Pattern 1: Single Pass / Linear Scan

**When to use:** Finding max/min, counting, checking conditions.

### Problem: Find the Largest Element
```python
def find_largest(arr):
    """Simply track the maximum as you scan."""
    largest = arr[0]
    for num in arr:
        if num > largest:
            largest = num
    return largest
# Time: O(n) | Space: O(1)
```

### Problem: Maximum Subarray Sum — Kadane's Algorithm (LeetCode #53)

This is the **#1 most important array algorithm** you must know.

```
arr = [-2, 1, -3, 4, -1, 2, 1, -5, 4]

The idea: At each position, decide:
  → Should I continue the previous subarray? (current_sum + num)
  → Or start a new subarray here? (num alone)

Walk through:
  num=-2: current=max(-2, 0+(-2))=-2, best=-2
  num= 1: current=max(1, -2+1)=1,     best= 1
  num=-3: current=max(-3, 1+(-3))=-2,  best= 1
  num= 4: current=max(4, -2+4)=4,     best= 4
  num=-1: current=max(-1, 4+(-1))=3,   best= 4
  num= 2: current=max(2, 3+2)=5,      best= 5
  num= 1: current=max(1, 5+1)=6,      best= 6  ← ANSWER
  num=-5: current=max(-5, 6+(-5))=1,   best= 6
  num= 4: current=max(4, 1+4)=5,      best= 6
```

```python
def maxSubArray(nums):
    """Kadane's Algorithm — THE most important array pattern."""
    current_sum = max_sum = nums[0]
    
    for num in nums[1:]:
        # Key decision: extend current subarray or start fresh?
        current_sum = max(num, current_sum + num)
        max_sum = max(max_sum, current_sum)
    
    return max_sum
# Time: O(n) | Space: O(1)
```

> **🎯 Interview Tip:** Kadane's is asked in almost every interview. Practice it until you can write it in your sleep.

---

## 🧩 Pattern 2: Prefix Sum

**When to use:** Subarray sum queries, range sum calculations.

### How it Works
```
arr:          [1,  2,  3,  4,  5]
prefix_sum:   [0,  1,  3,  6, 10, 15]
                         ↑
              prefix_sum[i] = sum of arr[0..i-1]

Sum of arr[2..4] = prefix_sum[5] - prefix_sum[2] = 15 - 3 = 12
                 = arr[2] + arr[3] + arr[4] = 3 + 4 + 5 = 12 ✅
```

### Problem: Subarray Sum Equals K (LeetCode #560)

```python
def subarraySum(nums, k):
    """Count subarrays with sum = k using prefix sum + hashmap."""
    count = 0
    prefix_sum = 0
    # Map: prefix_sum → how many times we've seen it
    prefix_map = {0: 1}  # Empty prefix has sum 0
    
    for num in nums:
        prefix_sum += num
        
        # If (prefix_sum - k) exists, there's a subarray with sum k
        if prefix_sum - k in prefix_map:
            count += prefix_map[prefix_sum - k]
        
        prefix_map[prefix_sum] = prefix_map.get(prefix_sum, 0) + 1
    
    return count
# Time: O(n) | Space: O(n)
```

**Why this works:**
```
If prefix_sum[j] - prefix_sum[i] = k
Then sum of arr[i+1..j] = k

So for each j, we ask: "Have I seen prefix_sum[j] - k before?"
```

---

# 🗃️ Part 3: Hashing — Your Secret Weapon (45 min)

## What is Hashing?

Hashing converts data into a fixed-size value for **O(1) average** lookup. Think of it as a **magic lookup table**.

```
Dictionary/HashMap:
┌────────────────────────────┐
│  Key     →  Value          │
│  "apple" →  5              │
│  "banana" → 3              │
│  "cherry" → 7              │
└────────────────────────────┘
Access: O(1) average!

Set/HashSet:
┌────────────────────────────┐
│  {1, 5, 3, 7, 9}          │
│  "Is 5 in the set?" → O(1)│
└────────────────────────────┘
```

### 🎬 **Visualize it:** [cs.usfca.edu - Hash Tables](https://www.cs.usfca.edu/~galles/visualization/OpenHash.html) — Watch how hash tables work

### Python Hash Data Structures

```python
# Dictionary (HashMap) — Key-Value pairs
freq = {}
freq["apple"] = 5
freq.get("banana", 0)  # Returns 0 if key doesn't exist

# Counter — Auto-counts frequencies
from collections import Counter
counts = Counter([1, 2, 2, 3, 3, 3])  # {3: 3, 2: 2, 1: 1}

# defaultdict — Dict with default values
from collections import defaultdict
graph = defaultdict(list)  # Default value is empty list
graph[1].append(2)  # No KeyError!

# Set — Unique elements, O(1) lookup
seen = set()
seen.add(5)
if 5 in seen:  # O(1) check
    print("Found!")
```

---

## 🧩 Pattern 3: Frequency Counting

### Problem: Two Sum (LeetCode #1) — 🔥 THE Most Asked Interview Question

```
Given nums = [2, 7, 11, 15], target = 9
Find two numbers that add up to target.
Return their indices.

Brute force: Check all pairs → O(n²)  ❌ Too slow

HashMap approach:
  For each number: "Does (target - number) exist in my map?"
  
  Step 1: num=2, need 9-2=7, not in map. Store {2: 0}
  Step 2: num=7, need 9-7=2, 2 IS in map at index 0! → Return [0, 1] ✅
```

```python
def twoSum(nums, target):
    """The #1 interview question. Must know this pattern cold."""
    seen = {}  # value → index
    
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    
    return []
# Time: O(n) | Space: O(n)
```

### Problem: Longest Consecutive Sequence (LeetCode #128)

```python
def longestConsecutive(nums):
    """Find the longest sequence of consecutive numbers."""
    num_set = set(nums)
    best = 0
    
    for num in num_set:
        # Only start counting from the BEGINNING of a sequence
        if num - 1 not in num_set:  # This is the start!
            length = 1
            while num + length in num_set:
                length += 1
            best = max(best, length)
    
    return best
# Time: O(n) | Space: O(n)
# Key insight: The "if num - 1 not in set" check ensures each
# sequence is counted only once, keeping it O(n).
```

### Problem: Group Anagrams (LeetCode #49)

```python
def groupAnagrams(strs):
    """Group words that are anagrams of each other."""
    groups = defaultdict(list)
    
    for word in strs:
        # Anagrams have the same sorted letters
        key = tuple(sorted(word))  # "eat" → ('a','e','t')
        groups[key].append(word)
    
    return list(groups.values())
# Time: O(n × k log k) where k = max word length | Space: O(n × k)
```

---

# 👆 Part 4: Two Pointers — Elegant O(n) Solutions (45 min)

## What is Two Pointers?

Two pointers move through data based on conditions. This reduces O(n²) brute force to O(n).

### 🎬 **Visualize it:** Think of two cursors moving through an array

```
Three main patterns:

1. OPPOSITE ENDS (converging):
   [1, 2, 3, 4, 5, 6, 7]
    L→                ←R

2. SAME DIRECTION (fast/slow):
   [1, 2, 3, 4, 5, 6, 7]
    S→ F→

3. TWO ARRAYS (merge):
   [1, 3, 5]    [2, 4, 6]
    i→            j→
```

---

## 🧩 Pattern 4: Opposite Direction Two Pointers

### Problem: Two Sum II — Sorted Array (LeetCode #167)

```
numbers = [2, 7, 11, 15], target = 9

Left=0 (value 2), Right=3 (value 15)
Sum = 2+15 = 17 > 9 → Move Right ←
Left=0 (value 2), Right=2 (value 11)  
Sum = 2+11 = 13 > 9 → Move Right ←
Left=0 (value 2), Right=1 (value 7)
Sum = 2+7 = 9 = target ✅ → Return [1, 2]
```

```python
def twoSumSorted(numbers, target):
    """When array is SORTED, use two pointers instead of hashmap."""
    left, right = 0, len(numbers) - 1
    
    while left < right:
        total = numbers[left] + numbers[right]
        if total == target:
            return [left + 1, right + 1]  # 1-indexed
        elif total < target:
            left += 1    # Need bigger sum → move left pointer right
        else:
            right -= 1   # Need smaller sum → move right pointer left
    
    return []
# Time: O(n) | Space: O(1)
```

### Problem: Container With Most Water (LeetCode #11)

```python
def maxArea(height):
    """Find two lines that form container holding most water."""
    left, right = 0, len(height) - 1
    max_water = 0
    
    while left < right:
        # Water limited by shorter line
        width = right - left
        h = min(height[left], height[right])
        max_water = max(max_water, width * h)
        
        # Move the shorter line inward (it's the bottleneck)
        if height[left] < height[right]:
            left += 1
        else:
            right -= 1
    
    return max_water
# Time: O(n) | Space: O(1)
```

### Problem: 3Sum — Find Three Numbers that Add to Zero (LeetCode #15)

```python
def threeSum(nums):
    """Fix one number, then use two pointers for the other two."""
    nums.sort()  # MUST sort first
    result = []
    
    for i in range(len(nums) - 2):
        # Skip duplicates for first number
        if i > 0 and nums[i] == nums[i - 1]:
            continue
        
        left, right = i + 1, len(nums) - 1
        
        while left < right:
            total = nums[i] + nums[left] + nums[right]
            
            if total == 0:
                result.append([nums[i], nums[left], nums[right]])
                # Skip duplicates
                while left < right and nums[left] == nums[left + 1]:
                    left += 1
                while left < right and nums[right] == nums[right - 1]:
                    right -= 1
                left += 1
                right -= 1
            elif total < 0:
                left += 1
            else:
                right -= 1
    
    return result
# Time: O(n²) | Space: O(1) excluding output
```

---

## 🧩 Pattern 5: Same Direction Two Pointers

### Problem: Remove Duplicates from Sorted Array (LeetCode #26)

```python
def removeDuplicates(nums):
    """'write' pointer only advances for unique elements."""
    if not nums:
        return 0
    
    write = 1  # Position to write next unique element
    
    for read in range(1, len(nums)):
        if nums[read] != nums[read - 1]:
            nums[write] = nums[read]
            write += 1
    
    return write
# Time: O(n) | Space: O(1)
```

### Problem: Move Zeroes (LeetCode #283)
```python
def moveZeroes(nums):
    """Move all zeros to end, maintaining order of non-zeros."""
    write = 0
    for read in range(len(nums)):
        if nums[read] != 0:
            nums[write], nums[read] = nums[read], nums[write]
            write += 1
# Time: O(n) | Space: O(1)
```

### Problem: Trapping Rain Water (LeetCode #42) — ⭐ Classic Hard

```
height = [0,1,0,2,1,0,1,3,2,1,2,1]

Visual:
       █
   █   ██ █
 █≈██≈█████≈█    (≈ = trapped water)
_____________
 0 1 0 2 1 0 1 3 2 1 2 1

Water at each position = min(left_max, right_max) - height
```

```python
def trap(height):
    """Two pointers from both ends. 
    The shorter side determines the water level."""
    left, right = 0, len(height) - 1
    left_max = right_max = 0
    water = 0
    
    while left < right:
        if height[left] <= height[right]:
            left_max = max(left_max, height[left])
            water += left_max - height[left]  # Water above current bar
            left += 1
        else:
            right_max = max(right_max, height[right])
            water += right_max - height[right]
            right -= 1
    
    return water
# Time: O(n) | Space: O(1)
# Why it works: When left_max < right_max, water at left position
# is determined by left_max (it's the bottleneck).
```

---

# 🪟 Part 5: Sliding Window — Subarray/Substring Problems (45 min)

## What is Sliding Window?

A **window** (subarray/substring) that slides across data, expanding or shrinking based on conditions.

```
FIXED SIZE WINDOW (size k=3):
[1, 3, 2, 6, 1, 4, 5]
 ├──────┤                  window = [1,3,2], sum=6
    ├──────┤               window = [3,2,6], sum=11
       ├──────┤            window = [2,6,1], sum=9
          ├──────┤         window = [6,1,4], sum=11
             ├──────┤      window = [1,4,5], sum=10

VARIABLE SIZE WINDOW:
[2, 3, 1, 2, 4, 3], target ≥ 7
 ├──────────┤              sum=8 ≥ 7, try shrinking...
    ├───────┤              sum=6 < 7, expand...
    ├──────────┤           sum=10 ≥ 7, try shrinking...
```

### 🎬 **Visualize it:** Think of a caterpillar 🐛 — the front expands, the back contracts

---

## 🧩 Pattern 6: Variable-Size Sliding Window

### The Universal Template
```python
def sliding_window(arr):
    left = 0
    result = ...
    window_state = ...  # sum, count, set, map, etc.
    
    for right in range(len(arr)):
        # 1. EXPAND: Add arr[right] to window
        update_window_state(arr[right])
        
        # 2. SHRINK: While window is invalid, remove from left
        while window_is_invalid():
            remove_from_window(arr[left])
            left += 1
        
        # 3. UPDATE: Record the best answer
        result = best(result, right - left + 1)
    
    return result
```

### Problem: Longest Substring Without Repeating Characters (LeetCode #3)

```
s = "abcabcbb"

Window:  a     → {a}       length=1
         ab    → {a,b}     length=2
         abc   → {a,b,c}   length=3
         abca  → 'a' repeats! Shrink: remove 'a' → bc a 
         bca   → {b,c,a}   length=3
         bcab  → 'b' repeats! Shrink: remove 'b' → cab
         cab   → {c,a,b}   length=3
         ...

Answer: 3
```

```python
def lengthOfLongestSubstring(s):
    """Find longest substring with all unique characters."""
    char_set = set()
    left = 0
    max_len = 0
    
    for right in range(len(s)):
        # Shrink while duplicate exists
        while s[right] in char_set:
            char_set.remove(s[left])
            left += 1
        
        char_set.add(s[right])
        max_len = max(max_len, right - left + 1)
    
    return max_len
# Time: O(n) | Space: O(min(n, alphabet_size))
```

### Problem: Minimum Window Substring (LeetCode #76) — ⭐ Hard

```python
from collections import Counter, defaultdict

def minWindow(s, t):
    """Smallest substring of s containing all characters of t."""
    if not s or not t:
        return ""
    
    t_count = Counter(t)
    required = len(t_count)       # Unique chars needed
    
    window_count = defaultdict(int)
    formed = 0                     # Unique chars with enough frequency
    
    result = (float('inf'), 0, 0)  # (length, left, right)
    left = 0
    
    for right in range(len(s)):
        # Expand
        char = s[right]
        window_count[char] += 1
        
        if char in t_count and window_count[char] == t_count[char]:
            formed += 1
        
        # Shrink while valid
        while formed == required:
            # Update result
            if right - left + 1 < result[0]:
                result = (right - left + 1, left, right)
            
            # Remove leftmost
            left_char = s[left]
            window_count[left_char] -= 1
            if left_char in t_count and window_count[left_char] < t_count[left_char]:
                formed -= 1
            left += 1
    
    return "" if result[0] == float('inf') else s[result[1]:result[2] + 1]
# Time: O(|s| + |t|) | Space: O(|s| + |t|)
```

### Problem: Maximum Consecutive Ones III (LeetCode #1004)

```python
def longestOnes(nums, k):
    """Longest subarray of 1s if you can flip at most k zeros."""
    left = 0
    zeros = 0
    max_len = 0
    
    for right in range(len(nums)):
        if nums[right] == 0:
            zeros += 1
        
        while zeros > k:
            if nums[left] == 0:
                zeros -= 1
            left += 1
        
        max_len = max(max_len, right - left + 1)
    
    return max_len
# Time: O(n) | Space: O(1)
```

### 🔥 The "Exactly K" Trick

Many problems ask for "exactly K." This is hard directly, but easy with:

```
exactly(K) = atMost(K) - atMost(K - 1)
```

```python
def subarraysWithKDistinct(nums, k):
    """Count subarrays with EXACTLY k distinct integers."""
    return atMost(nums, k) - atMost(nums, k - 1)

def atMost(nums, k):
    """Count subarrays with AT MOST k distinct integers."""
    count = defaultdict(int)
    left = result = 0
    distinct = 0
    
    for right in range(len(nums)):
        if count[nums[right]] == 0:
            distinct += 1
        count[nums[right]] += 1
        
        while distinct > k:
            count[nums[left]] -= 1
            if count[nums[left]] == 0:
                distinct -= 1
            left += 1
        
        result += right - left + 1  # All subarrays ending at 'right'
    
    return result
```

---

## ☕ Morning Review — Key Patterns Learned

| # | Pattern | Time → Optimized | Key Data Structure |
|---|---------|------------------|--------------------|
| 1 | Linear Scan | O(n) | None (variables) |
| 2 | Prefix Sum | O(n²) → O(n) | Array / HashMap |
| 3 | Frequency Count | O(n²) → O(n) | HashMap |
| 4 | Two Pointers (opposite) | O(n²) → O(n) | Two indices |
| 5 | Two Pointers (same dir) | O(n²) → O(n) | Read/Write pointers |
| 6 | Sliding Window | O(n²) → O(n) | Window + HashMap/Set |

> **🎯 Key Insight:** Most array/string problems use ONE of these 6 patterns. Learn to recognize which pattern fits!

---

*Take a lunch break! 🍕 Then continue to [day1-afternoon.md](day1-afternoon.md).*

[← Back to Schedule](README.md) | [Next: Day 1 Afternoon →](day1-afternoon.md)
