# ⚡ Day 1 — Fundamental Data Structures & Core Techniques

## Arrays → Strings → HashMaps → Linked Lists → Stacks & Queues → Sorting → Binary Search

> **Goal:** Master every fundamental data structure, understand when and why to use each, and learn the core patterns that solve 70% of interview problems.

---

## ⏱ Schedule

| Time | Topic | What You'll Learn |
|------|-------|-------------------|
| 0:00 - 0:10 | Big-O & Thinking | How to judge any solution's efficiency |
| 0:10 - 0:35 | Arrays | Traversal, prefix sum, Kadane's, two pointers, sliding window |
| 0:35 - 0:55 | Strings | Palindromes, anagrams, pattern matching, manipulation |
| 0:55 - 1:10 | HashMaps & Sets | O(1) lookup, frequency counting, grouping |
| 1:10 - 1:30 | Linked Lists | Traversal, reversal, slow/fast pointers, merge |
| 1:30 - 1:45 | Stacks & Queues | Matching, monotonic stack, BFS queues |
| 1:45 - 1:55 | Sorting & Binary Search | When sorting unlocks solutions, halving search space |
| 1:55 - 2:00 | Bit Manipulation | XOR tricks, power of 2, set bits |

---

# 🧠 0:00 — Big-O: How to Think About Efficiency

## What is Big-O?

Big-O notation describes **how your algorithm scales as the input grows**. It answers: *"If I double the input size, how much slower does my code get?"*

Think of it like estimating travel time — you don't count exact steps, you say "it's a 10-minute walk" vs "it's a 2-hour drive." Big-O gives you the **shape** of growth, ignoring constants.

### The Common Growth Rates

```
🟢 O(1)        → Constant     → Hash lookup, array[i]        → instant no matter the size
🟢 O(log n)    → Logarithmic  → Binary search                → 20 steps for 1,000,000 items
🟢 O(n)        → Linear       → Single loop                  → scales directly with input
🟡 O(n log n)  → Linearithmic → Sorting                      → slightly worse than linear
🟠 O(n²)       → Quadratic    → Nested loops                 → 10x input = 100x slower
🔴 O(2ⁿ)       → Exponential  → Brute-force subsets           → unusable for n > 25
🔴 O(n!)       → Factorial    → Brute-force permutations      → unusable for n > 12
```

### Why Does This Matter?

A computer does roughly **10⁸ operations/second**. So look at the constraint `n`:

| Constraint (n) | Max Complexity | What to Use |
|----------------|---------------|-------------|
| n ≤ 10 | O(n!) | Brute force, backtracking |
| n ≤ 20 | O(2ⁿ) | Bitmask, backtracking |
| n ≤ 1,000 | O(n²) | Nested loops OK |
| n ≤ 100,000 | O(n log n) | Sort, binary search, heap |
| n ≤ 10⁷ | O(n) | Single pass, hash map |
| n > 10⁷ | O(log n) / O(1) | Math or binary search |

> **💡 Golden Rule:** The FIRST thing to do with any problem — check the constraint `n`. It tells you which complexity you need, which tells you which patterns to try.

### The 5-Step Framework (Use for EVERY Problem)

```
1. UNDERSTAND  — Re-read the problem, walk through examples by hand
2. BRUTE FORCE — What's the "dumb" O(n²) or O(n³) way?
3. OPTIMIZE    — What data structure or pattern makes it faster?
4. CODE        — Write clean code, handle edge cases first
5. TEST        — Dry run with the example + at least one edge case
```

> 🎬 Bookmark: [bigocheatsheet.com](https://www.bigocheatsheet.com/)

---

# 📦 0:10 — Arrays (25 min)

## What is an Array?

An array is the **simplest and most fundamental** data structure — a contiguous block of memory where elements are stored side by side, each accessible by an index.

```
Index:   0    1    2    3    4
       ┌────┬────┬────┬────┬────┐
       │ 10 │ 20 │ 30 │ 40 │ 50 │
       └────┴────┴────┴────┴────┘
```

### Key Properties

| Operation | Time | Why |
|-----------|------|-----|
| Access by index `arr[i]` | O(1) | Direct memory address calculation |
| Search (unsorted) | O(n) | Must check each element |
| Insert at end | O(1) | Just append |
| Insert at middle | O(n) | Must shift everything after |
| Delete at middle | O(n) | Must shift everything after |

### When to Use Arrays

```
✅ You need fast access by index
✅ You know the size (or it doesn't change much)
✅ Data is sequential / ordered
✅ Cache-friendly operations (iterating)
```

---

## Pattern 1: Two Pointers — Avoid Nested Loops

### The Core Idea

> **"Use two indices that move through the data intelligently, skipping unnecessary comparisons."**

Instead of checking every pair (O(n²)), set up two pointers that converge based on a condition. There are two flavors:

**1. Opposite-end pointers** — start from both ends, move inward (works on sorted data)
**2. Same-direction pointers** — both start at beginning, one moves faster

### ⭐ Two Sum (LeetCode #1) — Most Asked Question Ever

**The Concept:** For each number, check if its complement (`target - num`) exists. Use a HashMap (covered later) for O(1) lookup, or sort + two pointers.

```python
def twoSum(nums, target):
    seen = {}                           # value → index
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
# O(n) time, O(n) space
```

### Container With Most Water (LeetCode #11)

**The Concept:** Start with the widest container (both ends). The shorter bar is the bottleneck — move that pointer inward to find potentially taller bars.

**Why it works:** Keeping the shorter bar and shrinking width can ONLY decrease area. Moving the shorter bar might find a taller one.

```python
def maxArea(height):
    lo, hi = 0, len(height) - 1
    best = 0
    while lo < hi:
        area = (hi - lo) * min(height[lo], height[hi])
        best = max(best, area)
        if height[lo] < height[hi]:
            lo += 1
        else:
            hi -= 1
    return best
# O(n) time, O(1) space
```

### 3Sum (LeetCode #15)

**The Concept:** Sort first. Fix one number, then use two pointers on the rest (sorted two-sum).

```python
def threeSum(nums):
    nums.sort()
    res = []
    for i in range(len(nums) - 2):
        if i > 0 and nums[i] == nums[i-1]: continue
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
# O(n²) — much better than O(n³) brute force
```

### ⭐ Trapping Rain Water (LeetCode #42) — Classic Hard

**The Concept:** Water at any position = `min(max_left, max_right) - height`. Use two pointers: the shorter side determines the water, so process that side.

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
# O(n) time, O(1) space
```

---

## Pattern 2: Sliding Window — Subarray / Substring Optimization

### The Core Idea

> **"Maintain a window [left, right] that expands and shrinks to track the best valid subarray."**

**Analogy:** Looking through a telescoping window — widen to see more, narrow when you see something invalid. Always track the best view.

### When to Use

```
✅ Problem asks about CONTIGUOUS subarrays or substrings
✅ Keywords: "longest," "shortest," "maximum sum of size k"
✅ There's a CONDITION that defines valid vs invalid windows
```

### The Universal Template

```python
def sliding_window(arr):
    left = 0
    window_state = ...  # set, counter, sum, etc.
    best = ...
    for right in range(len(arr)):
        # 1. EXPAND: add arr[right] to window
        while WINDOW_IS_INVALID:
            # 2. SHRINK: remove arr[left], left += 1
            pass
        # 3. UPDATE: check if current window is best
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

### Maximum Consecutive Ones III (LeetCode #1004)

**The Concept:** Window with at most `k` zeros. When zeros exceed `k`, shrink.

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

## Pattern 3: Prefix Sum & Kadane's

### Prefix Sum — Answer Subarray Sum Queries in O(1)

> **"Pre-compute cumulative sums so that any subarray sum becomes a single subtraction."**

**Analogy:** Car odometer — distance A to B = reading at B minus reading at A.

```
arr =        [1,  2,  3,  4,  5]
prefix =  [0, 1,  3,  6, 10, 15]
Sum(i..j) = prefix[j+1] - prefix[i]
```

### Subarray Sum Equals K (LeetCode #560)

```python
def subarraySum(nums, k):
    count = prefix = 0
    seen = {0: 1}
    for num in nums:
        prefix += num
        count += seen.get(prefix - k, 0)
        seen[prefix] = seen.get(prefix, 0) + 1
    return count
# O(n)
```

### Kadane's Algorithm — Maximum Subarray (LeetCode #53)

> **"At each step: extend the current subarray, or start fresh?"**

If the running sum goes negative, starting fresh is always better.

```python
def maxSubArray(nums):
    curr = best = nums[0]
    for num in nums[1:]:
        curr = max(num, curr + num)
        best = max(best, curr)
    return best
# O(n) time, O(1) space
```

---

# 🔤 0:35 — Strings (20 min)

## What is a String?

A string is an **array of characters**. This means most array techniques (two pointers, sliding window, hashing) apply directly. But strings have unique properties:

- **Immutable in most languages** — you can't modify in-place in Python/Java (create new strings instead)
- **Character set matters** — ASCII (128 chars), lowercase English (26 chars), Unicode
- **Built-in methods** — `.lower()`, `.split()`, `.join()`, `.isalpha()`, etc.

### Key String Operations & Their Costs

| Operation | Python | Time |
|-----------|--------|------|
| Access char | `s[i]` | O(1) |
| Slice | `s[i:j]` | O(j-i) |
| Concatenate | `s + t` | O(len(s) + len(t)) — creates NEW string |
| Search | `t in s` | O(n×m) worst case |
| Length | `len(s)` | O(1) |
| Compare | `s == t` | O(min(n,m)) |

> **⚠️ Common Pitfall:** Building a string with `+=` in a loop is O(n²) because each concatenation creates a new string! Use `''.join(list)` instead.

```python
# ❌ Slow: O(n²) — each += creates a new string
result = ""
for c in chars:
    result += c

# ✅ Fast: O(n) — join all at once
result = ''.join(chars)
```

---

## Pattern 4: Character Frequency Counting

### The Core Idea

> **"Many string problems reduce to: do two strings have the same character frequencies?"**

### Valid Anagram (LeetCode #242)

**The Concept:** Two strings are anagrams if they have identical character counts.

```python
from collections import Counter
def isAnagram(s, t):
    return Counter(s) == Counter(t)
# O(n), O(1) space (at most 26 keys)
```

### Group Anagrams (LeetCode #49)

**The Concept:** Sorting the letters of any anagram produces the same key. Use that as a HashMap key to group them.

```python
from collections import defaultdict
def groupAnagrams(strs):
    groups = defaultdict(list)
    for s in strs:
        key = tuple(sorted(s))
        groups[key].append(s)
    return list(groups.values())
# O(n × k log k) where k = max string length
```

### Valid Palindrome (LeetCode #125)

**The Concept:** A palindrome reads the same forwards and backwards. Use two pointers from both ends, skipping non-alphanumeric characters.

```python
def isPalindrome(s):
    s = ''.join(c.lower() for c in s if c.isalnum())
    return s == s[::-1]
# Or with two pointers for O(1) extra space:
def isPalindrome(s):
    l, r = 0, len(s) - 1
    while l < r:
        while l < r and not s[l].isalnum(): l += 1
        while l < r and not s[r].isalnum(): r -= 1
        if s[l].lower() != s[r].lower(): return False
        l += 1; r -= 1
    return True
```

---

## Pattern 5: Palindrome Techniques

### The Core Idea

> **"To find palindromes, either expand outward from a center, or use DP to track palindrome boundaries."**

### Longest Palindromic Substring (LeetCode #5) — Expand Around Center

**The Concept:** Start at each character (and each pair), expand outward as long as characters match. Check both odd-length ("aba") and even-length ("abba").

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
# O(n²) time, O(1) space
```

### Palindromic Substrings (LeetCode #647)

**The Concept:** Count ALL palindromic substrings. Same expand-around-center idea, but count instead of tracking the longest.

```python
def countSubstrings(s):
    count = 0
    def expand(l, r):
        nonlocal count
        while l >= 0 and r < len(s) and s[l] == s[r]:
            count += 1
            l -= 1; r += 1
    for i in range(len(s)):
        expand(i, i)       # odd-length
        expand(i, i + 1)   # even-length
    return count
```

---

## Pattern 6: String Manipulation & Building

### Reverse String (LeetCode #344)

```python
def reverseString(s):
    s.reverse()  # in-place
# Or: two pointers
    l, r = 0, len(s) - 1
    while l < r:
        s[l], s[r] = s[r], s[l]
        l += 1; r -= 1
```

### String to Integer — atoi (LeetCode #8)

**The Concept:** Parse character by character, handle signs, whitespace, overflow. Tests your attention to edge cases.

```python
def myAtoi(s):
    s = s.lstrip()
    if not s: return 0
    sign = -1 if s[0] == '-' else 1
    if s[0] in '+-': s = s[1:]
    result = 0
    for c in s:
        if not c.isdigit(): break
        result = result * 10 + int(c)
    result *= sign
    return max(-2**31, min(2**31 - 1, result))  # clamp to 32-bit
```

### ⭐ Minimum Window Substring (LeetCode #76) — Hard

**The Concept:** Sliding window on a string — expand right until all required characters are present, then shrink left to find the minimum.

```python
from collections import Counter, defaultdict
def minWindow(s, t):
    need = Counter(t)
    have = defaultdict(int)
    required, formed = len(need), 0
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

### Longest Common Prefix (LeetCode #14)

```python
def longestCommonPrefix(strs):
    if not strs: return ""
    prefix = strs[0]
    for s in strs[1:]:
        while not s.startswith(prefix):
            prefix = prefix[:-1]
            if not prefix: return ""
    return prefix
```

---

# 🗂️ 0:55 — Hash Maps & Sets (15 min)

## What is Hashing?

Hashing is arguably **the single most important concept** in DSA interviews. A HashMap (dictionary) takes any data and computes a "shelf code" (hash) that maps directly to a memory location.

**Analogy:** A library where every book has a shelf code. Instead of searching sequentially, you go directly to the right shelf.

### Key Operations — All O(1) Average

| Operation | HashMap (dict) | HashSet (set) |
|-----------|----------------|---------------|
| Insert | `d[key] = val` | `s.add(val)` |
| Lookup | `key in d` / `d[key]` | `val in s` |
| Delete | `del d[key]` | `s.remove(val)` |

### When to Use

```
HashMap: Need to store KEY → VALUE pairs     (count occurrences, index lookup)
HashSet: Need to check MEMBERSHIP only       (seen before? duplicate? exists?)
```

### ⭐ The Universal HashMap Pattern

> **"Am I checking `if X exists` inside a loop? → Use a HashMap to make it O(1)."**

```
Without HashMap: For each element, scan the rest  → O(n²)
With HashMap:    For each element, check the map   → O(n)
```

---

## Pattern 7: Frequency Counting & Lookup

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
# O(n) — each number visited at most twice
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
# O(n) — no sorting needed!
```

---

# 🔗 1:10 — Linked Lists (20 min)

## What is a Linked List?

Unlike arrays (contiguous memory), a linked list stores elements **scattered across memory**, connected by pointers — each node knows where the next one is.

```
Array:   [10][20][30][40]     → accessed by index (O(1))
Linked:  10 → 20 → 30 → 40 → None   → accessed by walking (O(n))
```

### Why Use Linked Lists?

| | Array | Linked List |
|---|-------|-------------|
| Access by index | O(1) ✅ | O(n) ❌ |
| Insert/delete at head | O(n) ❌ | O(1) ✅ |
| Insert/delete at known position | O(n) | O(1) |
| Memory | Fixed block | Scattered, flexible |

### The Node

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
```

> 🎬 Visualize: [visualgo.net/list](https://visualgo.net/en/list)

---

## Pattern 8: Slow & Fast Pointers (Floyd's Tortoise and Hare)

### The Core Idea

> **"Two pointers at different speeds: fast reaches the end in half the time → slow is at the middle. On a cycle, fast laps slow → cycle detected."**

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

## Pattern 9: Reverse a Linked List — The 3-Pointer Technique

### The Core Idea

> **"Walk through the list, reversing each arrow to point backwards. Need three pointers: where you came from (prev), where you are (curr), where you're going (nxt)."**

### ⭐ Reverse Linked List (LeetCode #206) — Top 5 Interview Q

```
prev=None  curr=1→2→3→None
Step 1: save nxt=2, point 1→None,  prev=1, curr=2
Step 2: save nxt=3, point 2→1,     prev=2, curr=3
Step 3: save nxt=None, point 3→2,  prev=3, curr=None → done!
Result: 3→2→1→None ✅
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
# O(n) time, O(1) space — MEMORIZE THIS
```

### Merge Two Sorted Lists (LeetCode #21)

**Key technique — Dummy Node:** Creates a fake start node to simplify edge cases.

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

**The Concept:** Combines THREE linked list patterns in one problem:
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

> **💡 Linked List Recipe:** Find middle → Reverse half → Merge/Compare

---

# 📚 1:30 — Stacks & Queues (15 min)

## What Are Stacks and Queues?

Both are **restricted access** data structures — you can only add/remove from specific ends:

### Stack = LIFO (Last In, First Out)

**Analogy:** Stack of plates — add and remove from the **top** only.

**Used for:** Undo operations, matching brackets, DFS, expression evaluation, function call stack.

### Queue = FIFO (First In, First Out)

**Analogy:** Line at a theater — first person in line is served first.

**Used for:** BFS traversal, task scheduling, level-by-level processing.

```python
# Stack — Python list
stack = []
stack.append(x)   # push O(1)
stack.pop()       # pop O(1)
stack[-1]         # peek O(1)

# Queue — ALWAYS use deque (list.pop(0) is O(n)!)
from collections import deque
q = deque()
q.append(x)       # enqueue O(1)
q.popleft()       # dequeue O(1)
```

---

## Pattern 10: Stack for Matching & Nesting

### The Core Idea

> **"When you see an opening element, push it. When you see a closing element, pop and check if they match. Stacks naturally handle nesting."**

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

## Pattern 11: Monotonic Stack — "Next Greater Element"

### The Core Idea

> **"Maintain a stack in decreasing order. When a bigger element arrives, pop — the popped element just found its answer."**

**Why O(n)?** Each element is pushed at most once and popped at most once → O(2n) = O(n) total.

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

### ⭐ Largest Rectangle in Histogram (LeetCode #84)

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

# 🔍 1:45 — Sorting & Binary Search (10 min)

## Why Sorting Matters

You rarely implement sorts, but **sorting as a preprocessing step** unlocks everything:

```
Sorted → Binary Search      O(n log n + log n)
Sorted → Two Pointers       O(n log n + n)
Sorted → Merge Intervals    O(n log n + n)
Sorted → Greedy decisions   O(n log n + n)
Sorted → Duplicates adjacent
```

### Dutch National Flag (LeetCode #75) — Sort 0s, 1s, 2s in One Pass

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

## Pattern 12: Binary Search — Halve the Search Space

### The Core Idea

> **"If you can determine which half contains the answer, throw away the other half. Repeat. O(log n)."**

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

**The Concept:** At any mid, ONE half is always sorted. Check if target is in the sorted half.

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

### ⭐ Binary Search on Answer

> **"Instead of searching an array, search the range of possible answers."**

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

# ⚡ 1:55 — Bit Manipulation (5 min)

## Four Tricks Worth Knowing

### 1. XOR Cancels Pairs — Single Number (LeetCode #136)

`a ^ a = 0` and `a ^ 0 = a`. XOR all numbers → pairs cancel → unique remains.

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

# ✅ Day 1 Summary — 12 Patterns

| # | Pattern | Core Insight | Key Problem |
|---|---------|-------------|-------------|
| 1 | **Two Pointers** | Converge from both ends | 3Sum #15, Trapping Rain #42 |
| 2 | **Sliding Window** | Expand right, shrink left | Longest Substring #3 |
| 3 | **Prefix Sum / Kadane's** | Pre-compute or extend/restart | Max Subarray #53 |
| 4 | **Char Frequency** | Same counts = same structure | Anagrams #242, #49 |
| 5 | **Palindrome Expand** | Expand from center | Longest Palindrome #5 |
| 6 | **String Building** | Use list + join, not += | Various |
| 7 | **HashMap Lookup** | O(1) existence check | Two Sum #1, Consecutive #128 |
| 8 | **Slow/Fast Pointers** | Cycle, middle detection | Cycle #141, Middle #876 |
| 9 | **Reverse LL** | Save → Reverse → Advance | Reverse LL #206 |
| 10 | **Stack Matching** | Push open, pop close | Valid Parentheses #20 |
| 11 | **Monotonic Stack** | Next greater/smaller | Daily Temps #739 |
| 12 | **Binary Search** | Halve the search space | Rotated Array #33, Koko #875 |

### 🏋️ Tonight's Homework (Pick 5-8)
```
🟢 #1    Two Sum              🟡 #3   Longest Substring
🟡 #5    Longest Palindrome   🟡 #15  3Sum
🟡 #49   Group Anagrams       🟡 #53  Max Subarray
🟡 #128  Longest Consecutive  🟢 #206 Reverse Linked List
🟡 #33   Search Rotated       🔴 #42  Trapping Rain Water
```

---

*Tomorrow: Recursion, Trees, Heaps, Tries, Graphs, Union-Find, Greedy, DP → [day2-2hrs.md](day2-2hrs.md)*
