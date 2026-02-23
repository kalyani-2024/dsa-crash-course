# 🎤 Interview Playbook — How to Crack Any Coding Interview

> **This is your secret weapon.** Read this the night before your interview.

---

## 🕐 The 45-Minute Interview Timeline

```
 0:00 ─ 3:00   │  Introductions & small talk
 3:00 ─ 8:00   │  Read problem, ask clarifying questions
 8:00 ─ 12:00  │  Discuss approach (out loud!)
12:00 ─ 30:00  │  Code the solution
30:00 ─ 37:00  │  Test with examples, fix bugs
37:00 ─ 42:00  │  Optimize, discuss trade-offs
42:00 ─ 45:00  │  Ask YOUR questions to the interviewer
```

---

## 📋 Step 1: Clarify the Problem (5 min)

**Never start coding immediately.** Ask these questions first:

### Must-Ask Questions
```
1. "Can you confirm the input format?"
   - Array of integers? Strings? Linked list nodes?

2. "What are the constraints?"
   - Input size (n)? Value range? 
   - This tells you the acceptable time complexity!

3. "Are there edge cases I should handle?"
   - Empty input? Single element? All same values?
   - Negative numbers? Zeros?

4. "Is the input sorted?"
   - Opens up binary search and two pointers

5. "Can I modify the input?"
   - In-place vs creating new data structures

6. "Are there duplicates?"
   - Affects set usage and skip logic
```

### Constraint → Complexity Mapping
```
n ≤ 10      → O(n!) OK         → Brute force / backtracking
n ≤ 20      → O(2ⁿ) OK        → Bitmask / backtracking
n ≤ 500     → O(n³) OK         → 3 nested loops
n ≤ 5,000   → O(n²) OK         → 2 nested loops
n ≤ 100,000 → O(n log n) needed → Sort / Binary Search / Heap
n ≤ 10⁷     → O(n) needed      → Single pass / HashMap
n > 10⁷     → O(log n) needed  → Math / Binary Search
```

---

## 💬 Step 2: Think Out Loud (5 min)

**The interviewer wants to see HOW you think, not just the answer.**

### The UMPIRE Method
```
U — Understand the problem
M — Match to a known pattern
P — Plan your approach
I — Implement the code
R — Review and test
E — Evaluate complexity
```

### What to Say

```
"Let me think about this... The brute force approach would be [X],
which is O(n²). But since n can be up to 10⁵, I need O(n log n) or better.

This looks like a [sliding window / two pointer / DP] problem because [reason].

My approach is:
1. [First step]
2. [Second step]
3. [Third step]

Does this approach make sense before I start coding?"
```

> **🎯 Pro Tip:** ALWAYS explain the brute force first, then optimize. This shows you can solve it AND think critically.

---

## 💻 Step 3: Code Clean (18 min)

### Coding Best Practices

```python
# ✅ GOOD: Clean, readable code
def two_sum(nums, target):
    seen = {}  # value → index
    
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    
    return []

# ❌ BAD: Messy, no comments, bad names
def f(a, t):
    d = {}
    for i in range(len(a)):
        if t - a[i] in d:
            return [d[t-a[i]], i]
        d[a[i]] = i
```

### Rules for Interview Code
```
1. Use MEANINGFUL variable names (not i, j, k everywhere)
2. Write COMMENTS for non-obvious logic
3. Handle EDGE CASES early (return early for empty/trivial)
4. Use HELPER FUNCTIONS to keep main logic clean
5. DON'T optimize prematurely — get it working first
```

---

## 🧪 Step 4: Test Your Code (7 min)

### How to Test

```
1. Trace through the given example BY HAND
   - Write variable values at each step
   - "nums = [2,7,11,15], target = 9"
   - "i=0: num=2, complement=7, not in seen, add {2:0}"
   - "i=1: num=7, complement=2, IS in seen! return [0,1] ✅"

2. Try edge cases:
   - Empty input: []
   - Single element: [5]
   - All same: [3, 3, 3]
   - Negative: [-1, 0, 1]
   - Already sorted / reverse sorted

3. Try large input MENTALLY:
   - "For n=10⁵, my O(n) solution should work fine"
```

---

## ⚡ Step 5: Optimize & Discuss (5 min)

### What Interviewers Love to Hear

```
"The current solution is O(n) time and O(n) space using a HashMap.
We could reduce space to O(1) by sorting first and using two pointers,
but that would make it O(n log n) time. The HashMap approach is better
for the general case since n can be large."
```

### Common Follow-up Questions & Answers

```
Q: "Can you do it without extra space?"
A: Sort in-place → two pointers, or use bit manipulation

Q: "What if the array is sorted?"
A: Two pointers → O(n) time, O(1) space

Q: "What if there are multiple valid answers?"
A: Return all pairs, or the first one found

Q: "What if the input is too large to fit in memory?"
A: External sort, divide and conquer, streaming

Q: "Can you do it in one pass?"
A: HashMap approach is already one pass
```

---

## 🎯 Pattern Recognition Cheat Sheet for Interviews

### When You See... → Use This Pattern

```
"Find a pair/triplet with sum X"          → Two Pointers / HashMap
"Find longest/shortest subarray"          → Sliding Window
"Find kth largest/smallest"               → Heap or QuickSelect
"Find in sorted array"                    → Binary Search
"Count elements / frequency"              → HashMap
"String matching / prefix"                → Trie
"Valid brackets / nested structures"       → Stack
"Level-by-level / shortest path"          → BFS
"All paths / connectivity / cycles"       → DFS
"Min/max with choices at each step"       → Dynamic Programming
"All combinations / subsets"              → Backtracking
"Minimum spanning tree / components"      → Union-Find
"Next greater/smaller element"            → Monotonic Stack
"Scheduling / intervals"                  → Sort + Greedy / Heap
```

---

## 🚫 Common Mistakes to Avoid

### Coding Mistakes
```
❌  Off-by-one errors in loops and binary search
     → Always check: should it be < or <=? lo+1 or lo?

❌  Modifying a list while iterating
     → Use a copy or iterate in reverse

❌  Integer overflow (in Java/C++)
     → Use long, or check before operations

❌  Not handling null/None
     → Check root, head, etc. at the start

❌  Forgetting to mark visited in graphs
     → Infinite loops in BFS/DFS
```

### Behavioral Mistakes
```
❌  Starting to code immediately without thinking
     → Always discuss approach first

❌  Going silent for long periods
     → Talk through your thought process

❌  Giving up when stuck
     → Start with brute force, then optimize

❌  Not asking clarifying questions
     → Shows you don't think critically

❌  Arguing with the interviewer
     → They're trying to help you. Listen.
```

---

## 🏆 The "I'm Stuck" Emergency Plan

```
Step 1: Re-read the problem. You might have missed something.

Step 2: Try the BRUTE FORCE. Even O(n³) is better than nothing.

Step 3: Think about what DATA STRUCTURE could help:
        → Need O(1) lookup? HashMap
        → Need sorted data? TreeMap / Sort
        → Need min/max? Heap
        → Need LIFO? Stack

Step 4: Try a SIMPLER VERSION of the problem:
        → What if the array had only 2 elements?
        → What if the string had no special characters?
        → What if the tree was a linked list?

Step 5: Think about RELATED PROBLEMS you've solved before.

Step 6: Ask the interviewer for a HINT. This is OK!
        "Could you give me a hint about the data structure?"
```

---

## 🌟 Top Interview Questions by Company (2024-2025)

### FAANG Favorites
```
🍎 Apple:        Two Sum, LRU Cache, Merge Intervals
📘 Meta:         Valid Parentheses, Binary Tree Paths, Word Break
🔍 Google:       Median Two Arrays, Course Schedule, Word Ladder
🛒 Amazon:       Number of Islands, LRU Cache, Best Time Stock
🤖 Microsoft:    Two Sum, Reverse LL, Spiral Matrix
```

### College Placement Favorites (India)
```
TCS/Infosys:     Two Sum, Palindrome, Sorting algorithms
Wipro/HCL:       Arrays, Strings, Basic recursion
Flipkart:        DP problems, Graph BFS/DFS
PhonePe/Razorpay: System Design + Medium LeetCode
Google/Microsoft: Hard LeetCode + System Design
```

---

## 📚 Resources for Continued Learning

### Free Resources
```
📺 NeetCode (YouTube + neetcode.io) — Best LeetCode explanations
📺 Striver (takeUforward) — A2Z DSA Sheet (this course is based on it)
📺 Abdul Bari (YouTube) — Algorithm theory explanations
📖 LeetCode Explore Cards — Structured learning paths
📖 GeeksForGeeks — Problem discussions and editorials
```

### Practice Platforms
```
💻 LeetCode.com — The gold standard
💻 HackerRank — Good for college contests
💻 Codeforces — Competitive programming
💻 InterviewBit — Interview-focused
💻 CodeStudio (Coding Ninjas) — Indian placement prep
```

### Books (Optional)
```
📕 "Cracking the Coding Interview" by Gayle McDowell
📕 "Introduction to Algorithms" (CLRS) — The bible of algorithms
📕 "Grokking Algorithms" — Visual, beginner-friendly
```

---

## 🎲 Mock Interview Checklist

Before your real interview, do at least 3 mock interviews:

- [ ] Practice explaining your approach OUT LOUD
- [ ] Practice writing code on a whiteboard or Google Doc (no autocomplete!)
- [ ] Practice under TIME PRESSURE (set a 25-minute timer)
- [ ] Practice TESTING your code manually
- [ ] Practice handling FOLLOW-UP questions
- [ ] Practice staying CALM when stuck

### Mock Interview Partners
```
🤝 Pramp.com — Free mock interviews with peers
🤝 Interviewing.io — Practice with engineers (some free)
🤝 LeetCode mock interview mode
🤝 Practice with a friend (switch roles)
```

---

## 💪 Final Words

```
┌────────────────────────────────────────────────────┐
│                                                    │
│   "Everyone struggles with DSA at first.           │
│    The difference between those who succeed         │
│    and those who don't is CONSISTENCY.              │
│                                                    │
│    Solve 2-3 problems daily for 3 months,          │
│    and you WILL crack any coding interview."        │
│                                                    │
│                          — Every successful SDE     │
│                                                    │
└────────────────────────────────────────────────────┘
```

> **You've completed the crash course. Now go practice!** 🚀

---

[← Cheatsheet](cheatsheet.md) | [Back to Schedule](README.md)
