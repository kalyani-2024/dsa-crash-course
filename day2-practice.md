# Day 2 Practice -- HashMaps, Linked Lists, Stacks, Queues, Sorting, and Binary Search

## 12 Must-Do Problems

**How to practice:** Spend 15-20 minutes per problem. If stuck for 10 minutes, read the hint. If still stuck, study the solution, then re-solve from scratch.

---

## Warm-Up

| # | Problem | LeetCode | Pattern | Day 2 Topic |
|---|---------|----------|---------|-------------|
| 1 | Contains Duplicate | [#217](https://leetcode.com/problems/contains-duplicate/) | HashSet | HashMap / Set |
| 2 | Reverse Linked List | [#206](https://leetcode.com/problems/reverse-linked-list/) | 3-Pointer Swap | Linked Lists |
| 3 | Valid Parentheses | [#20](https://leetcode.com/problems/valid-parentheses/) | Stack Matching | Stacks |
| 4 | Binary Search | [#704](https://leetcode.com/problems/binary-search/) | Halve Search Space | Binary Search |

---

## Core Practice

| # | Problem | LeetCode | Pattern | Day 2 Topic |
|---|---------|----------|---------|-------------|
| 5 | Longest Consecutive Sequence | [#128](https://leetcode.com/problems/longest-consecutive-sequence/) | HashSet Lookup | HashMap / Set |
| 6 | Linked List Cycle | [#141](https://leetcode.com/problems/linked-list-cycle/) | Slow/Fast Pointers | Linked Lists |
| 7 | Merge Two Sorted Lists | [#21](https://leetcode.com/problems/merge-two-sorted-lists/) | Dummy Node | Linked Lists |
| 8 | Daily Temperatures | [#739](https://leetcode.com/problems/daily-temperatures/) | Monotonic Stack | Stacks |
| 9 | Search in Rotated Sorted Array | [#33](https://leetcode.com/problems/search-in-rotated-sorted-array/) | Modified Binary Search | Binary Search |
| 10 | Merge Intervals | [#56](https://leetcode.com/problems/merge-intervals/) | Sort + Sweep | Sorting |

---

## Challenge

| # | Problem | LeetCode | Pattern | Day 2 Topic |
|---|---------|----------|---------|-------------|
| 11 | Largest Rectangle in Histogram | [#84](https://leetcode.com/problems/largest-rectangle-in-histogram/) | Monotonic Stack | Stacks |
| 12 | Reorder List | [#143](https://leetcode.com/problems/reorder-list/) | Find Middle + Reverse + Merge | Linked Lists |

---

## Approach Hints

<details>
<summary><b>Hint 5: Longest Consecutive (#128)</b></summary>

Put all numbers in a set. For each number, only start counting if `n-1` is NOT in the set (this is the beginning of a sequence). Then count forward.

</details>

<details>
<summary><b>Hint 6: Linked List Cycle (#141)</b></summary>

Use slow and fast pointers. Slow moves 1 step, fast moves 2 steps. If they ever meet, there's a cycle. If fast reaches null, there's no cycle.

</details>

<details>
<summary><b>Hint 8: Daily Temperatures (#739)</b></summary>

Monotonic decreasing stack of indices. When a warmer temperature arrives, pop -- the popped index just found how many days until a warmer day.

</details>

<details>
<summary><b>Hint 9: Search Rotated (#33)</b></summary>

At any mid, one half is always sorted. Check if the target is in the sorted half. If yes, search there. If no, search the other half.

</details>

<details>
<summary><b>Hint 11: Largest Rectangle (#84)</b></summary>

Monotonic increasing stack of indices. When a shorter bar arrives, pop and calculate the area using that bar's height and the width between the new top and current index.

</details>

<details>
<summary><b>Hint 12: Reorder List (#143)</b></summary>

Three steps: (1) Find middle with slow/fast pointers. (2) Reverse the second half. (3) Interleave the two halves.

</details>

---

## Self-Check: Day 2

- [ ] Can you explain when to use HashMap vs HashSet?
- [ ] Can you reverse a linked list with the 3-pointer technique?
- [ ] Can you explain why you should use `deque` instead of `list.pop(0)` for queues?
- [ ] Can you implement binary search without off-by-one bugs?
- [ ] Can you explain what a monotonic stack does and why it's O(n)?
- [ ] Can you explain sorting as a preprocessing step?

If you checked most boxes, move on to [Day 3](day3.md).

---

[Back to Day 2](day2-2hrs.md) | [Back to Course](README.md)
