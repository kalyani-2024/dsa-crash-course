# 🌙 Day 1 — Practice Problems

## 20 Must-Do Problems — Fundamental Data Structures

> **How to practice:** Spend 15-20 min per problem. If stuck for 10 min, read the hint. If still stuck, study the solution, then re-solve from scratch.

---

## 🟢 Warm-Up — 10 min each

| # | Problem | LeetCode | Pattern | Structure |
|---|---------|----------|---------|-----------|
| 1 | Two Sum | [#1](https://leetcode.com/problems/two-sum/) | HashMap Lookup | Arrays + HashMap |
| 2 | Valid Palindrome | [#125](https://leetcode.com/problems/valid-palindrome/) | Two Pointers | Strings |
| 3 | Reverse Linked List | [#206](https://leetcode.com/problems/reverse-linked-list/) | 3-Pointer Swap | Linked Lists |
| 4 | Valid Parentheses | [#20](https://leetcode.com/problems/valid-parentheses/) | Stack Matching | Stacks |
| 5 | Binary Search | [#704](https://leetcode.com/problems/binary-search/) | Halve Search Space | Arrays |

---

## 🟡 Core Practice — 15 min each

| # | Problem | LeetCode | Pattern | Structure |
|---|---------|----------|---------|-----------|
| 6 | Maximum Subarray | [#53](https://leetcode.com/problems/maximum-subarray/) | Kadane's Algorithm | Arrays |
| 7 | Longest Substring Without Repeating | [#3](https://leetcode.com/problems/longest-substring-without-repeating-characters/) | Sliding Window | Strings + HashSet |
| 8 | Group Anagrams | [#49](https://leetcode.com/problems/group-anagrams/) | Char Frequency | Strings + HashMap |
| 9 | Container With Most Water | [#11](https://leetcode.com/problems/container-with-most-water/) | Two Pointers | Arrays |
| 10 | Longest Consecutive Sequence | [#128](https://leetcode.com/problems/longest-consecutive-sequence/) | HashSet Lookup | HashMap / Set |
| 11 | 3Sum | [#15](https://leetcode.com/problems/3sum/) | Sort + Two Pointers | Arrays |
| 12 | Linked List Cycle | [#141](https://leetcode.com/problems/linked-list-cycle/) | Slow/Fast Pointers | Linked Lists |
| 13 | Daily Temperatures | [#739](https://leetcode.com/problems/daily-temperatures/) | Monotonic Stack | Stacks |
| 14 | Longest Palindromic Substring | [#5](https://leetcode.com/problems/longest-palindromic-substring/) | Expand Around Center | Strings |
| 15 | Search Rotated Sorted Array | [#33](https://leetcode.com/problems/search-in-rotated-sorted-array/) | Modified Binary Search | Arrays |

---

## 🔴 Challenge — 20 min each

| # | Problem | LeetCode | Pattern | Structure |
|---|---------|----------|---------|-----------|
| 16 | Trapping Rain Water | [#42](https://leetcode.com/problems/trapping-rain-water/) | Two Pointers | Arrays |
| 17 | Minimum Window Substring | [#76](https://leetcode.com/problems/minimum-window-substring/) | Sliding Window | Strings + HashMap |
| 18 | Largest Rectangle in Histogram | [#84](https://leetcode.com/problems/largest-rectangle-in-histogram/) | Monotonic Stack | Stacks |
| 19 | Merge Intervals | [#56](https://leetcode.com/problems/merge-intervals/) | Sort + Sweep | Arrays + Sorting |
| 20 | Subarray Sum Equals K | [#560](https://leetcode.com/problems/subarray-sum-equals-k/) | Prefix Sum + HashMap | Arrays + HashMap |

---

## 📝 Approach Hints

<details>
<summary><b>Hint 1: Two Sum (#1)</b></summary>

For each number, check if `target - number` exists in your hashmap. Store `{number: index}` as you go.

</details>

<details>
<summary><b>Hint 7: Longest Substring (#3)</b></summary>

Sliding window with a set. When you see a duplicate, shrink from the left until the duplicate is removed.

</details>

<details>
<summary><b>Hint 8: Group Anagrams (#49)</b></summary>

Sort the letters of each word — all anagrams produce the same sorted key. Group by that key in a HashMap.

</details>

<details>
<summary><b>Hint 11: 3Sum (#15)</b></summary>

Sort the array first. Fix one number, then use two pointers for the remaining pair. Skip duplicate values.

</details>

<details>
<summary><b>Hint 14: Longest Palindrome (#5)</b></summary>

For each index, expand outward while characters match. Try both odd-length (from single char) and even-length (from pair).

</details>

<details>
<summary><b>Hint 16: Trapping Rain Water (#42)</b></summary>

Two pointers from both ends. Track `left_max` and `right_max`. Water at each position = shorter max - current height. Process the shorter side.

</details>

<details>
<summary><b>Hint 17: Min Window Substring (#76)</b></summary>

Sliding window: expand right until all characters of `t` are present, then shrink left to find the minimum. Track character counts with a HashMap.

</details>

---

## ✅ Self-Check: Day 1

### Data Structures
- [ ] Can you explain the difference between Array, Linked List, Stack, and Queue?
- [ ] Can you explain when to use HashMap vs HashSet?
- [ ] Can you explain why strings are immutable and how to build strings efficiently?

### Patterns
- [ ] Can you write the Sliding Window template from memory?
- [ ] Can you implement binary search without off-by-one bugs?
- [ ] Can you reverse a linked list in your sleep?
- [ ] Can you explain two pointers on sorted vs unsorted data?
- [ ] Can you explain when to use a monotonic stack?

*If you checked most boxes, you're ready for Day 2! 🎉*

---

[← Day 1 Course](day1-2hrs.md) | [Back to Schedule](README.md) | [Next: Day 2 →](day2-2hrs.md)
