# 🌙 Day 1 — Evening Practice (5:00 PM - 7:00 PM)

## 20 Must-Do Problems — Day 1 Topics

> **How to practice:** Spend 15-20 min per problem. If stuck for 10 min, read the approach hint. If still stuck, study the solution, then re-solve from scratch.

---

## 🟢 Warm-Up (10 min each)

| # | Problem | LeetCode | Pattern | Difficulty |
|---|---------|----------|---------|------------|
| 1 | Two Sum | [#1](https://leetcode.com/problems/two-sum/) | HashMap | 🟢 Easy |
| 2 | Valid Palindrome | [#125](https://leetcode.com/problems/valid-palindrome/) | Two Pointers | 🟢 Easy |
| 3 | Best Time to Buy/Sell Stock | [#121](https://leetcode.com/problems/best-time-to-buy-and-sell-stock/) | Linear Scan | 🟢 Easy |
| 4 | Single Number | [#136](https://leetcode.com/problems/single-number/) | Bit XOR | 🟢 Easy |
| 5 | Binary Search | [#704](https://leetcode.com/problems/binary-search/) | Binary Search | 🟢 Easy |

---

## 🟡 Core Practice (15 min each)

| # | Problem | LeetCode | Pattern | Difficulty |
|---|---------|----------|---------|------------|
| 6 | Maximum Subarray (Kadane's) | [#53](https://leetcode.com/problems/maximum-subarray/) | Linear Scan | 🟡 Medium |
| 7 | 3Sum | [#15](https://leetcode.com/problems/3sum/) | Sort + Two Pointers | 🟡 Medium |
| 8 | Container With Most Water | [#11](https://leetcode.com/problems/container-with-most-water/) | Two Pointers | 🟡 Medium |
| 9 | Group Anagrams | [#49](https://leetcode.com/problems/group-anagrams/) | HashMap | 🟡 Medium |
| 10 | Longest Consecutive Sequence | [#128](https://leetcode.com/problems/longest-consecutive-sequence/) | HashSet | 🟡 Medium |
| 11 | Longest Substring Without Repeating Characters | [#3](https://leetcode.com/problems/longest-substring-without-repeating-characters/) | Sliding Window | 🟡 Medium |
| 12 | Search in Rotated Sorted Array | [#33](https://leetcode.com/problems/search-in-rotated-sorted-array/) | Binary Search | 🟡 Medium |
| 13 | Subarray Sum Equals K | [#560](https://leetcode.com/problems/subarray-sum-equals-k/) | Prefix Sum + HashMap | 🟡 Medium |
| 14 | Next Permutation | [#31](https://leetcode.com/problems/next-permutation/) | Array Algorithm | 🟡 Medium |
| 15 | Longest Palindromic Substring | [#5](https://leetcode.com/problems/longest-palindromic-substring/) | Expand Around Center | 🟡 Medium |

---

## 🔴 Challenge (20 min each)

| # | Problem | LeetCode | Pattern | Difficulty |
|---|---------|----------|---------|------------|
| 16 | Trapping Rain Water | [#42](https://leetcode.com/problems/trapping-rain-water/) | Two Pointers | 🔴 Hard |
| 17 | N-Queens | [#51](https://leetcode.com/problems/n-queens/) | Backtracking | 🔴 Hard |
| 18 | Median of Two Sorted Arrays | [#4](https://leetcode.com/problems/median-of-two-sorted-arrays/) | Binary Search | 🔴 Hard |
| 19 | Minimum Window Substring | [#76](https://leetcode.com/problems/minimum-window-substring/) | Sliding Window | 🔴 Hard |
| 20 | Merge Intervals | [#56](https://leetcode.com/problems/merge-intervals/) | Sort + Sweep | 🟡 Medium |

---

## 📝 Approach Hints (Read only if stuck!)

<details>
<summary><b>Hint 1: Two Sum (#1)</b></summary>

For each number, check if `target - number` exists in your hashmap. Store `{number: index}` as you go.

</details>

<details>
<summary><b>Hint 6: Maximum Subarray (#53)</b></summary>

Kadane's: `current_sum = max(num, current_sum + num)`. If the running sum goes negative, it's better to start fresh.

</details>

<details>
<summary><b>Hint 7: 3Sum (#15)</b></summary>

Sort the array first. Fix one number, then use two pointers for the remaining pair. Skip duplicate values.

</details>

<details>
<summary><b>Hint 11: Longest Substring (#3)</b></summary>

Sliding window with a set. When you see a duplicate, shrink from the left until the duplicate is removed.

</details>

<details>
<summary><b>Hint 16: Trapping Rain Water (#42)</b></summary>

Two pointers from both ends. Track `left_max` and `right_max`. Water at each position = shorter max - current height.

</details>

<details>
<summary><b>Hint 17: N-Queens (#51)</b></summary>

Place queens row by row. Use sets for columns, diagonals (row-col), and anti-diagonals (row+col).

</details>

---

## ✅ Self-Check: Can You...

- [ ] Explain Big-O of any solution you write?
- [ ] Solve Two Sum in under 5 minutes?
- [ ] Write Kadane's algorithm from memory?
- [ ] Implement binary search without bugs?
- [ ] Write the sliding window template from scratch?
- [ ] Generate all subsets using recursion?
- [ ] Explain when to use HashMap vs Two Pointers vs Sliding Window?

*If you checked all boxes, you're ready for Day 2! Get some rest. 😴*

---

[← Day 1 Afternoon](day1-afternoon.md) | [Back to Schedule](README.md) | [Next: Day 2 Morning →](day2-morning.md)
