# Day 1 Practice -- Arrays and Strings

## 10 Must-Do Problems

**How to practice:** Spend 15-20 minutes per problem. If stuck for 10 minutes, read the hint. If still stuck, study the solution, then re-solve from scratch.

---

## Warm-Up

| # | Problem | LeetCode | Pattern | Day 1 Topic |
|---|---------|----------|---------|-------------|
| 1 | Two Sum | [#1](https://leetcode.com/problems/two-sum/) | HashMap Lookup | Arrays |
| 2 | Valid Palindrome | [#125](https://leetcode.com/problems/valid-palindrome/) | Two Pointers | Strings |
| 3 | Best Time to Buy and Sell Stock | [#121](https://leetcode.com/problems/best-time-to-buy-and-sell-stock/) | Kadane's Variant | Arrays |

---

## Core Practice

| # | Problem | LeetCode | Pattern | Day 1 Topic |
|---|---------|----------|---------|-------------|
| 4 | Maximum Subarray | [#53](https://leetcode.com/problems/maximum-subarray/) | Kadane's Algorithm | Arrays |
| 5 | Longest Substring Without Repeating | [#3](https://leetcode.com/problems/longest-substring-without-repeating-characters/) | Sliding Window | Strings |
| 6 | Container With Most Water | [#11](https://leetcode.com/problems/container-with-most-water/) | Two Pointers | Arrays |
| 7 | 3Sum | [#15](https://leetcode.com/problems/3sum/) | Sort + Two Pointers | Arrays |
| 8 | Longest Palindromic Substring | [#5](https://leetcode.com/problems/longest-palindromic-substring/) | Expand Around Center | Strings |

---

## Challenge

| # | Problem | LeetCode | Pattern | Day 1 Topic |
|---|---------|----------|---------|-------------|
| 9 | Trapping Rain Water | [#42](https://leetcode.com/problems/trapping-rain-water/) | Two Pointers | Arrays |
| 10 | Minimum Window Substring | [#76](https://leetcode.com/problems/minimum-window-substring/) | Sliding Window | Strings |

---

## Approach Hints

<details>
<summary><b>Hint 1: Two Sum (#1)</b></summary>

For each number, check if `target - number` exists in your hashmap. Store `{number: index}` as you go. One pass, O(n).

</details>

<details>
<summary><b>Hint 5: Longest Substring (#3)</b></summary>

Sliding window with a set. Expand right. When you see a duplicate, shrink from the left until the duplicate is removed.

</details>

<details>
<summary><b>Hint 7: 3Sum (#15)</b></summary>

Sort the array first. Fix one number, then use two pointers for the remaining pair. Skip duplicate values to avoid duplicate triplets.

</details>

<details>
<summary><b>Hint 8: Longest Palindrome (#5)</b></summary>

For each index, expand outward while characters match. Try both odd-length (single center) and even-length (pair center).

</details>

<details>
<summary><b>Hint 9: Trapping Rain Water (#42)</b></summary>

Two pointers from both ends. Track `left_max` and `right_max`. Water at each position = shorter max - current height. Always process the shorter side.

</details>

<details>
<summary><b>Hint 10: Min Window Substring (#76)</b></summary>

Sliding window: expand right until all characters of `t` are present, then shrink left to find the minimum valid window. Track counts with a HashMap.

</details>

---

## Self-Check: Day 1

- [ ] Can you explain Big-O notation and compare O(n) vs O(n^2)?
- [ ] Can you write the Two Pointers template for sorted arrays?
- [ ] Can you write the Sliding Window template from memory?
- [ ] Can you explain when to use Kadane's algorithm vs Prefix Sum?
- [ ] Can you expand around center to find palindromes?

If you checked most boxes, move on to [Day 2](day2-2hrs.md).

---

[Back to Day 1](day1-2hrs.md) | [Back to Course](README.md)
