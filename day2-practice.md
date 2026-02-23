# 🌙 Day 2 — Evening Practice (5:00 PM - 7:00 PM)

## 20 Must-Do Problems — Day 2 Topics

---

## 🟢 Warm-Up (10 min each)

| # | Problem | LeetCode | Pattern | Difficulty |
|---|---------|----------|---------|------------|
| 1 | Reverse Linked List | [#206](https://leetcode.com/problems/reverse-linked-list/) | Three Pointers | 🟢 Easy |
| 2 | Valid Parentheses | [#20](https://leetcode.com/problems/valid-parentheses/) | Stack Matching | 🟢 Easy |
| 3 | Maximum Depth of Binary Tree | [#104](https://leetcode.com/problems/maximum-depth-of-binary-tree/) | Tree DFS | 🟢 Easy |
| 4 | Climbing Stairs | [#70](https://leetcode.com/problems/climbing-stairs/) | 1D DP | 🟢 Easy |
| 5 | Number of Islands | [#200](https://leetcode.com/problems/number-of-islands/) | BFS/DFS | 🟡 Medium |

---

## 🟡 Core Practice (15 min each)

| # | Problem | LeetCode | Pattern | Difficulty |
|---|---------|----------|---------|------------|
| 6 | Linked List Cycle | [#141](https://leetcode.com/problems/linked-list-cycle/) | Slow/Fast Pointers | 🟢 Easy |
| 7 | Binary Tree Level Order Traversal | [#102](https://leetcode.com/problems/binary-tree-level-order-traversal/) | Tree BFS | 🟡 Medium |
| 8 | Validate BST | [#98](https://leetcode.com/problems/validate-binary-search-tree/) | BST + Bounds | 🟡 Medium |
| 9 | Lowest Common Ancestor | [#236](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree/) | Tree DFS | 🟡 Medium |
| 10 | Top K Frequent Elements | [#347](https://leetcode.com/problems/top-k-frequent-elements/) | Heap / Bucket Sort | 🟡 Medium |
| 11 | Daily Temperatures | [#739](https://leetcode.com/problems/daily-temperatures/) | Monotonic Stack | 🟡 Medium |
| 12 | Course Schedule | [#207](https://leetcode.com/problems/course-schedule/) | Topological Sort | 🟡 Medium |
| 13 | House Robber | [#198](https://leetcode.com/problems/house-robber/) | 1D DP | 🟡 Medium |
| 14 | Coin Change | [#322](https://leetcode.com/problems/coin-change/) | Unbounded Knapsack DP | 🟡 Medium |
| 15 | Longest Common Subsequence | [#1143](https://leetcode.com/problems/longest-common-subsequence/) | 2D DP | 🟡 Medium |

---

## 🔴 Challenge (20 min each)

| # | Problem | LeetCode | Pattern | Difficulty |
|---|---------|----------|---------|------------|
| 16 | Merge K Sorted Lists | [#23](https://leetcode.com/problems/merge-k-sorted-lists/) | Heap + Linked List | 🔴 Hard |
| 17 | Largest Rectangle in Histogram | [#84](https://leetcode.com/problems/largest-rectangle-in-histogram/) | Monotonic Stack | 🔴 Hard |
| 18 | Binary Tree Maximum Path Sum | [#124](https://leetcode.com/problems/binary-tree-maximum-path-sum/) | Tree DFS | 🔴 Hard |
| 19 | Word Search II | [#212](https://leetcode.com/problems/word-search-ii/) | Trie + DFS | 🔴 Hard |
| 20 | Edit Distance | [#72](https://leetcode.com/problems/edit-distance/) | 2D DP | 🟡 Medium |

---

## 📝 Approach Hints

<details>
<summary><b>Hint 1: Reverse Linked List (#206)</b></summary>

Three pointers: `prev=None, curr=head`. At each step: save `next`, point `curr.next` to `prev`, then move `prev` and `curr` forward.

</details>

<details>
<summary><b>Hint 5: Number of Islands (#200)</b></summary>

For each '1' cell, run BFS/DFS to mark all connected '1's as visited. Count how many times you start a new BFS/DFS.

</details>

<details>
<summary><b>Hint 12: Course Schedule (#207)</b></summary>

Build a directed graph. Detect cycle using DFS with 3 states: unvisited(0), visiting(1), visited(2). Cycle = revisit a node in state 1.

</details>

<details>
<summary><b>Hint 14: Coin Change (#322)</b></summary>

`dp[amount] = min(dp[amount - coin] + 1)` for each coin. Initialize `dp[0] = 0`, all others = infinity.

</details>

<details>
<summary><b>Hint 17: Largest Rectangle (#84)</b></summary>

Maintain a monotonically increasing stack of indices. When a shorter bar appears, pop and calculate area with the popped height.

</details>

---

## ✅ Final Self-Check: Are You Interview-Ready?

### Data Structures
- [ ] Can you implement a Linked List from scratch?
- [ ] Can you explain when to use Stack vs Queue?
- [ ] Can you traverse a Binary Tree in all 4 ways?
- [ ] Can you explain Heap operations (insert, extract)?
- [ ] Can you implement a Trie?

### Algorithms
- [ ] Can you write BFS and DFS from memory?
- [ ] Can you solve any DP problem using the 4-step recipe?
- [ ] Can you detect a cycle in a linked list AND a graph?
- [ ] Can you explain Dijkstra's algorithm?
- [ ] Can you do topological sort?

### Problem-Solving
- [ ] Given a new problem, can you identify which pattern to use?
- [ ] Can you explain your approach before coding?
- [ ] Can you analyze the time and space complexity of your solution?
- [ ] Can you handle edge cases (empty input, single element, large input)?

*If you checked most boxes — you're ready! 🎉*

---

## 🚀 What's Next?

1. **Practice 5-10 problems daily** on LeetCode
2. **Read the [Cheatsheet](cheatsheet.md)** before each practice session
3. **Read the [Interview Playbook](interview-playbook.md)** before your interview
4. **Revisit the harder chapters** from the [full workshop](../README.md)
5. **Join communities:** r/leetcode, NeetCode Discord, Striver's community

---

[← Day 2 Afternoon](day2-afternoon.md) | [Back to Schedule](README.md) | [Next: Cheatsheet →](cheatsheet.md)
