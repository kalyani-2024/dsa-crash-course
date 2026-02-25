# Day 3 and Day 4 -- Practice Problems

## 25 Must-Do Problems -- Sorting, Searching, Recursion, Trees, Graphs, Greedy, and DP

**How to practice:** Spend 15-20 minutes per problem. If stuck for 10 minutes, read the hint. If still stuck, study the solution, then re-solve from scratch.

---

## Warm-Up

| # | Problem | LeetCode | Pattern | Topic |
|---|---------|----------|---------|-------|
| 1 | Climbing Stairs | [#70](https://leetcode.com/problems/climbing-stairs/) | 1D DP | Dynamic Programming |
| 2 | Maximum Depth of Binary Tree | [#104](https://leetcode.com/problems/maximum-depth-of-binary-tree/) | Tree DFS | Trees |
| 3 | Invert Binary Tree | [#226](https://leetcode.com/problems/invert-binary-tree/) | Tree DFS | Trees |
| 4 | Number of Islands | [#200](https://leetcode.com/problems/number-of-islands/) | BFS/DFS | Graphs |
| 5 | Subsets | [#78](https://leetcode.com/problems/subsets/) | Backtracking | Recursion |

---

## Core Practice

### Trees and BST
| # | Problem | LeetCode | Pattern | Topic |
|---|---------|----------|---------|-------|
| 6 | Binary Tree Level Order Traversal | [#102](https://leetcode.com/problems/binary-tree-level-order-traversal/) | Tree BFS | Trees |
| 7 | Validate BST | [#98](https://leetcode.com/problems/validate-binary-search-tree/) | BST Bounds Check | Trees |
| 8 | Lowest Common Ancestor | [#236](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree/) | Recursive Tree | Trees |
| 9 | Diameter of Binary Tree | [#543](https://leetcode.com/problems/diameter-of-binary-tree/) | Tree DFS | Trees |

### Heaps
| # | Problem | LeetCode | Pattern | Topic |
|---|---------|----------|---------|-------|
| 10 | Kth Largest Element | [#215](https://leetcode.com/problems/kth-largest-element-in-an-array/) | Heap / Top-K | Heaps |
| 11 | Top K Frequent Elements | [#347](https://leetcode.com/problems/top-k-frequent-elements/) | Heap + HashMap | Heaps |

### Graphs
| # | Problem | LeetCode | Pattern | Topic |
|---|---------|----------|---------|-------|
| 12 | Course Schedule | [#207](https://leetcode.com/problems/course-schedule/) | DFS Cycle Detection | Graphs |
| 13 | Rotting Oranges | [#994](https://leetcode.com/problems/rotting-oranges/) | Multi-source BFS | Graphs |
| 14 | Clone Graph | [#133](https://leetcode.com/problems/clone-graph/) | BFS/DFS + HashMap | Graphs |

### Greedy and Intervals
| # | Problem | LeetCode | Pattern | Topic |
|---|---------|----------|---------|-------|
| 15 | Jump Game | [#55](https://leetcode.com/problems/jump-game/) | Greedy (farthest reach) | Greedy |
| 16 | Non-overlapping Intervals | [#435](https://leetcode.com/problems/non-overlapping-intervals/) | Sort + Greedy | Greedy |

### Dynamic Programming
| # | Problem | LeetCode | Pattern | Topic |
|---|---------|----------|---------|-------|
| 17 | House Robber | [#198](https://leetcode.com/problems/house-robber/) | 1D DP | DP |
| 18 | Coin Change | [#322](https://leetcode.com/problems/coin-change/) | Knapsack DP | DP |
| 19 | Longest Common Subsequence | [#1143](https://leetcode.com/problems/longest-common-subsequence/) | 2D DP | DP |
| 20 | Longest Increasing Subsequence | [#300](https://leetcode.com/problems/longest-increasing-subsequence/) | 1D DP | DP |

---

## Challenge

| # | Problem | LeetCode | Pattern | Topic |
|---|---------|----------|---------|-------|
| 21 | N-Queens | [#51](https://leetcode.com/problems/n-queens/) | Constraint Backtracking | Recursion |
| 22 | Merge K Sorted Lists | [#23](https://leetcode.com/problems/merge-k-sorted-lists/) | Heap + Linked List | Heaps |
| 23 | Binary Tree Maximum Path Sum | [#124](https://leetcode.com/problems/binary-tree-maximum-path-sum/) | Tree DFS | Trees |
| 24 | Word Search II | [#212](https://leetcode.com/problems/word-search-ii/) | Trie + DFS | Tries |
| 25 | Edit Distance | [#72](https://leetcode.com/problems/edit-distance/) | 2D DP | DP |

---

## Approach Hints

<details>
<summary><b>Hint 5: Subsets (#78)</b></summary>

For each element, make two recursive calls: one including it, one excluding it. This creates a binary decision tree with 2^n leaves.

</details>

<details>
<summary><b>Hint 8: LCA (#236)</b></summary>

Recursively search left and right. If both return non-null, the current node is the LCA. If only one returns non-null, pass it up.

</details>

<details>
<summary><b>Hint 12: Course Schedule (#207)</b></summary>

Build a directed graph. Detect cycle using DFS with 3 states: unvisited(0), visiting(1), visited(2). Cycle = revisiting a node in state 1.

</details>

<details>
<summary><b>Hint 15: Jump Game (#55)</b></summary>

Track the farthest position you can reach. At each index, update `farthest = max(farthest, i + nums[i])`. If you ever cannot reach the current index, return False.

</details>

<details>
<summary><b>Hint 18: Coin Change (#322)</b></summary>

`dp[amount] = min(dp[amount - coin] + 1)` for each coin. Initialize `dp[0] = 0`, all others = infinity.

</details>

<details>
<summary><b>Hint 22: Merge K Lists (#23)</b></summary>

Put the head of each list into a min-heap. Pop the smallest, push its `.next`. The heap always gives you the globally smallest node.

</details>

<details>
<summary><b>Hint 24: Word Search II (#212)</b></summary>

Build a Trie from all words. DFS through the grid, following Trie branches. When no branch exists for a character, prune that path.

</details>

---

## Final Self-Check: Are You Interview-Ready?

### Data Structures
- [ ] Can you implement a Trie from scratch?
- [ ] Can you explain when to use a Heap vs sorting?
- [ ] Can you traverse a Binary Tree in all 4 ways?
- [ ] Can you implement Union-Find with path compression?

### Algorithms
- [ ] Can you write BFS and DFS from memory?
- [ ] Can you solve any DP problem using the 4-step recipe?
- [ ] Can you detect cycles in both linked lists and directed graphs?
- [ ] Can you explain when Greedy works and when you need DP instead?
- [ ] Can you implement topological sort?

### Problem-Solving
- [ ] Given a new problem, can you identify which pattern to use?
- [ ] Can you explain your approach before coding?
- [ ] Can you analyze the time and space complexity of your solution?
- [ ] Can you handle edge cases (empty input, single element, large input)?

If you checked most boxes, you are ready.

---

## What's Next?

1. **Practice 5-10 problems daily** on LeetCode
2. **Read the [Cheatsheet](cheatsheet.md)** before each practice session
3. **Read the [Interview Playbook](interview-playbook.md)** before your interview
4. **Use [NeetCode Roadmap](https://neetcode.io/roadmap)** for structured progression
5. **Join communities:** r/leetcode, NeetCode Discord, Striver's community

---

[Back to Day 3](day3.md) | [Back to Day 4](day4.md) | [Back to Course](README.md) | [Cheatsheet](cheatsheet.md)
