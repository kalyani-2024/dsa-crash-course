# Day 4 Practice -- Tries, Graphs, Greedy, and Dynamic Programming

## 12 Must-Do Problems

**How to practice:** Spend 15-20 minutes per problem. If stuck for 10 minutes, read the hint. If still stuck, study the solution, then re-solve from scratch.

---

## Warm-Up

| # | Problem | LeetCode | Pattern | Day 4 Topic |
|---|---------|----------|---------|-------------|
| 1 | Climbing Stairs | [#70](https://leetcode.com/problems/climbing-stairs/) | 1D DP | Dynamic Programming |
| 2 | Number of Islands | [#200](https://leetcode.com/problems/number-of-islands/) | BFS/DFS | Graphs |
| 3 | Jump Game | [#55](https://leetcode.com/problems/jump-game/) | Greedy | Greedy |

---

## Core Practice

| # | Problem | LeetCode | Pattern | Day 4 Topic |
|---|---------|----------|---------|-------------|
| 4 | Course Schedule | [#207](https://leetcode.com/problems/course-schedule/) | DFS Cycle Detection | Graphs |
| 5 | Rotting Oranges | [#994](https://leetcode.com/problems/rotting-oranges/) | Multi-source BFS | Graphs |
| 6 | House Robber | [#198](https://leetcode.com/problems/house-robber/) | 1D DP | Dynamic Programming |
| 7 | Coin Change | [#322](https://leetcode.com/problems/coin-change/) | Knapsack DP | Dynamic Programming |
| 8 | Longest Increasing Subsequence | [#300](https://leetcode.com/problems/longest-increasing-subsequence/) | 1D DP | Dynamic Programming |
| 9 | Non-overlapping Intervals | [#435](https://leetcode.com/problems/non-overlapping-intervals/) | Sort + Greedy | Greedy |
| 10 | Redundant Connection | [#684](https://leetcode.com/problems/redundant-connection/) | Union-Find | Union-Find |

---

## Challenge

| # | Problem | LeetCode | Pattern | Day 4 Topic |
|---|---------|----------|---------|-------------|
| 11 | Word Search II | [#212](https://leetcode.com/problems/word-search-ii/) | Trie + DFS | Tries |
| 12 | Edit Distance | [#72](https://leetcode.com/problems/edit-distance/) | 2D DP | Dynamic Programming |

---

## Approach Hints

<details>
<summary><b>Hint 2: Number of Islands (#200)</b></summary>

Iterate through the grid. When you find a '1', increment count and BFS/DFS to mark all connected '1's as visited (set to '0'). Each BFS/DFS explores one full island.

</details>

<details>
<summary><b>Hint 4: Course Schedule (#207)</b></summary>

Build a directed graph. Detect cycles using DFS with 3 states: unvisited(0), visiting(1), visited(2). If you revisit a node in state 1, there's a cycle.

</details>

<details>
<summary><b>Hint 7: Coin Change (#322)</b></summary>

`dp[amount] = min(dp[amount - coin] + 1)` for each coin denomination. Initialize `dp[0] = 0`, all others = infinity. If `dp[amount]` is still infinity, return -1.

</details>

<details>
<summary><b>Hint 8: LIS (#300)</b></summary>

`dp[i]` = length of longest increasing subsequence ending at index `i`. For each `i`, check all `j < i`: if `nums[j] < nums[i]`, then `dp[i] = max(dp[i], dp[j] + 1)`.

</details>

<details>
<summary><b>Hint 9: Non-overlapping Intervals (#435)</b></summary>

Sort by end time. Greedily keep intervals that end earliest -- they leave the most room for future intervals. Count the ones you have to remove.

</details>

<details>
<summary><b>Hint 10: Redundant Connection (#684)</b></summary>

Use Union-Find. Add edges one by one. If `union` returns False (already connected), that edge creates a cycle and is the answer.

</details>

<details>
<summary><b>Hint 11: Word Search II (#212)</b></summary>

Build a Trie from all target words. DFS through the grid, following Trie branches. When no branch exists for a character, prune that entire search path.

</details>

<details>
<summary><b>Hint 12: Edit Distance (#72)</b></summary>

2D DP where `dp[i][j]` = min operations to convert `word1[:i]` to `word2[:j]`. If characters match, no cost. Otherwise, take the minimum of insert, delete, replace (all +1).

</details>

---

## Final Self-Check: Are You Interview-Ready?

### Data Structures
- [ ] Can you implement a Trie from scratch?
- [ ] Can you implement Union-Find with path compression?
- [ ] Can you build an adjacency list from an edge list?

### Algorithms
- [ ] Can you write BFS and DFS from memory?
- [ ] Can you solve any DP problem using the 4-step recipe (state, recurrence, base case, direction)?
- [ ] Can you detect cycles in directed graphs using 3-state DFS?
- [ ] Can you explain when Greedy works and when you need DP instead?
- [ ] Can you implement topological sort?

### Problem-Solving
- [ ] Given a new problem, can you identify which pattern to use?
- [ ] Can you explain your approach before coding?
- [ ] Can you analyze the time and space complexity of your solution?
- [ ] Can you handle edge cases (empty input, single element, large input)?

If you checked most boxes, you are interview-ready.

---

## What's Next?

1. **Practice 5-10 problems daily** on LeetCode
2. **Read the [Cheatsheet](cheatsheet.md)** before each practice session
3. **Read the [Interview Playbook](interview-playbook.md)** before your interview
4. **Use [NeetCode Roadmap](https://neetcode.io/roadmap)** for structured progression
5. **Join communities:** r/leetcode, NeetCode Discord, Striver's community

---

[Back to Day 4](day4.md) | [Back to Course](README.md) | [Cheatsheet](cheatsheet.md) | [Interview Playbook](interview-playbook.md)
