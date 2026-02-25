# Day 3 Practice -- Recursion, Backtracking, Trees, BST, and Heaps

## 12 Must-Do Problems

**How to practice:** Spend 15-20 minutes per problem. If stuck for 10 minutes, read the hint. If still stuck, study the solution, then re-solve from scratch.

---

## Warm-Up

| # | Problem | LeetCode | Pattern | Day 3 Topic |
|---|---------|----------|---------|-------------|
| 1 | Maximum Depth of Binary Tree | [#104](https://leetcode.com/problems/maximum-depth-of-binary-tree/) | Recursive Tree | Trees |
| 2 | Invert Binary Tree | [#226](https://leetcode.com/problems/invert-binary-tree/) | Recursive Tree | Trees |
| 3 | Subsets | [#78](https://leetcode.com/problems/subsets/) | Backtracking | Recursion |

---

## Core Practice

| # | Problem | LeetCode | Pattern | Day 3 Topic |
|---|---------|----------|---------|-------------|
| 4 | Permutations | [#46](https://leetcode.com/problems/permutations/) | Backtracking | Recursion |
| 5 | Combination Sum | [#39](https://leetcode.com/problems/combination-sum/) | Backtracking | Recursion |
| 6 | Word Search | [#79](https://leetcode.com/problems/word-search/) | DFS + Backtracking | Recursion |
| 7 | Binary Tree Level Order Traversal | [#102](https://leetcode.com/problems/binary-tree-level-order-traversal/) | Tree BFS | Trees |
| 8 | Validate BST | [#98](https://leetcode.com/problems/validate-binary-search-tree/) | BST Bounds Check | Trees |
| 9 | Lowest Common Ancestor | [#236](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree/) | Recursive Tree | Trees |
| 10 | Kth Largest Element | [#215](https://leetcode.com/problems/kth-largest-element-in-an-array/) | Heap / Top-K | Heaps |

---

## Challenge

| # | Problem | LeetCode | Pattern | Day 3 Topic |
|---|---------|----------|---------|-------------|
| 11 | N-Queens | [#51](https://leetcode.com/problems/n-queens/) | Constraint Backtracking | Recursion |
| 12 | Merge K Sorted Lists | [#23](https://leetcode.com/problems/merge-k-sorted-lists/) | Heap + Linked List | Heaps |

---

## Approach Hints

<details>
<summary><b>Hint 3: Subsets (#78)</b></summary>

For each element, make two recursive calls: one including it, one excluding it. This creates a binary decision tree with 2^n leaves.

</details>

<details>
<summary><b>Hint 6: Word Search (#79)</b></summary>

DFS from each cell. At each step, check if the current cell matches the current character. Mark visited, recurse in 4 directions, then unmark (backtrack).

</details>

<details>
<summary><b>Hint 8: Validate BST (#98)</b></summary>

Pass lower and upper bounds down the recursion. Each node must satisfy `lo < val < hi`. Update bounds: left child gets `hi = node.val`, right child gets `lo = node.val`.

</details>

<details>
<summary><b>Hint 9: LCA (#236)</b></summary>

Recursively search left and right subtrees. If both return non-null, the current node is the LCA. If only one returns non-null, pass it up.

</details>

<details>
<summary><b>Hint 10: Kth Largest (#215)</b></summary>

Use a min-heap of size k. After processing all numbers, the root of the heap is the kth largest element. O(n log k).

</details>

<details>
<summary><b>Hint 11: N-Queens (#51)</b></summary>

Place queens row by row. Track attacked columns, diagonals (`row-col`), and anti-diagonals (`row+col`) with sets. If a column is safe, place and recurse.

</details>

<details>
<summary><b>Hint 12: Merge K Lists (#23)</b></summary>

Put the head of each list into a min-heap. Pop the smallest, append to result, push its `.next`. The heap always gives you the globally smallest node.

</details>

---

## Self-Check: Day 3

- [ ] Can you write the backtracking template (choose, explore, undo)?
- [ ] Can you traverse a tree in all 4 ways (inorder, preorder, postorder, level order)?
- [ ] Can you explain the recursive tree pattern (solve left, solve right, combine)?
- [ ] Can you validate a BST using bounds?
- [ ] Can you explain when to use a heap vs sorting?
- [ ] Can you implement the "Top-K" pattern with a min-heap?

If you checked most boxes, move on to [Day 4](day4.md).

---

[Back to Day 3](day3.md) | [Back to Course](README.md)
