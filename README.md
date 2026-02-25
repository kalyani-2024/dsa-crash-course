# DSA Crash Course -- Master Data Structures and Algorithms in 4 Days

## From Zero to Interview-Ready

**Who is this for?** Complete beginners, college students preparing for placements, anyone who wants to crack coding interviews at top tech companies.

**What you will learn:** Every major data structure, every core algorithm pattern, interview strategies, and -- most importantly -- how to think algorithmically.

**Prerequisites:** Basic programming knowledge in any language (Python/C++/Java). We use Python in examples, but concepts apply to all languages.

---

## What Makes This Course Different?

Most DSA resources dump code and expect you to memorize it. This course teaches you to think.

Every topic follows a concept-first approach:

```
1. WHAT is it?       -- Plain-English explanation and real-world analogy
2. WHY does it work? -- The intuition behind the pattern
3. WHEN do I use it? -- Clear signals to recognize the pattern
4. HOW to code it    -- Clean implementation with inline explanations
5. WALKTHROUGH       -- Step-by-step trace through an example
```

---

## 4-Day Schedule

### Day 1 -- Arrays and Strings

Learn the most fundamental data structures and the patterns built on top of them.

| Resource | File | Topics |
|----------|------|--------|
| **Course Content** | [day1-2hrs.md](day1-2hrs.md) | Big-O, Arrays (Two Pointers, Sliding Window, Prefix Sum, Kadane's), Strings (Frequency Counting, Palindromes, String Manipulation) |

**Patterns Covered:** 6 | **Structures:** Arrays, Strings

---

### Day 2 -- HashMaps, Linked Lists, Stacks, Queues, Sorting, and Binary Search

Master the core data structures and essential searching/sorting techniques.

| Resource | File | Topics |
|----------|------|--------|
| **Course Content** | [day2-2hrs.md](day2-2hrs.md) | Hash Maps and Sets, Linked Lists (Slow/Fast, Reversal, Merge), Stacks (Matching, Monotonic), Queues, Sorting, Binary Search, Bit Manipulation |

**Patterns Covered:** 6 | **Structures:** Hash Maps, Hash Sets, Linked Lists, Stacks, Queues

---

### Day 3 -- Recursion, Backtracking, Trees, and Heaps

Learn recursive thinking, then apply it to trees and priority queues.

| Resource | File | Topics |
|----------|------|--------|
| **Course Content** | [day3.md](day3.md) | Recursion, Backtracking (Subsets, Permutations, Combinations, N-Queens), Trees and BST (Traversals, Recursive Properties, Validation), Heaps and Priority Queues (Top-K, Merge K Sorted, Median) |

**Patterns Covered:** 4 | **Structures:** Trees, BST, Heaps

---

### Day 4 -- Tries, Graphs, Greedy, and Dynamic Programming

The most advanced material -- graph algorithms and algorithm paradigms.

| Resource | File | Topics |
|----------|------|--------|
| **Course Content** | [day4.md](day4.md) | Tries (Prefix Trees), Graphs (BFS, DFS, Topological Sort, Dijkstra), Union-Find, Greedy Algorithms, Dynamic Programming (1D, 2D, Knapsack) |

**Patterns Covered:** 4+ | **Structures:** Tries, Graphs, Union-Find

---

### Reference Materials

| Resource | File | Purpose |
|----------|------|---------|
| **Cheatsheet** | [cheatsheet.md](cheatsheet.md) | Quick-reference for all patterns during practice |
| **Interview Playbook** | [interview-playbook.md](interview-playbook.md) | How to approach any coding interview question |

---

## Complete Topic Coverage

```
Day 1 -- ARRAYS AND STRINGS          Day 2 -- DATA STRUCTURES + SEARCH
+-------------------------+          +-------------------------+
| Arrays                  |          | Hash Maps / Sets        |
| Strings                 |          | Linked Lists            |
| Two Pointers            |          | Stacks / Queues         |
| Sliding Window          |          | Monotonic Stack         |
| Prefix Sum / Kadane's   |          | Slow/Fast Pointers      |
| Palindrome Techniques   |          | Sorting (preprocessing) |
|                         |          | Binary Search           |
| 6 Patterns              |          | Bit Manipulation        |
| 15+ Problems            |          |                         |
+-------------------------+          | 6 Patterns              |
                                     | 16+ Problems            |
                                     +-------------------------+

Day 3 -- RECURSION + TREES           Day 4 -- GRAPHS + PARADIGMS
+-------------------------+          +-------------------------+
| Recursion               |          | Tries (Prefix Trees)    |
| Backtracking            |          | Graphs (BFS/DFS)        |
| Trees (Binary, BST)     |          | Topological Sort        |
| Tree Traversals         |          | Dijkstra's Algorithm    |
| Heaps / Priority Q      |          | Union-Find              |
|                         |          | Greedy Algorithms       |
| 4 Patterns              |          | Dynamic Programming     |
| 15+ Problems            |          |                         |
+-------------------------+          | 4+ Patterns             |
                                     | 12+ Problems            |
                                     +-------------------------+
```

---

## Visual Learning Tools (Bookmark These)

| Tool | URL | Best For |
|------|-----|----------|
| **VisuAlgo** | [visualgo.net](https://visualgo.net/) | Sorting, Trees, Graphs, DP -- animated |
| **Algorithm Visualizer** | [algorithm-visualizer.org](https://algorithm-visualizer.org/) | Interactive code + animation |
| **USFCA Visualizations** | [cs.usfca.edu/~galles](https://www.cs.usfca.edu/~galles/visualization/Algorithms.html) | BST, Heaps, Hash Tables, Tries |
| **Pathfinding Visualizer** | [pathfinding.js.org](https://qiao.github.io/PathFinding.js/visual/) | BFS, DFS, Dijkstra on grids |
| **Sorting Visualizer** | [toptal.com/sorting](https://www.toptal.com/developers/sorting-algorithms) | Compare sorting algorithms |
| **Python Tutor** | [pythontutor.com](https://pythontutor.com/) | Visualize your code step-by-step |
| **NeetCode Roadmap** | [neetcode.io/roadmap](https://neetcode.io/roadmap) | Problem roadmap with videos |

---

## The 5-Step Problem Solving Framework

```
+------------------------------------------+
|  1. UNDERSTAND  -- Read problem 2-3x     |
|     - What are the inputs/outputs?       |
|     - What are the constraints?          |
|     - Walk through examples by hand      |
+------------------------------------------+
|  2. PLAN  -- Think before coding          |
|     - What pattern does this match?      |
|     - What data structure helps?         |
|     - What's the brute force?            |
|     - Can I optimize?                    |
+------------------------------------------+
|  3. CODE  -- Write clean code             |
|     - Start with function signature      |
|     - Handle edge cases first            |
|     - Write the core logic               |
+------------------------------------------+
|  4. TEST  -- Verify with examples         |
|     - Dry run with given examples        |
|     - Try edge cases (empty, 1 element)  |
|     - Try large inputs mentally          |
+------------------------------------------+
|  5. OPTIMIZE  -- Can we do better?        |
|     - Better time complexity?            |
|     - Better space complexity?           |
|     - Cleaner code?                      |
+------------------------------------------+
```

---

## Quick Pattern Recognition

```
"Find pair with property X"          -> HashMap or Two Pointers
"Longest/shortest subarray"          -> Sliding Window
"Find in sorted data"               -> Binary Search
"Search answer range"               -> Binary Search on Answer
"All subsets/combos/perms"           -> Backtracking
"Cycle in linked list"              -> Slow/Fast Pointers
"Matching brackets/nesting"          -> Stack
"Next greater/smaller"              -> Monotonic Stack
"Level-by-level / shortest path"    -> BFS
"All paths / cycle detection"       -> DFS
"Connected components"              -> Union-Find
"Schedule/select intervals"         -> Greedy
"Min/max with overlapping choices"  -> Dynamic Programming
"Prefix matching / autocomplete"    -> Trie
"Top K / streaming min/max"         -> Heap
```

---

## Learning Path Map

```
                    +------------------+
                    |   START HERE     |
                    |   (Big-O)        |
                    +--------+---------+
                             |
         +-------------------+-------------------+
         v                   v                   v
   +-----------+      +-----------+      +-----------+
   |  Arrays   |      |  Strings  |      |  HashMap  |
   |Two Pointer|      |Palindrome |      | Frequency |
   |Sliding Win|      | Matching  |      |  Lookup   |
   +-----+-----+      +-----+-----+      +-----+-----+
         |                   |                   |
         v                   v                   v
   +-----------+      +-----------+      +-----------+
   |  Linked   |      |  Stacks   |      |  Sorting  |
   |  Lists    |      |  Queues   |      | Bin Search|
   +-----+-----+      +-----+-----+      +-----+-----+
         |                   |                   |
         +-------------------+-------------------+
                             v
                    +---------------+
                    |  Recursion &  |
                    |  Backtracking |
                    +-------+-------+
                            |
         +------------------+------------------+
         v                  v                  v
   +-----------+     +-----------+     +-----------+
   |   Trees   |     |   Heaps   |     |   Tries   |
   |    BST    |     |  Top-K    |     |  Prefix   |
   +-----+-----+     +-----------+     +-----------+
         |
         v
   +-----------+     +-----------+     +-----------+
   |  Graphs   |     |  Greedy   |     | Union-Find|
   |  BFS/DFS  |     | Intervals |     |Components |
   +-----+-----+     +-----------+     +-----------+
         |
         v
   +---------------+
   |    Dynamic    |
   |  Programming  |
   +---------------+
```

---

*Let's begin! Open [day1-2hrs.md](day1-2hrs.md) to start your journey.*
