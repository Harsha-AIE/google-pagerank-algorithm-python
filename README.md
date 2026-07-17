# Google PageRank Algorithm Implementation

A Python implementation of Google's **PageRank Algorithm**, demonstrating how search engines rank webpages according to their importance using graph theory, adjacency matrices, and iterative probability calculations.

---

## Overview

PageRank is one of the foundational algorithms behind Google's search engine. It determines the importance of webpages by analyzing the hyperlink structure between them.

This project demonstrates:

- Representation of webpages as a graph
- Construction of an adjacency matrix
- Conversion into a stochastic matrix
- Iterative PageRank computation
- Ranking webpages based on their importance

This implementation is intended for educational purposes to understand the mathematical principles behind PageRank.

---

## Features

- Python implementation of PageRank
- Matrix-based computation using NumPy
- Configurable damping factor
- Iterative convergence algorithm
- Final webpage ranking output
- Easy to modify for larger web graphs

---

## Algorithm Workflow

1. Create webpages and hyperlinks
2. Build the adjacency matrix
3. Convert it into a column stochastic matrix
4. Initialize equal page ranks
5. Iteratively update page ranks
6. Check convergence
7. Display ranked webpages

---

## Mathematical Formula

The PageRank vector is updated using:

PR = β × M × PR + (1 − β) × C

Where:

- **β** = Damping Factor
- **M** = Stochastic Matrix
- **PR** = PageRank Vector
- **C** = Teleportation Vector

The iterations continue until the rank values converge.

---

## Technologies Used

- Python 3
- NumPy

---

## Project Structure

```
PageRank/
│
├── pagerank.py
├── README.md
├── report.pdf
└── requirements.txt
```

---

## Installation

Clone the repository

```bash
git clone https://github.com/yourusername/google-pagerank-algorithm-python.git
```

Navigate to the project

```bash
cd google-pagerank-algorithm-python
```

Install dependencies

```bash
pip install numpy
```

Run

```bash
python pagerank.py
```

---

## Sample Output

```
Summation of matrix:
[3,2,1]

Column Stochastic Matrix

Initial Rank Vector

Iteration 1
Iteration 2
...

Final Rank Vector

[0.48
 0.27
 0.25]
```

---

## Applications

- Search Engines
- Link Analysis
- Web Mining
- Recommendation Systems
- Social Network Analysis
- Citation Ranking

---

## Learning Outcomes

This project helps understand:

- Graph Theory
- Markov Chains
- Probability
- Matrix Operations
- Search Engine Ranking
- Iterative Algorithms

---

## Future Improvements

- Parse real HTML webpages
- Web crawling
- Query-based search engine
- Interactive GUI
- NetworkX visualization
- Support larger web graphs
- Compare PageRank variants
- Performance optimization

---

## References

- Google's Original PageRank Paper
- Stanford CS246 - Mining Massive Datasets
- Towards Data Science - Visualized PageRank
- Search Engine Land - What is PageRank?

---

## Author

G. Harshavardhan

Department of Artificial Intelligence

Amrita Vishwa Vidyapeetham

---

## License

This project is intended for educational and academic purposes.
