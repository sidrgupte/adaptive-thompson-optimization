# Adaptive Thompson Optimization

## Overview
This project explores the use of **Thompson Sampling** for adaptive optimization in noisy, non-convex functions. The objective is to identify the global maximum of a function with minimal sampling, leveraging entropy-based partitioning and proposal distributions.

## Key Features
- Implementation of Thompson Sampling for optimization.
- Adaptive partitioning of the search space to focus on promising regions.
- Entropy-based stopping and repartitioning criteria.

## Project Structure
- `code/`: Contains all Python scripts.
- `docs/`: Detailed explanations of methods and theory.
- `results/`: Plots and outputs from the experiments.

## Goals
- Develop a robust method for noisy optimization.
- Visualize the impact of entropy-based learning.
- Create a framework for adaptive search applicable to real-world problems.

## Tasks
- **Part 1**: Implement basic Thompson Sampling.
- **Part 2**: Apply Thompson Sampling to noisy, non-convex maximization.
- **Part 3**: Visualize results and analyze entropy decline.

## Getting Started
### Prerequisites
- Python 3.x
- Libraries: `numpy`, `matplotlib`

### Running the Project
1. Clone this repository:
   ```bash
   git clone <your-repo-link>
   cd adaptive-thompson-optimization
