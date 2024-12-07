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
2. Ensure you have the required Python libraries installed
   ```bash
      pip install numpy matplotlib

3. Create a parameter file (`params.txt`) in the root directory with the following format:
   ```plaintext
   M=200
   SEARCH_SPACE_L=0
   SEARCH_SPACE_H=1
   NUM_REFINEMENT=5
   THOMPSON_LOOP_LEN=1000
   refinement_type=1
   func=2
4. Run the script
   ```bash
   python code/thompson_optimizer.py --param_file params.txt

### Parameter Details

| Parameter            | Description                                                         | Default Value |
|----------------------|---------------------------------------------------------------------|---------------|
| `M`                  | Number of initial intervals in the search space.                   | `200`         |
| `SEARCH_SPACE_L`     | Lower bound of the search space.                                    | `0`           |
| `SEARCH_SPACE_H`     | Upper bound of the search space.                                    | `1`           |
| `NUM_REFINEMENT`     | Number of refinements to perform.                                   | `5`           |
| `THOMPSON_LOOP_LEN`  | Number of iterations per refinement in the Thompson Sampling loop.  | `1000`        |
| `refinement_type`    | Type of refinement (`1` for neighbors, `2` for CDF-based).          | `1`           |
| `func`               | Objective function to optimize (`1` for `phi_1`, `2` for `phi_2`). | `2`           |

### Output
- The refined intervals are displayed in the terminal after each refinement.
- Example output in the terminal:
  ```plaintext
  =========== Refinement 1 ===========
  Iteration 0: H(t) = 1.5000, H(0) = 2.0000
  Iteration 1: H(t) = 1.0000, H(0) = 2.0000
  Stopping Thompson Sampling at iteration 2.
  Refined Intervals: [[0.0, 0.5], [0.5, 1.0]]; HPIdx: 0

  =========== Refinement 2 ===========
  Iteration 0: H(t) = 1.2000, H(0) = 1.5000
  Iteration 1: H(t) = 0.8000, H(0) = 1.5000
  Stopping Thompson Sampling at iteration 3.
  Refined Intervals: [[0.0, 0.25], [0.25, 0.5], [0.5, 1.0]]; HPIdx: 1

- Plots of the noisy function and final refined intervals are saved to `results/plots/` as:
  ```plaintext
  function_maximization1.png
  
  ```plaintext
  function_maximization2.png
