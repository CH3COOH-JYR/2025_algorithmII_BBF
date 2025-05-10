# Nearest Neighbor Search Algorithms Comparison

This project implements and compares three nearest neighbor search algorithms:
1. Brute Force Search
2. K-D Tree Search
3. Best Bin First (BBF) Search

## Project Structure

- `bruteforce.py`: Brute force nearest neighbor search implementation
- `kdtree.py`: K-D tree construction and search implementation
- `bbf.py`: Best Bin First search implementation
- `main.py`: Main script to run experiments on all data files and generate result plots
- `data/`: Directory containing data files (1.txt to 100.txt)
- `results/`: Directory where results will be stored

## Requirements

```
numpy
pandas
matplotlib
```

Install the required packages using:
```
pip install -r requirements.txt
```

## Data Format

Each data file in the `data/` directory has the following format:
- First line: `n m d` where:
  - `n`: Number of data points
  - `m`: Number of query points
  - `d`: Dimensionality of the points
- Next `n` lines: Data points (each point has `d` coordinates)
- Next `m` lines: Query points (each point has `d` coordinates)

## Usage

To run the full comparison on all data files:

```
python main.py
```

This will:
1. Process all data files in the `data/` directory
2. Run all three search algorithms on each file
3. Collect performance metrics (time, memory, accuracy)
4. Save results to CSV files in the `results/` directory
5. Generate comparison plots

### Parameters

You can modify the following parameters in the `main.py` file:

- `max_files`: Maximum number of data files to process
- `max_queries`: Maximum number of queries to test for each file
- `bbf_t_value`: Number of leaf nodes to visit in BBF search

## Results

The program generates the following results:

- `results/all_results.csv`: Detailed results for each query
- `results/summary_results.csv`: Aggregated results for each data file
- Various plots comparing:
  - Query time
  - Memory usage
  - Accuracy
  - Performance vs dimensions
  - Performance vs number of data points

## Algorithm Comparison

The comparison evaluates the algorithms based on:
1. **Search Time**: How fast the algorithm finds the nearest neighbor
2. **Memory Usage**: How much memory is used during the search
3. **Accuracy**: How close the result is to the true nearest neighbor (found by brute force) 