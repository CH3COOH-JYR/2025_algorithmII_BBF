import numpy as np

def euclidean_distance(point1, point2):
    """Calculates the Euclidean distance between two points."""
    return np.sqrt(np.sum((point1 - point2)**2))

def bruteforce_search(data, query_point):
    """
    Performs a brute-force search to find the nearest neighbor.
    Returns the nearest neighbor and its distance to the query point.
    """
    if data is None or len(data) == 0:
        return None, float('inf')
        
    best_neighbor = None
    min_distance = float('inf')
    
    for i, neighbor in enumerate(data):
        distance = euclidean_distance(query_point, neighbor)
        if distance < min_distance:
            min_distance = distance
            best_neighbor = neighbor
            # Storing index might be useful if you need to refer back to original dataset index
            # best_neighbor_idx = i 
            
    return best_neighbor, min_distance

if __name__ == '__main__':
    # Example Usage
    data_points = np.array([
        [1, 2],
        [3, 4],
        [5, 0],
        [0, 5]
    ])
    query = np.array([0.5, 1.5])
    
    nn, dist = bruteforce_search(data_points, query)
    print(f"Query point: {query}")
    print(f"Nearest neighbor (Brute-force): {nn}")
    print(f"Distance: {dist}")
    
    # Test with empty data
    empty_data = np.array([])
    nn_empty, dist_empty = bruteforce_search(empty_data, query)
    print(f"NN for empty data: {nn_empty}, Distance: {dist_empty}") 