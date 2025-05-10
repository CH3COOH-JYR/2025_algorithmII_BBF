import numpy as np
from bruteforce import euclidean_distance # Using the already defined euclidean_distance

class KDNode:
    def __init__(self, point=None, axis=None, left=None, right=None, is_leaf=False, points=None):
        self.point = point          # Splitting point for internal nodes, single data point for leaves (if not bucketed)
        self.axis = axis            # Splitting dimension
        self.left = left            # Left child (KDNode)
        self.right = right          # Right child (KDNode)
        self.is_leaf = is_leaf      # Boolean flag
        # If leaves store multiple points (buckets), use self.points. For this implementation, leaf stores one point in self.point.

def build_kdtree(points, depth=0, leaf_size=1):
    """
    Builds a k-d tree from a list of points.
    points: A numpy array of shape (n, k) where n is number of points and k is dimension.
    depth: Current depth of the tree.
    leaf_size: Maximum number of points in a leaf node (for simplicity, current default is 1 point per leaf).
    """
    if points is None or len(points) == 0:
        return None

    n, k = points.shape

    if n <= leaf_size:
        # Create a leaf node. If leaf_size > 1, self.point could be a list of points.
        # For simplicity, if leaf_size=1, self.point is the single point.
        return KDNode(point=points[0] if n == 1 else points.tolist(), is_leaf=True)

    # Select axis based on depth
    axis = depth % k

    # Sort points by the selected axis and choose median
    # Using np.median and then finding the element can be slow. Sorting is more direct.
    sorted_indices = np.argsort(points[:, axis])
    points = points[sorted_indices]
    median_idx = n // 2
    median_point = points[median_idx]

    node = KDNode(point=median_point, axis=axis)
    node.left = build_kdtree(points[:median_idx], depth + 1, leaf_size)
    node.right = build_kdtree(points[median_idx + 1:], depth + 1, leaf_size) # Exclude median itself
    
    return node

def kdtree_search_recursive(node, query_point, best_dist, best_node_point, depth, k_features):
    """Recursive helper for k-d tree search."""
    if node is None:
        return best_dist, best_node_point

    if node.is_leaf:
        # If leaf stores multiple points (bucket), iterate through them.
        # For current implementation (leaf_size=1), node.point is the single point.
        current_dist = euclidean_distance(query_point, node.point)
        if current_dist < best_dist:
            best_dist = current_dist
            best_node_point = node.point
        return best_dist, best_node_point

    # Internal node
    axis = node.axis
    current_node_dist = euclidean_distance(query_point, node.point)
    if current_node_dist < best_dist:
        best_dist = current_node_dist
        best_node_point = node.point

    # Decide which subtree to visit first
    if query_point[axis] < node.point[axis]:
        nearer_child = node.left
        farther_child = node.right
    else:
        nearer_child = node.right
        farther_child = node.left

    best_dist, best_node_point = kdtree_search_recursive(nearer_child, query_point, best_dist, best_node_point, depth + 1, k_features)

    # Check if the other subtree could contain a closer point
    # This is the "ball within bounds" check
    if farther_child is not None:
        # Distance from query point to the splitting hyperplane
        dist_to_hyperplane_sq = (query_point[axis] - node.point[axis])**2
        if dist_to_hyperplane_sq < best_dist**2: # Compare squared distances to avoid sqrt if possible
            best_dist, best_node_point = kdtree_search_recursive(farther_child, query_point, best_dist, best_node_point, depth + 1, k_features)
            
    return best_dist, best_node_point

def kdtree_search(root, query_point):
    """
    Performs a standard k-d tree search.
    Returns the nearest neighbor point and its distance.
    """
    if root is None:
        return None, float('inf')
    
    k_features = len(query_point)
    best_dist, best_node_point = kdtree_search_recursive(root, query_point, float('inf'), None, 0, k_features)
    return best_node_point, best_dist


if __name__ == '__main__':
    # Example Usage
    points_data = np.array([
        [2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]
    ])
    
    print("Building k-d tree...")
    kdtree_root = build_kdtree(points_data)
    print("k-d tree built.")

    query = np.array([9, 5])
    print(f"\nQuery point: {query}")
    
    nn_kdtree, dist_kdtree = kdtree_search(kdtree_root, query)
    print(f"Nearest neighbor (k-d tree): {nn_kdtree}")
    print(f"Distance (k-d tree): {dist_kdtree}")

    # Compare with brute-force for verification
    from bruteforce import bruteforce_search
    nn_bf, dist_bf = bruteforce_search(points_data, query)
    print(f"Nearest neighbor (Brute-force): {nn_bf}")
    print(f"Distance (Brute-force): {dist_bf}")

    assert np.allclose(dist_kdtree, dist_bf), "k-d tree search result mismatch with brute-force!"
    if nn_kdtree is not None and nn_bf is not None:
         assert np.allclose(nn_kdtree, nn_bf), "k-d tree search NN mismatch with brute-force!"
    print("\nK-d tree search matches brute-force search.")

    points_data_3d = np.array([
        [2,3,1], [5,4,2], [9,6,3], [4,7,4], [8,1,5], [7,2,6], [1,1,1]
    ])
    kdtree_root_3d = build_kdtree(points_data_3d)
    query_3d = np.array([8,2,4])
    nn_kdtree_3d, dist_kdtree_3d = kdtree_search(kdtree_root_3d, query_3d)
    nn_bf_3d, dist_bf_3d = bruteforce_search(points_data_3d, query_3d)
    print(f"\nQuery point 3D: {query_3d}")
    print(f"NN (k-d tree 3D): {nn_kdtree_3d}, Dist: {dist_kdtree_3d}")
    print(f"NN (Brute-force 3D): {nn_bf_3d}, Dist: {dist_bf_3d}")
    assert np.allclose(dist_kdtree_3d, dist_bf_3d), "3D k-d tree search result mismatch!"
    if nn_kdtree_3d is not None and nn_bf_3d is not None:
        assert np.allclose(nn_kdtree_3d, nn_bf_3d), "3D k-d tree search NN mismatch!"
    print("3D K-d tree search matches brute-force search.") 