import numpy as np
import heapq
from kdtree import KDNode # Assuming KDNode is defined in kdtree.py
from bruteforce import euclidean_distance # Using the already defined euclidean_distance

def bbf_search(kdtree_root, query_point, t_max_leaves):
    """
    Best Bin First (BBF) search algorithm based on the provided pseudocode.
    kdtree_root: The root of the k-d tree.
    query_point: The point to search for.
    t_max_leaves: Maximum number of leaf nodes to search.
    Returns the best neighbor found and its distance.
    """
    if kdtree_root is None:
        return None, float('inf')

    # Priority queue: stores (-priority, item_id, node)
    # item_id is for tie-breaking to ensure FIFO for equal priorities if needed, and stability.
    # We use -priority because heapq is a min-heap, and we need a max-priority queue.
    pq = []
    item_id_counter = 0 # Ensures unique items in PQ for tie-breaking

    # Pseudocode: "PriorityQueue <- k-d tree root with priority=0"
    # For a max-priority queue where higher numbers are higher priority, and children get 1/d (positive),
    # a priority of 0 for the root means it would be processed AFTER children if not handled carefully.
    # To ensure root is processed first if it's the only item, or based on its actual conceptual priority:
    # Let's assign a very high priority to the root to ensure it's explored first.
    # Or, if priority 0 is *defined* as highest by problem setter, then (-0, ...) is (0, ...)
    # and children priorities 1/d would become -(1/d) (negative).
    # A min-heap pops negative values before 0. So children would be popped before root. This is not correct.
    # Therefore, the root must have the numerically highest priority value.
    # Let's assign initial priority for the root to be conceptually very high.
    # For practical purposes, if it's the first node, its actual priority value might not matter as much
    # as long as it gets processed. The pseudocode's "priority=0" might be a simplification.
    # Let's use a very large number for initial priority, negated for min-heap.
    heapq.heappush(pq, (-float('inf'), item_id_counter, kdtree_root))
    item_id_counter += 1
    
    best_dist = float('inf')
    best_nn_point = None
    searched_leaves_count = 0

    while pq and searched_leaves_count < t_max_leaves:
        neg_priority, _, current_node = heapq.heappop(pq)
        # current_priority = -neg_priority # Actual priority value

        if current_node.is_leaf:
            searched_leaves_count += 1
            # Assuming leaf node stores a single point in current_node.point
            # If it were a bucket: for point_in_leaf in current_node.points:
            dist_to_leaf_point = euclidean_distance(query_point, current_node.point)
            if dist_to_leaf_point < best_dist:
                best_dist = dist_to_leaf_point
                best_nn_point = current_node.point
        else: # Internal node
            # Check the point at the internal node itself
            dist_to_internal_point = euclidean_distance(query_point, current_node.point)
            if dist_to_internal_point < best_dist:
                best_dist = dist_to_internal_point
                best_nn_point = current_node.point

            split_dim = current_node.axis
            query_val_at_split_dim = query_point[split_dim]
            node_val_at_split_dim = current_node.point[split_dim]

            if query_val_at_split_dim < node_val_at_split_dim:
                nearer_subtree = current_node.left
                farther_subtree = current_node.right
            else:
                nearer_subtree = current_node.right
                farther_subtree = current_node.left
            
            # Priority for children based on distance to splitting plane of current_node
            # distance_to_split = abs(query_val_at_split_dim - node_val_at_split_dim)
            # The pseudocode says: priority=1/(distance_to_split)
            # This distance is from the query point to the splitting hyperplane of the *current_node*.
            # Both children get this priority when added from this parent.

            dist_to_hyperplane = abs(query_val_at_split_dim - node_val_at_split_dim)
            
            child_priority = 0
            if dist_to_hyperplane < 1e-9: # Effectively zero distance, query on hyperplane
                child_priority = float('inf') # Assign very high priority
            else:
                child_priority = 1.0 / dist_to_hyperplane

            # Enqueue children as per pseudocode, without explicit pruning before adding
            # The pruning is through best_dist updates and the t_max_leaves limit.
            if farther_subtree is not None:
                # Standard BBF/k-d tree often checks if the ball intersects the farther half-space:
                # if dist_to_hyperplane < best_dist:
                # However, the provided pseudocode does not show this check *before* insertion.
                # It just inserts. We will follow the pseudocode.
                heapq.heappush(pq, (-child_priority, item_id_counter, farther_subtree))
                item_id_counter += 1
            
            if nearer_subtree is not None:
                heapq.heappush(pq, (-child_priority, item_id_counter, nearer_subtree))
                item_id_counter += 1
                
    return best_nn_point, best_dist


if __name__ == '__main__':
    from kdtree import build_kdtree # For testing
    from bruteforce import bruteforce_search # For comparison

    # Example Usage
    points_data = np.array([
        [2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2],
        [1, 1], [2, 2], [6, 5], [5, 6], [0, 9], [9,0]
    ])
    
    print("Building k-d tree for BBF test...")
    kdtree_r = build_kdtree(points_data)
    print("k-d tree built.")

    query = np.array([5.5, 3.5])
    t_leaves = 5 # Max leaves to search
    print(f"\nQuery point: {query}, t_max_leaves = {t_leaves}")
    
    nn_bbf, dist_bbf = bbf_search(kdtree_r, query, t_leaves)
    print(f"Nearest neighbor (BBF): {nn_bbf}")
    print(f"Distance (BBF): {dist_bbf}")

    # Compare with brute-force for verification of nearest possible
    nn_bf, dist_bf = bruteforce_search(points_data, query)
    print(f"Nearest neighbor (Brute-force): {nn_bf}")
    print(f"Distance (Brute-force): {dist_bf}")

    if nn_bbf is not None:
        accuracy_ratio = dist_bbf / dist_bf if dist_bf > 0 else (0 if dist_bbf == 0 else float('inf'))
        print(f"BBF Accuracy Ratio: {accuracy_ratio:.4f} (Target <= 1.05 for success)")
    else:
        print("BBF did not find a neighbor.")

    query2 = np.array([0,0])
    t_leaves2 = 3
    print(f"\nQuery point: {query2}, t_max_leaves = {t_leaves2}")
    nn_bbf2, dist_bbf2 = bbf_search(kdtree_r, query2, t_leaves2)
    print(f"Nearest neighbor (BBF): {nn_bbf2}")
    print(f"Distance (BBF): {dist_bbf2}")
    nn_bf2, dist_bf2 = bruteforce_search(points_data, query2)
    print(f"Nearest neighbor (Brute-force): {nn_bf2}")
    print(f"Distance (Brute-force): {dist_bf2}")
    if nn_bbf2 is not None:
        accuracy_ratio2 = dist_bbf2 / dist_bf2 if dist_bf2 > 0 else (0 if dist_bbf2 == 0 else float('inf'))
        print(f"BBF Accuracy Ratio: {accuracy_ratio2:.4f}") 