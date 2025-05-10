import numpy as np

def load_data(filepath):
    """
    Loads data from a specified file format.
    Expected first line format:
    1. num_points ignored_value dimensions (e.g., "100000 100 8")
    OR
    2. num_points dimensions (e.g., "1000 2")
    Subsequent num_points lines: dimensions numbers per line.
    """
    with open(filepath, 'r') as f:
        first_line_content = f.readline()
        if not first_line_content:
            raise ValueError(f"File {filepath} is empty or first line is missing.")
        first_line_parts = first_line_content.split()
        
        if len(first_line_parts) >= 3:
            # Assuming format: num_points ignored_value dimensions
            num_points = int(first_line_parts[0])
            dimensions = int(first_line_parts[2]) # Use the third part as dimensions
            # print(f"DEBUG: Parsed 3-part header from '{filepath}': num_points={num_points}, ignored='{first_line_parts[1]}', dimensions={dimensions}")
        elif len(first_line_parts) == 2:
            # Assuming format: num_points dimensions
            num_points = int(first_line_parts[0])
            dimensions = int(first_line_parts[1])
            # print(f"DEBUG: Parsed 2-part header from '{filepath}': num_points={num_points}, dimensions={dimensions}")
        else:
            raise ValueError(
                f"Unexpected first line format in {filepath}: '{first_line_content.strip()}'. "
                f"Expected 'num_points dimensions' or 'num_points ignored_value dimensions'."
            )
            
        if num_points <= 0:
            # Allow empty datasets (0 points) but they should be handled by evaluation logic
            # print(f"Warning: Dataset {filepath} has {num_points} points as per header.")
            return np.empty((0, dimensions)), dimensions


        data = np.zeros((num_points, dimensions))
        for i in range(num_points):
            line_content = f.readline()
            if not line_content: # Check for premature end of file
                raise ValueError(f"Premature end of file in {filepath}. Expected {num_points} data lines, got {i}.")
            
            line_parts = line_content.split()
            
            # Check if the number of parts matches the expected dimensions
            if len(line_parts) != dimensions:
                raise ValueError(
                    f"Data line {i+2} in {filepath} (index {i}) has {len(line_parts)} values, but expected {dimensions} dimensions. Line content: '{line_content.strip()}'"
                )
            
            try:
                data[i] = [float(x) for x in line_parts]
            except ValueError as e:
                raise ValueError(f"Error converting data on line {i+2} in {filepath} (index {i}) to float: {e}. Line content: '{line_content.strip()}'")
            
    return data, dimensions

if __name__ == '__main__':
    # Example usage:
    # Create a dummy data file for testing (2-part header)
    dummy_data_path_2part = "data/dummy_dataset_2part.txt"
    # Create a dummy data file for testing (3-part header)
    dummy_data_path_3part = "data/dummy_dataset_3part.txt"
    # Create a dummy empty data file
    dummy_data_path_empty = "data/dummy_dataset_empty.txt"
    
    import os
    if not os.path.exists("data"):
        os.makedirs("data")
        
    with open(dummy_data_path_2part, 'w') as f:
        f.write("3 2\n") # num_points dimensions
        f.write("1.0 2.0\n")
        f.write("3.0 4.0\n")
        f.write("5.0 6.0\n")

    with open(dummy_data_path_3part, 'w') as f:
        f.write("4 100 3\n") # num_points ignored_value dimensions
        f.write("1.1 2.1 3.1\n")
        f.write("4.1 5.1 6.1\n")
        f.write("7.1 8.1 9.1\n")
        f.write("0.1 0.2 0.3\n")

    with open(dummy_data_path_empty, 'w') as f:
        f.write("0 8\n") # 0 points, 8 dimensions

    print("Testing 2-part header dummy file:")
    try:
        points, dims = load_data(dummy_data_path_2part)
        print(f"Loaded {len(points)} points with {dims} dimensions.")
        if len(points) > 0: print("First point:", points[0])
    except Exception as e:
        print(f"An error occurred with {dummy_data_path_2part}: {e}")

    print("\nTesting 3-part header dummy file:")
    try:
        points, dims = load_data(dummy_data_path_3part)
        print(f"Loaded {len(points)} points with {dims} dimensions.")
        if len(points) > 0: print("First point:", points[0])
    except Exception as e:
        print(f"An error occurred with {dummy_data_path_3part}: {e}")

    print("\nTesting empty dataset dummy file:")
    try:
        points, dims = load_data(dummy_data_path_empty)
        print(f"Loaded {len(points)} points with {dims} dimensions.")
        # This should print "Loaded 0 points with 8 dimensions."
    except Exception as e:
        print(f"An error occurred with {dummy_data_path_empty}: {e}") 