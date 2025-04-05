import json
import matplotlib.pyplot as plt
import numpy as np
from tkinter import *
from tkinter import simpledialog, messagebox

def place_in_crop_region(arr, crop_coords, target_shape=None):
    """
    Place a 2D array within a specified crop region
    
    Parameters:
    ----------
    arr : ndarray
        2D array to be placed in the crop region
    crop_coords : list or tuple
        Crop coordinates as [min_row, max_row, min_col, max_col]
    target_shape : tuple, optional
        Final output shape. If None, uses maximum crop coordinates
    
    Returns:
    -------
    ndarray
        Array containing original data in the cropped region
    """
    arr = np.asarray(arr)
    
    try:
        min_row, max_row, min_col, max_col = map(int, crop_coords)
    except Exception as e:
        raise ValueError(f"Invalid crop coordinates: {crop_coords}") from e
    
    # Calculate crop region dimensions
    crop_height = max_row - min_row + 1
    crop_width = max_col - min_col + 1
    
    # Check dimension compatibility
    if arr.shape != (crop_height, crop_width):
        raise ValueError(f"Input array shape {arr.shape} doesn't match crop region dimensions {(crop_height, crop_width)}")
    
    # Determine output dimensions
    if target_shape is None:
        output_height = max_row + 1
        output_width = max_col + 1
    else:
        output_height, output_width = target_shape
    
    # Create output array
    output = np.zeros((output_height, output_width), dtype=arr.dtype)
    
    # Place input array in specified region
    output[min_row:max_row+1, min_col:max_col+1] = arr
    
    return output

def auto_crop(arr, margin=0, crop_coords=None):
    """
    Final crop function with all issues resolved
    
    Parameters:
        arr: Input array to crop
        margin: Additional margin around the crop (default 0)
        crop_coords: Predefined crop coordinates [min_row, max_row, min_col, max_col] (optional)
    
    Returns:
        Tuple of (cropped array, crop coordinates)
    """
    arr = np.asarray(arr)
    
    if crop_coords is not None:
        try:
            min_row, max_row, min_col, max_col = map(int, crop_coords)
            return arr[min_row:max_row+1, min_col:max_col+1], crop_coords
        except Exception as e:
            raise ValueError(f"Invalid crop coordinates: {crop_coords}") from e
    
    rows, cols = np.where(arr != 0)
    
    if len(rows) == 0:
        return arr.copy(), [0, 1, 0, 1]
    
    min_row = max(int(np.min(rows)) - margin, 0)
    max_row = min(int(np.max(rows)) + 1 + margin, arr.shape[0])
    min_col = max(int(np.min(cols)) - margin, 0)
    max_col = min(int(np.max(cols)) + 1 + margin, arr.shape[1])
    
    crop_coords = [min_row, max_row-1, min_col, max_col-1]
    return arr[min_row:max_row, min_col:max_col], crop_coords

def plot_multiple_matrices(matrices, titles, figsize=(15, 5), background_color='lightgray'):
    """
    Display multiple binary matrices as black-and-white images in a single plot
    
    Parameters:
        matrices: List of 2D binary matrices (0 and 1 values)
        titles: List of titles for each matrix
        figsize: Overall plot size (default (15, 5))
        background_color: Window background color (default 'lightgray')
    """
    n = len(matrices)
    if n != len(titles):
        raise ValueError("Number of matrices and titles must match")
    
    plt.figure(figsize=figsize, facecolor=background_color)
    
    for i, (matrix, title) in enumerate(zip(matrices, titles)):
        plt.subplot(1, n, i+1)
        plt.imshow(matrix, cmap='binary', interpolation='nearest')
        plt.title(title)
        plt.axis('off')
    
    plt.tight_layout(pad=2)  # Increase padding around subplots
    plt.show()

def plot_binary_matrix(matrix, title='Binary Matrix', background_color='lightgray'):
    """
    Display a binary matrix as a black-and-white image
    
    Parameters:
        matrix: 2D binary matrix (0 and 1 values)
        title: Image title (default 'Binary Matrix')
        background_color: Window background color (default 'lightgray')
    """
    plt.figure(figsize=(8, 8), facecolor=background_color)
    plt.imshow(matrix, cmap='binary', interpolation='nearest')
    plt.title(title)
    plt.axis('off')
    plt.tight_layout(pad=2)  # Increase padding around image
    plt.show()

def string_to_matrix(indices, size=20):
    """
    Convert a list of indices to a binary matrix
    
    Parameters:
        indices: List of indices to set to 1
        size: Output matrix size (default 20x20)
    
    Returns:
        Binary matrix of size size x size
    """
    # Create zero matrix
    matrix = [[0 for _ in range(size)] for _ in range(size)]
    
    # Set specified indices to 1
    for idx in indices:
        if 0 <= idx < size*size:
            row = idx // size
            col = idx % size
            matrix[row][col] = 1
    
    return matrix

def remove_duplicate_rows_columns(matrix, min_s=(2,2)):
    """
    Remove duplicate rows and columns from a matrix
    
    Parameters:
        matrix: Input matrix to process
        min_s: Minimum size constraints (rows, columns)
    
    Returns:
        Tuple of (cleaned matrix, removal path dictionary)
    """
    path = {}
    row_path = []
    col_path = []
    
    # Remove duplicate rows
    rows_to_remove = []
    if len(matrix) > 1:
        for i in range(1, len(matrix)):
            if all(matrix[i][j] == matrix[i-1][j] for j in range(len(matrix[i]))):
                rows_to_remove.append(i)
                if any(cell == 1 for cell in matrix[i]):
                    row_path.append(i)
    
    # Create new matrix without duplicate rows
    new_matrix = [row for idx, row in enumerate(matrix) if idx not in rows_to_remove]
    
    if not new_matrix:
        return []
    
    # Remove duplicate columns
    cols_to_remove = []
    if len(matrix[0]) > 1:
        for j in range(1, len(new_matrix[0])):
            if all(new_matrix[i][j] == new_matrix[i][j-1] for i in range(len(new_matrix))):
                cols_to_remove.append(j)
                if any(new_matrix[i][j] == 1 for i in range(len(new_matrix))):
                    col_path.append(j)
    
    # Create final matrix without duplicate columns
    result = []
    for row in new_matrix:
        new_row = [val for idx, val in enumerate(row) if idx not in cols_to_remove]
        result.append(new_row)
    
    path['rows_to_remove'] = rows_to_remove
    path['cols_to_remove'] = cols_to_remove

    return result, path

def apply_path(matrix, path):
    """
    Apply removal path to a matrix
    
    Parameters:
        matrix: Input matrix (NumPy array or Python list)
        path: Dictionary containing rows_to_remove and cols_to_remove
    
    Returns:
        Tuple of (modified matrix, path)
    """
    # For NumPy arrays
    if isinstance(matrix, np.ndarray):
        # Remove rows (in descending order)
        rows_to_remove = sorted(path.get('rows_to_remove', []), reverse=True)
        modified = np.delete(matrix, rows_to_remove, axis=0)
        
        # Remove columns (in descending order)
        cols_to_remove = sorted(path.get('cols_to_remove', []), reverse=True)
        modified = np.delete(modified, cols_to_remove, axis=1)
        
        return modified, path
    
    # For Python lists
    else:
        modified = [row.copy() for row in matrix]
        
        # Remove rows (in descending order)
        for idx in sorted(path.get('rows_to_remove', []), reverse=True):
            if idx < len(modified):
                del modified[idx]
        
        # Remove columns (in descending order)
        for idx in sorted(path.get('cols_to_remove', []), reverse=True):
            if modified and idx < len(modified[0]):
                for row in modified:
                    del row[idx]
        
        return modified, path

def compare_images(img1, img2):
    """
    Compare two images and check for similarity
    
    Parameters:
        img1, img2: Two 2D numpy arrays (images)
    
    Returns:
        1 if identical from start
        2 if identical after 90° rotation
        3 if identical after 180° rotation
        4 if identical after 270° rotation
        0 otherwise
    """
    img1 = np.array(img1)
    img2 = np.array(img2)

    # 1. Check if identical from start
    if np.array_equal(img1, img2):
        return 1
    
    # 2. Check after 90° rotation
    img2_rotated = np.rot90(img2)
    if np.array_equal(img1, img2_rotated):
        return 2
    
    # 3. Check after 180° rotation
    img2_rotated = np.rot90(img2, 2)
    if np.array_equal(img1, img2_rotated):
        return 3
    
    # 4. Check after 270° rotation
    img2_rotated = np.rot90(img2, 3)
    if np.array_equal(img1, img2_rotated):
        return 4
    
    return 0

def duplicate_index(arr, index, axis=0):
    """
    Duplicate a row/column at specified index and insert into array
    
    Parameters:
        arr: Input array
        index: Index to duplicate before
        axis: 0 for rows, 1 for columns
    
    Returns:
        New array with duplicated row/column
    """
    arr = np.array(arr)

    if axis not in [0, 1]:
        raise ValueError("axis must be 0 (rows) or 1 (columns)")
    
    n = arr.shape[axis]
    
    if index < 0 or index > n:
        print(index, n)
        raise IndexError(f"Index {index} out of bounds for axis {axis}")
    
    if axis == 0:
        row_to_copy = arr[index-1:index, :]  # Shape (1, cols)
        return np.insert(arr, index, row_to_copy, axis=0)
    else:
        col_to_copy = arr[:, index-1:index]  # Shape (rows, 1)
        col_to_copy = np.rot90(col_to_copy, -1)
        return np.insert(arr, index, col_to_copy, axis=1)
    
def add_col_row(src, path):
    """
    Add columns and rows back based on removal path
    
    Parameters:
        src: Source matrix
        path: Dictionary containing removal information
    
    Returns:
        Modified matrix with added rows/columns
    """
    for idx in path['cols_to_remove']:
        src = duplicate_index(src, idx, 1)

    for idx in path['rows_to_remove']:
        src = duplicate_index(src, idx, 0)

    return np.array(src)
    
def process_shape_transformation(src_indices, new_indices, src_prompt_indices, new_prompt_indices, size=20):
    """
    Process shape transformation from source to target pattern
    
    Parameters:
        src_indices: List of indices for source shape
        new_indices: List of indices for target shape
        src_prompt_indices: List of indices for source prompt
        new_prompt_indices: List of indices for target prompt
        size: Matrix size (default 20)
    
    Returns:
        Tuple of (resulting matrix placed in cropped prompt region, cropped prompt matrix)
    """
    # Convert indices to matrices
    src_prompt_matrix = string_to_matrix(src_prompt_indices, size)
    new_prompt_matrix = string_to_matrix(new_prompt_indices, size)
    src_matrix = string_to_matrix(src_indices, size)

    # Auto-crop matrices
    src_prompt_cropped, c_cords = auto_crop(np.array(src_prompt_matrix))
    new_prompt_cropped, n_cords = auto_crop(np.array(new_prompt_matrix))
    src_cropped, _ = auto_crop(np.array(src_matrix), crop_coords=c_cords)

    # Remove duplicate rows/columns
    src_prompt_cleaned, path_src = remove_duplicate_rows_columns(src_prompt_cropped)
    new_prompt_cleaned, path_new = remove_duplicate_rows_columns(new_prompt_cropped)
    src_cleaned, _ = apply_path(src_cropped, path_src)

    # Compare and align orientation
    flag = compare_images(new_prompt_cleaned, src_prompt_cleaned)
    if flag > 1:
        rotations = flag - 1
        src_prompt_cleaned = np.rot90(src_prompt_cleaned, rotations)
        src_cleaned = np.rot90(src_cleaned, rotations)

    # Apply transformation
    result = add_col_row(src_cleaned, path_new)
    
    # Place the result in the original cropped prompt region
    final_result = place_in_crop_region(result, n_cords, target_shape=np.array(new_prompt_matrix).shape)
    
    return final_result, new_prompt_matrix

def add_zero_padding_advanced(array, padding=(1,1,1,1)):
    """
    Add zero padding with different sizes for each side
    
    Parameters:
        array: 2D NumPy array
        padding: Tuple of (top, bottom, left, right) padding sizes
    
    Returns:
        Padded array
    """
    if not isinstance(array, np.ndarray) or array.ndim != 2:
        raise ValueError("Input must be a 2D NumPy array")
    
    top, bottom, left, right = padding
    
    new_shape = (
        array.shape[0] + top + bottom,
        array.shape[1] + left + right
    )
    padded_array = np.zeros(new_shape, dtype=array.dtype)
    
    padded_array[
        top:-bottom if bottom !=0 else None,
        left:-right if right !=0 else None
    ] = array
    
    return padded_array

class DrawingApp:
    """
    Main application class for the Shape Transformer Tool
    
    Provides a GUI for:
    - Drawing shapes on a grid
    - Saving shapes as transformation tools
    - Applying transformations based on prompts
    - Managing saved tools
    """
    def __init__(self, master):
        self.master = master
        master.title("Shape Transformer Tool")
        
        # Initial settings
        self.grid_size = 20
        self.cell_size = 20
        self.tools_file = "tools.json"
        self.current_indices = []
        self.short_term_memory = []
        self.selected_tool = None
        self.tools = self.load_tools()
        
        # Create UI
        self.create_widgets()
        self.create_grid()

    def create_widgets(self):
        """Create the main UI widgets"""
        # Control frame
        control_frame = Frame(self.master)
        control_frame.pack(pady=10)
        
        # Control buttons
        Button(control_frame, text="New Tool", command=self.save_as_tool).pack(side=LEFT, padx=5)
        Button(control_frame, text="Clear", command=self.clear_grid).pack(side=LEFT, padx=5)
        Button(control_frame, text="Delete Selected Tool", command=self.delete_selected_tool).pack(side=LEFT, padx=5)

        # Tools listbox
        self.tool_listbox = Listbox(self.master, width=30, height=10)
        self.tool_listbox.pack(pady=10)
        self.update_tool_list()
        
        # Tool selection event
        self.tool_listbox.bind('<<ListboxSelect>>', self.select_tool)

    def clear_prediction(self):
        """Clear predicted cells from the grid"""
        for idx in self.prediction_indices:
            i = idx // self.grid_size
            j = idx % self.grid_size
            if idx not in self.current_indices:
                self.cells[i][j].config(bg="white")
        self.prediction_indices = []
        
    def create_grid(self):
        """Create the drawing grid"""
        grid_frame = Frame(self.master)
        grid_frame.pack()
        
        # Create 20x20 grid
        self.cells = []
        for i in range(self.grid_size):
            row = []
            for j in range(self.grid_size):
                cell = Button(grid_frame, width=2, height=1, background="#ffffff",
                            command=lambda i=i, j=j: self.toggle_cell(i, j))
                cell.grid(row=i, column=j)
                row.append(cell)
            self.cells.append(row)

    def predict_shape(self):
        """Predict and apply shape transformation based on selected tool"""
        if not self.selected_tool:
            messagebox.showerror("Error", "Please select a tool first!")
            return
        
        if not self.current_indices:
            messagebox.showerror("Error", "Please draw a shape first!")
            return
        
        try:
            # Transformation parameters
            n = self.selected_tool['n']
            src_full = self.selected_tool['indices']
            
            # Select last n points from current drawing
            if len(self.current_indices) < n:
                messagebox.showerror("Error", f"At least {n} points needed for prediction!")
                return
            
            new_full = self.current_indices[-n:]
            src_prompt = src_full[:n]
            new_prompt = new_full
            
            # Perform shape transformation
            result, _ = process_shape_transformation(
                src_indices=src_full,
                new_indices=new_full,
                src_prompt_indices=src_prompt,
                new_prompt_indices=new_prompt
            )
            
            # Convert result to indices
            new_indices = []
            for i in range(result.shape[0]):
                for j in range(result.shape[1]):
                    if result[i][j] == 1:
                        idx = i * self.grid_size + j
                        new_indices.append(idx)
            
            # Add new points to grid
            for idx in new_indices:
                if idx not in self.current_indices:
                    self.current_indices.append(idx)
                    row = idx // self.grid_size
                    col = idx % self.grid_size
                    self.cells[row][col].config(bg="black")
            
        except Exception as e:
            pass
    
    def save_as_tool(self):
        """Save current drawing as a new transformation tool"""
        n = simpledialog.askinteger("Input", "Enter number of initial points (n):", 
                                   parent=self.master)
        if n is not None and len(self.current_indices) >= n:
            tool_name = simpledialog.askstring("Input", "Enter tool name:", 
                                              parent=self.master)
            if tool_name:
                new_tool = {
                    "name": tool_name,
                    "n": n,
                    "indices": self.current_indices.copy()
                }
                self.tools.append(new_tool)
                self.save_tools()
                self.update_tool_list()
                messagebox.showinfo("Success", "Tool saved successfully!")
        else:
            messagebox.showerror("Error", f"At least {n} points needed!" if n else "Invalid input!")
    
    def toggle_cell(self, i, j):
        """Toggle cell state between black/white"""
        cell = self.cells[i][j]
        index = i * self.grid_size + j
        
        if cell.cget("bg") == "black":
            # Remove from short-term memory
            if index in self.short_term_memory:
                self.short_term_memory.remove(index)
            cell.config(bg="white")
            self.current_indices.remove(index)
        else:
            cell.config(bg="black")
            self.current_indices.append(index)
            
            # Add to short-term memory
            self.short_term_memory.append(index)
            
            # Check number of points in short-term memory
            if self.selected_tool:
                required_n = self.selected_tool['n']
                if len(self.short_term_memory) > required_n:
                    # Remove oldest point
                    removed_idx = self.short_term_memory.pop(0)
                
                # Automatically predict when enough points
                if len(self.short_term_memory) == required_n:
                    self.auto_predict()
                    self.short_term_memory.clear()

    def auto_predict(self):
        """Automatically predict shape when enough points are drawn"""
        try:
            if not self.selected_tool:
                return
            
            n = self.selected_tool['n']
            src_full = self.selected_tool['indices']
            new_full = self.short_term_memory[-n:]  # Use last n points
            
            src_prompt = src_full[:n]
            new_prompt = new_full
            
            # Perform shape transformation
            result, _ = process_shape_transformation(
                src_indices=src_full,
                new_indices=new_full,
                src_prompt_indices=src_prompt,
                new_prompt_indices=new_prompt
            )
            
            # Convert result to indices
            new_indices = []
            for i in range(result.shape[0]):
                for j in range(result.shape[1]):
                    if result[i][j] == 1:
                        idx = i * self.grid_size + j
                        new_indices.append(idx)
            
            # Add new points to grid
            for idx in new_indices:
                if idx not in self.current_indices:
                    self.current_indices.append(idx)
                    row = idx // self.grid_size
                    col = idx % self.grid_size
                    self.cells[row][col].config(bg="black")
            
        except Exception as e:
            pass

    def select_tool(self, event):
        """Handle tool selection from listbox"""
        selection = self.tool_listbox.curselection()
        if selection:
            self.selected_tool = self.tools[selection[0]]
            # Reset short-term memory when changing tools
            self.short_term_memory.clear()

    def clear_grid(self):
        """Clear the drawing grid"""
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                self.cells[i][j].config(bg="white")
        self.current_indices = []
        self.short_term_memory.clear()
    
    def load_tools(self):
        """Load saved tools from JSON file"""
        try:
            with open(self.tools_file, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []
    
    def save_tools(self):
        """Save tools to JSON file"""
        with open(self.tools_file, 'w') as f:
            json.dump(self.tools, f, indent=2)
    
    def update_tool_list(self):
        """Update the tools listbox"""
        self.tool_listbox.delete(0, END)
        for tool in self.tools:
            self.tool_listbox.insert(END, f"{tool['name']} (n={tool['n']})")

    def delete_selected_tool(self):
        """Delete the currently selected tool"""
        selection = self.tool_listbox.curselection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a tool to delete!")
            return
        
        tool_name = self.tools[selection[0]]['name']
        if True:  # Simplified confirmation
            # Remove tool from list
            del self.tools[selection[0]]
            
            # Save changes and update list
            self.save_tools()
            self.update_tool_list()
            self.selected_tool = None

if __name__ == "__main__":
    root = Tk()
    app = DrawingApp(root)
    root.mainloop()