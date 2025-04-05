
# Shape Transformer Tool

## Description
A Python application that enables shape transformation through a graphical interface. Users can draw shapes, save them as reusable tools, and apply transformations to automatically complete shapes based on initial input patterns.

## Features
- Interactive 20×20 drawing grid
- Custom tool creation with configurable initial points
- Real-time shape prediction
- Tool management system
- Visual drawing feedback

## Requirements
- Python 3.6+
- Required packages:
  ```bash
  pip install numpy matplotlib
  ```

## Installation
1. Clone or download the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage Instructions
### Basic Operations
1. Launch the application:
   ```bash
   python shape_transformer.py
   ```

2. Drawing:
   - Left-click cells to toggle between active/inactive states
   - Drawing sequence is tracked automatically

### Tool Management
**Creating Tools:**
1. Draw your desired shape
2. Click "New Tool"
3. Configure:
   - `n`: Number of initial points required
   - Name: Tool identifier

**Using Tools:**
1. Select tool from list
2. Begin drawing initial points
3. System auto-completes shape after `n` points

**Deleting Tools:**
1. Select tool from list
2. Click "Delete Selected Tool"

## Technical Details
### Core Components
- **Transformation Engine**:
  - Shape matching with rotation detection
  - Matrix cropping and normalization
  - Pattern reconstruction

- **Data Management**:
  - Tools stored in `tools.json`
  - Automatic saving/loading

### Key Functions
```python
# Transformation core
process_shape_transformation()
compare_images()
auto_crop()

# Matrix operations
remove_duplicate_rows_columns()
add_col_row()
string_to_matrix()
```

## Examples
### Creating a Square Tool
1. Draw complete square
2. Save with n=2 (two initial points)
3. Name: "Basic Square"

### Using the Tool
1. Select "Basic Square"
2. Draw two adjacent points
3. System completes square pattern

## Troubleshooting
| Issue | Solution |
|-------|----------|
| Corrupted tools.json | Delete file and recreate tools |
| Missing dependencies | Run `pip install -r requirements.txt` |
| Prediction errors | Increase initial points (n) value |

## Support
For assistance, please open an issue in the project repository.

## License
Open-source under MIT License.
```
