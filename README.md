# Sudoku Detector and Solver using Convolutional Neural Networks

The system detects Sudoku puzzles from images, extracts the grid, recognizes digits using a Convolutional Neural Network (CNN), and solves the puzzle using the Exact Cover Algorithm. The solution is then overlaid onto the original image for a seamless user experience.

## Features

- **Sudoku Grid Detection**: Utilizes OpenCV to detect and extract the Sudoku grid from images.
- **Digit Recognition**: Employs a CNN trained on a printed digit dataset to recognize digits within the grid.
- **Sudoku Solving**: Implements the Exact Cover Algorithm with backtracking to solve the puzzle efficiently.
- **Perspective Adjustment**: Corrects image distortions to ensure accurate grid extraction and solution overlay.
- **User-Friendly GUI**: Provides a graphical interface for easy image upload and solution display.

## Results

Below are examples of input images and their corresponding solved Sudoku puzzles:

| Input Image | Solved Sudoku |
|-------------|---------------|
| <img width="597" alt="Screenshot 2025-03-02 at 10 15 21 PM" src="https://github.com/user-attachments/assets/46c0503a-6d8f-4ffa-8ce1-6ef5d7a16eba" /> | <img width="597" alt="Screenshot 2025-03-02 at 10 15 37 PM" src="https://github.com/user-attachments/assets/8313c9cc-c7e5-46e8-9bbb-7ffeb19071d3" /> |
| <img width="596" alt="Screenshot 2025-03-02 at 10 16 26 PM" src="https://github.com/user-attachments/assets/5d33fc01-d717-4353-a655-a54e17317b27" /> | <img width="597" alt="Screenshot 2025-03-02 at 10 16 36 PM" src="https://github.com/user-attachments/assets/2ab8ace1-e5fc-4b6d-9870-a35af25b53ac" /> |
