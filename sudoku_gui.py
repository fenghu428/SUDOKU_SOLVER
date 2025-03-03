import cv2
import os
import numpy as np
from tkinter import *
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from keras.models import load_model
from image_cv import *
from sudoku_solver import solve_wrapper

# check if the image contains a Sudoku grid contour
def check_contour(img):
    prep_img = image_preprocess(img)
    frame, contour, contour_line, thresh = extract_sudoku_grid(prep_img)
    contour_exist = len(contour) == 4

    return contour_exist,prep_img, frame, contour, contour_line, thresh

# predict the Sudoku grid from image
def predict_sudoku(img,model):
    numbers_image, number_stats, number_centroids = get_numbers(img) # get numbers and their positions in the grid
    centered_numbers_image, number_positions_mask = position_num_in_centre(numbers_image, number_stats, number_centroids) # put numbers in centre of cell
    predicted_numbers_grid = predict_numbers(centered_numbers_image, number_positions_mask, model)
    # solve the sudoku
    solved_grid = solve_wrapper(predicted_numbers_grid.copy())

    return numbers_image, centered_numbers_image, predicted_numbers_grid, solved_grid

# overlay the digits on the original image
def highlight_solution(background_mask, original_image, predicted_digits, solved_digits, corner_points):
    # put the solved numbers on a separate mask
    solved_numbers_image = display_numbers(background_mask, predicted_digits, solved_digits)
    
    # overlay the solved numbers onto the original image
    overlay_image = restore_image(original_image, solved_numbers_image, corner_points)
    
    # blend the original image and the numbers
    final_image = cv2.addWeighted(original_image, 1, overlay_image, 1, 0)
    
    return final_image, solved_numbers_image

if not os.path.exists('solved_images'):
    os.makedirs('solved_images')

model = load_model('model.h5')

# function to upload the image and display it in the GUI
def upload_image():
    global img_path, img_display
    img_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if img_path:
        try:
            img = Image.open(img_path)
            img = img.resize((500, 500), Image.Resampling.LANCZOS)  # resize for display
            img_display = ImageTk.PhotoImage(img)
            panel.configure(image=img_display)
            panel.image = img_display
        except IOError:
            messagebox.showerror("Image Load Error", "Error loading image!")

# load test images for app_photo from individual image files or directories
def load_sudoku_image(image_path):
    loaded_images = []
    if os.path.isdir(image_path):  # check if the path is a directory
        for filename in os.listdir(image_path):
            full_path = os.path.join(image_path, filename)
            image_array = cv2.imread(full_path)
            if image_array is not None:
                resized_image = cv2.resize(image_array, (540, 540), interpolation=cv2.INTER_LINEAR)
                loaded_images.append(resized_image)
    else:  # just a single file, not a directory
        image_array = cv2.imread(image_path)
        if image_array is not None:
            resized_image = cv2.resize(image_array, (540, 540), interpolation=cv2.INTER_LINEAR)
            loaded_images.append(resized_image)

    return np.array(loaded_images)

# function to solve the Sudoku puzzle using the uploaded image and display the result
def solve_sudoku():
    try:
        sudoku_images = load_sudoku_image(img_path) 
        for image in sudoku_images:
            if image is None:
                continue

            # check for contours in the image
            contour_found, processed_image, perspective_frame, contour, contour_outline, threshold_image = check_contour(image)
            if not contour_found:
                continue

            corners = calculate_contour_corners(contour) # calculate corners from the largest contour
            flattened_image = transform_image_to_array(perspective_frame, (450, 450), corners)
            digits_image, centered_digits_image, predicted_grid, solved_grid = predict_sudoku(flattened_image, model) # predict and solve Sudoku
            mask = np.zeros_like(flattened_image) 
            solved_image, solution_highlighted_image = highlight_solution(mask, image, predicted_grid, solved_grid, corners)
            
            cv2.imwrite('solved_images/solved_sudoku.jpg', solved_image)
            show_result('solved_images/solved_sudoku.jpg')

    except Exception as e:
        messagebox.showerror("Processing Error", str(e))

# display the processed image in the GUI
def show_result(image_path):
    solved_img = Image.open(image_path)
    solved_img = solved_img.resize((500, 500), Image.Resampling.LANCZOS)
    solved_display = ImageTk.PhotoImage(solved_img)
    panel.configure(image=solved_display)
    panel.image = solved_display

# setup the main window
window = Tk()
window.title("Sudoku Solver")
window.geometry('600x1000')

# configure the grid layout to center the content
window.columnconfigure(0, weight=1)  # left padding column
window.columnconfigure(1, weight=1)  # main content column
window.columnconfigure(2, weight=1)  # right padding column

# create a label and place it in the center column
panel = Label(window)
panel.grid(column=1, row=0)  # placed in the center column

# create buttons and place them in the center column
upload_btn = Button(window, text="Upload Sudoku Image", command=upload_image)
upload_btn.grid(column=1, row=1)  # centered horizontally

solve_btn = Button(window, text="Solve Sudoku", command=solve_sudoku)  # updated function name for clarity
solve_btn.grid(column=1, row=2)  # centered horizontally

# run the main event loop
window.mainloop()
