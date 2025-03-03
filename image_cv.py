import cv2
import numpy as np
import tensorflow as tf

# pre-process the input image by applying Gaussian blur and converting it to grayscale
def image_preprocess(img):
    blurred = cv2.GaussianBlur(img, (3, 3), 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    return gray

# extract the sudoku grid from the input image, return the biggest contour
def extract_sudoku_grid(image):
    frame_mask = np.zeros(image.shape, np.uint8)

    thresh = cv2.adaptiveThreshold(image, 255, 0, 1, 9, 5)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    largest_contour = []
    result_image = []
    max_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        approx_vertices = cv2.approxPolyDP(contour, 0.01 * perimeter, True)
        if (len(approx_vertices) == 4) and (area > max_area) and (area > 40000):
            max_area = area
            largest_contour = approx_vertices
    if len(largest_contour) > 0:
        cv2.drawContours(frame_mask, [largest_contour], 0, 255, -1)
        cv2.drawContours(frame_mask, [largest_contour], 0, 0, 2)
        result_image = cv2.bitwise_and(image, frame_mask)
    return result_image, largest_contour, frame_mask, thresh

# obtain the arranged coordinates of the corners in the Sudoku grid
def calculate_contour_corners(contour):
    reshaped_contour = contour.reshape(len(contour), 2)
    sum_vectors = reshaped_contour.sum(axis=1)
    intermediate_contour = np.delete(reshaped_contour, [np.argmax(sum_vectors), np.argmin(sum_vectors)], axis=0)

    corners = np.float32([
        reshaped_contour[np.argmin(sum_vectors)],  # Min sum vector point
        intermediate_contour[np.argmax(intermediate_contour[:, 0])],  # Max x after deleting extremes
        intermediate_contour[np.argmin(intermediate_contour[:, 0])],  # Min x after deleting extremes
        reshaped_contour[np.argmax(sum_vectors)]   # Max sum vector point
    ])

    return corners

# apply perspective transformation to transform image into a numPy array
def transform_image_to_array(img, shape, corners):
    pt = np.float32(
        [[0, 0], [shape[0], 0], [0, shape[1]], [shape[0], shape[1]]])  
    #  keep the original structure with PT

    matrix = cv2.getPerspectiveTransform(corners, pt)
    result = cv2.warpPerspective(img, matrix, (shape[0], shape[1]))

    return result

# extract numbers out of the image with countour coordinates
def get_numbers(img):
    processed_image = prepare_numbers(img)
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(processed_image)
    visualization = np.zeros_like(processed_image, np.uint8)

    number_centroids = []
    number_stats = []

    for i, stat in enumerate(stats):
        if i == 0:
            continue
        # conditions for selecting valid number regions
        area = stat[4]
        width = stat[2]
        height = stat[3]
        x_position = stat[0]
        y_position = stat[1]
        aspect_ratio = int(height / width)

        if area > 50 and width in range(5, 40) and height in range(5, 40) and x_position > 0 and y_position > 0 and aspect_ratio in range(1, 5):
            visualization[labels == i] = 255
            number_centroids.append(centroids[i])
            number_stats.append(stat)

    number_stats = np.array(number_stats)
    number_centroids = np.array(number_centroids)
    return visualization, number_stats, number_centroids


# use thresholing and morphology to process numbers
def prepare_numbers(img):
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 9, 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=1)

    return img

# position numbers in the centre
def position_num_in_centre(image, bounding_boxes, centroids):
    centered_numbers_image = np.zeros_like(image, np.uint8)
    number_presence_mask = np.zeros((9, 9), dtype='uint8')
    for index, bbox in enumerate(bounding_boxes):
        left, top, width, height, area = bbox
        # calculate new top-left corner to center the number in its grid cell
        new_left = int((left // 50) * 50 + ((50 - width) / 2))
        new_top = int((top // 50) * 50 + ((50 - height) / 2))
        
        # place the number in the new centered position
        centered_numbers_image[new_top:new_top + height, new_left:new_left + width] = \
            image[top:top + height, left:left + width]
        
        # calculate grid coordinates based on the centroid of the number
        grid_y = int(np.round((centroids[index][0] + 5) / 50, 1))
        grid_x = int(np.round((centroids[index][1] + 5) / 50, 1))
        number_presence_mask[grid_x, grid_y] = 1  # Mark the cell as filled

    return centered_numbers_image, number_presence_mask

# crop and inverse the image
def crop_and_inverse_image(img):
    cropped_img = img[5:img.shape[0] - 5, 5:img.shape[0] - 5]
    resized = cv2.resize(cropped_img, (40, 40))
    return resized

# predict the numbers
def predict_numbers(numbers_image, mask_matrix, model):
    prediction_inputs = []
    
    for row in range(9):
        for col in range(9):
            # if there is a number to predict, mark as 1
            if mask_matrix[row, col] == 1:
                cell_image = numbers_image[50 * row: (50 * row) + 50, 50 * col: (50 * col) + 50] # extract the cell segment from the original image 
                cell_image = crop_and_inverse_image(cell_image)
                cell_image = cell_image / 255  # image data normalization
                cell_image = cell_image.reshape(1, 40, 40, 1)
                prediction_inputs.append(cell_image)
    
    # use modal for prediction
    all_predictions = model.predict(tf.reshape(np.array(prediction_inputs), (np.sum(mask_matrix), 40, 40, 1)))
    # confidence level for each prediction
    prediction_confidence = [np.max(prediction) for prediction in all_predictions]
    predicted_digits = list(map(np.argmax, all_predictions))
    
    flat_mask_matrix = list(mask_matrix.flatten())

    # replace masked cells with predicted values
    index = 0
    for number_index, value in enumerate(flat_mask_matrix):
        if value == 1:
            flat_mask_matrix[number_index] = predicted_digits[index]
            index += 1

    # convert the list back to a 9x9 grid
    final_matrix = np.array(flat_mask_matrix).reshape(9, 9)
    return final_matrix


# show image of solved sudoku with numbers on image
def display_numbers(image, initial_grid, solved_grid, text_color=(0, 255, 0)):
    cell_width = int(image.shape[1] / 9)
    cell_height = int(image.shape[0] / 9)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  # convert grayscale to BGR for coloring text

    for row in range(9):
        for col in range(9):
            if initial_grid[col, row] == 0:  # only draw where there were no initial numbers
                number_text = str(solved_grid[col, row])
                text_x = row * cell_width + int(cell_width / 2) - int(cell_width / 4)
                text_y = int((col + 0.7) * cell_height)
                cv2.putText(image, number_text, (text_x, text_y),
                            cv2.FONT_HERSHEY_COMPLEX, 1, text_color,
                            1, cv2.LINE_AA)
    return image

# show image of solved sudoku with numbers on the original image
def restore_image(original_image, masked_numbers_image, corner_points, target_height=450, target_width=450):
    # define points in the final image that correspond to the corners of the desired perspective
    destination_points = np.float32([[0, 0], [target_width, 0], [0, target_height], [target_width, target_height]])
    # points in the original image where the numbers will be placed
    source_points = np.float32([corner_points[0], corner_points[1], corner_points[2], corner_points[3]])
    
    transformation_matrix = cv2.getPerspectiveTransform(destination_points, source_points)
    inversed_perspective_image = cv2.warpPerspective(masked_numbers_image, transformation_matrix, 
                                                     (original_image.shape[1], original_image.shape[0]))
    
    return inversed_perspective_image

# draw the corners
def draw_corners(image, corner_points):
    # loop through each corner point and draw a circle at each point
    for corner in corner_points:
        x_coordinate, y_coordinate = corner  
        # draw a green circle at each corner point
        cv2.circle(image, (int(x_coordinate), int(y_coordinate)), radius=2, color=(0, 255, 0), thickness=-1)
    return image

# search for rectangles 
def draw_rectangle(image, iteration):
    # calculate the corners of the rectangle based on the iteration number
    top_left_corner = (75 + (2 * iteration), 75 + (2 * iteration))
    bottom_right_corner = (725 - (2 * iteration), 525 - (2 * iteration))
    
    # draw a green rectangle on the image
    cv2.rectangle(image, top_left_corner, bottom_right_corner, (0, 255, 0), 2)

    return image, top_left_corner[0]