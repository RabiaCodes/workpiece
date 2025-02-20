import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def contour_overlap(contour1, contour2):
    # Create empty masks for the contours
    mask1 = np.zeros((500, 500), dtype=np.uint8)  # Adjust size accordingly
    mask2 = np.zeros((500, 500), dtype=np.uint8)

    # Draw the contours on the masks
    cv2.drawContours(mask1, [contour1], -1, 255, thickness=cv2.FILLED)
    cv2.drawContours(mask2, [contour2], -1, 255, thickness=cv2.FILLED)

    # Calculate the intersection (bitwise AND) of the two contours
    intersection = cv2.bitwise_and(mask1, mask2)

    # Calculate the union (bitwise OR) of the two contours
    union = cv2.bitwise_or(mask1, mask2)

    # Compute the overlap ratio as intersection over union
    overlap_ratio = np.sum(intersection) / np.sum(union)
    return overlap_ratio

# Function to filter contours
def filter_contours(contours, contour_boxes):
    filtered_contours = []
    for contour in contours:
        is_duplicate = False
        for existing_contour in filtered_contours:
            overlap = contour_overlap(contour, existing_contour)
            if overlap > 0.5:  # Eliminate if overlap is more than 90%
                is_duplicate = True
                break

        if not is_duplicate:
            filtered_contours.append(contour)
    return filtered_contours


# Updated process_contour to handle rectangles
def process_contour(contour, contour_image, image):
    # Approximate the contour and calculate the area
    approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
    area = cv2.contourArea(contour)

    # Initialize shape details
    shape_name = "Unknown"
    circle_specs = np.nan
    ellipse_specs = np.nan

    # Identify shapes
    if len(approx) > 4:  # More than 4 vertices might indicate a circle or ellipse
        (x, y), (MA, ma), angle = cv2.fitEllipse(contour)
        circularity = (4 * np.pi * area) / (cv2.arcLength(contour, True) ** 2)

        if 0.85 < circularity <= 1.0:  # Circle criteria
            shape_name = "Circle"
            circle_specs = (x, y, MA / 2)  # (Center x, Center y, Radius)
            cv2.drawContours(contour_image, [contour], -1, (0, 255, 0), 2)
            cv2.putText(contour_image, "Circle", (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        elif 0.6 > MA / ma > 0.4:  # Ellipse criteria
            shape_name = "Ellipse"
            ellipse_specs = (x, y, MA, ma, angle)  # (Center x, Center y, Major Axis, Minor Axis, Angle)
            cv2.drawContours(contour_image, [contour], -1, (0, 255, 0), 2)
            cv2.putText(contour_image, "Ellipse", (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

    return {
        "Shape": shape_name,
        "Contour": contour,
        "Circle Specs": circle_specs,
        "Ellipse Specs": ellipse_specs,
        "Area": area
    }

# Function to analyze detected circles
def analyze_circles(shape_data, contour_image):
    for i, circle1 in enumerate(shape_data):
        for j, circle2 in enumerate(shape_data):
            if i != j and (circle1["Shape"] == "Circle" and circle2["Shape"] == "Circle"):
                x1, y1, r1 = circle1["Circle Specs"]
                x2, y2, r2 = circle2["Circle Specs"]
                center_distance = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                if center_distance < 0.5:
                    if r2 < r1:
                        if r2 < 0.5 * r1:
                            circle1["Shape"] = "Cone"
                            cv2.putText(contour_image, "Cone", (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                        elif r2 < 0.8 * r1:
                            circle1["Shape"] = "Hole"
                            cv2.putText(contour_image, "Hole", (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

# Main function to detect shapes and create dataframe
def detect_shapes_and_create_dataframe(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)
    cv2.imshow("Thresh", thresh)
    cv2.waitKey(0)

    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour_image = np.copy(image)

    contour_boxes = []
    filtered_contours = filter_contours(contours, contour_boxes)

    shape_data = []
    for contour in filtered_contours:
        shape_info = process_contour(contour, contour_image, thresh)
        shape_data.append(shape_info)

    analyze_circles(shape_data, contour_image)

    cv2.imshow("Detected Shapes", contour_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    data = []
    for shape in shape_data:
        x, y, r, Ma, ma, angle = (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
        if not np.isnan(shape["Circle Specs"]).any():
            x, y, r = shape["Circle Specs"]
        if not np.isnan(shape["Ellipse Specs"]).any():
            x, y, Ma, ma, angle = shape["Ellipse Specs"]
        data.append({
            "Shape": shape["Shape"],
            "Area": shape["Area"],
            "Radius": r,
            "Center": (x, y),
            "Major Axis": Ma,
            "Minor Axis": ma,
            "Angle": angle,
            "Contour Points": shape["Contour"].tolist()})

    df = pd.DataFrame(data)
    print(df)
    return df

def group_and_visualize_shapes(df, original_image_path):
    # Thresholds for grouping
    radius_threshold = 5  # Radius difference allowed for grouping circles
    area_threshold = 20   # Area difference allowed for grouping circles
    ellipse_ratio_threshold = 0.1  # Ratio difference allowed for grouping ellipses
    ellipse_area_threshold = 20    # Area difference allowed for grouping ellipses

    # Group shapes (Circles and Ellipses separately)
    grouped_shapes = {"Circles": {}, "Ellipses": {}}

    # Process Circles
    circle_data = df[df["Shape"] == "Circle"].copy()
    circle_data["Group"] = -1
    circle_group_id = 0

    for index, circle in circle_data.iterrows():
        if circle_data.at[index, "Group"] == -1:  # Not yet grouped
            circle_data.at[index, "Group"] = circle_group_id
            for idx, other_circle in circle_data.iterrows():
                if idx != index and circle_data.at[idx, "Group"] == -1:
                    radius_diff = abs(circle["Radius"] - other_circle["Radius"])
                    area_diff = abs(circle["Area"] - other_circle["Area"])
                    if radius_diff <= radius_threshold and area_diff <= area_threshold:
                        circle_data.at[idx, "Group"] = circle_group_id
            circle_group_id += 1

    grouped_shapes["Circles"] = {
        f"Group_{group}": circle_data[circle_data["Group"] == group] for group in range(circle_group_id)
    }

    # Process Ellipses
    ellipse_data = df[df["Shape"] == "Ellipse"].copy()
    ellipse_data["Group"] = -1
    ellipse_group_id = 0

    for index, ellipse in ellipse_data.iterrows():
        if ellipse_data.at[index, "Group"] == -1:  # Not yet grouped
            ellipse_data.at[index, "Group"] = ellipse_group_id
            for idx, other_ellipse in ellipse_data.iterrows():
                if idx != index and ellipse_data.at[idx, "Group"] == -1:
                    ratio_diff = abs(ellipse["Major Axis"] / ellipse["Minor Axis"] -
                                     other_ellipse["Major Axis"] / other_ellipse["Minor Axis"])
                    area_diff = abs(ellipse["Area"] - other_ellipse["Area"])
                    if ratio_diff <= ellipse_ratio_threshold and area_diff <= ellipse_area_threshold:
                        ellipse_data.at[idx, "Group"] = ellipse_group_id
            ellipse_group_id += 1

    grouped_shapes["Ellipses"] = {
        f"Group_{group}": ellipse_data[ellipse_data["Group"] == group] for group in range(ellipse_group_id)
    }

    # Visualization
    image = cv2.imread(original_image_path)
    color_map = plt.cm.get_cmap("tab20", max(circle_group_id, ellipse_group_id))
    colors = [tuple([int(c * 255) for c in color_map(i)[:3]]) for i in range(max(circle_group_id, ellipse_group_id))]

    # Draw Circles and Ellipses by Contours
    for group, group_df in grouped_shapes["Circles"].items():
        group_color = colors[int(group.split("_")[1])]
        for _, row in group_df.iterrows():
            contour = np.array(row["Contour Points"], dtype=np.int32)
            cv2.drawContours(image, [contour], -1, group_color, 2)

    for group, group_df in grouped_shapes["Ellipses"].items():
        group_color = colors[int(group.split("_")[1])]
        for _, row in group_df.iterrows():
            contour = np.array(row["Contour Points"], dtype=np.int32)
            cv2.drawContours(image, [contour], -1, group_color, 2)

    # Display the grouped image
    cv2.imshow("Grouped Shapes by Contours", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return image
