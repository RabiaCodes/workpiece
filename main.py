from utils import detect_shapes_and_create_dataframe, group_and_visualize_shapes


# Call the function with the uploaded image path
image_path = "image.png"
df_shapes = detect_shapes_and_create_dataframe(image_path)
output_image = group_and_visualize_shapes(df_shapes, image_path)
