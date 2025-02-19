import cv2
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import os
# import supervision as sv
from datetime import datetime
import pytz
import uuid
import xml.etree.ElementTree as ET
import numpy as np
# from shapely.geometry import Polygon
import json , random
import colorsys
from PIL import Image

# print("OpenCV version:", cv2.__version__)
# print(torch.cuda.is_available())
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MODEL_TYPE = "vit_h"
CHECKPOINT_PATH= os.getenv("CHECKPOINT_PATH")

def get_segment(image_path):
    # Initialize the model
    sam = sam_model_registry[MODEL_TYPE](checkpoint=None).to(device=DEVICE)
    # Load and preprocess the state dict
    state_dict = torch.load(CHECKPOINT_PATH)
    new_state_dict = {}
    for key in state_dict.keys():
        new_key = key.replace('sam.', '')
        new_state_dict[new_key] = state_dict[key]

    # Load the processed state dict
    sam.load_state_dict(new_state_dict)
    mask_generator = SamAutomaticMaskGenerator(sam)
    IMAGE_PATH = image_path
    image_bgr = cv2.imread(IMAGE_PATH)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    sam_result = mask_generator.generate(image_rgb)
    return sam_result   

########################################################################

# def generate_random_color():
#     """Generate a random hex color code."""
#     r = random.randint(0, 255)
#     g = random.randint(0, 255)
#     b = random.randint(0, 255)
#     return f"#{r:02X}{g:02X}{b:02X}"

# def sam_masks_to_pdf_annotations(sam_result, target_width, target_height, page_number=0):
#     """
#     Convert SAM segmentation masks to PDF annotation format with resized coordinates.
#     Each mask gets a randomly generated color.
#     Args:
#         sam_result (list): List of dictionaries containing SAM segmentation results
#         target_width (int): Target width for resizing annotations
#         target_height (int): Target height for resizing annotations
#         page_number (int): PDF page number for the annotations
#     Returns:
#         list: List of dictionaries in PDF annotation format
#     """
#     annotations = []
#     # Sort masks by area (largest to smallest)
#     sorted_masks = sorted(sam_result, key=lambda x: x['area'], reverse=True)
#     # Calculate scaling factors
#     original_size = sorted_masks[0]['segmentation'].shape
#     scale_x = target_width / original_size[1]
#     scale_y = target_height / original_size[0]
#     # Generate current date in PDF format
#     current_date = datetime.now().strftime("D:%Y%m%d%H%M%S+02'00'")
#     for idx, mask_data in enumerate(sorted_masks):
#         # Get the binary mask
#         original_mask = mask_data['segmentation'].astype(np.uint8)
#         # Find contours in the original mask
#         contours, _ = cv2.findContours(
#             original_mask,
#             cv2.RETR_EXTERNAL,
#             cv2.CHAIN_APPROX_SIMPLE
#         )
#         # Skip if no contours found
#         if not contours:
#             continue
#         # Get the largest contour
#         contour = max(contours, key=cv2.contourArea)
#         # Simplify contour to reduce number of points
#         epsilon = 0.005 * cv2.arcLength(contour, True)
#         approx = cv2.approxPolyDP(contour, epsilon, True)
#         # Apply scaling to coordinates without flipping Y coordinates
#         resized_vertices = []
#         for point in approx:
#             x = float(point[0][0] * scale_x)
#             y = float(point[0][1] * scale_y)  # No longer flipping Y coordinates
#             resized_vertices.append(f"{x:.1f},{y:.1f}")
#         vertices_str = ";" .join(resized_vertices)
#         # Calculate bounding box for the annotation
#         x_values = [float(point.split(',')[0]) for point in resized_vertices]
#         y_values = [float(point.split(',')[1]) for point in resized_vertices]
#         min_x = min(x_values)
#         max_x = max(x_values)
#         min_y = min(y_values)
#         max_y = max(y_values)
#         # Generate a random color for this annotation
#         random_color = generate_random_color()
#         # Create annotation in the required format
#         annotation = {
#             "color": random_color,
#             "creationdate": current_date,
#             "date": current_date,
#             "interior-color": random_color,
#             "name": str(uuid.uuid4()),
#             "page": page_number,
#             "rect": f"{min_x:.1f},{min_y:.1f},{max_x:.1f},{max_y:.1f}",
#             "vertices": vertices_str
#         }
#         annotations.append(annotation)
#     return annotations
# def process_sam_to_pdf_annotations(
#     sam_result,
#     target_width,
#     target_height,
#     page_number=0
# ):
#     """
#     Process SAM results and save as PDF annotation format with random colors.
#     Args:
#         sam_result (list): List of SAM segmentation results
#         target_width (int): Desired width for output annotations
#         target_height (int): Desired height for output annotations
#         output_file (str): Path to save the annotations
#         page_number (int): PDF page number
#     Returns:
#         list: List of annotation dictionaries
#     """

#     annotations = sam_masks_to_pdf_annotations(
#         sam_result,
#         target_width,
#         target_height,
#         page_number
#     )
#     # import json
#     # with open(output_file, 'w') as f:
#     #     json.dump(annotations, f, indent=4)
#     return annotations


########################################################################
def generate_random_color():
    """Generate a random color in hex format."""
    return "#{:06x}".format(random.randint(0, 0xFFFFFF))

def format_rect(bbox):
    """Convert bbox [x_min, y_min, width, height] to 'x_min,y_min,x_max,y_max' string format."""
    return f"{bbox[0]:.1f},{bbox[1]:.1f},{bbox[0] + bbox[2]:.1f},{bbox[1] + bbox[3]:.1f}"

def format_vertices(polygon):
    """Convert polygon points to 'x,y;x,y;...' string format without nesting."""
    return ";".join([f"{point[0]:.1f},{point[1]:.1f}" for point in polygon])

def get_current_date_string():
    """Generate date string in the format D:YYYYMMDDHHMMSSz'00'"""
    now = datetime.now()
    # Adjust timezone offset as needed
    timezone_offset = "+02'00'"
    date_str = f"D:{now.strftime('%Y%m%d%H%M%S')}{timezone_offset}"
    return date_str
def resize_and_convert_masks(sam_result, target_width, target_height, page_number=1):
    """
    Resize masks and convert to custom annotation format matching the provided structure.
    Args:
        sam_result (list): List of dictionaries containing SAM segmentation results
        target_width (int): Desired width for output annotations
        target_height (int): Desired height for output annotations
        page_number (int): Page number for the annotations
    Returns:
        list: List of dictionaries containing custom format annotations
    """
    annotations = []
    # Sort masks by area (largest to smallest)
    sorted_masks = sorted(sam_result, key=lambda x: x['area'], reverse=True)
    # Calculate scaling factors
    original_size = sorted_masks[0]['segmentation'].shape
    scale_x = target_width / original_size[1]
    scale_y = target_height / original_size[0]
    current_date = get_current_date_string()
    for idx, mask_data in enumerate(sorted_masks, 1):
        # Get the binary mask and resize it
        original_mask = mask_data['segmentation'].astype(np.uint8)
        mask_image = Image.fromarray(original_mask)
        resized_mask = np.array(mask_image.resize(
            (target_width, target_height),
            Image.Resampling.NEAREST
        ))
        # Find contours in the resized mask
        contours, _ = cv2.findContours(
            resized_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        if contours:
            # Get the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            # Simplify contour
            epsilon = 0.005 * cv2.arcLength(largest_contour, True)
            approx = cv2.approxPolyDP(largest_contour, epsilon, True)
            # Convert contour points to list format (without nesting)
            polygon = [float(point[0][0]) for point in approx] + [float(point[0][1]) for point in approx]
            # Restructure polygon to pairs of [x1, y1, x2, y2, ...]
            polygon_pairs = []
            for i in range(0, len(polygon)//2):
                polygon_pairs.append([polygon[i], polygon[i + len(polygon)//2]])
            # Calculate bounding box
            x_coords = [p[0] for p in polygon_pairs]
            y_coords = [p[1] for p in polygon_pairs]
            x_min, y_min = min(x_coords), min(y_coords)
            x_max, y_max = max(x_coords), max(y_coords)
            bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
            # Generate random color
            color = generate_random_color()
            annotation = {
                'color': color,
                'creationdate': current_date,
                'date': current_date,
                'interior-color': color,
                'name': str(uuid.uuid4()),
                'page': page_number,
                'rect': format_rect(bbox),
                'vertices': format_vertices(polygon_pairs)
            }
            annotations.append(annotation)
    return annotations

# def save_custom_annotations(annotations, output_file):
#     """
#     Save annotations in custom format.
#     Args:
#         annotations (list): List of annotation dictionaries
#         output_file (str): Path to output JSON file
#     """
#     with open(output_file, 'w') as f:
#         json.dump(annotations, f, indent=4)

# def process_sam_to_custom(
#     sam_result,
#     target_width,
#     target_height,
#     output_file='custom_annotations.json',
#     page_number=1
# ):
#     """
#     Process SAM results and save as custom format annotations.
#     Args:
#         sam_result (list): List of SAM segmentation results
#         target_width (int): Desired width for output annotations
#         target_height (int): Desired height for output annotations
#         output_file (str): Path to save the annotations
#         page_number (int): Page number for the annotations
#     Returns:
#         list: List of annotation dictionaries
#     """
#     annotations = resize_and_convert_masks(
#         sam_result,
#         target_width,
#         target_height,
#         page_number
#     )
#     save_custom_annotations(annotations, output_file)
#     return annotations