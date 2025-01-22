import cv2
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import os
import supervision as sv

print("OpenCV version:", cv2.__version__)
print(torch.cuda.is_available())
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MODEL_TYPE = "vit_h"
CHECKPOINT_PATH= "D:\Melbin\G5_ML_model\SAM_model\Models\sam_vit_h_4b8939.pth"
print("Model checkpoint path:", CHECKPOINT_PATH)
def get_segment(image_path):
    sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device=DEVICE)
    mask_generator = SamAutomaticMaskGenerator(sam)

    image_bgr = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    sam_result = mask_generator.generate(image_rgb)

    return sam_result

########################################################################

import numpy as np
from shapely.geometry import Polygon
import json

def mask_to_polygons(mask_results):
    """
    Convert segmentation masks to polygon annotations.
    
    Args:
        mask_results (list): List of dictionaries containing segmentation results
        
    Returns:
        list: List of dictionaries containing polygon annotations
    """
    annotations = []
    
    # Sort masks by area (largest to smallest)
    sorted_masks = sorted(mask_results, key=lambda x: x['area'], reverse=True)
    
    for idx, mask_data in enumerate(sorted_masks):
        # Get the binary mask
        mask = mask_data['segmentation'].astype(np.uint8)
        
        # Find contours in the mask
        contours, _ = cv2.findContours(mask, 
                                     cv2.RETR_EXTERNAL, 
                                     cv2.CHAIN_APPROX_SIMPLE)
        
        # Convert contours to polygons
        polygons = []
        for contour in contours:
            # Simplify contour to reduce number of points
            epsilon = 0.005 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Convert to list of [x,y] coordinates
            polygon = [[float(point[0][0]), float(point[0][1])] for point in approx]
            
            # Only add polygons with enough points
            if len(polygon) >= 3:  # minimum 3 points to form a polygon
                polygons.append(polygon)
        
        if polygons:
            # Calculate bounding box
            all_points = np.concatenate([np.array(poly) for poly in polygons])
            x_min, y_min = np.min(all_points, axis=0)
            x_max, y_max = np.max(all_points, axis=0)
            
            annotation = {
                'id': idx,
                'image_id': mask_data.get('image_id', 0),
                'category_id': mask_data.get('category_id', 1),
                'segmentation': polygons,
                'area': float(mask_data['area']),
                'bbox': [float(x_min), float(y_min), 
                        float(x_max - x_min), float(y_max - y_min)],
                'iscrowd': 0
            }
            
            annotations.append(annotation)
    
    return annotations

def save_annotations(annotations, output_file):
    """
    Save annotations to a COCO-format JSON file.
    
    Args:
        annotations (list): List of annotation dictionaries
        output_file (str): Path to output JSON file
    """
    coco_format = {
        'info': {
            'description': 'Converted from segmentation masks',
            'version': '1.0',
        },
        'images': [{'id': 0, 'file_name': 'image.jpg'}],  
        'categories': [{'id': 1, 'name': 'object'}],  
        'annotations': annotations
    }
    
    with open(output_file, 'w') as f:
        json.dump(coco_format, f, indent=2)


def process_segmentation_masks(sam_result, output_file='annotations.json'):
    """
    Process SAM results and save as polygon annotations.
    
    Args:
        sam_result (list): List of SAM segmentation results
        output_file (str): Path to save the annotations
    """
    annotations = mask_to_polygons(sam_result)
    save_annotations(annotations, output_file)
    return annotations


########################################################################

from datetime import datetime
import pytz
import uuid
import xml.etree.ElementTree as ET

def create_xfdf_from_masks(mask_results, output_file='output.xfdf'):
    """
    Convert segmentation masks to XFDF format for PDF annotations.
    
    Args:
        mask_results (list): List of dictionaries containing segmentation results
        output_file (str): Path to output XFDF file
    """
    # Create the root XFDF element
    root = ET.Element("xfdf", {
        "xmlns": "http://ns.adobe.com/xfdf/",
        "xml:space": "preserve"
    })
    
    # Add pdf-info element
    pdf_info = ET.SubElement(root, "pdf-info", {
        "xmlns": "http://www.pdftron.com/pdfinfo",
        "version": "2"
    })
    
    # Add fields element
    fields = ET.SubElement(root, "fields")
    
    # Add annots element
    annots = ET.SubElement(root, "annots")
    
    # Get current timestamp in PDF format
    now = datetime.now(pytz.utc)
    pdf_date = now.strftime("D:%Y%m%d%H%M%S+00'00'")
    
    # Process each mask
    for idx, mask_data in enumerate(mask_results):
        # Convert mask to polygon vertices
        contours, _ = cv2.findContours(
            mask_data['segmentation'].astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        for contour in contours:
            # Simplify contour
            epsilon = 0.005 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Convert points to vertices string
            vertices = ";".join(
                f"{float(point[0][0])},{float(point[0][1])}"
                for point in approx
            )
            
            if len(approx) >= 3:  # Only create polygon if we have at least 3 points
                # Calculate bounding rect for the 'rect' attribute
                x_min = float(np.min(approx[:, :, 0]))
                y_min = float(np.min(approx[:, :, 1]))
                x_max = float(np.max(approx[:, :, 0]))
                y_max = float(np.max(approx[:, :, 1]))
                
                # Create polygon annotation
                polygon = ET.SubElement(annots, "polygon", {
                    "color": "#239123",  # Green color
                    "creationdate": pdf_date,
                    "date": pdf_date,
                    "flags": "print",
                    "interior-color": "#239123",
                    "name": str(uuid.uuid4()),
                    "opacity": "1",
                    "page": "0",
                    "rect": f"{x_min},{y_min},{x_max},{y_max}",
                    "subject": "Polygon",
                    "title": str(idx),
                    "width": "0.5"
                })
                
                # Add vertices
                vertices_elem = ET.SubElement(polygon, "vertices")
                vertices_elem.text = vertices
    
    # Add pages element with defmtx
    pages = ET.SubElement(root, "pages")
    ET.SubElement(pages, "defmtx", {
        "matrix": "1,0,0,-1,0,842"
    })
    
    # Create XML string
    tree = ET.ElementTree(root)
    
    # Add XML declaration
    xml_str = '<?xml version="1.0" encoding="UTF-8" standalone="no"?>'
    
    # Convert tree to string and combine with declaration
    tree_str = ET.tostring(root, encoding='unicode')
    complete_xml = xml_str + tree_str
    
    # Save to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(complete_xml)
    
    return complete_xml

def process_masks_to_xfdf(sam_result, output_file='annotations.xfdf'):
    """
    Process SAM results and save as XFDF annotations.
    
    Args:
        sam_result (list): List of SAM segmentation results
        output_file (str): Path to save the XFDF file
    """
    xfdf_content = create_xfdf_from_masks(sam_result, output_file)
    return xfdf_content

