import cv2
import os
import numpy as np
from pathlib import Path
import torch
from transformers import DetrImageProcessor, DetrForObjectDetection

def detect_traffic_light_color(roi):
    # Get dimensions of ROI
    height, width = roi.shape[:2]
    
    # If ROI is too small, return unknown
    if height < 10 or width < 10:
        return 'unknown'
    
    # Focus on the top portion of traffic light where the signal is
    # For vertical traffic lights (taller than wide)
    if height > width:
        # Take the top third of the traffic light
        y_offset = int(height * 0.2)
        roi_signal = roi[:y_offset, :]
    else:
        # For horizontal traffic lights, take the entire image
        roi_signal = roi.copy()
    
    # Convert ROI to HSV color space
    hsv = cv2.cvtColor(roi_signal, cv2.COLOR_BGR2HSV)
    
    # Define color ranges in HSV with broader ranges for red
    # Red color ranges (red wraps around in HSV)
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    
    # Yellow color range
    lower_yellow = np.array([15, 70, 50])
    upper_yellow = np.array([35, 255, 255])
    
    # Green color range
    lower_green = np.array([35, 70, 50])
    upper_green = np.array([90, 255, 255])
    
    # Create masks for each color
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    
    # Apply morphological operations to reduce noise
    kernel = np.ones((3, 3), np.uint8)
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)
    mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_OPEN, kernel)
    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel)
    
    # Calculate the percentage of each color
    total_pixels = roi_signal.shape[0] * roi_signal.shape[1]
    if total_pixels == 0:
        return 'unknown'
    
    red_ratio = np.sum(mask_red > 0) / total_pixels
    yellow_ratio = np.sum(mask_yellow > 0) / total_pixels
    green_ratio = np.sum(mask_green > 0) / total_pixels
    
    # Adjust threshold for color detection
    # Increase threshold for red to reduce false positives
    red_threshold = 0.15
    other_threshold = 0.15
    
    # Boosting red detection by checking for brightness and color intensity
    # If we find bright pixels in the red mask
    if red_ratio > red_threshold:
        red_pixels = cv2.bitwise_and(roi_signal, roi_signal, mask=mask_red)
        if np.mean(red_pixels) > 50:  # If average brightness is significant
            red_ratio *= 1.2  # Boost red ratio
    
    # Additional validation for red detection to reduce false positives
    if red_ratio > red_threshold:
        # Check the spatial distribution of red pixels (should be concentrated)
        non_zero_coords = cv2.findNonZero(mask_red)
        if non_zero_coords is not None and len(non_zero_coords) > 10:
            x, y, w, h = cv2.boundingRect(non_zero_coords)
            compactness = (w * h) / total_pixels
            if compactness > 0.3:  # Red pixels should form a compact region
                red_ratio *= 1.1  # Boost for well-formed red signals
            else:
                red_ratio *= 0.8  # Penalize scattered red pixels
    
    # Detect the dominant color
    if red_ratio > red_threshold and red_ratio > yellow_ratio and red_ratio > green_ratio:
        return 'red'
    elif yellow_ratio > other_threshold and yellow_ratio > red_ratio and yellow_ratio > green_ratio:
        return 'yellow'
    elif green_ratio > other_threshold and green_ratio > red_ratio and green_ratio > yellow_ratio:
        return 'green'
    
    # Check for significant red even if not dominant - higher threshold to reduce false positives
    if red_ratio > red_threshold * 0.9:
        # Additional validation: check for roundness which is common in traffic lights
        non_zero_coords = cv2.findNonZero(mask_red)
        if non_zero_coords is not None and len(non_zero_coords) > 15:
            # Check if red area forms a circular or compact shape
            x, y, w, h = cv2.boundingRect(non_zero_coords)
            aspect_ratio = float(w) / h if h > 0 else 0
            if 0.7 < aspect_ratio < 1.3:  # Close to circular
                return 'red'
    
    return 'unknown'

def detect_traffic_lights():
    # Load DETR model and processor
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Create output directory for annotated images
    output_dir = Path('detected_images')
    output_dir.mkdir(exist_ok=True)
    
    # Color mapping for visualization
    color_map = {
        'red': (0, 0, 255),    # BGR format: Red
        'yellow': (0, 255, 255),  # BGR format: Yellow
        'green': (0, 255, 0),  # BGR format: Green
        'unknown': (128, 128, 128)  # BGR format: Gray
    }
    
    # Process all images in the image folder
    image_dir = Path('image')
    for img_path in image_dir.glob('*'):
        if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            # Read the image
            image = cv2.imread(str(img_path))
            
            # Convert BGR to RGB for DETR
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process image for DETR with higher threshold
            inputs = processor(images=rgb_image, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Perform detection
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Process detections with lower threshold for better recall
            target_sizes = torch.tensor([rgb_image.shape[:2]]).to(device)
            results = processor.post_process_object_detection(
                outputs, target_sizes=target_sizes, threshold=0.4
            )[0]
            
            # Create a copy of the image for drawing
            annotated_image = image.copy()
            
            # Track if any red traffic lights were detected
            red_lights_detected = False
            
            # Process each detection
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                # Get class name
                class_name = model.config.id2label[label.item()]
                
                # Only process if it's a traffic light
                if "traffic light" in class_name.lower():
                    # Get box coordinates
                    x1, y1, x2, y2 = box.tolist()
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                    
                    # Make sure the box is valid
                    if x2 <= x1 or y2 <= y1 or x1 < 0 or y1 < 0 or x2 >= image.shape[1] or y2 >= image.shape[0]:
                        continue
                    
                    # Extract ROI (Region of Interest)
                    roi = image[y1:y2, x1:x2]
                    
                    # Check if ROI is empty
                    if roi.size == 0:
                        continue
                    
                    # Detect the color of the traffic light
                    light_color = detect_traffic_light_color(roi)
                    
                    # Get the appropriate color and label
                    color = color_map[light_color]
                    label = f"{light_color.capitalize()} Traffic Light"
                    
                    # Track red light detection
                    if light_color == 'red':
                        red_lights_detected = True
                    
                    # Draw bounding box
                    cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
                    
                    # Add confidence text
                    text = f'{label}: {score:.2f}'
                    cv2.putText(annotated_image, text, 
                              (x1, y1 - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 
                              0.5, color, 2)
            
            # Add a global indicator for red traffic light
            if red_lights_detected:
                cv2.putText(annotated_image, 'RED LIGHT DETECTED', 
                          (20, 40),
                          cv2.FONT_HERSHEY_SIMPLEX, 
                          1, (0, 0, 255), 2)
            
            # Save the annotated image
            output_path = output_dir / f'detected_{img_path.name}'
            cv2.imwrite(str(output_path), annotated_image)
            print(f'Processed {img_path.name} - Saved to {output_path}')

if __name__ == '__main__':
    detect_traffic_lights() 