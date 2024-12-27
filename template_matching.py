import math
import cv2
import numpy as np
import torch
from torchvision.transforms import ToTensor
import config
from data import get_predicted_points, pair_marking_points, calc_point_squre_dist, pass_through_third_point
from model import DirectionalPointDetector
from util import Timer

# Step 1: Load Pretrained Model and Perform Initial Inference
def detect_marking_points(detector, image, thresh, device):
    """
    Given an image, predict parking slots using the detector model.
    Args:
        detector: The trained DirectionalPointDetector model.
        image: Input image.
        thresh: Confidence threshold for predictions.
        device: Torch device (CPU or CUDA).
    Returns:
        predictions: List of predicted marking points with confidence.
    """
    prediction = detector(preprocess_image(image).to(device))
    return get_predicted_points(prediction[0], thresh)

# Step 2: Create Edge Template from Model Predictions
def create_edge_template(image, slots):
    """
    Create an edge template based on the predicted parking slots.
    Args:
        image: Original image.
        slots: List of slot coordinates [(x1, y1), (x2, y2), ...]
    Returns:
        template: Binary edge template image.
    """
    template = np.zeros_like(image, dtype=np.uint8)
    for slot in slots:
        for i in range(len(slot)):
            pt1 = tuple(slot[i])
            pt2 = tuple(slot[(i + 1) % len(slot)])
            cv2.line(template, pt1, pt2, 255, 2)  # Draw edges in white
    return template

# Step 3: Extract Edges and Directions from the Original Image
def extract_edges_and_directions(image):
    """
    Extract edges and corresponding edge orientations from the image.
    Args:
        image: Input image.
    Returns:
        edges: Binary edge map.
        directions: Gradient orientations at edge points.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    
    # Calculate gradients for orientation extraction
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    directions = cv2.phase(grad_x, grad_y, angleInDegrees=True)
    
    return edges, directions

# Step 4: Directional Chamfer Matching
def directional_chamfer_matching(template, image_edges, image_directions):
    """
    Perform directional chamfer matching to refine the position and orientation of the parking slots.
    Args:
        template: Binary edge template of the parking slot.
        image_edges: Edge map of the input image.
        image_directions: Gradient directions at each pixel in the input image.
    Returns:
        refined_template: Refined template after matching.
    """
    # Compute distance transform of the image edges
    dist_transform = cv2.distanceTransform(255 - image_edges, cv2.DIST_L2, 5)
    
    best_score = float('inf')
    best_template = template
    
    # Try small rotations (-10 to +10 degrees) to find the best match
    for angle in range(-10, 11, 2):
        # Rotate template
        center = (template.shape[1] // 2, template.shape[0] // 2)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_template = cv2.warpAffine(template, rot_mat, (template.shape[1], template.shape[0]))
        
        # Calculate chamfer distance
        chamfer_score = cv2.matchTemplate(dist_transform, rotated_template, cv2.TM_CCOEFF_NORMED)
        score = np.sum(chamfer_score)
        
        # Update best match
        if score < best_score:
            best_score = score
            best_template = rotated_template

    return best_template

# Step 5: Preprocess Image
def preprocess_image(image):
    """
    Preprocess numpy image to torch tensor.
    Args:
        image: Input image.
    Returns:
        tensor: Torch tensor of the image.
    """
    if image.shape[0] != 512 or image.shape[1] != 512:
        image = cv2.resize(image, (512, 512))
    return torch.unsqueeze(ToTensor()(image), 0)

# Step 6: Main Function to Perform Inference and Refinement
def main():
    args = config.get_parser_for_inference().parse_args()
    # Load the pretrained model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    detector = DirectionalPointDetector(3, args.depth_factor, config.NUM_FEATURE_MAP_CHANNEL).to(device)
    detector.load_state_dict(torch.load(args.detector_weights, map_location=device))
    detector.eval()
    
    # Load the image
    image = cv2.imread("frame_002637.jpg")
    
    # Step 1: Predict parking slots
    pred_points = detect_marking_points(detector, image, args.thresh, device)
    
    # Step 2: Create edge template from predictions
    slots = inference_slots(pred_points)
    template = create_edge_template(image, slots)
    
    # Step 3: Extract edges and directions from the original image
    image_edges, image_directions = extract_edges_and_directions(image)
    
    # Step 4: Perform directional chamfer matching
    refined_template = directional_chamfer_matching(template, image_edges, image_directions)
    
    # Step 5: Display the results
    plot_points(image, pred_points)
    plot_slots(image, pred_points, slots)
    cv2.imshow("Original Image", image)
    cv2.imshow("Initial Template", template)
    cv2.imshow("Refined Template", refined_template)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
