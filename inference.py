"""Inference demo of directional point detector."""
import math
import cv2 as cv
import numpy as np
import torch
from torchvision.transforms import ToTensor
import config
from data import get_predicted_points, pair_marking_points, calc_point_squre_dist, pass_through_third_point
from model import DirectionalPointDetector
from util import Timer

from data.struct import MarkingPoint
# shape < 0.5: T, >=0.5 L
# Define templates globally to avoid redefining them in every function call
templates = {}
roi_size = 50
template_size = roi_size * 2
# T-Shape Template
t_shape_template = np.zeros((template_size, template_size), dtype=np.float32)
cv.line(t_shape_template, (roi_size, roi_size-50), (roi_size, roi_size+50), 255, 2)  # Horizontal line
cv.line(t_shape_template, (roi_size, roi_size), (roi_size+50, roi_size), 255, 2)     
templates['t_shape'] = t_shape_template / 255.0 # binarize

# L-Shape Template
l_shape_template = np.zeros((template_size, template_size), dtype=np.float32)
cv.line(l_shape_template, (roi_size, roi_size), (roi_size + 50, roi_size), 255, 2)
cv.line(l_shape_template, (roi_size, roi_size), (roi_size, roi_size - 50), 255, 2)
templates['l_shape'] = l_shape_template / 255.0 # binarize 


def plot_points(image, pred_points):
    """Plot marking points on the image."""
    if not pred_points:
        return
    height = image.shape[0]
    width = image.shape[1]
    for confidence, marking_point in pred_points:
        p0_x = width * marking_point.x - 0.5
        p0_y = height * marking_point.y - 0.5
        cos_val = math.cos(marking_point.direction)
        sin_val = math.sin(marking_point.direction)
        p1_x = p0_x + 150*cos_val
        p1_y = p0_y + 150*sin_val
        p2_x = p0_x - 50*sin_val
        p2_y = p0_y + 50*cos_val
        p3_x = p0_x + 50*sin_val
        p3_y = p0_y - 50*cos_val
        p0_x = int(round(p0_x))
        p0_y = int(round(p0_y))
        p1_x = int(round(p1_x))
        p1_y = int(round(p1_y))
        p2_x = int(round(p2_x))
        p2_y = int(round(p2_y))
        cv.line(image, (p0_x, p0_y), (p1_x, p1_y), (0, 0, 255), 2)
        cv.putText(image, str(confidence), (p0_x, p0_y),
                   cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 0))
        if marking_point.shape > 0.5:
            cv.line(image, (p0_x, p0_y), (p2_x, p2_y), (0, 0, 255), 2)
        else:
            p3_x = int(round(p3_x))
            p3_y = int(round(p3_y))
            cv.line(image, (p2_x, p2_y), (p3_x, p3_y), (0, 0, 255), 2)


def plot_slots(image, pred_points, slots):
    """Plot parking slots on the image."""
    if not pred_points or not slots:
        return

    # Extract marking points from prediction
    marking_points = list(list(zip(*pred_points))[1])
    height, width = image.shape[:2]

    for slot in slots:
        point_a = marking_points[slot[0]]
        point_b = marking_points[slot[1]]

        # Extract coordinates for points A and B
        p0_x = width * point_a.x - 0.5
        p0_y = height * point_a.y - 0.5
        p1_x = width * point_b.x - 0.5
        p1_y = height * point_b.y - 0.5

        # Calculate direction vector from orientation angle (assuming the direction attribute is in radians)
        orientation_a = point_a.direction  # Assuming both points have the same orientation
        direction_a = np.array([math.cos(orientation_a), math.sin(orientation_a)])

        orientation_b = point_b.direction  # Assuming both points have the same orientation
        direction_b = np.array([math.cos(orientation_b), math.sin(orientation_b)])
        # Normalize direction vector
        direction_a = direction_a / np.linalg.norm(direction_a)
        direction_b = direction_b / np.linalg.norm(direction_b)


        # Calculate distance to decide on separator length
        distance = calc_point_squre_dist(point_a, point_b)

        if config.VSLOT_MIN_DIST <= distance <= config.VSLOT_MAX_DIST:  # Vertical slot
            separating_length = config.LONG_SEPARATOR_LENGTH
        elif config.HSLOT_MIN_DIST <= distance <= config.HSLOT_MAX_DIST:  # Horizontal slot
            separating_length = config.SHORT_SEPARATOR_LENGTH
        else:
            # If distance does not match any condition, skip drawing
            continue

        # Compute p2 and p3 based on direction vector
        p2_x = p0_x + height * separating_length * direction_a[0]
        p2_y = p0_y + height * separating_length * direction_a[1]
        p3_x = p1_x + height * separating_length * direction_b[0]
        p3_y = p1_y + height * separating_length * direction_b[1]

        # Convert coordinates to integer for drawing
        p0_x, p0_y = int(round(p0_x)), int(round(p0_y))
        p1_x, p1_y = int(round(p1_x)), int(round(p1_y))
        p2_x, p2_y = int(round(p2_x)), int(round(p2_y))
        p3_x, p3_y = int(round(p3_x)), int(round(p3_y))

        # Draw the parking slot lines
        # Entrance line
        cv.line(image, (p0_x, p0_y), (p1_x, p1_y), (255, 0, 0), 2)
        # Side boundaries extending along the orientation direction
        cv.line(image, (p0_x, p0_y), (p2_x, p2_y), (255, 0, 0), 2)
        cv.line(image, (p1_x, p1_y), (p3_x, p3_y), (255, 0, 0), 2)

    return image



def preprocess_image(image):
    """Preprocess numpy image to torch tensor."""
    if image.shape[0] != 512 or image.shape[1] != 512:
        image = cv.resize(image, (512, 512))
    return torch.unsqueeze(ToTensor()(image), 0)


def detect_marking_points(detector, image, thresh, device):
    """Given image read from opencv, return detected marking points."""
    prediction = detector(preprocess_image(image).to(device))
    return get_predicted_points(prediction[0], thresh)


def inference_slots(marking_points):
    """Inference slots based on marking points."""
    num_detected = len(marking_points)
    slots = []
    for i in range(num_detected - 1):
        for j in range(i + 1, num_detected):
            point_i = marking_points[i]
            point_j = marking_points[j]
            # Step 1: length filtration.
            distance = calc_point_squre_dist(point_i, point_j)
            # print('->vslot min dist:',config.VSLOT_MIN_DIST,',vslot max dist:',config.VSLOT_MAX_DIST)
            # print('->hslot min dist:',config.HSLOT_MIN_DIST,',hslot max dist:',config.HSLOT_MAX_DIST)


            if not (config.VSLOT_MIN_DIST <= distance <= config.VSLOT_MAX_DIST
                    or config.HSLOT_MIN_DIST <= distance <= config.HSLOT_MAX_DIST):
                continue
            # Step 2: pass through filtration.
            # print('pass through 3rd point', pass_through_third_point(marking_points,i,j))
            if pass_through_third_point(marking_points, i, j):
                continue
            result = pair_marking_points(point_i, point_j)
            # print('results:',result)
            if result == 1:
                slots.append((i, j))
            elif result == -1:
                slots.append((j, i))
    return slots


def detect_video(detector, device, args):
    """Demo for detecting video."""
    timer = Timer()
    input_video = cv.VideoCapture(args.video)
    frame_width = int(input_video.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(input_video.get(cv.CAP_PROP_FRAME_HEIGHT))
    output_video = cv.VideoWriter()
    if args.save:
        output_video.open('topview_1666775453007_median.avi', cv.VideoWriter_fourcc(*'XVID'),
                          input_video.get(cv.CAP_PROP_FPS),
                          (frame_width, frame_height), True)
    frame = np.empty([frame_height, frame_width, 3], dtype=np.uint8)
    while input_video.read(frame)[0]:
        timer.tic()
        # Bilatral filter
        # frame = cv.bilateralFilter(frame, 15, 75, 75) 
        # Median blur
        frame = cv.medianBlur(frame,5)
        # Normal blur
        # frame = cv.blur(frame,(5,5),0)
        # Gaussian blur
        # frame = cv.GaussianBlur(frame,(1,1),0)
        pred_points = detect_marking_points(
            detector, frame, args.thresh, device)
        slots = None
        if pred_points and args.inference_slot:
            # pred_points = directional_chamfer_matching(frame, pred_points)
            pred_points = post_process(frame, pred_points)
            marking_points = list(list(zip(*pred_points))[1])
            slots = inference_slots(marking_points)
        timer.toc()
        plot_points(frame, pred_points)
        plot_slots(frame, pred_points, slots)
        cv.imshow('demo', frame)
        cv.waitKey(1)
        if args.save:
            output_video.write(frame)
    print("Average time: ", timer.calc_average_time(), "s.")
    input_video.release()
    output_video.release()


def detect_image(detector, device, args):
    """Demo for detecting images."""
    timer = Timer()
    a = 0
    while a==0:

        # image_file = input('Enter image file path: ')
        image_file = 'frame_002658.jpg'
        image = cv.imread(image_file)
        timer.tic()
        # Step 1: Detect marking points 
        pred_points = detect_marking_points(
            detector, image, args.thresh, device)
        slots = None
        # Step 2: inference splots 
        if pred_points and args.inference_slot:
            # Step 3: Perform directional chamfer matching 
            # pred_points = directional_chamfer_matching(image, pred_points)
            pred_points = post_process(image, pred_points)

            # marking_points already been refined 
            marking_points = list(list(zip(*pred_points))[1])
            slots = inference_slots(marking_points)
        timer.toc()
        # Step 5: Display the results 
        plot_points(image, pred_points)
        plot_slots(image, pred_points, slots)
        # cv.imshow('demo', image)

        cv.waitKey(1)
        if args.save:
            cv.imwrite('demo.jpg', image, [int(cv.IMWRITE_JPEG_QUALITY), 100])

        a = 1
def post_process(image, pred_points): 
    """
    Post processing based on Hough Transform. Given the prediction of the model
    """
    height, width,_ = image.shape 
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY) 
    edges = cv.Canny(gray, 50, 150) 
    refined_pred_points = []
    for confidence,marking_point in pred_points:
        p0_x = int(width * marking_point.x - 0.5)
        p0_y = int(height * marking_point.y - 0.5) 
        x_min = max(p0_x - roi_size, 0) 
        x_max = min(p0_x +  roi_size, width)  
        y_min = max(p0_y - roi_size, 0) 
        y_max = min(p0_y + roi_size, height)
        roi = edges[y_min:y_max, x_min:x_max] 
        lines = cv.HoughLines(roi, 1 ,np.pi/180,30) # roi, pixel_resolution, angle_resolution, number of points to form a line 
        if lines is not None: 
            # Since hough line returns the angle of the normal line 
            thetas = lines[:,0,1]
            min_theta = np.min(thetas)
            max_theta = np.max(thetas) 


            # min_theta, max_theta = np.min(thetas) + np.pi/2, np.max(thetas) - np.pi/2

            point_theta = marking_point.direction
            thresh = 0 * np.pi / 180 
            print(f'min: {min_theta * 180 / np.pi}, max: {max_theta * 180 / np.pi}, predicted: {point_theta * 180 / np.pi}, confidence: {confidence}')

            # Compare between the angle of the fitting line and the actual angle
            if (abs(point_theta - min_theta) < thresh): 
                best_angle = min_theta
            elif (abs(point_theta - max_theta) < thresh): 
                best_angle = max_theta 
            else: 
                best_angle = point_theta
            refined_marking_point = MarkingPoint(x=marking_point.x,
                                                y=marking_point.y,
                                                direction=best_angle,
                                                shape=marking_point.shape
                                                )
            refined_pred_points.append((confidence,refined_marking_point))
    return refined_pred_points

def directional_chamfer_matching(image, pred_points):
    """
    Perform directional chamfer matching to refine the position and orientation of the parking slots.
    Args:
        imge: input image 
        pred_points: list of predicted marking points 

    Returns:
        pred_points: List of refined angle marking points 
    """
    height, width, _ = image.shape
    # 1. Edge detection with Canny operator
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray, 50, 150) 
    dist_transform = cv.distanceTransform(255 - edges, cv.DIST_L2, 5) 
    dist_transform = dist_transform.astype(np.float32)
    refined_pred_points = []
    # cv.imshow('edge', edges) 
    for confidence, marking_point in pred_points:
        p0_x = int(width * marking_point.x - 0.5) 
        p0_y = int(height * marking_point.y - 0.5) 

        # 2. Define a region of interest around the marking point
        # roi_size = 50
        x_min = max(p0_x - roi_size, 0) 
        x_max = min(p0_x + roi_size, width) 
        y_min = max(p0_y - roi_size, 0) 
        y_max = min(p0_y + roi_size, height) 

        # 3. Extract the ROI from distance transform 
        dist_transform_roi = dist_transform[y_min:y_max, x_min:x_max]
        # cv.imshow('dist_trans', dist_transform_roi)

        # How to create a template here? 
        if marking_point.shape < 0.5: # T shape 
            template = templates['t_shape']
        else: # L shape 
            template = templates['l_shape']
        
        best_angle_deg = 0
        # Initialized the current prediction of the model as the best 
        angle_rad = marking_point.direction 
        rotation_matrix = cv.getRotationMatrix2D((roi_size, roi_size), angle_rad, 1.0)
        rotated_template = cv.warpAffine(template, rotation_matrix, (template_size, template_size))
        binary_template = rotated_template > 0 
        best_score = np.sum(dist_transform_roi[binary_template > 0])
        best_angle = angle_rad 
        print('og angle:',best_angle / math.pi * 180)
        for angle in range(-2,2,1): 
            angle_rad = marking_point.direction + math.radians(angle) 

            # 5. Calculate chamfer score within the ROI 
            if dist_transform_roi.shape[0] >= template.shape[0] and dist_transform_roi.shape[1] >= template.shape[1]: 
                # ROI of template and edge detection kernel equal
                rotation_matrix = cv.getRotationMatrix2D((roi_size/2, roi_size/2), angle_rad, 1.0)
                rotated_template = cv.warpAffine(template, rotation_matrix, (template_size, template_size))

                # Compute Chamfer Distance 
                binary_template = (rotated_template > 0)
                score = np.sum(dist_transform_roi[binary_template > 0])

                print(angle, score)
                if score < best_score: 
                    best_score = score 
                    best_angle = angle_rad
                    best_angle_deg = angle
        print('best_score:',best_score,', best_angle degree:',best_angle_deg, 'confidence:',confidence)
        refined_marking_point = MarkingPoint(x=marking_point.x,
                                             y=marking_point.y,
                                             direction=best_angle,
                                             shape=marking_point.shape
                                            )
        # cv.waitKey(0)
        refined_pred_points.append((confidence,refined_marking_point))
#  N = namedtuple("N", ['ind', 'set', 'v']) 
# items[node.ind].v = node.v
#  items[node.ind] = items[node.ind]._replace(v=node.v)
#  MarkingPoint = namedtuple('MarkingPoint', ['x', 'y', 'direction', 'shape'])
# marking_point = marking_point._replace(direction=best_angle)
    # Named tuple is immutable -> use this
    print('->refined_pred_points',refined_pred_points)
    return refined_pred_points
 


def inference_detector(args):
    """Inference demo of directional point detector."""
    args.cuda = not args.disable_cuda and torch.cuda.is_available()
    device = torch.device('cuda:' + str(args.gpu_id) if args.cuda else 'cpu')
    torch.set_grad_enabled(False)
    dp_detector = torch.nn.DataParallel(DirectionalPointDetector(
        3, args.depth_factor, config.NUM_FEATURE_MAP_CHANNEL)).to(device)
    # dp_detector = DirectionalPointDetector(
    #     3, args.depth_factor, config.NUM_FEATURE_MAP_CHANNEL).to(device)
    dp_detector.load_state_dict(torch.load(args.detector_weights))
    dp_detector.eval()
    if args.mode == "image":
        detect_image(dp_detector, device, args)
    elif args.mode == "video":
        detect_video(dp_detector, device, args)


if __name__ == '__main__':
    inference_detector(config.get_parser_for_inference().parse_args())
