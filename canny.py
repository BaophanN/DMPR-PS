import cv2
import numpy as np
import argparse

def unsharp_mask(image, sigma=1.5, strength=1.5):
    # Apply Gaussian Blur to create a blurred version of the image
    blurred = cv2.GaussianBlur(image, (5, 5), sigma)
    
    # Subtract the blurred image from the original to enhance edges
    sharpened = cv2.addWeighted(image, 1 + strength, blurred, -strength, 0)
    
    return sharpened
def apply_canny_edge_detection(input_path, output_path, low_threshold, high_threshold,blur_kernel_size=5):
    # Load the image
    image = cv2.imread(input_path)

    # Check if the image was loaded properly
    if image is None:
        print("Error: Unable to load the image.")
        return


    sharpened_image = unsharp_mask(image)
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(sharpened_image, cv2.COLOR_BGR2GRAY)
    # deblur 
    deblurred_image = cv2.GaussianBlur(gray_image, (blur_kernel_size, blur_kernel_size), 0)
    # Apply Canny edge detection
    edges = cv2.Canny(deblurred_image, low_threshold, high_threshold)

    # Display the original and edge-detected images
    cv2.imshow("Original Image", image)
    cv2.imshow("Sharpend Image", sharpened_image)
    cv2.imshow("Edge Detected Image", edges)

    # Wait until a key is pressed, then close the windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save the edge-detected image to a file
    cv2.imwrite(output_path, edges)
    print(f"Edge-detected image saved to {output_path}")

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Apply Canny edge detection on an image.")
    parser.add_argument("--image_name", type=str, help="Path to the input image.")
    parser.add_argument("--low_threshold", type=int, default=100, help="Lower threshold for Canny edge detection.")
    parser.add_argument("--high_threshold", type=int, default=200, help="Higher threshold for Canny edge detection.")

    # Parse the arguments
    args = parser.parse_args()
    
    input_path = f'sample_image_input/edge_detection/{args.image_name}'
    output_path = f'sample_image_output/edge_detection/output_{args.image_name}'
    # Apply Canny edge detection
    apply_canny_edge_detection(input_path,output_path, args.low_threshold, args.high_threshold)
