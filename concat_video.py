import cv2
import numpy as np

# Video files and their labels
def concat_video():
    video_files = [
                "record.avi", 
                #    "record_gaussian.avi", 
                "record_median.avi",
                #    "record_bilateral.avi"
                ]
    video_labels = [
                    "No filter", 
                    # "Gaussian", 
                    "Median",
                    # "Bilateral"
                ]

    # Open video captures
    caps = [cv2.VideoCapture(video) for video in video_files]

    # Get properties of the first video to define the output video
    frame_width = int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(caps[0].get(cv2.CAP_PROP_FPS))

    # Define the codec and output file
    output_width = frame_width * len(caps)
    output_height = frame_height
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4', fourcc, fps, (output_width, output_height))

    # Font settings for labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (255, 255, 255)
    font_thickness = 2

    while True:
        frames = []
        for i, cap in enumerate(caps):
            ret, frame = cap.read()
            if not ret:
                # End when any video ends
                frames = []
                break

            # Add label to the frame
            label_size = cv2.getTextSize(video_labels[i], font, font_scale, font_thickness)[0]
            text_x = (frame.shape[1] - label_size[0]) // 2
            text_y = 30
            cv2.putText(frame, video_labels[i], (text_x, text_y), font, font_scale, font_color, font_thickness)

            # Resize to ensure uniform height
            frame = cv2.resize(frame, (frame_width, frame_height))
            frames.append(frame)

        if not frames:
            break

        # Concatenate frames horizontally
        concatenated_frame = np.hstack(frames)
        out.write(concatenated_frame)

        # Optionally display the video
        cv2.imshow('Concatenated Video', concatenated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    #Release resources
    for cap in caps:
        cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":

    import cv2
    import numpy as np
    import matplotlib.pyplot as plt

    # Load the image
    image_path = 'frame_002648.jpg'  # Replace with your image path
    original_image = cv2.imread(image_path)
    if original_image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    # Convert the image from BGR to RGB (for displaying correctly with matplotlib)
    original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    # Apply median filtering
    kernel_size = 5  # Define the kernel size (must be odd)
    filtered_image = cv2.medianBlur(original_image, kernel_size)

    # Convert the filtered image to RGB
    filtered_image_rgb = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2RGB)

    # Display the images side by side
    plt.figure(figsize=(10, 5))

    # Original image
    plt.subplot(1, 2, 1)
    plt.imshow(original_image_rgb)
    plt.title("Original Image")
    plt.axis("off")

    # Filtered image
    plt.subplot(1, 2, 2)
    plt.imshow(filtered_image_rgb)
    plt.title("Median Filtered Image")
    plt.axis("off")

    plt.tight_layout()
    plt.show()
