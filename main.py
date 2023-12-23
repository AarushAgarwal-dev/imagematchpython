# Import the necessary libraries
import cv2  # For image processing
import numpy as np  # For numerical operations
import pyttsx3  # For text-to-speech
import requests  # For downloading images from the web

# Initialize the text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Set the speech rate
engine.setProperty('volume', 1.0)  # Set the volume


# Define a function to compare two images and return the similarity score
def compare_images(img1, img2):
    # Convert the images to grayscale
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    # Resize the images to the same size
    img1_resized = cv2.resize(img1_gray, (256, 256))
    img2_resized = cv2.resize(img2_gray, (256, 256))
    # Compute the mean squared error between the images
    mse = np.mean((img1_resized - img2_resized) ** 2)
    # Return the inverse of the mse, scaled by 100
    return 100 / (1 + mse)


# Define a function to download an image from a URL and return it as a numpy array
def download_image(url):
    # Send a GET request to the URL
    response = requests.get(url)
    # Check if the response is successful
    if response.status_code == 200:
        # Convert the response content to a numpy array
        img = np.frombuffer(response.content, dtype=np.uint8)
        # Decode the array as an image
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        # Return the image
        return img
    else:
        # Raise an exception if the response is not successful
        raise Exception(f"Failed to download image from {url}")


# Define the URL of the image to be uploaded
uploaded_image_url = "https://th.bing.com/th/id/OIP.W0RlTOGhgXu0Er1ALiKcDAHaHa?rs=1&pid=ImgDetMain"  # Replace this with your own URL

# Download the image from the URLq
uploaded_image = download_image(uploaded_image_url)

# Create a video capture object to access the laptop camera
cap = cv2.VideoCapture(0)

# Loop until the user presses the 'q' key
while True:
    # Read a frame from the camera
    ret, frame = cap.read()
    # Check if the frame is valid
    if ret:
        # Compare the frame with the uploaded image and get the similarity score
        score = compare_images(frame, uploaded_image)
        # Print the score on the frame
        cv2.putText(frame, f"Similarity: {score:.2f}%", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # Show the frame on the screen
        cv2.imshow("Camera", frame)
        # If the score is above a threshold, say "Match found" using the text-to-speech engine
        if score > 1.5:
            engine.say("Match found")
            engine.runAndWait()
    # Wait for 1 millisecond for a key press
    key = cv2.waitKey(1)
    # If the 'q' key is pressed, break the loop
    if key == ord('q'):
        break

# Release the video capture object
cap.release()
# Destroy all the windows
cv2.destroyAllWindows()
