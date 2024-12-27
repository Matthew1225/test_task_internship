import cv2
import matplotlib.pyplot as plt

def match_images(img1_path, img2_path):
    # Load the images from the file paths. 
    # The images are read in color (BGR format).
    img1 = cv2.imread(img1_path, cv2.IMREAD_COLOR)
    img2 = cv2.imread(img2_path, cv2.IMREAD_COLOR)

    # Convert the images to grayscale.
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Create an ORB detector.
    orb = cv2.ORB_create()

    # Detect keypoints and compute descriptors for both images using the ORB detector.
    keypoints1, descriptors1 = orb.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(gray2, None)

    # Create a BFMatcher (Brute-Force Matcher) to find the best matches between keypoints.
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match the descriptors from the two images.
    matches = bf.match(descriptors1, descriptors2)

    # Sort the matches by distance.
    # The smaller the distance, the better the match. 
    matches = sorted(matches, key=lambda x: x.distance)

    # Visualize the top 20 matches.
    img_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches[:20], None, 
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Display the result.
    plt.figure(figsize=(15, 15))
    plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))  # Convert the image to RGB format for matplotlib
    plt.axis('off')  
    plt.show()  


image1_path = r"C:\Users\Matthew\Desktop\test_task_internship\image_matching\Sentinel-2 images\2024-11-17-00_00_2024-11-17-23_59_Sentinel-2_L2A_True_color.jpg"  
image2_path = r"C:\Users\Matthew\Desktop\test_task_internship\image_matching\Sentinel-2 images\2024-11-25-00_00_2024-11-25-23_59_Sentinel-2_L2A_True_color.jpg" 

match_images(image1_path, image2_path)