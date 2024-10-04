import numpy as np
import cv2
import matplotlib.pyplot as plt

def decomp_essential_mat(E):
    # Decompose the Essential matrix using OpenCV
    R1, R2, t = cv2.decomposeEssentialMat(E)
    return R1, R2, t

def form_transf(R, t):
    """Form a 4x4 transformation matrix from rotation R and translation t."""

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t.flatten()
    return T

def transform_image(image, R, t):
    """Transform image based on rotation R and translation t."""
    # Get image dimensions
    h, w = image.shape[:2]

    # Define the camera intrinsic matrix (for visualization purposes)
    K = np.array([[0.718856 ,   0.    , 0.6071928],
                    [  0.    , 0.718856 , 0.1852157],
                    [  0.    ,   0.    ,   1.    ]])

    # Create the transformation matrix from R and t
    T = form_transf(R, t)

    # Apply the projection to the image
    P = K @ T[:3, :]  # Projection matrix

    # Perform the image warp
    transformed_image = cv2.warpPerspective(image, P[:3, :3], (w, h))
    return transformed_image

def visualize_images(img_a, img_b, img_b_transformed):
    """Visualize original images and transformed image in a Matplotlib figure."""
    plt.figure(figsize=(15, 5))

    # Panel 1: Original Image A
    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(img_a, cv2.COLOR_BGR2RGB))
    plt.title('Image A')
    plt.axis('off')

    # Panel 2: Original Image B
    plt.subplot(2, 2, 2)
    plt.imshow(cv2.cvtColor(img_b, cv2.COLOR_BGR2RGB))
    plt.title('Image B')
    plt.axis('off')

    # Panel 3: Transformed Image B
    plt.subplot(2, 2, 3)
    plt.imshow(cv2.cvtColor(img_b_transformed, cv2.COLOR_BGR2RGB))
    plt.title('Transformed Image B')
    plt.axis('off')

    # Panel 4: Transformed Image B
    plt.subplot(2, 2, 4)
    plt.imshow(cv2.cvtColor(img_b_transformed, cv2.COLOR_BGR2RGB))
    plt.title('Transformed Image B')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# Main function to load images and apply transformations
def main():
    # Load images (replace with your actual image paths)
    img_a = cv2.imread('KITTI_sequence_1/image_l/000000.png')
    img_b = cv2.imread('KITTI_sequence_1/image_l/000001.png')

    # Convert images to grayscale for essential matrix calculation
    gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)

    # Detect ORB features and compute descriptors
    orb = cv2.ORB_create()
    keypoints_a, descriptors_a = orb.detectAndCompute(gray_a, None)
    keypoints_b, descriptors_b = orb.detectAndCompute(gray_b, None)

    # Use FLANN to match descriptors
    index_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(descriptors_a, descriptors_b, k=2)

    # Apply Lowe's ratio test
    good_matches = [m[0] for m in matches if len(m) > 1 and m[0].distance < 0.75 * m[1].distance]

    # Get matching keypoints
    q1 = np.float32([keypoints_a[m.queryIdx].pt for m in good_matches])
    q2 = np.float32([keypoints_b[m.trainIdx].pt for m in good_matches])

    # Compute essential matrix
    E, _ = cv2.findEssentialMat(q1, q2, focal=1000, pp=(img_a.shape[1]/2, img_a.shape[0]/2))

    # Decompose essential matrix
    R,_,  t = decomp_essential_mat(E)

    # Transform image B using R and t
    img_b_transformed = transform_image(img_b, R, t)

    # Visualize the results
    visualize_images(img_a, img_b, img_b_transformed)

if __name__ == "__main__":
    main()
