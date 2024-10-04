import os
import numpy as np
import cv2
from matplotlib import pyplot as plt 

from lib.visualization import plotting
from lib.visualization.video import play_trip

from tqdm import tqdm


class VisualOdometry():
    def __init__(self, data_dir):

        # K 3x3 matrix (fx,fy,1 along diagonal,  cx,cy), P 3x4 matrix
        # K is the intrinsic camera matrix, focal lenght, center points, to translate pixels to meters
        # P is the projection matrix, to translate 3D points to 2D points
        # P = K[R|t] which is the projection of K onto a matrix of shape 3x4, which has rotation 3x3 and translation 3x1 stacked
        self.K, self.P = self._load_calib(os.path.join(data_dir, 'calib.txt'))

        # see _load_poses for more information
        self.gt_poses = self._load_poses(os.path.join(data_dir, 'poses.txt'))

        # # make a 3D scatter plot of the x,y,z of the poses
        # fig= plt.figure()
        # ax = fig.add_subplot(projection='3d')
        # [ax.scatter(self.gt_poses[i][0,-1],self.gt_poses[i][1,-1],self.gt_poses[i][2,-1],s=100/(i+1) ) for i in range(50)]
        # plt.title("Ground truth translation pose of the camera")
        # ax.set_xlabel('X')
        # ax.set_ylabel('Y')
        # ax.set_zlabel('Z')
        # plt.show()

        self.images = self._load_images(os.path.join(data_dir, 'image_l'))

        # create an ORB object that can detect 3000 keypoints
        # they are selected not randomly but based on an algorithm (Harris corner detection)
        self.orb = cv2.ORB_create(3000)

        # use Locality Sensitive Hashing (LSH) for binary descriptors like ORB, or KD-trees for floating-point descriptors like SIFT
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)

        # search N times for the best match -> higher N, more accurate but slower
        # it seems that 50 already gives the best results, so higher is not necessary
        search_params = dict(checks=50) 

        # define the matcher, which is used later
        self.flann = cv2.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)

    @staticmethod
    def _load_calib(filepath):
        """
        Loads the calibration of the camera
        Parameters
        ----------
        filepath (str): The file path to the camera file

        Returns
        -------
        K (ndarray): Intrinsic parameters
        P (ndarray): Projection matrix
        """
        with open(filepath, 'r') as f:
            params = np.fromstring(f.readline(), dtype=np.float64, sep=' ')
            P = np.reshape(params, (3, 4))
            K = P[0:3, 0:3]
        return K, P

    @staticmethod
    def _load_poses(filepath):
        """
        Loads the GT poses

        Parameters
        ----------
        filepath (str): The file path to the poses file

        Returns
        -------
        poses (ndarray): The GT poses

        The GT poses is the actual position and orientation of the vehicle camera
        It is a 4x4 homogeneous transformation matrix that has rotation and translation stored

        |r11 r12 r13 tx|
        |r21 r22 r23 ty|
        |r31 r32 r33 tz|
        |0   0   0   1 |
        
        """

        poses = []
        with open(filepath, 'r') as f:
            for line in f.readlines():
                T = np.fromstring(line, dtype=np.float64, sep=' ')
                T = T.reshape(3, 4)
                T = np.vstack((T, [0, 0, 0, 1]))
                poses.append(T)
        return poses

    @staticmethod
    def _load_images(filepath):
        """
        Loads the images

        Parameters
        ----------
        filepath (str): The file path to image dir

        Returns
        -------
        images (list): grayscale images
        """
        image_paths = [os.path.join(filepath, file) for file in sorted(os.listdir(filepath))]
        return [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in image_paths]

    @staticmethod
    def _form_transf(R, t):
        """
        Makes a transformation matrix from the given rotation matrix and translation vector

        Parameters
        ----------
        R (ndarray): The rotation matrix
        t (list): The translation vector

        Returns
        -------
        T (ndarray): The transformation matrix
        """
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t
        return T

    def get_matches(self, i):
        """
        This function detect and compute keypoints and descriptors from the i-1'th and i'th image using the class orb object

        Parameters
        ----------
        i (int): The current frame

        Returns
        -------
        q1 (ndarray): The good keypoints matches position in i-1'th image
        q2 (ndarray): The good keypoints matches position in i'th image
        """

        # keypoints have the information of:
        # - x,y position
        # - size of the meaningful neighborhood
        # - orientation
        # - response that measures the strength of the keypoint
        # - octave (pyramid layer)

        # descriptors are the feature vectors that describe the keypoints
        # they are 32 bytes long and are used to match keypoints between images (e.g. shape is 3000x32 if all ORB keypoints are used)

        mask = None

        # use 2 frames for the matching (tracking)
        keypoints1, descriptors1 = self.orb.detectAndCompute(self.images[i - 1], mask)
        keypoints2, descriptors2 = self.orb.detectAndCompute(self.images[i], mask)


        # every frame returns a list of descriptors that have keypoints
        # they should be matched 
        matches = self.flann.knnMatch(descriptors1, descriptors2, k=2)
        # the variable matches stores (maximum) 3000 match pairs
        # every match pair has Idx of the descriptor1 and descriptor2, as well as the distance
        # the smalles distance is the best match (and also defined as the 1st match)
        # in the following the best match is m and the second best match is n



        # since orb is 3000 it means that i will have many matches, some good, some questionable
        # since we have 3000 of them, we can easily drop the questionable ones
        # so we take only the ones where the hamming distance is much better from one to the other
        good = []
        for m,n in matches:
            # hamming distance
            if m.distance < 0.5*n.distance:
                good.append(m)

        # the matching points of frame i-1 and i are stored in q1 and q2
        # q1 and q2 are the x,y pixels (with subpixel accuracy)
        # e.g. (412.06591796875, 175.57591247558594) are q1 points      
                
        q1 = np.float32([ keypoints1[m.queryIdx].pt for m in good ])
        q2 = np.float32([ keypoints2[m.trainIdx].pt for m in good ])

        

        # draw_params = dict(matchColor = -1, # draw matches in green color
        #         singlePointColor = None,
        #         matchesMask = None, # draw only inliers
        #         flags = 2)

        # img3 = cv2.drawMatches(self.images[i], keypoints1, self. images[i-1],keypoints2, good ,None,**draw_params)
        # cv2.imshow("image", img3)
        # cv2.waitKey(0)
        # plt.imshow(img3, 'gray'),plt.show()
        # plt.imshow(self.images[i]),plt.show()
        # plt.imshow(self.images[i-1]),plt.show() 
        return q1, q2


        # This function should detect and compute keypoints and descriptors from the i-1'th and i'th image using the class orb object
        # The descriptors should then be matched using the class flann object (knnMatch with k=2)
        # Remove the matches not satisfying Lowe's ratio test
        # Return a list of the good matches for each image, sorted such that the n'th descriptor in image i matches the n'th descriptor in image i-1
        # https://docs.opencv.org/master/d1/de0/tutorial_py_feature_homography.html
        

    def get_pose(self, q1, q2):
        """
        Calculates the transformation matrix

        Parameters
        ----------
        q1 (ndarray): The good keypoints matches position in i-1'th image
        q2 (ndarray): The good keypoints matches position in i'th image

        Returns
        -------
        transformation_matrix (ndarray): The transformation matrix
        """

        Essential, mask = cv2.findEssentialMat(q1, q2, self.K)
        # print ("\nEssential matrix:\n" + str(Essential))

        # get rotation and translation components from Essential matrix
        R, t = self.decomp_essential_mat(Essential, q1, q2)


        # side note: The step findEssentialMat followed by decomp_essential_mat is more
        # stable then directly caluclating R,t from q1,q2
        # because findEssentialMat uses some outlier removal and RANSAC

        return self._form_transf(R,t)

        # Estimate the Essential matrix using built in OpenCV function
        # Use decomp_essential_mat to decompose the Essential matrix into R and t
        # Use the provided function to convert R and t to a transformation matrix T

    def decomp_essential_mat(self, E, q1, q2):
        """
        Decompose the Essential matrix

        Parameters
        ----------
        E (ndarray): Essential matrix
        q1 (ndarray): The good keypoints matches position in i-1'th image
        q2 (ndarray): The good keypoints matches position in i'th image

        Returns
        -------
        right_pair (list): Contains the rotation matrix and translation vector
        """


        R1, R2, t = cv2.decomposeEssentialMat(E)

        # all combinations (forward or backward) of translations have to be tested 
        # for all rotations
        T1 = self._form_transf(R1,np.ndarray.flatten(t))
        T2 = self._form_transf(R2,np.ndarray.flatten(t))
        T3 = self._form_transf(R1,np.ndarray.flatten(-t))
        T4 = self._form_transf(R2,np.ndarray.flatten(-t))
        transformations = [T1, T2, T3, T4]
        
        # Homogenize K
        # just add a column of zeros to K to give it shape 3x4
        K = np.concatenate(( self.K, np.zeros((3,1)) ), axis = 1)

        # Project the Tranformations 4x4 onto the camera intrinsics 3x4
        # which results in a variable that represents how the 3D rotation/translation
        # are projected onto the 2D world of the camera
        projections = [K @ T1, K @ T2, K @ T3, K @ T4]

        np.set_printoptions(suppress=True)

        # now use triangulation to find which of the transformations is the correct one
        positives = []
        for P, T in zip(projections, transformations):

            # triangulate points q1 and q2 with respect to the first camera (frame i-1)
            # get q1 amount of points 
            # for X, Y, Z, W (weight/scale)
            hom_Q1 = cv2.triangulatePoints(self.P, P, q1.T, q2.T)

            # apply current transformation T to hom_Q1 to get the second cameras frame (frame i)
            hom_Q2 = T @ hom_Q1

            # Un-homogenize the points X, Y, Z and normalize with W
            Q1 = hom_Q1[:3, :] / hom_Q1[3, :] # frame i-1
            Q2 = hom_Q2[:3, :] / hom_Q2[3, :] # frame i

            # count how many points have a positive z value (are in front of the camera)
            total_sum = sum(Q2[2, :] > 0) + sum(Q1[2, :] > 0)

            # transpose Q to have shape (Num_q, 3) which is basically x,y,z coordinate for every q1
            # then see which difference is between the frames i-1 and i
            # and normalize it by Q2

            # distance Q1(q1_a) - Q1(q1_b) in frame t relative to the distance that the same
            # matched points Q2(q2_a) - Q2(q2_b) have in frame t-1 
            # e.g. the tree should have the same relative distance to the street in frame t and t-1
            relative_scale = np.mean(np.linalg.norm(Q1.T[:-1] - Q1.T[1:], axis=-1)/
                                     np.linalg.norm(Q2.T[:-1] - Q2.T[1:], axis=-1))
            # the higher the score, the more points are in front of the camera
            # and the more relative scales are preserved
            positives.append(total_sum + relative_scale)
            

        # Decompose the Essential matrix using built in OpenCV function
        # Form the 4 possible transformation matrix T from R1, R2, and t
        # Create projection matrix using each T, and triangulate points hom_Q1
        # Transform hom_Q1 to second camera using T to create hom_Q2
        # Count how many points in hom_Q1 and hom_Q2 with positive z value
        # Return R and t pair which resulted in the most points with positive z

        max = np.argmax(positives)

        if (max == 0):
            # print(t)
            return R1, np.ndarray.flatten(t)
        elif (max == 1):
            # print(t)
            return R2, np.ndarray.flatten(t)
        elif (max == 2):
            # print(-t)
            return R1, np.ndarray.flatten(-t)
        elif (max == 3):
            # print(-t)
            return R2, np.ndarray.flatten(-t)


def main():
    data_dir = 'KITTI_sequence_2'  # Try KITTI_sequence_2 too
    vo = VisualOdometry(data_dir)


    # play_trip(vo.images)  # Comment out to not play the trip

    # make a 3D scatter plot of the x,y,z of the poses
    fig= plt.figure()
    ax = fig.add_subplot(projection='3d')

    gt_path = []
    estimated_path = []

    gt_3d = []
    estimated_3d = []
    for i, gt_pose in enumerate(tqdm(vo.gt_poses, unit="pose")):
        if i == 0:
            cur_pose = gt_pose
        else:
            q1, q2 = vo.get_matches(i)

            # Transformation matrix 4x4
            # |R t|
            # |0 1|
            # with R is 3x3 and t is 3x1
            transf = vo.get_pose(q1, q2)

            # transformation matrix goes from i-1 to i and 
            # since we want to represent i in the coordinate system of i-1
            # we need to invert the transformation matrix (so that it points from i to i-1)
            cur_pose = np.matmul(cur_pose, np.linalg.inv(transf))

            print ("\nGround truth pose:\n" + str(gt_pose))
            print ("\n Current pose:\n" + str(cur_pose))
            print ("The current pose used x,y: \n" + str(cur_pose[0,3]) + "   " + str(cur_pose[2,3]) )


        # now this is just for plotting results
        gt_path.append((gt_pose[0, 3], gt_pose[2, 3]))
        estimated_path.append((cur_pose[0, 3], cur_pose[2, 3]))

        gt_3d.append((gt_pose[0, 3], gt_pose[1, 3], gt_pose[2, 3]))
        estimated_3d.append((cur_pose[0, 3], cur_pose[1, 3], cur_pose[2, 3]))
        
    ax.scatter(*zip(*gt_3d), s=10)
    ax.scatter(*zip(*estimated_3d), s=10)
    plt.title("Ground truth translation pose of the camera")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()
    
    plotting.visualize_paths(gt_path, estimated_path, "Visual Odometry", file_out=os.path.basename(data_dir) + ".html")


if __name__ == "__main__":
    main()
