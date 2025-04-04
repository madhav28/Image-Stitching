import numpy as np
from matplotlib import pyplot as plt
from common import *


def homography_transform(X, H):
    # Perform homography transformation on a set of points X
    # using homography matrix H
    # Input - a set of 2D points in an array with size (N,2)
    #         a 3*3 homography matrix 
    # Output - a set of 2D points in an array with size (N,2)

    X = np.hstack((X, np.ones((X.shape[0], 1))))
    Y_hat = X @ H.T
    Y = Y_hat / Y_hat[:, 2:3]
    return Y[:, :2]


def fit_homography(XY):
    # Given two set of points X, Y in one array,
    # fit a homography matrix from X to Y
    # Input - an array with size(N,4), each row contains two
    #         points in the form[x^T_i,y^T_i]1×4
    # Output - a 3*3 homography matrix
    X = XY[:, :2]
    Y = XY[:, 2:4]

    A = []
    for i in range(X.shape[0]):
        x, y = X[i]
        x_prime, y_prime = Y[i]
        A.append([-x, -y, -1, 0, 0, 0, x * x_prime, y * x_prime, x_prime])
        A.append([0, 0, 0, -x, -y, -1, x * y_prime, y * y_prime, y_prime])
    A = np.array(A)

    U, S, Vh = np.linalg.svd(A)
    H = Vh[-1].reshape(3, 3)
    
    return H


def stitchimage(imgleft, imgright, savename):
    # 1. extract descriptors from images
    #    you may use SIFT/SURF of opencv
    sift = cv2.SIFT_create()
    keypoints_left, descriptors_left = sift.detectAndCompute(imgleft, None)
    keypoints_right, descriptors_right = sift.detectAndCompute(imgright, None)
    imgleft_with_keypoints = cv2.drawKeypoints(imgleft, keypoints_left, None)
    imgright_with_keypoints = cv2.drawKeypoints(imgright, keypoints_right, None)
    save_img(savename+'_left_sift.png', imgleft_with_keypoints)
    save_img(savename+'_right_sift.png', imgright_with_keypoints)

    kp_left_loc = np.array([[keypoint.pt[0], keypoint.pt[1]] for keypoint in keypoints_left])
    kp_right_loc = np.array([[keypoint.pt[0], keypoint.pt[1]] for keypoint in keypoints_right])

    descriptors_left_normalized = ((descriptors_left - np.mean(descriptors_left, axis = 0)) / np.std(descriptors_left, axis = 0))
    descriptors_right_normalized = ((descriptors_right - np.mean(descriptors_right, axis = 0)) / np.std(descriptors_right, axis = 0))
    distances_norm_euclidean = np.linalg.norm(descriptors_left_normalized[:, np.newaxis] - descriptors_right_normalized, axis=2)
    # distances_norm_correlation = np.dot(descriptors_left_normalized, descriptors_right_normalized.T)
    distances = distances_norm_euclidean

    # 2. select paired descriptors
    threshold = 0.5  
    matches = []

    for i in range(len(descriptors_left)):
        distances_i = distances[i]
        
        sorted_indices = np.argsort(distances_i)
        if distances_i[sorted_indices[0]] / distances_i[sorted_indices[1]] < threshold:
            matches.append((i, sorted_indices[0]))

    print(f"Number of putative matches selected for {savename} = {len(matches)}")

    r = [idx[0] for idx in matches]
    c = [idx[1] for idx in matches]
    XY = np.hstack((kp_left_loc[r, :], kp_right_loc[c, :]))

    # 3. run RANSAC to find a transformation
    #    matrix which has most innerliers
    best_h = None
    best_dist = None
    bestInlierCount = -1
    bestInlierAvgResidual = -1
    bestInlierMatchIdx = None
    for itr in range(1000):
        rand_idx = np.random.choice(len(XY), 20, replace=False)
        H = fit_homography(XY[rand_idx, :])
        Y_hat = homography_transform(XY[:, :2], H)
        residuals = np.sum((Y_hat - XY[:, 2:])**2, axis = 1)
        inlierCount = np.sum(residuals < 5)
        if inlierCount > bestInlierCount:
            best_h = H
            bestInlierCount = inlierCount
            bestInlierAvgResidual = np.mean(residuals[residuals < 5])
            bestInlierMatchIdx = np.where(residuals < 5)[0]
            best_dist = np.sqrt(residuals)
    print(f"Number of inliers for the best homography mapping of {savename} = {bestInlierCount}")
    print(f"Average residual of inliers for the best homography mapping of {savename} = {bestInlierAvgResidual}")

    kp_left = []
    kp_right = []
    matchLR = []
    for i, idx in enumerate(bestInlierMatchIdx):
        kp_left.append(cv2.KeyPoint(kp_left_loc[r[i]][0], kp_left_loc[r[i]][1], 1))
        kp_right.append(cv2.KeyPoint(kp_right_loc[c[i]][0], kp_right_loc[c[i]][1], 1))
        matchLR.append(cv2.DMatch(i, i, best_dist[idx]))
    
    matched_features = cv2.drawMatches(imgleft, kp_left, imgright, kp_right, matchLR, None)
    save_img(savename+'_matched_features.png', matched_features)

    # 4. warp one image by your transformation 
    #    matrix
    #
    #    Hint: 
    #    a. you can use opencv to warp image
    #    b. Be careful about final image size
    savename_temp = savename.split('/')[-1]
    imgleft = read_colorimg('./data/'+savename_temp+'_left.jpg')
    imgright = read_colorimg('./data/'+savename_temp+'_right.jpg')
    translation = np.array([[1, 0, 650],[0 ,1, 250],[0, 0, 1]], dtype=np.float32)
    best_h = translation @ best_h
    imgleft_warped = cv2.warpPerspective(imgleft, best_h, (2048, 1024))
    save_img(savename+'_left_warped_image.png', imgleft_warped)

    # # 5. combine two images, use average of them
    # #    in the overlap area
    imgright = cv2.warpPerspective(imgright, translation, (2048, 1024))
    composite_img = (imgleft_warped.astype(np.int32) + imgright.astype(np.int32))
    mask_overlap = np.logical_and(imgleft_warped, imgright)
    composite_img[mask_overlap]  = (0.5*composite_img[mask_overlap]).astype(np.int32)
    save_img(savename+'_composite.png', composite_img)


def p2(p1, p2, savename):
    # read left and right images
    imgleft = read_img(p1)
    imgright = read_img(p2)
    save_img(savename+'_left_grayscale.png', imgleft)
    save_img(savename+'_right_grayscale.png', imgright)

    # stitch image
    output = stitchimage(imgleft, imgright, savename)
    # # save stitched image
    # save_img('./' + savename + '.jpg', output)


if __name__ == "__main__":
    p2('./data/uttower_left.jpg', './data/uttower_right.jpg', './results/uttower')
    p2('./data/bbb_left.jpg', './data/bbb_right.jpg', './results/bbb')