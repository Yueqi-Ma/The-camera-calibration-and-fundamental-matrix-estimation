from utils import dehomogenize, homogenize, draw_epipolar, visualize_pcd
import numpy as np
import cv2
import pdb
import os


def find_fundamental_matrix(shape, pts1, pts2):
    # Normalize the data
    pts1_normalized = (pts1 - np.mean(pts1, axis=0)) / np.std(pts1, axis=0)
    pts2_normalized = (pts2 - np.mean(pts2, axis=0)) / np.std(pts2, axis=0)

    # Scale the image size
    T1 = np.array([[2/shape[1], 0, -1],
                   [0, 2/shape[0], -1],
                   [0, 0, 1]])
    T2 = np.array([[2/shape[1], 0, -1],
                   [0, 2/shape[0], -1],
                   [0, 0, 1]])

    # Apply normalization to the points
    pts1_normalized_homogeneous = homogenize(pts1_normalized)
    pts2_normalized_homogeneous = homogenize(pts2_normalized)

    # Construct the A matrix
    A = np.zeros((len(pts1), 9))
    for i in range(len(pts1)):
        x1, y1, _ = pts1_normalized_homogeneous[i]
        x2, y2, _ = pts2_normalized_homogeneous[i]
        A[i] = [x2 * x1, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, 1]

    # Solve for the fundamental matrix
    _, _, V = np.linalg.svd(A)
    F_normalized = V[-1].reshape((3, 3))

    # Enforce rank 2 constraint
    U, S, V = np.linalg.svd(F_normalized)
    S[-1] = 0
    F_normalized = U @ np.diag(S) @ V

    # Denormalize the fundamental matrix
    F = T2.T @ F_normalized @ T1

    return F


def compute_epipoles(F):

    _, _, V = np.linalg.svd(F.T)
    e1 = V[-1]
    _, _, V = np.linalg.svd(F)
    e2 = V[-1]
    return e1, e2


def find_triangulation(K1, K2, F, pts1, pts2):
    pcd = np.zeros((pts1.shape[0], 4))

    # Construct camera projection matrices
    M1 = K1 @ np.eye(3, 4)
    M2 = K2 @ np.hstack((np.eye(3), np.dot(K2, np.array([[0], [0], [-1]]))))

    # Triangulate each point
    for i in range(pts1.shape[0]):
        x1 = pts1[i][0]
        y1 = pts1[i][1]
        x2 = pts2[i][0]
        y2 = pts2[i][1]
        A = np.array([
            y1 * M1[2] - M1[1],
            M1[0] - x1 * M1[2],
            y2 * M2[2] - M2[1],
            M2[0] - x2 * M2[2]
        ])
        _, _, V = np.linalg.svd(A)
        X = V[-1]
        pcd[i] = X / X[3]

    return pcd


if __name__ == '__main__':

    # You can run it on one or all the examples
    names = os.listdir("task23")
    output = "results/"

    if not os.path.exists(output):
        os.mkdir(output)

    for name in names:
        print(name)

        # load the information
        img1 = cv2.imread(os.path.join("task23", name, "im1.png"))
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2 = cv2.imread(os.path.join("task23", name, "im2.png"))
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        data = np.load(os.path.join("task23", name, "data.npz"))
        pts1 = data['pts1'].astype(float)
        pts2 = data['pts2'].astype(float)
        K1 = data['K1']
        K2 = data['K2']
        shape = img1.shape

        # compute F
        F = find_fundamental_matrix(shape, pts1, pts2)
        # compute the epipoles
        e1, e2 = compute_epipoles(F)
        print(e1, e2)
        # to get the real coordinates, divide by the last entry
        print(e1[:2]/e1[-1], e2[:2]/e2[-1])

        outname = os.path.join(output, name + "_us.png")
        # If filename isn't provided or is None, this plt.shows().
        # If it's provided, it saves it
        draw_epipolar(img1, img2, F, pts1[::10, :], pts2[::10, :],
                      epi1=e1, epi2=e2, filename=outname)

        if 1:
            # you can turn this on or off
            pcd = find_triangulation(K1, K2, F, pts1, pts2)
            visualize_pcd(pcd, filename=os.path.join(output, name + "_rec.png"))