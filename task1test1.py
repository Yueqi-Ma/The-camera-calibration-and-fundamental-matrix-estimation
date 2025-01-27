import numpy as np
import utils


def find_projection(pts2d, pts3d):
    
    N = pts2d.shape[0]
    A = np.zeros((2 * N, 12))
    for i in range(N):
        X, Y, Z = pts3d[i]
        u, v = pts2d[i]

        A[2*i] = [X, Y, Z, 1, 0, 0, 0, 0, -u*X, -u*Y, -u*Z, -u]
        A[2*i+1] = [0, 0, 0, 0, X, Y, Z, 1, -v*X, -v*Y, -v*Z, -v]

    _, _, V = np.linalg.svd(A)
    M = V[-1].reshape(3, 4)

    return M

if __name__ == '__main__':
    pts2d = np.loadtxt("task1/pts2d.txt")
    pts3d = np.loadtxt("task1/pts3d.txt")

    M = find_projection(pts2d, pts3d)
    print("Projection Matrix:")
    print(M)



def compute_distance(pts2d, pts3d):
   
    M = find_projection(pts2d, pts3d)
    N = pts2d.shape[0]
    homogeneous_pts3d = np.hstack((pts3d, np.ones((N, 1))))
    reprojected_pts2d = np.dot(M, homogeneous_pts3d.T).T
    reprojected_pts2d /= reprojected_pts2d[:, 2][:, np.newaxis]
    distances = np.linalg.norm(pts2d - reprojected_pts2d[:, :2], axis=1)
    average_distance = np.mean(distances)

    return average_distance

if __name__ == '__main__':
    pts2d = np.loadtxt("task1/pts2d.txt")
    pts3d = np.loadtxt("task1/pts3d.txt")

   
    foundDistance = compute_distance(pts2d, pts3d)
    print("Distance: %f" % foundDistance)