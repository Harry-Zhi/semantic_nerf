import numpy as np
import quaternion
import trimesh

def habitat_world_transformations():
    import habitat_sim
    # Transforms between the habitat frame H (y-up) and the world frame W (z-up).
    T_wh = np.identity(4)

    # https://stackoverflow.com/questions/1171849/finding-quaternion-representing-the-rotation-from-one-vector-to-another
    T_wh[0:3, 0:3] = quaternion.as_rotation_matrix(habitat_sim.utils.common.quat_from_two_vectors(
            habitat_sim.geo.GRAVITY, np.array([0.0, 0.0, -1.0])))

    T_hw = np.linalg.inv(T_wh)

    return T_wh, T_hw

def opencv_to_opengl_camera(transform=None):
    if transform is None:
        transform = np.eye(4)
    return transform @ trimesh.transformations.rotation_matrix(
        np.deg2rad(180), [1, 0, 0]
    )

def opengl_to_opencv_camera(transform=None):
    if transform is None:
        transform = np.eye(4)
    return transform @ trimesh.transformations.rotation_matrix(
        np.deg2rad(-180), [1, 0, 0]
    )

def Twc_to_Thc(T_wc):  # opencv-camera to world transformation ---> habitat-caemra to habitat world transformation
    T_wh, T_hw = habitat_world_transformations()
    T_hc = T_hw @ T_wc @ opengl_to_opencv_camera()
    return T_hc


def Thc_to_Twc(T_hc):  # habitat-caemra to habitat world transformation --->  opencv-camera to world transformation
    T_wh, T_hw = habitat_world_transformations()
    T_wc = T_wh @ T_hc @ opencv_to_opengl_camera()
    return T_wc


def combine_pose(t: np.array, q: quaternion.quaternion) -> np.array:
    T = np.identity(4)
    T[0:3, 3] = t
    T[0:3, 0:3] = quaternion.as_rotation_matrix(q)
    return T