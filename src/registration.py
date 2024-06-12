import numpy as np
import open3d as o3d

from src.utils import numpy_2_pcd
from src.visualizer import PickPointVisualizer


class PickedPointsHandler:
    # tracks the points picked by a PickPointVisualizer
    # when update() is called, if the list of picked points has changed,
    # it returns true
    # uses get_picked_points() to get the current set of picked points
    def __init__(self, point_picker: PickPointVisualizer):
        self.point_picker = point_picker
        self.picked_points = np.zeros((0, 3))  # needs this shape for np.array_equal

    def update(self) -> bool:
        # Check if the picked points have changed
        new_picked_points = self.point_picker.get_picked_points()
        if (
            len(new_picked_points) == len(self.picked_points)
            and np.linalg.norm(new_picked_points - self.picked_points) < 1e-6
        ):
            return False

        # Update picked points
        self.picked_points = new_picked_points
        return True

    def get_picked_points(self) -> np.ndarray:
        return self.picked_points


class Point2PointRegistrationHandler:
    def __init__(self, with_scaling: bool = False):
        self.transformation_pipeline = (
            o3d.pipelines.registration.TransformationEstimationPointToPoint()
        )
        self.transformation_pipeline.with_scaling = with_scaling

        self.transform_mtx = np.eye(4)
        self.ransac_result = None

    def register(
        self,
        source_points_3d: np.ndarray,
        target_points_3d: np.ndarray,
        with_ransac=False,
        max_correspondence_distance=5,  # in mm
    ) -> np.ndarray:
        # Check input shapes
        assert source_points_3d.shape[1] == 3, "source_points_3d must have shape (N, 3)"
        assert target_points_3d.shape[1] == 3, "target_points_3d must have shape (N, 3)"
        assert (
            source_points_3d.shape[0] == target_points_3d.shape[0]
        ), "source_points_3d and target_points_3d must have the same number of points"

        # Assign correspondences
        n_points = source_points_3d.shape[0]
        source = numpy_2_pcd(source_points_3d)
        target = numpy_2_pcd(target_points_3d)
        corr = o3d.utility.Vector2iVector(
            np.stack([np.arange(n_points), np.arange(n_points)]).T
        )

        # Compute transformation
        if with_ransac:
            self.ransac_result = (
                o3d.pipelines.registration.registration_ransac_based_on_correspondence(
                    source,
                    target,
                    corr,
                    max_correspondence_distance=max_correspondence_distance,
                    estimation_method=self.transformation_pipeline,
                )
            )
            self.transform_mtx = (
                self.ransac_result.transformation.copy()
            )  # copy b/c returned matrix is read-only
        else:
            self.transform_mtx = self.transformation_pipeline.compute_transformation(
                source, target, corr
            ).copy()  # copy b/c returned matrix is read-only
        return self.transform_mtx

    def get_transform(self) -> np.ndarray:
        return self.transform_mtx


class ICPRegistrationHandler:
    def __init__(self):
        pass

    def register(
        self,
        source_pcd: o3d.geometry.PointCloud,
        target_pcd: o3d.geometry.PointCloud,
        initial_transform: np.ndarray = np.identity(4),
    ) -> np.ndarray:
        # # estimate normals
        # radius = 0.002 * 1000
        # target_pcd.estimate_normals(
        #     o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30)
        # )
        # source_pcd.estimate_normals(
        #     o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30)
        # )

        # run ICP pipeline
        threshold = 0.004 * 1000
        reg = o3d.pipelines.registration.registration_icp(
            source_pcd,
            target_pcd,
            threshold,
            initial_transform,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50),
        )
        self.ICP_result = reg
        self.transform_mtx = reg.transformation[:]
        return self.transform_mtx
