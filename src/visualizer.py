import json
import multiprocessing as mp
from dataclasses import dataclass, field
from functools import partial

import numpy as np
import open3d as o3d
import pyautogui


# dataclass for window settings
@dataclass
class WindowSettings:
    name: str = "Open3D"
    width: int = pyautogui.size()[0]
    height: int = pyautogui.size()[1]
    left: int = 0
    top: int = 0

    def hsplit(self, splits: list[float] = [0.5, 0.5]) -> list["WindowSettings"]:
        windows = []
        for idx, split in enumerate(splits):
            window = WindowSettings()
            window.name = self.name
            window.width = int(self.width * split)
            window.height = self.height
            window.left = int(self.left + self.width * sum(splits[:idx]))
            window.top = self.top
            windows.append(window)
        return windows


# dataclass for view settings
@dataclass
class ViewSettings:
    field_of_view: float = 5.0
    front: list[float] = field(default_factory=lambda: [0.0, 0.3, -1.0])
    lookat: list[float] = None
    up: list[float] = field(default_factory=lambda: [0.0, -0.95, -0.3])
    zoom: float = 1.3

    def apply_to_vis(self, vis: o3d.visualization.Visualizer) -> None:
        view = json.loads(vis.get_view_status())
        for key, value in self.__dict__.items():
            if value is not None:
                view["trajectory"][0][key] = value
        vis.set_view_status(json.dumps(view))


# class to visualize picked points in a point cloud
class PickPointVisualizer:
    def __init__(
        self,
        point_cloud: o3d.geometry.PointCloud,
        window_settings: WindowSettings = WindowSettings(),
        view_settings: ViewSettings = ViewSettings(),
    ):
        # set window and view settings
        self.window_settings = window_settings
        self.view_settings = view_settings

        # set point cloud and initialize list of picked points
        self.point_cloud = point_cloud
        self.vertices = np.asarray(
            point_cloud.points
        ).copy()  # copy to avoid changes if the point cloud is modified by another process
        self.picked_idx = []

        # set up process to run visualizer
        self.queue = mp.SimpleQueue()
        self.process = mp.Process(target=self.__run_process, args=(self.queue,))
        self.process.start()

    def __run_process(self, queue: mp.SimpleQueue):
        # set verbosity level to ignore picked point info
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Warning)

        # create window and add point cloud
        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window(
            window_name=self.window_settings.name,
            width=self.window_settings.width,
            height=self.window_settings.height,
            left=self.window_settings.left,
            top=self.window_settings.top,
        )
        vis.add_geometry(self.point_cloud)
        self.view_settings.apply_to_vis(vis)

        # register callback to update picked points
        vis.register_animation_callback(
            partial(self.__update_picked_points, queue=queue)
        )

        # run visualizer
        vis.run()
        vis.destroy_window()

    def __update_picked_points(
        self, vis: o3d.visualization.Visualizer, queue: mp.SimpleQueue
    ):
        picked_idx = vis.get_picked_points()
        queue.put(picked_idx)
        return False

    def get_picked_points(self) -> np.ndarray:
        if not self.queue.empty():
            self.picked_idx = self.queue.get()
        return self.vertices[self.picked_idx]

    def is_running(self):
        return self.process.is_alive()

    def close(self):
        self.process.terminate()


# class to visualize geometries with a non-blocking, per-frame display function
class NonBlockingVisualizer:
    def __init__(
        self,
        window_settings: WindowSettings = WindowSettings(),
        view_settings: ViewSettings = ViewSettings(),
    ):
        # create visualizer and window
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(
            window_name=window_settings.name,
            width=window_settings.width,
            height=window_settings.height,
            left=window_settings.left,
            top=window_settings.top,
        )
        self.view_settings = view_settings

        # handle quit key press
        self.__quit_pressed = False
        self.vis.register_key_callback(ord("Q"), self.__key_callback_quit)

    def add_geometry(
        self, geometry: o3d.geometry.Geometry, reset_bounding_box: bool = True
    ) -> None:
        self.vis.add_geometry(geometry, reset_bounding_box=reset_bounding_box)
        self.view_settings.apply_to_vis(self.vis)

    def __key_callback_quit(self, *args, **kwargs) -> None:
        self.__quit_pressed = True

    def render_frame(self) -> bool:
        if self.__quit_pressed:
            return False
        self.vis.poll_events()
        self.vis.update_renderer()
        return not self.__quit_pressed

    def close(self):
        self.__quit_pressed = True
        self.vis.destroy_window()


# class to visualize geometries with a toggleable visibility
class ToggleVisualizer(NonBlockingVisualizer):
    def __init__(
        self,
        window_settings: WindowSettings = WindowSettings(),
        view_settings: ViewSettings = ViewSettings(),
    ):
        # initialize visualizer window
        super().__init__(window_settings, view_settings)

        # initialize list of geometry visibilities
        self.geometry_visibiity = []

        # set key callbacks for toggling visibility of geometries (1-9)
        for idx in range(9):
            self.vis.register_key_callback(
                ord(str(idx)),
                partial(self.__key_callback_toggle_visibility, idx=idx - 1),
            )

        # Add key command in alignment visualizer to alt-tab to point picker visualizers
        # self.vis.register_key_callback(ord("A"), self.__key_callback_switch_windows)

    def add_geometry(
        self, geometry: o3d.geometry.Geometry, reset_bounding_box: bool = True
    ) -> None:
        self.geometry_visibiity.append({"geometry": geometry, "visible": True})
        super().add_geometry(geometry, reset_bounding_box)

    def update_geometry(self, geometry: o3d.geometry.Geometry) -> None:
        self.vis.update_geometry(geometry)

    def remove_geometry(
        self, geometry: o3d.geometry.Geometry, reset_bounding_box: bool = True
    ) -> None:
        self.vis.remove_geometry(geometry, reset_bounding_box)

    def __key_callback_toggle_visibility(
        self, __vis: o3d.visualization.Visualizer, idx: int
    ) -> bool:
        if idx > len(self.geometry_visibiity):
            return False
        geometry = self.geometry_visibiity[idx]
        if geometry["visible"]:
            self.vis.remove_geometry(geometry["geometry"], reset_bounding_box=False)
        else:
            self.vis.add_geometry(geometry["geometry"], reset_bounding_box=False)
        geometry["visible"] = not geometry["visible"]
        self.vis.update_geometry(geometry["geometry"])
        return True

    # def __key_callback_switch_windows(self, *args, **kwargs):
    #     pyautogui.hotkey("ctrl", "tab")
    #     pyautogui.keyDown("ctrl")
    #     pyautogui.press("tab")
    #     pyautogui.press("tab", _pause=False)
    #     pyautogui.keyUp("ctrl", _pause=False)


class RegistrationVisualizer(ToggleVisualizer):
    def __init__(
        self,
        window_settings: WindowSettings = WindowSettings(),
        view_settings: ViewSettings = ViewSettings(),
        # use_ransac: bool = False,
    ):
        super().__init__(window_settings, view_settings)
        # self.use_ransac = use_ransac

        # set key callback for running ICP registration
        self.icp_requested = False
        self.vis.register_key_callback(ord("I"), self.__key_callback_run_icp)

        # set key callback for toggling ransac in point-to-point registration
        # self.vis.register_key_callback(ord("R"), self.__key_callback_toggle_ransac)

    # def __key_callback_toggle_ransac(self, *args, **kwargs):
    #     self.use_ransac = not self.use_ransac

    def __key_callback_run_icp(self, *args, **kwargs):
        self.icp_requested = True


# Manually crop point cloud
def crop_manually(
    pcd: o3d.geometry.PointCloud,
    view_settings: ViewSettings = ViewSettings(),
) -> o3d.geometry.PointCloud:
    crop_vis = o3d.visualization.VisualizerWithEditing()
    crop_vis.create_window()
    crop_vis.add_geometry(pcd)
    view_settings.apply_to_vis(crop_vis)
    crop_vis.run()
    crop_vis.destroy_window()
    return crop_vis.get_cropped_geometry()
