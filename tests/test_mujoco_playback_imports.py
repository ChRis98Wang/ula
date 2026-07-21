import importlib
import sys


class BlockMediapyImport:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "mediapy" or fullname.startswith("mediapy."):
            raise ModuleNotFoundError("blocked mediapy")
        return None


def test_mujoco_playback_import_does_not_require_mediapy():
    previous_module = sys.modules.pop("upper_body_skeleton.mujoco_playback", None)
    previous_mediapy = sys.modules.pop("mediapy", None)
    blocker = BlockMediapyImport()
    sys.meta_path.insert(0, blocker)
    try:
        module = importlib.import_module("upper_body_skeleton.mujoco_playback")
        assert hasattr(module, "render_motion")
    finally:
        sys.meta_path.remove(blocker)
        sys.modules.pop("upper_body_skeleton.mujoco_playback", None)
        if previous_module is not None:
            sys.modules["upper_body_skeleton.mujoco_playback"] = previous_module
        if previous_mediapy is not None:
            sys.modules["mediapy"] = previous_mediapy
