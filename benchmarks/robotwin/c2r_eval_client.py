"""Isolated C2R client for the installed RoboTwin revision.

Reuse the standard manifest/rollout protocol. Native scene sampling, planners,
success predicates and step limits are unchanged; recording is disabled.
"""

from __future__ import annotations

import importlib
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmarks.robotwin import eval_policy_wrapper as wrapper  # noqa: E402


def prepare_private_runtime():
    import yaml

    native = Path(os.environ["ROBOTWIN_PATH"]).resolve()
    runtime = Path(os.environ["ROBOTWIN_RUNTIME_ROOT"]).resolve()
    runtime.mkdir(parents=True, exist_ok=True)
    config = runtime / "task_config"
    if config.is_symlink():
        raise ValueError("C2R runtime task_config must be a private copy")
    if not config.exists():
        shutil.copytree(native / "task_config", config)
    mode = "demo_randomized"
    path = config / f"{mode}.yml"
    settings = yaml.safe_load(path.read_text())
    settings.update(collect_data=False, save_data=False)
    # Expert manifest construction has no video rollout. Evaluation keeps the
    # native eval_video_log setting and writes only to the private runtime.
    if "--operation" in sys.argv and sys.argv[sys.argv.index("--operation") + 1] == "build_manifest":
        settings["eval_video_log"] = False
    path.write_text(yaml.safe_dump(settings, sort_keys=False))
    os.environ["ASSETS_PATH"] = str(native)


def main():
    if sys.argv[1:] == ["--render-probe"]:
        # Run only after training releases the GPU. CUDA availability alone does
        # not establish that the Vulkan loader can create a render device.
        wrapper._prewarm_cuda_for_curobo()
        import sapien

        renderer = sapien.render.RenderSystem(device="cuda:0")
        scene = sapien.Scene([sapien.physx.PhysxCpuSystem(), renderer])
        scene.set_ambient_light([0.5, 0.5, 0.5])
        builder = scene.create_actor_builder()
        builder.add_box_visual(half_size=[0.1, 0.1, 0.1])
        builder.build_static()
        camera = scene.add_camera("probe", 64, 64, 1.0, 0.1, 10)
        camera.set_entity_pose(sapien.Pose([-1, 0, 0]))
        scene.update_render()
        camera.take_picture()
        picture = camera.get_picture("Color")
        assert picture.shape == (64, 64, 4)
        assert float(picture[..., :3].std()) > 0
        print(
            json.dumps(
                {
                    "renderer": str(renderer.device),
                    "pci": renderer.device.pci_string,
                    "shape": list(picture.shape),
                    "passed": True,
                }
            ),
            flush=True,
        )
        return 0
    prepare_private_runtime()
    original_load = wrapper._load_robotwin_eval_module

    def load_with_native_compat(path):
        module = original_load(path)
        # This RoboTwin revision contains a legacy place_fan Reward constructor.
        # Use the same narrow constructor adapter validated during collection.
        sys.path.insert(0, os.environ["RAX_ROOT"])
        from services.robotwin.native_compat import compatibility_report, install

        install(Path(path), importlib.import_module("envs.reward"))
        print("[native_compat] " + json.dumps(compatibility_report()), flush=True)
        return module

    wrapper._load_robotwin_eval_module = load_with_native_compat
    return wrapper.main()


if __name__ == "__main__":
    raise SystemExit(main())
