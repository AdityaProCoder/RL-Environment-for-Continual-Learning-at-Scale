import open_continual_env
import open_continual_env.env
import open_continual_env.trajectory
import open_continual_env.controller
import open_continual_env.baselines
import open_continual_env.benchmark


def test_package_import():
    assert open_continual_env.__version__ == "0.1.0"


def test_submodules_exist():
    assert open_continual_env.env is not None
    assert open_continual_env.trajectory is not None
    assert open_continual_env.controller is not None
    assert open_continual_env.baselines is not None
    assert open_continual_env.benchmark is not None
