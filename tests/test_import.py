import importlib

def test_package_import():
    pkg = importlib.import_module("torch_camera_design")
    assert hasattr(pkg, "__version__")
