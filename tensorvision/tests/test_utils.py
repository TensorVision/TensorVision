"""Test the utils module of TensorVision."""


def test_set_dirs():
    """Test if setting plugins works."""
    hype_file = "examples/cifar10_minimal.json"
    with open(hype_file, 'r') as f:
        import json
        import os
        hypes = json.load(f)

    from tensorvision.utils import set_dirs

    set_dirs(hypes, hype_file)


def test_cfg():
    """Test if one can call the cfg as expected."""
    from tensorvision.utils import cfg
    cfg.plugin_dir


def test_load_plugins():
    """Test if loading plugins works."""
    from tensorvision.utils import load_plugins
    load_plugins()
