"""Test the utils module of TensorVision."""


def test_get_train_dir():
    """Test if something breaks when get_train_dir is called."""
    from tensorvision.utils import get_train_dir

    get_train_dir()


def test_cfg():
    """Test if one can call the cfg as expected."""
    from tensorvision.utils import cfg
    cfg.data_dir


def test_load_plugins():
    """Test if loading plugins works."""
    from tensorvision.utils import load_plugins
    load_plugins()
