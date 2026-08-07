import auricle


def test_version_string():
    assert isinstance(auricle.__version__, str)
    assert auricle.__version__.count(".") == 2
