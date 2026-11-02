import contextgc


def test_version_and_public_api():
    assert contextgc.__version__ == "0.1.0"
    for name in contextgc.__all__:
        assert hasattr(contextgc, name), name
