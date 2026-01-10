import importlib.metadata
import swiftgalaxy


class TestVersion:
    """Check that version numbering is consistent."""

    def test_code_version(self):
        """Check that code version matches pyproject.toml version."""
        assert importlib.metadata.version("swiftgalaxy") == swiftgalaxy.__version__

    def test_codemeta_version(self):
        """Check that the version in codemeta.json matches pyproject.toml version."""
        with open("codemeta.json") as f:
            codemeta_content = f.read()
        assert (
            f'"version": "{importlib.metadata.version("swiftgalaxy")}",'
            in codemeta_content
        )
