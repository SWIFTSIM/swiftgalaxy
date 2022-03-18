class MaskCollection(object):

    # Could use dataclasses module, but requires python 3.7+

    def __init__(self, **kwargs) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)
        return

    def __getattr__(self, attr: str) -> None:
        # If attribute does not exist.
        return None
