from copy import copy
from typing import Optional


class MaskCollection(object):
    """
    Barebones container for mask objects.

    Takes a set of kwargs at initialisation and assigns their values to
    attributes of the object. Attempts to access a non-existent attribute
    returns :obj:`None` instead of raising an :exc:`AttributeError`.

    This is intended to hold masks that can be applied to
    :class:`~swiftsimio.objects.cosmo_array` objects under the names of
    particle types (e.g. ``gas``, ``dark_matter``, etc.), but this is not
    checked or enforced.

    Parameters
    ----------

    **kwargs
        Any items passed as kwargs will have their values passed to
        correspondingly named attributes of this object.

    Notes
    -----

    .. note::
        The :mod:`velociraptor.swift.swift` module makes some use of a
        :obj:`namedtuple` called ``MaskCollection``. These objects are not
        valid where :mod:`swiftgalaxy` functions expect a :obj:`MaskCollection`
        because :obj:`namedtuple` objects are immutable.

    Examples
    --------
    ::

        n_dm = 123  # suppose this is number of dark matter particles
        # all these masks select all particles:
        MaskCollection(
            gas=np.s_[...],
            dark_matter=np.ones(n_dm, dtype=bool),
            stars=None
        )
    """

    # Could use dataclasses module, but requires python 3.7+

    def __init__(self, **kwargs) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)
        return

    def __getattr__(self, attr: str) -> None:
        # If attribute does not exist.
        return None

    def __copy__(self) -> "MaskCollection":
        return MaskCollection(**self.__dict__)

    def __deepcopy__(self, memo: Optional[dict] = None) -> "MaskCollection":
        return MaskCollection(**{k: copy(v) for k, v in self.__dict__.items()})
