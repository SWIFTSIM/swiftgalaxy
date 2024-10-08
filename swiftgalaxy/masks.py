"""
The :mod:`swiftsimio` masking features are optimized for spatial masking, that is
selecting regions made up of a subset of the SWIFT "top-level cells" in a simulation.
:mod:`swiftgalaxy` masking features aim to support refining these relatively coarse
spatial masks to select particles belonging to individual structures, or other
arbitrary sets of particles, and therefore needs its own masking tools. In their
long term the hope is to merge the two together, but for now the
:class:`~swiftgalaxy.masks.MaskCollection` is the recommended way to define a
selection of particles of different types for use with
:class:`~swiftgalaxy.reader.SWIFTGalaxy`.
"""

from copy import deepcopy
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
        """
        Return ``None`` if an attribute of the object doesn't exist.

        This function is called if an attribute is asked for and not found.
        Instead of the usual behaviour of raising a :exc:`AttributeError`,
        ``None`` is returned.

        Parameters
        ----------
        attr : :obj:`str`
            The name of the requested attribute.

        Returns
        -------
        out : None
            If we reach calling this function the attribute is not found and we
            return ``None``.
        """
        return None

    def __copy__(self) -> "MaskCollection":
        """
        Make a copy of the :class:`~swiftgalaxy.masks.MaskCollection` without copying
        data (a "shallow" copy).

        Returns
        -------
        out : :class:`~swiftgalaxy.masks.MaskCollection`
            The copy of the :class:`~swiftgalaxy.masks.MaskCollection`.
        """
        return MaskCollection(**self.__dict__)

    def __deepcopy__(self, memo: Optional[dict] = None) -> "MaskCollection":
        """
        Make a copy of the :class:`~swiftgalaxy.masks.MaskCollection`, copying data
        (a "deep" copy).

        Parameters
        ----------
        memo : :obj:`dict` (optional), default: ``None``
            For the copy operation to keep a record of already copied objects.

        Returns
        -------
        out : :class:`~swiftgalaxy.masks.MaskCollection`
            The copy of the :class:`~swiftgalaxy.masks.MaskCollection`.
        """
        return MaskCollection(**{k: deepcopy(v) for k, v in self.__dict__.items()})
