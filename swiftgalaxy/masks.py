"""
Mask particles to select galaxies. Supplements :mod:`swiftsimio`'s spatial masking.

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
from typing import Optional, Union, Callable
from types import EllipsisType
from numpy.typing import ArrayLike


class LazyMask(object):
    """
    A class to hold a function to evaluate a mask until it is needed.

    This class can contain either an explicitly evaluated mask (boolean array,
    slice, etc.) or a reference to a function that returns such a mask when
    called. When the ``mask`` property is accessed, if the mask is already
    evaluated it is returned, otherwise it is evaluated and returned.

    The ``_evaluated`` attribute tracks whether the explicitly evaluated
    mask is available.

    Parameters
    ----------
    mask : slice, default: ``None``
        An object that can be used to mask an array (could be a slice, boolean array, etc)

    mask_function : Callable, default: ``None``
        A reference to a function that returns a mask when called.
    """

    _mask_function: Optional[Callable]
    _mask: Optional[Union[slice, EllipsisType, ArrayLike]]
    _evaluated: bool

    def __init__(
        self,
        mask: Optional[Union[slice, EllipsisType, ArrayLike]] = None,
        mask_function: Optional[Callable] = None,
    ) -> None:
        if mask_function is None and mask is None:
            self._mask = None
            self._evaluated = True
        elif mask_function is not None and mask is None:
            self._mask_function = mask_function
            self._evaluated = False
            # leave self._mask unset
        else:
            self._mask = mask
            self._mask_function = mask_function
            self._evaluated = True
        return

    def _evaluate(self) -> None:
        """Force evaluation the mask function."""
        if not self._evaluated:
            assert self._mask_function is not None  # placate mypy
            self._mask = self._mask_function()
            self._evaluated = True

    @property
    def mask(self) -> Optional[Union[slice, EllipsisType, ArrayLike]]:
        """
        Get the explicitly evaluated mask, evaluating it if necessary.

        Returns
        -------
        out : slice
            The explicitly evaluated mask.
        """
        if not self._evaluated:
            self._evaluate()
        return self._mask

    def __copy__(self) -> "LazyMask":
        """
        Make a copy of the :class:`~swiftgalaxy.masks.LazyMask`.

        This is without copying data (a "shallow" copy).

        Returns
        -------
        out : :class:`~swiftgalaxy.masks.LazyMask`
            The copy of the :class:`~swiftgalaxy.masks.LazyMask`.
        """
        if self._evaluated:
            return LazyMask(mask=self._mask, mask_function=self._mask_function)
        else:
            return LazyMask(mask_function=self._mask_function)

    def __deepcopy__(self, memo: Optional[dict] = None) -> "LazyMask":
        """
        Make a copy of the :class:`~swiftgalaxy.masks.LazyMask`.

        This copies data (a "deep" copy).

        Parameters
        ----------
        memo : :obj:`dict` (optional), default: ``None``
            For the copy operation to keep a record of already copied objects.

        Returns
        -------
        out : :class:`~swiftgalaxy.masks.LazyMask`
            The copy of the :class:`~swiftgalaxy.masks.LazyMask`.
        """
        if self._evaluated:
            return LazyMask(
                mask=deepcopy(self._mask), mask_function=deepcopy(self._mask_function)
            )
        else:
            return LazyMask(mask_function=deepcopy(self._mask_function))

    def __eq__(self, other: object) -> bool:
        """
        Check this mask for equality with another.

        If compared with another :class:`~swiftgalaxy.masks.LazyMask` then the comparison
        of the two explicitly evaluated masks is returned. If compared to any other
        object, comparison is attempted with the explicitly evaluated mask.

        If the mask has not been evaluated, no evaluation is triggered.

        Parameters
        ----------
        other : :obj:`object`
            The mask to compare with.

        Returns
        -------
        out : :obj:`bool`
            Comparison result.

        Raises
        ------
        ValueError
            If the internal mask is not already evaluated. Also raised if the compared
            object is a :class:`~swiftgalaxy.masks.LazyMask` and its mask is not
            evaluated.
        """
        if isinstance(other, LazyMask):
            if hasattr(self, "_mask") and hasattr(other, "_mask"):
                masks_equal = self._mask == other._mask
            else:
                raise ValueError(
                    "Cannot compare when one or more masks are not evaluated."
                )
        else:
            if hasattr(self, "_mask"):
                masks_equal = self._mask == other
            else:
                raise ValueError(
                    "Cannot compare when one or more masks are not evaluated."
                )
        if type(masks_equal) is not bool:
            masks_equal = all(masks_equal)
        return masks_equal

    def __ne__(self, other: object) -> bool:
        """
        Check this mask for inequality with another.

        If compared with another :class:`~swiftgalaxy.masks.LazyMask` then the comparison
        of the two explicitly evaluated masks is returned. If compared to any other
        object, comparison is attempted with the explicitly evaluated mask.

        If the mask has not been evaluated, no evaluation is triggered.

        Parameters
        ----------
        other : :obj:`object`
            The mask to compare with.

        Returns
        -------
        out : :obj:`bool`
            Comparison result.

        Raises
        ------
        ValueError
            If the internal mask is not already evaluated. Also raised if the compared
            object is a :class:`~swiftgalaxy.masks.LazyMask` and its mask is not
            evaluated.
        """
        return not self.__eq__(other)


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

    def __init__(
        self,
        **kwargs: Optional[Union[slice, EllipsisType, ArrayLike, LazyMask]],
    ) -> None:
        for k, v in kwargs.items():
            if isinstance(v, LazyMask):
                setattr(self, k, v)
            elif v is None:
                pass
            else:
                setattr(self, k, LazyMask(mask=v))
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
        Make a copy of the :class:`~swiftgalaxy.masks.MaskCollection`.

        This is without copying data (a "shallow" copy).

        Returns
        -------
        out : :class:`~swiftgalaxy.masks.MaskCollection`
            The copy of the :class:`~swiftgalaxy.masks.MaskCollection`.
        """
        return MaskCollection(**self.__dict__)

    def __deepcopy__(self, memo: Optional[dict] = None) -> "MaskCollection":
        """
        Make a copy of the :class:`~swiftgalaxy.masks.MaskCollection`.

        This copies data (a "deep" copy).

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
