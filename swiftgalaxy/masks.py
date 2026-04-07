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
from typing import Optional, Union, Callable, TYPE_CHECKING
from types import EllipsisType
import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:  # pragma: no cover
    from swiftgalaxy import SWIFTGalaxy

MaskType = Optional[Union[slice, EllipsisType, NDArray]]


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
        An object that can be used to mask an array (slice, boolean array, etc.).

    mask_function : Callable, default: ``None``
        A reference to a function that returns a mask when called.

    sg : ~swiftgalaxy.reader.SWIFTGalaxy, default: ``None``
        A reference to a :class:`~swiftgalaxy.reader.SWIFTGalaxy` that is made available
        when evaluating the mask. Stored separately from the mask evaluation function so
        that it can be modified, for example when the :class:`~swiftgalaxy.masks.LazyMask`
        is copied to a new object.

    combinable : bool
        If ``True``, it declares that for this mask ``data[this_mask][other_mask]`` is
        equivalent to ``data[this_mask[other_mask]]``. This usually means that it is an
        array of integer indices to select from ``data``.

    mask_type : str, default: ``None``
        The :mod:`swiftsimio` "group_type" (e.g. ``"gas"``, ``"dark_matter"``, etc.) that
        this mask applies to.
    """

    _mask_function: Optional[Callable]
    _mask: MaskType
    _evaluated: bool
    _sg: "Optional[SWIFTGalaxy]"
    _mask_type: "Optional[str]"
    _combinable: bool

    def __init__(
        self,
        mask: MaskType = None,
        mask_function: Optional[Callable] = None,
        sg: "Optional[SWIFTGalaxy]" = None,
        combinable: bool = False,
        mask_type: Optional[str] = None,
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
        self._sg = sg
        self._mask_type = mask_type
        self._combinable = combinable
        return

    def _evaluate(self) -> None:
        """Force evaluation the mask function."""
        if not self._evaluated:
            assert self._mask_function is not None  # placate mypy
            self._mask = self._mask_function()
            self._evaluated = True

    def _make_combinable(self) -> None:
        """
        Ensure that the mask can have an arbitrary second mask applied to combine them.

        This is done implicitly if the mask is not already evaluated, or explicitly
        otherwise.
        """
        # need to convert to an integer mask to combine
        # (boolean is insufficient in case of re-ordering masks)
        assert self._sg is not None  # placate mypy
        if self._sg._spatial_mask is None:
            # get a count of particles in the box
            num_part = getattr(self._sg.metadata, f"n_{self._mask_type}")
        else:  # sg._spatial_mask is not None
            # get a count of particles in the spatial mask region
            num_part = np.sum(
                self._sg._spatial_mask.get_masked_counts_offsets()[0][self._mask_type]
            )
        if self._mask_function is not None:
            old_mask_function = self._mask_function  # need reference to the current one
            self._mask_function = lambda: np.arange(num_part)[old_mask_function()]
        if self._evaluated:
            self._mask = np.arange(num_part)[self._mask]
        self._combinable = True

    def _combined_with(self, other_mask: "LazyMask") -> "LazyMask":
        """
        Combine two lazy masks into one, avoiding evaluating them.

        The first mask may be "combinable", which means that the second mask can be
        applied directly to the first. If this flag is not set we first need to make it
        combinable.

        Parameters
        ----------
        other_mask : ~swiftgalaxy.masks.LazyMask
            The second mask to combine.

        Returns
        -------
        ~swiftgalaxy.masks.LazyMask
            The combined mask.
        """
        if self._sg is None:
            raise ValueError(
                "LazyMask must have associated SWIFTGalaxy to use _combined_with."
            )

        # may as well always defer evaluating combination until it's asked for
        def lazy_mask() -> NDArray:
            """
            Evaluate a mask combining two existing masks.

            Returns
            -------
            :class:`~numpy.ndarray`
                The combined mask.
            """
            if not self._combinable:
                self._make_combinable()
            assert isinstance(self.mask, np.ndarray)  # placate mypy
            assert self.mask.dtype == int
            return self.mask[other_mask.mask]

        return LazyMask(
            mask_function=lazy_mask,
            sg=self._sg,
            combinable=True,
            mask_type=self._mask_type,
        )

    @property
    def mask(self) -> MaskType:
        """
        Get the explicitly evaluated mask, evaluating it if necessary.

        Returns
        -------
        slice
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
        :class:`~swiftgalaxy.masks.LazyMask`
            The copy of the :class:`~swiftgalaxy.masks.LazyMask`.
        """
        if self._evaluated:
            return LazyMask(
                mask=self._mask,
                mask_function=self._mask_function,
                sg=self._sg,
                combinable=self._combinable,
                mask_type=self._mask_type,
            )
        else:
            return LazyMask(
                mask_function=self._mask_function,
                sg=self._sg,
                combinable=self._combinable,
                mask_type=self._mask_type,
            )

    def __deepcopy__(self, memo: Optional[dict] = None) -> "LazyMask":
        """
        Make a copy of the :class:`~swiftgalaxy.masks.LazyMask`.

        This copies data (a "deep" copy). Does not deep-copy the reference to the
        swiftgalaxy object, which should be replaced after copying if required.

        Parameters
        ----------
        memo : :obj:`dict` (optional), default: ``None``
            For the copy operation to keep a record of already copied objects.

        Returns
        -------
        :class:`~swiftgalaxy.masks.LazyMask`
            The copy of the :class:`~swiftgalaxy.masks.LazyMask`.
        """
        if self._evaluated:
            return LazyMask(
                mask=deepcopy(self._mask),
                mask_function=deepcopy(self._mask_function),
                sg=self._sg,
                combinable=deepcopy(self._combinable),
                mask_type=deepcopy(self._mask_type),
            )
        else:
            return LazyMask(
                mask_function=deepcopy(self._mask_function),
                sg=self._sg,
                combinable=deepcopy(self._combinable),
                mask_type=deepcopy(self._mask_type),
            )

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
        :obj:`bool`
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
        :obj:`bool`
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

    _masks: dict[str, LazyMask]

    def __init__(
        self,
        **kwargs: Optional[Union[MaskType, LazyMask]],
    ) -> None:
        self._masks = {}
        for k, v in kwargs.items():
            if isinstance(v, LazyMask):
                self._masks[k] = v
                assert getattr(self, k)._mask_type == k, (
                    f"`LazyMask` for {k} has non-matching "
                    f"`mask_type` {getattr(self, k)._mark_type}."
                )
            elif v is None:  # a literal `None` mask would resolve like `np.newaxis`
                self._masks[k] = LazyMask(mask=Ellipsis, mask_type=k)
            else:
                self._masks[k] = LazyMask(mask=v, mask_type=k)
        return

    @classmethod
    def _blank_from_mask_types(
        cls, mask_types: tuple[str], sg: "Optional[SWIFTGalaxy]" = None
    ) -> "MaskCollection":
        """
        Make a set of masks for a list of types where all the masks are just ``Ellipsis``.

        Parameters
        ----------
        mask_types : tuple
            The list of mask types (strings, e.g. ``"gas"``, ``"dark_matter"``, etc.).

        sg : ~swiftgalaxy.reader.SWIFTGalaxy
            The :class:`~swiftgalaxy.reader.SWIFTGalaxy` to associate with the masks.

        Returns
        -------
        MaskCollection
            The collection of masks with all masks set to ``Ellipsis``.
        """
        return cls._from_mask_types_and_values(mask_types=mask_types, sg=sg)

    @classmethod
    def _from_mask_types_and_values(
        cls,
        mask_types: tuple[str],
        sg: "Optional[SWIFTGalaxy]" = None,
        masks: dict[str, MaskType] = {},
    ) -> "MaskCollection":
        """
        Make a set of masks for a list of mask types, defaulting to ``Ellipsis``.

        Parameters
        ----------
        mask_types : tuple
            The list of mask types (strings, e.g. ``"gas"``, ``"dark_matter"``, etc.).

        sg : ~swiftgalaxy.reader.SWIFTGalaxy
            The :class:`~swiftgalaxy.reader.SWIFTGalaxy` to associate with the masks.

        masks : dict
            A dictionary with keys corresponding to (some of) ``mask_types`` and values
            containing the masks (boolean array, slice, index array, etc. - not
            :class:`~swiftgalaxy.masks.LazyMask`) to use for those keys. Any elements of
            ``mask_types`` without a corresponding entry in this dictionary get a default
            mask value of ``Ellipsis``.

        Returns
        -------
        MaskCollection
            The collection of masks set to provided values, or the default ``Ellipsis``.
        """
        return cls(
            **{
                k: LazyMask(mask=masks.get(k, Ellipsis), sg=sg, mask_type=k)
                for k in mask_types
            }
        )

    def __getattr__(self, attr: str) -> LazyMask:
        """
        Access masks as attributes.

        This function is called if an attribute is asked for and not found. In this case
        the ``_masks`` dictionary is checked for a key matching the requested
        attribute. It is returned if found, else a ``AttributeError`` is raised as usual.

        Parameters
        ----------
        attr : :obj:`str`
            The name of the requested attribute.

        Returns
        -------
        ~swiftgalaxy.masks.LazyMask
            The requested :class:`~swiftgalaxy.masks.LazyMask` from the ``_masks``.
        """
        try:
            return self._masks[attr]
        except KeyError:
            raise AttributeError(
                f"Attribute {attr} not found (and not a key of `_masks`)"
            )

    def __copy__(self) -> "MaskCollection":
        """
        Make a copy of the :class:`~swiftgalaxy.masks.MaskCollection`.

        This is without copying data (a "shallow" copy).

        Returns
        -------
        :class:`~swiftgalaxy.masks.MaskCollection`
            The copy of the :class:`~swiftgalaxy.masks.MaskCollection`.
        """
        return MaskCollection(**self._masks)

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
        :class:`~swiftgalaxy.masks.MaskCollection`
            The copy of the :class:`~swiftgalaxy.masks.MaskCollection`.
        """
        return MaskCollection(**{k: deepcopy(v) for k, v in self._masks.items()})

    def combine(self, other_mask_collection: "MaskCollection") -> "MaskCollection":
        """
        Combine this :class:`~swiftgalaxy.masks.MaskCollection` with another.

        ``data[this_mask.<type>.mask][other_mask.<type>.mask]`` and
        ``data[combined_mask.<type>.mask]`` are equivalent, where
        ``combined_mask = this_mask.combine(other_mask)``.

        Parameters
        ----------
        other_mask_collection : ~swiftgalaxy.masks.MaskCollection
            The other mask collection to combine with this one.
        """
        return_collection = {}
        all_keys = set(self._masks.keys()) | set(other_mask_collection._masks.keys())
        for k in all_keys:
            this_mask = getattr(self, k)
            other_mask = getattr(other_mask_collection, k, None)
            return_collection[k] = (
                this_mask._combined_with(other_mask)
                if other_mask is not None
                else this_mask
            )
        return MaskCollection(**return_collection)
