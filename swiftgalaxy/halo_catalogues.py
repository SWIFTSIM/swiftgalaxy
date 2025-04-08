"""
This module contains classes providing interfaces to halo catalogues used with
SWIFT so that :mod:`swiftgalaxy` can obtain the information it requires in a
streamlined way. Currently SOAP_ and Velociraptor_ halo catalogues are supported,
as well as Caesar_ catalogues, others can be supported on request.

.. _Velociraptor: https://ui.adsabs.harvard.edu/abs/2019PASA...36...21E/\
abstract
.. _Caesar: https://caesar.readthedocs.io/en/latest/
.. _SOAP: https://github.com/SWIFTSIM/SOAP
"""

from warnings import warn
from abc import ABC, abstractmethod
from collections.abc import Sequence
import numpy as np
import unyt as u
from swiftsimio import SWIFTMask, SWIFTDataset, mask
from swiftgalaxy.masks import MaskCollection
from swiftsimio.objects import cosmo_array, cosmo_factor, a

from typing import Any, Union, Optional, TYPE_CHECKING, List, Set, Dict
from numpy.typing import NDArray

if TYPE_CHECKING:
    from swiftgalaxy.reader import SWIFTGalaxy
    from velociraptor.catalogue.catalogue import Catalogue as VelociraptorCatalogue
    from caesar.loader import Galaxy as CaesarGalaxy, Halo as CaesarHalo


class _MaskHelper:
    """
    Class to mask halo catalogue attributes when requested for those halo catalogue
    interfaces that need additional functionality to support this.

    Parameters
    ----------
    data : :obj:`object`
        The halo catalogue data to be masked.

    mask : :obj:`int` or :obj:`slice`
        The row, index or slice to select from the data.
    """

    def __init__(self, data: u.unyt_array, mask: Union[int, slice]):
        self._mask_helper_data = data
        self._mask_helper_mask = mask

    def __getattr__(self, attr: str) -> Any:
        """
        Get an attribute of the object, applying the mask.

        Looks up the requested attribute in the halo catalogue data and applies the
        mask provided at initialization before returning.

        Parameters
        ----------
        attr : :obj:`str`
            The name of the attribute.

        Returns
        -------
        out : :obj:`object`
            The requested attributed with mask applied.
        """
        return getattr(self._mask_helper_data, attr)[self._mask_helper_mask]

    def __getitem__(self, key: str) -> Any:
        """
        Enables looking up attributes with square-bracket syntax.

        Redirects requests for items to
        :func:`~swiftgalaxy.halo_catalogues._MaskHelper.__getattr__`.

        Parameters
        ----------
        key : :obj:`str`
            The name of the attribute to retrieve.

        Returns
        -------
        out : :obj:`object`
            The requested attributed with mask applied.
        """
        return self.__getattr__(key)

    def __repr__(self) -> str:
        """
        Get a string representation of the object.

        Redirects the request to the ``data`` object provided at initialization.

        Returns
        -------
        out : :obj:`str`
            The string representation.
        """
        return self._mask_helper_data.__repr__()


class _HaloCatalogue(ABC):
    """
    Abstract base class for halo catalogue interface classes.

    Handles common operations needed for the interface between halo catalogues and the
    main :class:`~swiftgalaxy.reader.SWIFTGalaxy` class. Also defines what further
    functions interface classes need to provide.

    Parameters
    ----------
    extra_mask : :obj:`str` or :class:`~swiftgalaxy.masks.MaskCollection` (optional), \
    default: ``"bound_only"``
        The "extra" mask to apply to data when it is read (extra in the sense of in
        addition to the spatial mask applied by the
        :class:`~swiftgalaxy.reader.SWIFTGalaxy`).
    """

    _user_spatial_offsets: Optional[cosmo_array] = None
    _multi_galaxy: bool = False
    _multi_galaxy_catalogue_mask: Optional[int] = None
    _multi_galaxy_index_mask: Optional[Union[int, slice]] = None
    _multi_count: int
    _index_attr: Optional[str]

    def __init__(
        self, extra_mask: Optional[Union[str, MaskCollection]] = "bound_only"
    ) -> None:
        self.extra_mask = extra_mask
        self._check_multi()
        if self._index_attr is not None:
            if isinstance(getattr(self, self._index_attr), u.unyt_array):
                setattr(
                    self,
                    self._index_attr,
                    getattr(self, self._index_attr)
                    .to_value(u.dimensionless)
                    .astype(int),
                )
            if self._multi_galaxy:
                if len(set(getattr(self, self._index_attr))) < len(
                    getattr(self, self._index_attr)
                ):
                    raise ValueError(
                        f"{self._index_attr[1:]} must not contain duplicates."
                    )
        self._load()
        return

    def _mask_multi_galaxy(self, index) -> None:
        """
        Switch on restricting the catalogue to a single row.

        Used when the halo catalogue interface is in multi-galaxy mode (when
        iterating over many :class:`~swiftgalaxy.reader.SWIFTGalaxy`'s) to
        restrict the catalogue to a single row, for use during one such iteration.

        In most cases the mask into the list of indices (targets provided by user)
        and the mask into the catalogue (from the relevant halo catalogue) are the
        same, but SOAP for example implicitly sorts the target list when defining
        the swiftsimio mask for the halo catalogue, so the two indices differ in
        that case. The SOAP class therefore implements its own _mask_multi_galaxy
        function that overrides this one.

        Parameters
        ----------
        index : :obj:`int`
            The position in the list of selected catalogue objects to mask down to.

        See Also
        --------
        swiftgalaxy.halo_catalogues._HaloCatalogue._unmask_multi_galaxy
        """
        if self._index_attr is None:
            self._multi_galaxy_index_mask = index
        else:
            self._multi_galaxy_index_mask = int(
                np.argmax(np.array(getattr(self, self._index_attr)) == index)
            )
        self._multi_galaxy_catalogue_mask = index
        return

    def _unmask_multi_galaxy(self) -> None:
        """
        Switch off restricting the catalogue to a single row.

        Used when the halo catalogue interface is in multi-galaxy mode (when
        iterating over many :class:`~swiftgalaxy.reader.SWIFTGalaxy`'s) to
        remove the mask on the catalogue to a single row.

        See Also
        --------
        swiftgalaxy.halo_catalogues._HaloCatalogue._mask_multi_galaxy
        """
        self._multi_galaxy_catalogue_mask = None
        self._multi_galaxy_index_mask = None
        return

    def _get_user_spatial_mask(self, snapshot_filename: str) -> SWIFTMask:
        """
        Turn a bounding box provided by the user into a mask object.

        A :class:`~swiftgalaxy.reader.SWIFTGalaxy` allows a user to define a
        spatial mask as three coordinate pairs defining a bounding box. This
        gets stored in ``self._user_spatial_offsets``. This function uses the
        bounding box to initialize a :class:`~swiftsimio.masks.SWIFTMask` object
        defining the spatial mask.

        Parameters
        ----------
        snapshot_filename : :obj:`str`
            The location of the SWIFT snapshot file.

        Returns
        -------
        out : :class:`~swiftsimio.masks.SWIFTMask`
            The spatial mask to select particles in the region of interest.
        """
        sm = mask(snapshot_filename, spatial_only=True)
        region = self._user_spatial_offsets
        if region is not None:
            for ax in range(3):
                region[ax] = (
                    [self.centre[ax] + region[ax][0], self.centre[ax] + region[ax][1]]
                    if region[ax] is not None
                    else None
                )
            sm.constrain_spatial(region)
        return sm

    def _get_spatial_mask(self, snapshot_filename: str) -> SWIFTMask:
        """
        Helper function to verify that when in multi-galaxy mode the halo catalogue
        interface has been masked (with
        :func:`~swiftgalaxy.halo_catalogues._HaloCatalogue._mask_multi_galaxy`) before
        attempting to evaluate the spatial mask.

        Parameters
        ----------
        snapshot_filename : :obj:`str`
            The location of the SWIFT snapshot file.

        Returns
        -------
        out : :class:`~swiftsimio.masks.SWIFTMask`
            The spatial mask to select particles in the region of interest.
        """
        if self._multi_galaxy and self._multi_galaxy_catalogue_mask is None:
            raise RuntimeError(
                "Halo catalogue has multiple galaxies and is not currently masked."
            )
        return self._generate_spatial_mask(snapshot_filename)

    def _get_extra_mask(self, sg: "SWIFTGalaxy") -> MaskCollection:
        """
        Evaluate the extra (in the sense of in addition to spatial masking) mask.

        If the user requested a ``bound_only`` mask at initialization evaluate this
        using the ``_generate_bound_only_mask`` method of the derived class. Otherwise
        set the mask to do nothing if no mask was provided, or to the mask that the
        user provided if they provided one.

        Parameters
        ----------
        sg : :class:`~swiftgalaxy.reader.SWIFTGalaxy`
            The :class:`~swiftgalaxy.reader.SWIFTGalaxy` to which the mask will apply.

        Returns
        -------
        out : :class:`~swiftgalaxy.masks.MaskCollection`
            The extra mask to be applied after spatial masking.
        """
        if self.extra_mask == "bound_only":
            if self._multi_galaxy and self._multi_galaxy_catalogue_mask is None:
                raise RuntimeError(
                    "Halo catalogue has multiple galaxies and is not currently masked."
                )
            return self._generate_bound_only_mask(sg)
        elif self.extra_mask == "fof":
            return self._generate_fof_mask(sg)
        elif self.extra_mask is None:
            return MaskCollection(**{k: None for k in sg.metadata.present_group_names})
        else:
            # Keep user provided mask. If no mask provided for a particle type
            # use None (no mask).
            return MaskCollection(
                **{
                    name: getattr(self.extra_mask, name, None)
                    for name in sg.metadata.present_group_names
                }
            )

    def _mask_index(self) -> Optional[Union[int, list[int]]]:
        """
        Get the index into the halo catalogue, applying a mask if needed.

        Each derived class is free to define the name of its "index attribute" and defines
        this name in ``self._index_attr``. We first retrieve this, and if in multi-galaxy
        mode and currently masking down to a single row we apply the mask before returning
        the result.

        Returns
        -------
        out : :obj:`int` or :obj:`list`
            The index or list of indices into the halo catalogue.
        """
        if self._index_attr is None:
            return None
        else:
            index = getattr(self, self._index_attr)
            return (
                index[self._multi_galaxy_index_mask]
                if self._multi_galaxy_index_mask is not None
                else index
            )

    def _check_multi(self) -> None:
        """
        Determine whether the catalogue interface is in multi-galaxy mode.

        Checks whether there is more than one object of interest and set state
        variables accordingly during initialization.
        """
        if self._index_attr is None:
            # for now this means we're in Standalone
            if self._centre.ndim > 1:
                self._multi_galaxy = True
                self._multi_count = len(self._centre)
            else:
                if len(self._centre) == 0:
                    self._multi_galaxy = True
                    self._multi_count = 0
                else:
                    self._multi_galaxy = False
                    self._multi_count = 1
        else:
            index = getattr(self, self._index_attr)
            if isinstance(index, (Sequence, np.ndarray)):
                self._multi_galaxy = True
                if not isinstance(index, int):  # placate mypy
                    self._multi_count = len(index)
            else:
                self._multi_galaxy = False
                self._multi_count = 1
        if self._multi_galaxy:
            assert self.extra_mask in (None, "bound_only")
        return

    def __dir__(self) -> list[str]:
        """
        Supply a list of attributes of the halo catalogue.

        The regular ``dir`` behaviour doesn't index the names of catalogue attributes
        because they're attached to the internally maintained ``_catalogue`` attribute,
        so we customize the ``__dir__`` method to list the attribute names. They will
        then appear in tab completion, for example.

        Returns
        -------
        out : list
            The list of catalogue attribute names.
        """
        # use getattr to default to None, e.g. for Standalone
        return dir(getattr(self, "_catalogue", None))

    def __getattr__(self, attr: str) -> Any:
        """
        Exposes the masked halo catalogue.

        Invoked only if the attribute is not found on the interface class (it is then
        assumed to be a request for a halo catalogue property and delegated). If in
        multi-galaxy mode, use a :class:`~swiftgalaxy.halo_catalogues._MaskHelper` to
        enable any needed masking.

        Parameters
        ----------
        attr : :obj:`str`
            The name of the requested attribute.

        Returns
        -------
        out : :obj:`object`
            The requested attribute.
        """
        if attr == "_catalogue":  # guard infinite recursion
            try:
                return object.__getattribute__(self, "_catalogue")
            except AttributeError:
                return None
        obj = getattr(self._catalogue, attr)
        if self._multi_galaxy_index_mask is not None:
            return _MaskHelper(obj, self._multi_galaxy_index_mask)
        else:
            return obj

    @property
    def count(self) -> int:
        """
        Number of galaxies in the catalogue.

        When in multi-galaxy mode and not masked to a single row this can be >1.

        Returns
        -------
        out : :obj:`int`
            The number of galaxies in the catalogue.
        """
        if self._multi_galaxy and self._multi_galaxy_catalogue_mask is None:
            return self._multi_count
        else:
            return 1

    @abstractmethod
    def _load(self) -> None:
        """
        Abstract method.

        Derived classes shoud put any non-trivial i/o operations needed at
        initialization here. Method will be called during ``__init__``.
        """
        pass

    @abstractmethod
    def _generate_spatial_mask(self, snapshot_filename: str) -> SWIFTMask:
        """
        Abstract method.

        Derived classes should implement a function that accepts the filename of
        the swift snapshot file and returns a mask object to select the particles
        from the snapshot in the region of interest.

        Parameters
        ----------
        snapshot_filename : :obj:`str`
            The location of the SWIFT snapshot file.

        Returns
        -------
        out : :class:`~swiftsimio.masks.SWIFTMask`
            The spatial mask to select particles in the region of interest.
        """
        pass

    @abstractmethod
    def _generate_bound_only_mask(self, sg: "SWIFTGalaxy") -> MaskCollection:
        """
        Abstract method.

        Derived classes should implement a function that accepts the
        :class:`~swiftgalaxy.reader.SWIFTGalaxy` instance that the halo catalogue
        interface instance is associated to and returns a mask object to select
        the particles from the spatially-masked set of particles that correspond
        to the gravitationally-bound object of interest when the user specifies a
        ``"bound_only"`` extra mask.

        Parameters
        ----------
        sg : :class:`~swiftgalaxy.reader.SWIFTGalaxy`
            The :class:`~swiftgalaxy.reader.SWIFTGalaxy` that this halo finder
            interface is associated to.

        Returns
        -------
        out : :class:`~swiftgalaxy.masks.MaskCollection`
            The mask object that selects bound particles from the spatially-masked
            set of particles.
        """
        pass

    @abstractmethod
    def _get_preload_fields(self, sg: "SWIFTGalaxy") -> Set[str]:
        """
        Abstract method.

        Derived classes should implement a function that specifies which particle
        properties need to be loaded in order to evaluate masks with the
        :func:`~swiftgalaxy.halo_catalogues._HaloCatalogue._generate_spatial_mask`
        and :func:`~swiftgalaxy.halo_catalogues._HaloCatalogue._generate_bound_only_mask`.
        This is so that the data do not get repeatedly read when iterating over multiple
        :class:`~swiftgalaxy.reader.SWIFTGalaxy` objects in multi-galaxy mode.

        Parameters
        ----------
        sg : :class:`~swiftgalaxy.reader.SWIFTGalaxy`
            The :class:`~swiftgalaxy.reader.SWIFTGalaxy` that this halo finder
            interface is associated to.

        Returns
        -------
        out : :obj:`set`
            A set specifying the data that need to be read as strings, such as
            ``{"gas.particle_ids", ...}``.
        """
        pass

    @property
    @abstractmethod
    def centre(self) -> cosmo_array:
        """
        Abstract method.

        Derived classes should implement a property method that returns the coordinate
        centre of the object of interest (or of all of the objects of interest in
        multi-galaxy mode, or of the masked object of interest in multi-galaxy mode
        when a mask is active).

        Returns
        -------
        out : :class:`~swiftsimio.objects.cosmo_array`
            The coordinate centre(s) of the object(s) of interest.
        """
        pass

    @property
    @abstractmethod
    def velocity_centre(self) -> cosmo_array:
        """
        Abstract method.

        Derived classes should implement a property method that returns the velocity
        centre of the object of interest (or of all of the objects of interest in
        multi-galaxy mode, or of the masked object of interest in multi-galaxy mode
        when a mask is active).

        Returns
        -------
        out : :class:`~swiftsimio.objects.cosmo_array`
            The velocity centre(s) of the object(s) of interest.
        """
        pass

    @property
    @abstractmethod
    def _region_centre(self) -> cosmo_array:
        """
        Abstract method.

        Derived classes should implement a property method that returns the centre of the
        bounding box that defines a suitable spatial mask for the object(s) of interest,
        one per object in multi-galaxy mode.

        Returns
        -------
        out : :class:`~swiftsimio.objects.cosmo_array`
            The coordinates of the centres of the spatial mask regions.
        """
        pass

    @property
    @abstractmethod
    def _region_aperture(self) -> cosmo_array:
        """
        Abstract method.

        Derived classes should implement a property method that returns the half-length
        of the bounding box (e.g. the maximum radius of any particle of interest from the
        ``_region_centre``) for the spatial mask for the object(s) of interest, one per
        object in multi-galaxy mode.

        Returns
        -------
        out : :class:`~swiftsimio.objects.cosmo_array`
            The half-length of the bounding box to use to construct the spatial mask
            regions.
        """
        pass


class SOAP(_HaloCatalogue):
    """
    Interface to SOAP halo catalogues for use with :mod:`swiftgalaxy`.

    Takes a set of ``SOAP`` output files and configuration options and provides an
    interface that :mod:`swiftgalaxy` understands. Also exposes the galaxy properties
    computed by ``SOAP`` for a single object of interest through the :mod:`swiftsimio`
    interface.

    Parameters
    ----------
    soap_file : :obj:`str`, default: ``None``
        The filename of a SOAP catalogue file, possibly including the path.
    soap_index : :obj:`int` or :obj:`list`, default: ``None``
        The position (row) in the SOAP catalogue corresponding to the object of interest.
        Duplicate entries are not allowed.
    extra_mask : :obj:`str` or :class:`~swiftgalaxy.masks.MaskCollection` (optional), \
    default: ``"bound_only"``
        Mask to apply to particles after spatial masking. If ``"bound_only"``,
        then the galaxy is masked to include only the gravitationally bound
        particles as determined by ``SOAP``. A user-defined mask can also be provided
        as an an object (such as a :class:`swiftgalaxy.masks.MaskCollection`) that has
        attributes with names corresponding to present particle names (e.g. gas,
        dark_matter, etc.), each containing a mask.
    centre_type : :obj:`str` (optional), default: ``"input_halos.halo_centre"``
        Type of centre, chosen from those provided by ``SOAP``. This should be
        expressed as a string analogous to what would be written in
        :mod:`swiftsimio` code (or :mod:`swiftgalaxy`) to access that property in the
        SOAP catalogue. The default takes the ``"input_halos.halo_centre"`` (usually
        the centre of potential, e.g. the HBT+ halo finder defines it in this way;
        another option amongst many more is ``"bound_subhalo.centre_of_mass"``.
    velocity_centre_type : :obj:`str` (optional), \
    default: ``"bound_subhalo.centre_of_mass_velocity"``
        Type of velocity centre, chosen from those provided by ``SOAP``. This should be
        expressed as a string analogous to what would be written in
        :mod:`swiftsimio` code (or :mod:`swiftgalaxy`) to access that property in the
        SOAP catalogue. The default takes the ``"bound_subhalo.centre_of_mass_velocity"``;
        note that there is no velocity corresponding to the centre of potential
        (``"input_halos.halo_centre_velocity"`` is not defined). Another useful option
        could be ``"exclusive_sphere_1kpc.centre_of_mass_velocity"`` to choose the
        velocity of bound particles in the central 1 kpc.
    custom_spatial_offsets : :class`~swiftsimio.objects.cosmo_array` (optional), \
    default: ``None``
        A region to override the automatically-determined region enclosing
        group member particles. May be used in conjunction with ``extra_mask``,
        for example to select all simulation particles in an aperture around
        the object of interest (see 'Masking' section of documentation for a
        cookbook example). Provide a pair of offsets from the object's centre
        along each axis to define the region, for example for a cube extending
        +/- 1 Mpc from the centre:
        ``cosmo_array([[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]], u.Mpc)``.

    Notes
    -----

    .. note::
        ``SOAP`` only supports index access to catalogue arrays, not
        identifier access. This means that the ``soap_index`` is simply the
        position of the object of interest in the SOAP catalogue arrays.

    Examples
    --------
    Given a file :file:`halo_properties_0050.hdf5`, the following creates a :class:`SOAP`
    object for the entry at index ``0`` in the catalogue (i.e. the 1st row, indexed
    from 0) and demonstrates retrieving its virial mass (M200crit).

    ::

        >>> cat = SOAP(
        >>>     soap_file="/output/path/halo_properties_0050.hdf5"
        >>>     soap_index=0,  # 1st entry in catalogue (indexed from 0)
        >>> )
        >>> cat.spherical_overdensity_200_crit.total_mass.to(u.Msun)
        cosmo_array([6.72e+12], dtype=float32, units='1.98841586e+30*kg', comoving=False)
    """

    soap_file: str
    _soap_index: Union[int, Sequence[int]]
    centre_type: str
    velocity_centre_type: str
    _catalogue: SWIFTDataset
    _index_attr = "_soap_index"

    def __init__(
        self,
        soap_file: Optional[str] = None,
        soap_index: Optional[Union[int, Sequence[int]]] = None,
        extra_mask: Union[str, MaskCollection] = "bound_only",
        centre_type: str = "input_halos.halo_centre",
        velocity_centre_type: str = "bound_subhalo.centre_of_mass_velocity",
        custom_spatial_offsets: Optional[cosmo_array] = None,
    ) -> None:
        if soap_file is not None:
            self.soap_file = soap_file
        else:
            raise ValueError("Provide a soap_file.")

        if soap_index is not None:
            self._soap_index = soap_index
        else:
            raise ValueError("Provide a soap_index.")
        self.centre_type = centre_type
        self.velocity_centre_type = velocity_centre_type
        self._user_spatial_offsets = custom_spatial_offsets
        super().__init__(extra_mask=extra_mask)
        return

    def _load(self) -> None:
        """
        Do non-trivial I/O operations needed at initialization.

        Set up the :class:`~swiftsimio.reader.SWIFTDataset` that will handle the SOAP
        catalogue, including the appropriate mask to select only the rows of interest.
        """
        sm = mask(self.soap_file, spatial_only=not self._multi_galaxy)
        if self._multi_galaxy:
            sm.constrain_indices(self._soap_index)
        else:
            sm.constrain_index(self._soap_index)
        self._catalogue = SWIFTDataset(self.soap_file, mask=sm)

    @property
    def soap_index(self) -> Union[int, List[int]]:
        """
        The index (or indices in multi-galaxy mode when no mask is active) of the
        objects of interest in the halo catalogue. This is just the position in the
        arrays stored in the SOAP :class:`~swiftsimio.reader.SWIFTDataset` (that
        could have a :class:`~swiftsimio.masks.SWIFTMask`).

        Returns
        -------
        out : :obj:`int` or :obj:`list`
            The index or indices of the object(s) of interest in the halo catalogue.
        """
        index = self._mask_index()
        if index is not None:
            squeezed_index = np.squeeze(index)
            return (
                int(squeezed_index)
                if squeezed_index.ndim == 0
                else list(squeezed_index)
            )
        else:
            raise ValueError("SOAP definition of _index_attr unset!")

    def _mask_multi_galaxy(self, index) -> None:
        """
        Switch on restricting the catalogue to a single row.

        Used when the halo catalogue interface is in multi-galaxy mode (when
        iterating over many :class:`~swiftgalaxy.reader.SWIFTGalaxy`'s) to
        restrict the catalogue to a single row, for use during one such iteration.

        Because :mod:`swiftsimio`'s masking used to mask the SOAP catalogue does not
        respect the ordering of selected items, :class:`~swiftgalaxy.halo_catalogues.SOAP`
        needs its own implementation of this function to mask the correct row.

        Parameters
        ----------
        index : :obj:`int`
            The position in the input list of selected catalogue objects to mask down to.

        See Also
        --------
        swiftgalaxy.halo_catalogues._HaloCatalogue._unmask_multi_galaxy
        """
        self._multi_galaxy_catalogue_mask = np.argsort(np.argsort(self._soap_index))[
            index
        ]
        self._multi_galaxy_index_mask = np.s_[index : index + 1]
        return

    @property
    def _region_centre(self) -> cosmo_array:
        """
        Centre(s) of the bounding box regions for spatial masking.

        The default centre for SOAP is the ``bound_subhalo.centre_of_mass``. The
        maximum radius of any bound particle is also stored with reference to this
        centre.

        Returns
        -------
        out : :class:`~swiftsimio.objects.cosmo_array`
            The coordinates of the centres of the spatial mask regions.
        """
        # should not need to box wrap here but there's a bug upstream
        boxsize = self._catalogue.metadata.boxsize
        coords = self.bound_subhalo.centre_of_mass.squeeze()
        return coords % boxsize

    @property
    def _region_aperture(self) -> cosmo_array:
        """
        Half-length(s) of the bounding box regions for spatial masking.

        SOAP stores the maximum radius of any bound particle with respect to the
        ``input_halos.halo_centre``, as the ``bound_subhalo.enclose_radius``.
        Use this to define the bounding box for spatial masking.

        Returns
        -------
        out : :class:`~swiftsimio.objects.cosmo_array`
            The half-length of the bounding box to use to construct the spatial mask
            regions.
        """
        return self.bound_subhalo.enclose_radius.squeeze()

    def _get_preload_fields(self, sg: "SWIFTGalaxy") -> Set[str]:
        """
        Preload data needed to evaluate masks when in multi-galaxy mode.

        For SOAP, we need to know the ``gas.group_nr_bound`` values to evaluate
        masks (and similarly for other particle types), so pre-load these. Only needed
        in ``bound_only`` mode, so skip otherwise.

        Parameters
        ----------
        sg : :class:`~swiftgalaxy.reader.SWIFTGalaxy`
            The :class:`~swiftgalaxy.reader.SWIFTGalaxy` that this halo finder
            interface is associated to.

        Returns
        -------
        out : :obj:`set`
            The ``group_nr_bound`` dataset identifiers for all present particle types.
        """
        if self.extra_mask == "bound_only":
            return {
                f"{group_name}.group_nr_bound"
                for group_name in sg.metadata.present_group_names
            }
        else:
            return set()

    def _generate_spatial_mask(self, snapshot_filename: str) -> SWIFTMask:
        """
        Evaluate the spatial mask.

        Returns a mask object to select the particles from the snapshot in the region of
        interest.

        Parameters
        ----------
        snapshot_filename : :obj:`str`
            The location of the SWIFT snapshot file.

        Returns
        -------
        out : :class:`~swiftsimio.masks.SWIFTMask`
            The spatial mask to select particles in the region of interest.
        """
        pos, rmax = (self._region_centre, self._region_aperture)
        sm = mask(snapshot_filename, spatial_only=True)
        load_region = cosmo_array([pos - rmax, pos + rmax]).T
        sm.constrain_spatial(load_region)
        return sm

    def _generate_bound_only_mask(self, sg: "SWIFTGalaxy") -> MaskCollection:
        """
        Evaluate the mask for spatially selected particles to gravitationally bound
        particles.

        SOAP stores the halo catalogue index (e.g. from HBT+) that each particle
        is associated with. Check what halo catalogue index the object of interest
        is and use this to define the mask.

        Parameters
        ----------
        sg : :class:`~swiftgalaxy.reader.SWIFTGalaxy`
            The :class:`~swiftgalaxy.reader.SWIFTGalaxy` that this halo finder
            interface is associated to.

        Returns
        -------
        out : :class:`~swiftgalaxy.masks.MaskCollection`
            The mask object that selects bound particles from the spatially-masked
            set of particles.
        """
        masks = MaskCollection(
            **{
                group_name: getattr(
                    sg, group_name
                )._particle_dataset.group_nr_bound.to_value(u.dimensionless)
                == self.input_halos.halo_catalogue_index.to_value(u.dimensionless)
                for group_name in sg.metadata.present_group_names
            }
        )
        if not self._multi_galaxy:
            for group_name in sg.metadata.present_group_names:
                del getattr(sg, group_name)._particle_dataset.group_nr_bound
        return masks

    def _generate_fof_mask(self, sg: "SWIFTGalaxy") -> MaskCollection:
        """
        Will wants to select FOF particles. Let's select FOF particles for him.

        Parameters
        ----------
        sg : :class:`~swiftgalaxy.reader.SWIFTGalaxy`
            The :class:`~swiftgalaxy.reader.SWIFTGalaxy` that this halo finder
            interface is associated to.

        Returns
        -------
        out : :class:`~swiftgalaxy.masks.MaskCollection`
            The mask object that selects FOF particles from the spatially-masked
            set of particles.
        """
        masks = MaskCollection(
            **{
                group_name: getattr(
                    sg, group_name
                )._particle_dataset.fofgroup_ids.to_value(u.dimensionless)
                == self.input_halos_hbtplus.host_fofid.to_value(u.dimensionless)
                for group_name in sg.metadata.present_group_names
            }
        )
        if not self._multi_galaxy:
            for group_name in sg.metadata.present_group_names:
                del getattr(sg, group_name).fofgroup_ids
        return masks

    @property
    def centre(self) -> cosmo_array:
        """
        Obtain the centre specified by the ``centre_type`` from the halo catalogue.

        In multi-galaxy mode if no mask is active return the centres of all objects of
        interest, otherwise return the centre of the (current) object of interest.

        Returns
        -------
        out : :class:`~swiftsimio.objects.cosmo_array`
            The centre(s) of the object(s) of interest.
        """
        obj = self._catalogue
        for attr in self.centre_type.split("."):
            obj = getattr(obj, attr)
        if self._multi_galaxy_index_mask is not None:
            return obj[self._multi_galaxy_index_mask]
        return obj.squeeze()

    @property
    def velocity_centre(self) -> cosmo_array:
        """
        Obtain the velocity centre specified by the ``velocity_centre_type`` from the halo
        catalogue.

        In multi-galaxy mode if no mask is active return the centres of all objects of
        interest, otherwise return the centre of the (current) object of interest.

        Returns
        -------
        out : :class:`~swiftsimio.objects.cosmo_array`
            The centre(s) of the object(s) of interest.
        """
        obj = self._catalogue
        for attr in self.velocity_centre_type.split("."):
            obj = getattr(obj, attr)
        if self._multi_galaxy_index_mask is not None:
            return obj[self._multi_galaxy_index_mask]
        return obj.squeeze()

    def __repr__(self) -> str:
        """
        Expose the catalogue ``__repr__`` for interactive use.

        Delegates creating a string representation to the internal
        :class:`~swiftsimio.reader.SWIFTDataset` holding the SOAP data.

        Returns
        -------
        out : :obj:`str`
            The string representation of the catalogue.
        """
        return self._catalogue.__repr__()

    def __dir__(self) -> list[str]:
        """
        Supply a list of attributes of the halo catalogue.

        The regular ``dir`` behaviour doesn't index the names of catalogue attributes
        because they're attached to the internally maintained ``_catalogue`` attribute,
        so we custimize the ``__dir__`` method to list the attribute names. They will
        then appear in tab completion, for example.

        Returns
        -------
        out : list
            The list of catalogue attribute names.
        """
        return list(self._catalogue.metadata.present_group_names)


class Velociraptor(_HaloCatalogue):
    """
    Interface to velociraptor halo catalogues for use with :mod:`swiftgalaxy`.

    Takes a set of :mod:`velociraptor` output files and configuration options
    and provides an interface that :mod:`swiftgalaxy` understands. Also exposes
    the halo/galaxy properties computed by :mod:`velociraptor` for a single
    object of interest with the `API`_ provided by the :mod:`velociraptor`
    python package. Reading of properties is done on-the-fly, and only rows
    corresponding to the object of interest are read from disk.

    .. _API: https://velociraptor-python.readthedocs.io/en/latest/

    Parameters
    ----------

    velociraptor_filebase : :obj:`str`
        The initial part of the velociraptor filenames (possibly including
        path), e.g. if there is a :file:`{halos}.properties` file, pass
        ``halos`` as this argument. Provide this or `velociraptor_files`,
        not both.

    velociraptor_files : :obj:`dict`
        A dictionary containing the names of the velociraptor files (possibly
        including paths). There should be two entries, with keys `properties`
        and `catalog_groups` containing locations of the `{halos}.properties`
        and `{halos}.catalog_groups` files, respectively. Provide this or
        `velociraptor_filebase`, not both.

    halo_index : :obj:`int` or :obj:`list`
        Position(s) of the object(s) of interest in the catalogue arrays. In the case
        of multiple objects, duplicate entries are not allowed.

    extra_mask : :obj:`str` or :class:`~swiftgalaxy.masks.MaskCollection` (optional), \
    default: ``"bound_only"``
        Mask to apply to particles after spatial masking. If ``"bound_only"``,
        then the galaxy is masked to include only the gravitationally bound
        particles as determined by :mod:`velociraptor`. A user-defined mask
        can also be provided as an an object (such as a
        :class:`swiftgalaxy.masks.MaskCollection`) that has attributes with
        names corresponding to present particle names (e.g. gas, dark_matter,
        etc.), each containing a mask.

    centre_type : :obj:`str` (optional), default: ``"minpot"``
        Type of centre, chosen from those provided by :mod:`velociraptor`.
        Default is the position of the particle with the minimum potential,
        ``"minpot"``; other possibilities may include ``""``, ``"_gas"``,
        ``"_star"``, ``"mbp"`` (most bound particle).

    custom_spatial_offsets : `~swiftsimio.objects.cosmo_array` (optional), \
    default: ``None``
        A region to override the automatically-determined region enclosing
        group member particles. May be used in conjunction with ``extra_mask``,
        for example to select all simulation particles in an aperture around
        the object of interest (see 'Masking' section of documentation for a
        cookbook example). Provide a pair of offsets from the object's centre
        along each axis to define the region, for example for a cube extending
        +/- 1 Mpc from the centre:
        ``cosmo_array([[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]], u.Mpc)``.

    Notes
    -----

    .. note::
        :mod:`velociraptor` only supports index access to catalogue arrays, not
        identifier access. This means that the ``halo_index`` is simply the
        position of the object of interest in the catalogue arrays.

    Examples
    --------
    Given a file :file:`{halos}.properties` (and also
    :file:`{halos}.catalog_groups`, etc.) at :file:`/output/path/`, the
    following creates a :class:`Velociraptor` object for the entry at index
    ``3`` in the catalogue (i.e. the 4th row, indexed from 0) and demonstrates
    retrieving its virial mass.

    ::

        >>> cat = Velociraptor(
        >>>     velociraptor_filebase="/output/path/halos",  # halos.properties file is at
        >>>                                                  # /output/path/
        >>>     halo_index=3,  # 4th entry in catalogue (indexed from 0)
        >>> )
        >>> cat
        Masked velociraptor catalogue at /path/to/output/out.properties.
        Contains the following field collections: metallicity, ids, energies,
        stellar_age, spherical_overdensities, rotational_support,
        star_formation_rate, masses, eigenvectors, radii, temperature, veldisp,
        structure_type, velocities, positions, concentration, rvmax_quantities,
        angular_momentum, projected_apertures, apertures,
        element_mass_fractions, dust_mass_fractions, number,
        hydrogen_phase_fractions, black_hole_masses, stellar_birth_densities,
        snii_thermal_feedback_densities, species_fractions,
        gas_hydrogen_species_masses, gas_H_and_He_masses,
        gas_diffuse_element_masses, dust_masses_from_table, dust_masses,
        stellar_luminosities, cold_dense_gas_properties,
        log_element_ratios_times_masses, lin_element_ratios_times_masses,
        element_masses_in_stars, fail_all
        >>> cat.masses
        Contains the following fields: mass_200crit, mass_200crit_excl,
        mass_200crit_excl_gas, mass_200crit_excl_gas_nsf,
        mass_200crit_excl_gas_sf, mass_200crit_excl_star, mass_200crit_gas,
        mass_200crit_gas_nsf, mass_200crit_gas_sf, mass_200crit_star,
        mass_200mean, mass_200mean_excl, mass_200mean_excl_gas,
        mass_200mean_excl_gas_nsf, mass_200mean_excl_gas_sf,
        mass_200mean_excl_star, mass_200mean_gas, mass_200mean_gas_nsf,
        mass_200mean_gas_sf, mass_200mean_star, mass_bn98, mass_bn98_excl,
        mass_bn98_excl_gas, mass_bn98_excl_gas_nsf, mass_bn98_excl_gas_sf,
        mass_bn98_excl_star, mass_bn98_gas, mass_bn98_gas_nsf,
        mass_bn98_gas_sf, mass_bn98_star, mass_fof, mass_bh, mass_gas,
        mass_gas_30kpc, mass_gas_500c, mass_gas_rvmax, mass_gas_hight_excl,
        mass_gas_hight_incl, mass_gas_incl, mass_gas_nsf, mass_gas_nsf_incl,
        mass_gas_sf, mass_gas_sf_incl, mass_star, mass_star_30kpc,
        mass_star_500c, mass_star_rvmax, mass_star_incl, mass_tot,
        mass_tot_incl, mvir
        >>> cat.masses.mvir
        unyt_array(14.73875777, '10000000000.0*Msun')
    """

    velociraptor_filebase: str
    velociraptor_files: Dict[str, str]
    _halo_index: Union[int, Sequence[int]]
    centre_type: str
    velocity_centre_type: str
    _catalogue: "VelociraptorCatalogue"
    _index_attr = "_halo_index"

    def __init__(
        self,
        velociraptor_filebase: Optional[str] = None,
        velociraptor_files: Optional[dict] = None,
        halo_index: Optional[Union[int, Sequence[int]]] = None,
        extra_mask: Union[str, MaskCollection] = "bound_only",
        centre_type: str = "minpot",  # _gas _star mbp minpot
        custom_spatial_offsets: Optional[cosmo_array] = None,
    ) -> None:
        if velociraptor_filebase is not None and velociraptor_files is not None:
            raise ValueError(
                "Provide either velociraptor_filebase or velociraptor_files, not both."
            )
        elif velociraptor_files is not None:
            self.velociraptor_files = velociraptor_files
        elif velociraptor_filebase is not None:
            self.velociraptor_files = dict(
                properties=f"{velociraptor_filebase}.properties",
                catalog_groups=f"{velociraptor_filebase}.catalog_groups",
            )
        else:
            raise ValueError(
                "Provide one of velociraptor_filebase or velociraptor_files."
            )
        if halo_index is None:
            raise ValueError("Provide a halo_index.")
        else:
            self._halo_index: int = halo_index
        self.centre_type: str = centre_type
        self._user_spatial_offsets = custom_spatial_offsets
        super().__init__(extra_mask=extra_mask)
        # currently velociraptor_python works with a halo index, not halo_id
        # self.catalogue_mask = (catalogue.ids.id == halo_id).nonzero()
        return

    def _load(self) -> None:
        """
        Do non-trivial I/O operations needed at initialization.

        Set up the :class:`~velociraptor.catalogue.catalogue.Catalogue` that will handle
        the Velociraptor catalogue, including the appropriate mask to select only the rows
        of interest.
        """
        import h5py
        from velociraptor import load as load_catalogue
        from velociraptor.particles import load_groups

        with h5py.File(self.velociraptor_files["properties"]) as propfile:
            self.scale_factor = (
                float(propfile["SimulationInfo"].attrs["ScaleFactor"])
                if propfile["SimulationInfo"].attrs["Cosmological_Sim"]
                else 1.0
            )

        self._catalogue = load_catalogue(
            self.velociraptor_files["properties"], mask=np.array(self._halo_index)
        )
        groups = load_groups(
            self.velociraptor_files["catalog_groups"],
            catalogue=load_catalogue(self.velociraptor_files["properties"]),
        )
        if self._multi_galaxy:
            if not isinstance(self._halo_index, int):  # placate mypy
                self._particles = [
                    groups.extract_halo(halo_index=hi)[0] for hi in self._halo_index
                ]
        else:
            self._particles, unbound_particles_unused = groups.extract_halo(
                halo_index=self._halo_index
            )
        return

    def _generate_spatial_mask(self, snapshot_filename: str) -> SWIFTMask:
        """
        Evaluate the spatial mask.

        Returns a mask object to select the particles from the snapshot in the region of
        interest.

        Parameters
        ----------
        snapshot_filename : :obj:`str`
            The location of the SWIFT snapshot file.

        Returns
        -------
        out : :class:`~swiftsimio.masks.SWIFTMask`
            The spatial mask to select particles in the region of interest.
        """
        from velociraptor.swift.swift import generate_spatial_mask

        # super()._get_spatial_mask guards getting here in multi-galaxy mode
        # without a self._multi_galaxy_catalogue_mask
        return generate_spatial_mask(
            (
                self._particles[self._multi_galaxy_catalogue_mask]
                if self._multi_galaxy_catalogue_mask is not None
                else self._particles
            ),
            snapshot_filename,
        )

    def _generate_bound_only_mask(self, sg: "SWIFTGalaxy") -> MaskCollection:
        """
        Evaluate the mask for spatially selected particles to gravitationally bound
        particles.

        Velociraptor stores a list of particle IDs that belong to each bound object.
        The :mod:`velociraptor` package provides tools to read these and evaluate the
        mask for us, so we just call the relevant function from that package for each
        particle type.

        Parameters
        ----------
        sg : :class:`~swiftgalaxy.reader.SWIFTGalaxy`
            The :class:`~swiftgalaxy.reader.SWIFTGalaxy` that this halo finder
            interface is associated to.

        Returns
        -------
        out : :class:`~swiftgalaxy.masks.MaskCollection`
            The mask object that selects bound particles from the spatially-masked
            set of particles.
        """
        from velociraptor.swift.swift import generate_bound_mask

        return MaskCollection(
            **generate_bound_mask(
                sg,
                (
                    self._particles[self._multi_galaxy_catalogue_mask]
                    if self._multi_galaxy_catalogue_mask is not None
                    else self._particles
                ),
            )._asdict()
        )

    @property
    def halo_index(self) -> Union[List[int], int]:
        """
        The index (or indices in multi-galaxy mode when no mask is active) of the
        objects of interest in the halo catalogue. This is just the position in the
        arrays stored in the Velociraptor catalogue files.

        Returns
        -------
        out : :obj:`int` or :obj:`list`
            The index or indices of the object(s) of interest in the halo catalogue.
        """
        # a bit of logic to placate mypy
        index = self._mask_index()
        if index is not None:
            return index
        else:
            raise ValueError("Velociraptor _index_attr is unset!")

    @property
    def _region_centre(self) -> cosmo_array:
        """
        Centre(s) of the bounding box regions for spatial masking.

        Velociraptor stores the "default" centre coordinates in the ``x``, ``y`` and
        ``z`` attributes of the ``self._particles``, retrieve these for the object(s)
        of interest.

        Returns
        -------
        out : :class:`~swiftsimio.objects.cosmo_array`
            The coordinates of the centres of the spatial mask regions.
        """
        length_factor = (
            self._particles[0].groups_instance.catalogue.units.a
            if not self._particles[0].groups_instance.catalogue.units.comoving
            else 1.0
        )
        if self._multi_galaxy_catalogue_mask is None:
            return cosmo_array(
                [
                    [
                        particles.x.to(u.Mpc) / length_factor,
                        particles.y.to(u.Mpc) / length_factor,
                        particles.z.to(u.Mpc) / length_factor,
                    ]
                    for particles in self._particles
                ],
                u.Mpc,
                comoving=True,
                cosmo_factor=cosmo_factor(a**1, length_factor),
            ).squeeze()
        else:
            return cosmo_array(
                [
                    self._particles[self._multi_galaxy_catalogue_mask].x.to_value(u.Mpc)
                    / length_factor,
                    self._particles[self._multi_galaxy_catalogue_mask].y.to_value(u.Mpc)
                    / length_factor,
                    self._particles[self._multi_galaxy_catalogue_mask].z.to_value(u.Mpc)
                    / length_factor,
                ],
                u.Mpc,
                comoving=True,
                cosmo_factor=cosmo_factor(a**1, length_factor),
            )

    @property
    def _region_aperture(self) -> cosmo_array:
        """
        Half-length(s) of the bounding box regions for spatial masking.

        Velociraptor stores the maximum radius of any bound particle with respect to the
        ``x``, ``y`` and ``z`` centre as the ``r_size``. Use this to define the bounding
        box for spatial masking.

        Returns
        -------
        out : :class:`~swiftsimio.objects.cosmo_array`
            The half-length of the bounding box to use to construct the spatial mask
            regions.
        """
        length_factor = (
            self._particles[0].groups_instance.catalogue.units.a
            if not self._particles[0].groups_instance.catalogue.units.comoving
            else 1.0
        )
        if self._multi_galaxy_catalogue_mask is None:
            return cosmo_array(
                [
                    particles.r_size.to_value(u.Mpc) / length_factor
                    for particles in self._particles
                ],
                u.Mpc,
                comoving=True,
                cosmo_factor=cosmo_factor(a**1, length_factor),
            ).squeeze()
        else:
            return cosmo_array(
                self._particles[self._multi_galaxy_catalogue_mask].r_size.to_value(
                    u.Mpc
                )
                / length_factor,
                u.Mpc,
                comoving=True,
                cosmo_factor=cosmo_factor(a**1, length_factor),
            )

    def _get_preload_fields(self, sg: "SWIFTGalaxy") -> Set[str]:
        """
        Preload data needed to evaluate masks when in multi-galaxy mode.

        For Velociraptor, we need to know the ``gas.particle_ids`` values to evaluate
        masks (and similarly for other particle types), so pre-load these. Only
        needed in ``bound_only`` mode, so skip otherwise.

        Parameters
        ----------
        sg : :class:`~swiftgalaxy.reader.SWIFTGalaxy`
            The :class:`~swiftgalaxy.reader.SWIFTGalaxy` that this halo finder
            interface is associated to.

        Returns
        -------
        out : :obj:`set`
            The ``particle_ids`` dataset identifiers for all present particle types.
        """
        if self.extra_mask == "bound_only":
            return {
                f"{group_name}.particle_ids"
                for group_name in sg.metadata.present_group_names
            }
        else:
            return set()

    @property
    def centre(self) -> cosmo_array:
        """
        Obtain the centre specified by the ``centre_type`` from the halo catalogue.

        In multi-galaxy mode if no mask is active return the centres of all objects of
        interest, otherwise return the centre of the (current) object of interest.

        Returns
        -------
        out : :class:`~swiftsimio.objects.cosmo_array`
            The centre(s) of the object(s) of interest.
        """
        # According to Velociraptor documentation:
        if self.centre_type in ("_gas", "_stars"):
            # {XYZ}c_gas and {XYZ}c_stars are relative to {XYZ}c
            relative_to = np.hstack(
                [
                    cosmo_array(
                        getattr(self._catalogue.positions, "{:s}c".format(c)),
                        comoving=self._catalogue.units.comoving,
                        cosmo_factor=cosmo_factor(a**1, self._catalogue.units.a),
                    )
                    for c in "xyz"
                ]
            ).T
        else:
            # {XYZ}cmbp, {XYZ}cminpot and {XYZ}c are absolute
            # comoving doesn't matter for origin, arbitrarily set False
            relative_to = cosmo_array(
                [0.0, 0.0, 0.0],
                u.Mpc,
                comoving=False,
                cosmo_factor=cosmo_factor(a**1, self.scale_factor),
            )
        centre = cosmo_array(
            (
                relative_to
                + np.hstack(
                    [
                        cosmo_array(
                            getattr(
                                self._catalogue.positions,
                                "{:s}c{:s}".format(c, self.centre_type),
                            ),
                            comoving=self._catalogue.units.comoving,
                            cosmo_factor=cosmo_factor(a**1, self._catalogue.units.a),
                        )
                        for c in "xyz"
                    ]
                ).T
            ).to_value(u.Mpc),
            u.Mpc,
            comoving=False,  # velociraptor gives physical centres!
            cosmo_factor=cosmo_factor(a**1, self.scale_factor),
        ).to_comoving()
        if self._multi_galaxy and self._multi_galaxy_catalogue_mask is None:
            return centre
        elif self._multi_galaxy and self._multi_galaxy_catalogue_mask is not None:
            return centre[self._multi_galaxy_catalogue_mask]
        else:
            return centre.squeeze()

    @property
    def velocity_centre(self) -> cosmo_array:
        """
        Obtain the velocity centre specified by the ``velocity_centre_type`` from the halo
        catalogue.

        In multi-galaxy mode if no mask is active return the centres of all objects of
        interest, otherwise return the centre of the (current) object of interest.

        Returns
        -------
        out : :class:`~swiftsimio.objects.cosmo_array`
            The centre(s) of the object(s) of interest.
        """
        # According to Velociraptor documentation:
        if self.centre_type in ("_gas", "_stars"):
            # V{XYZ}c_gas and V{XYZ}c_stars are relative to {XYZ}c
            relative_to = np.hstack(
                [
                    cosmo_array(
                        getattr(self._catalogue.velocities, "v{:s}c".format(c)),
                        comoving=self._catalogue.units.comoving,
                        cosmo_factor=cosmo_factor(a**0, self._catalogue.units.a),
                    )
                    for c in "xyz"
                ]
            ).T
        else:
            # V{XYZ}cmbp, V{XYZ}cminpot and V{XYZ}c are absolute
            relative_to = cosmo_array(
                [0.0, 0.0, 0.0],
                u.km / u.s,
                comoving=False,
                cosmo_factor=cosmo_factor(a**0, self.scale_factor),
            )
        vcentre = cosmo_array(
            (
                relative_to
                + np.hstack(
                    [
                        cosmo_array(
                            getattr(
                                self._catalogue.velocities,
                                "v{:s}c{:s}".format(c, self.centre_type),
                            ),
                            comoving=self._catalogue.units.comoving,
                            cosmo_factor=cosmo_factor(a**0, self._catalogue.units.a),
                        )
                        for c in "xyz"
                    ]
                ).T
            ).to_value(u.km / u.s),
            u.km / u.s,
            comoving=False,
            cosmo_factor=cosmo_factor(a**0, self.scale_factor),
        ).to_comoving()
        if self._multi_galaxy and self._multi_galaxy_catalogue_mask is None:
            return vcentre
        elif self._multi_galaxy and self._multi_galaxy_catalogue_mask is not None:
            return vcentre[self._multi_galaxy_catalogue_mask]
        else:
            return vcentre.squeeze()

    def __repr__(self) -> str:
        """
        Expose the catalogue ``__repr__`` for interactive use.

        Delegates creating a string representation to the internal
        :class:`~velociraptor.catalogue.catalogue.Catalogue` holding the Velociraptor
        data.

        Returns
        -------
        out : :obj:`str`
            The string representation of the catalogue.
        """
        return self._catalogue.__repr__()


class Caesar(_HaloCatalogue):
    """
    Interface to Caesar halo catalogues for use with :mod:`swiftgalaxy`.

    Takes a :mod:`caesar` output file and configuration options and provides
    an interface that :mod:`swiftgalaxy` understands. Also exposes the halo/galaxy
    properties computed by CAESAR for a single object of interest with
    the same `interface`_ provided by the :class:`~loader.Group` class
    in the :mod:`caesar` python package. Reading of properties is done on-the-fly, and
    only rows corresponding to the object of interest are read from disk.

    .. _interface: https://caesar.readthedocs.io/en/latest/

    Parameters
    ----------

    caesar_file : :obj:`str`
        The catalogue file (hdf5 format) output by caesar.

    group_type : :obj:`str`
        The category of the object of interest, either ``"halo"`` or ``"galaxy"``.

    group_index : :obj:`int`
        Position(s) of the object(s) of interest in the catalogue arrays. In the case
        of multiple objects, duplicate entries are not allowed.

    centre_type : :obj:`str` (optional), default: ``"minpot"``
        Type of centre, chosen from those provided by :mod:`caesar`.
        Default is the position of the particle with the minimum potential,
        ``"minpot"``, alternatively ``""`` can be used for the centre of mass.

    extra_mask : :obj:`str` or :class:`~swiftgalaxy.masks.MaskCollection` (optional), \
    default: ``"bound_only"``
        Mask to apply to particles after spatial masking. If ``"bound_only"``,
        then the galaxy is masked to include only the gravitationally bound
        particles as provided by :mod:`caesar`. A user-defined mask can also be
        as an an object (such as a :class:`swiftgalaxy.masks.MaskCollection`) that has
        attributes with names corresponding to present particle names (e.g. gas,
        dark_matter, etc.), each containing a mask.

    custom_spatial_offsets : :class:`~swiftsimio.objects.cosmo_array` (optional), \
    default: ``None``
        A region to override the automatically-determined region enclosing
        group member particles. May be used in conjunction with ``extra_mask``,
        for example to select all simulation particles in an aperture around
        the object of interest (see 'Masking' section of documentation for a
        cookbook example). Provide a pair of offsets from the object's centre
        along each axis to define the region, for example for a cube extending
        +/- 1 Mpc from the centre:
        ``cosmo_array([[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]], u.Mpc)``.

    Notes
    -----

    .. note::
        :class:`~loader.CAESAR` only supports index access to catalogue lists, not
        identifier access. This means that the ``group_index`` is simply the
        position of the object of interest in the catalogue list.

    Examples
    --------
    Given a file :file:`s12.5n128_0012.hdf5` at :file:`/output/path/`, the
    following creates a :class:`Caesar` object for the entry at index
    ``3`` in the catalogue (i.e. the 4th row, indexed from 0) and demonstrates
    retrieving its virial mass.

    ::

        >>> cat = Caesar(
        >>>     caesar_file="/output/path/s12.5n128_0012.hdf5",
        >>>     group_type="halo",
        >>>     group_index=3,  # 4th entry in catalogue (indexed from 0)
        >>> )
        >>> cat.info()
        {'GroupID': 3,
        'ages': {'mass_weighted': unyt_quantity(2.26558173, 'Gyr'),
                 'metal_weighted': unyt_quantity(2.21677032, 'Gyr')},
        'bh_fedd': unyt_quantity(3.97765937, 'dimensionless'),
        'bhlist_end': 12,
        'bhlist_start': 11,
        ...
        'virial_quantities': {'circular_velocity': unyt_quantity(158.330253, 'km/s'),
                              'm200c': unyt_quantity(1.46414384e+12, 'Msun'),
                              'm2500c': unyt_quantity(8.72801239e+11, 'Msun'),
                              'm500c': unyt_quantity(1.23571772e+12, 'Msun'),
                              'r200': unyt_quantity(425.10320408, 'kpccm'),
                              'r200c': unyt_quantity(327.46600342, 'kpccm'),
                              'r2500c': unyt_quantity(118.77589417, 'kpccm'),
                              'r500c': unyt_quantity(228.03265381, 'kpccm'),
                              'spin_param': unyt_quantity(2.28429179,'s/(Msun*km*kpccm)'),
                              'temperature': unyt_quantity(902464.88453405, 'K')}}
        >>> cat.virial_quantities
        {'circular_velocity': unyt_quantity(158.330253, 'km/s'),
         'm200c': unyt_quantity(1.46414384e+12, 'Msun'),
         'm2500c': unyt_quantity(8.72801239e+11, 'Msun'),
         'm500c': unyt_quantity(1.23571772e+12, 'Msun'),
         'r200': unyt_quantity(425.10320408, 'kpccm'),
         'r200c': unyt_quantity(327.46600342, 'kpccm'),
         'r2500c': unyt_quantity(118.77589417, 'kpccm'),
         'r500c': unyt_quantity(228.03265381, 'kpccm'),
         'spin_param': unyt_quantity(2.28429179, 's/(Msun*km*kpccm)'),
         'temperature': unyt_quantity(902464.88453405, 'K')}
        >>> cat.virial_quantities["m200c"]
        unyt_quantity(1.46414384e+12, 'Msun')
    """

    caesar_file: Optional[str]
    group_type: str
    _group_index: Union[int, Sequence[int]]
    centre_type: str
    velocity_centre_type: str
    _catalogue: Union[
        "CaesarHalo", "CaesarGalaxy", List[Union["CaesarHalo", "CaesarGalaxy"]]
    ]
    _index_attr = "_group_index"

    def __init__(
        self,
        caesar_file: Optional[str] = None,
        group_type: Optional[str] = None,  # halos galaxies
        group_index: Optional[Union[int, Sequence[int]]] = None,
        centre_type: str = "minpot",  # "" "minpot"
        extra_mask: Union[str, MaskCollection] = "bound_only",
        custom_spatial_offsets: Optional[cosmo_array] = None,
    ) -> None:
        import caesar
        import logging
        from yt.utilities import logger as yt_logger

        log_level = logging.getLogger("yt").level  # cache the log level before we start
        yt_logger.set_log_level("warning")  # disable INFO log messages
        self.caesar_file = caesar_file
        self._caesar = caesar.load(caesar_file)
        yt_logger.set_log_level(log_level)  # restore old log level

        valid_group_types = dict(halo="halos", galaxy="galaxies")
        if group_type not in valid_group_types:
            raise ValueError(
                "group_type required, valid values are 'halo' or 'galaxy'."
            )
        self.group_type = group_type
        if group_index is None:
            raise ValueError("group_index (int or list) required.")
        else:
            self._group_index = group_index

        self.centre_type = centre_type
        self._user_spatial_offsets = custom_spatial_offsets

        super().__init__(extra_mask=extra_mask)
        self._catalogue = getattr(self._caesar, valid_group_types[group_type])
        if self._multi_galaxy:  # set in super().__init__
            if not isinstance(group_index, int):  # placate mypy
                self._catalogue = [self._catalogue[gi] for gi in group_index]
        else:
            self._catalogue = self._catalogue[group_index]
        return

    def _load(self) -> None:
        """
        Do non-trivial I/O operations needed at initialization.

        Nothing needed for Caesar catalogues, do nothing.
        """
        pass

    def _generate_spatial_mask(self, snapshot_filename: str) -> SWIFTMask:
        """
        Evaluate the spatial mask.

        Returns a mask object to select the particles from the snapshot in the region of
        interest.

        Parameters
        ----------
        snapshot_filename : :obj:`str`
            The location of the SWIFT snapshot file.

        Returns
        -------
        out : :class:`~swiftsimio.masks.SWIFTMask`
            The spatial mask to select particles in the region of interest.
        """
        cat = self._mask_catalogue()
        sm = mask(snapshot_filename, spatial_only=True)
        if "total_rmax" in cat.radii.keys():
            # spatial extent information is present, define the mask
            pos = cosmo_array(
                cat.pos.to(u.kpc),  # maybe comoving, ensure physical
                comoving=False,
                cosmo_factor=cosmo_factor(a**1, self._caesar.simulation.scale_factor),
            ).to_comoving()
            rmax = cosmo_array(
                cat.radii["total_rmax"].to(u.kpc),  # maybe comoving, ensure physical
                comoving=False,
                cosmo_factor=cosmo_factor(a**1, self._caesar.simulation.scale_factor),
            ).to_comoving()
            load_region = cosmo_array([pos - rmax, pos + rmax]).T
        else:
            # probably an older caesar output file, not enough information to define mask
            # so we read the entire box and warn
            boxsize = sm.metadata.boxsize
            load_region = [[0.0 * b, 1.0 * b] for b in boxsize]
            warn(
                "CAESAR catalogue does not contain group extent information, so spatial "
                "mask defaults to entire box. Reading will be inefficient. See "
                "https://github.com/dnarayanan/caesar/issues/92"
            )
        sm.constrain_spatial(load_region)
        return sm

    def _generate_bound_only_mask(self, sg: "SWIFTGalaxy") -> MaskCollection:
        """
        Evaluate the mask for spatially selected particles to gravitationally bound
        particles.

        Caesar stores the starts and ends of ranges containing the particles belonging
        to each object. We define a function that can efficiently (CPU efficient, at
        the cost of memory efficiency) check which particles are in any of the specified
        ranges and use it to evaluate the masks.

        Parameters
        ----------
        sg : :class:`~swiftgalaxy.reader.SWIFTGalaxy`
            The :class:`~swiftgalaxy.reader.SWIFTGalaxy` that this halo finder
            interface is associated to.

        Returns
        -------
        out : :class:`~swiftgalaxy.masks.MaskCollection`
            The mask object that selects bound particles from the spatially-masked
            set of particles.
        """

        def in_one_of_ranges(
            ints: NDArray[np.int_], int_ranges: NDArray[np.int_]
        ) -> NDArray[np.bool_]:
            """
            Produces a boolean mask corresponding to `ints`.

            For each element in `ints`, the mask is `True` if the value is between (at
            least) one of the pairs of integers in `int_ranges`. This is potentially
            memory intensive with a footprint proportional to
            `ints.size * int_ranges.size`.

            Parameters
            ----------
            ints : :class:`~numpy.ndarray`
                Array of integers for which to check membership in the ranges.
            int_ranges : :class:`~numpy.ndarray`
                2D array with shape (N, 2) of half-open ranges `[min, max[` in
                which to check membership.

            Returns
            -------
            out : :class:`~numpy.ndarray`
                Boolean array with same shape as `ints`, `True` if the integer is in
                at least one of the ranges, `False` otherwise.
            """
            return np.logical_and(
                ints >= int_ranges[:, 0, np.newaxis],
                ints < int_ranges[:, 1, np.newaxis],
            ).any(axis=0)

        cat = self._mask_catalogue()
        null_slice = np.s_[:0]  # mask that selects no particles
        if hasattr(cat, "glist"):
            gas_mask = cat.glist
            gas_mask = gas_mask[in_one_of_ranges(gas_mask, sg.mask.gas)]
            gas_mask = np.isin(
                np.concatenate([np.arange(start, end) for start, end in sg.mask.gas]),
                gas_mask,
            )
        else:
            gas_mask = null_slice
        # seems like name could be dmlist or dlist?
        if hasattr(cat, "dlist") or hasattr(cat, "dmlist"):
            dark_matter_mask = getattr(cat, "dlist", cat.dmlist)
            dark_matter_mask = dark_matter_mask[
                in_one_of_ranges(dark_matter_mask, sg.mask.dark_matter)
            ]
            dark_matter_mask = np.isin(
                np.concatenate(
                    [np.arange(start, end) for start, end in sg.mask.dark_matter]
                ),
                dark_matter_mask,
            )
        else:
            dark_matter_mask = null_slice

        if hasattr(cat, "slist"):
            stars_mask = cat.slist
            stars_mask = stars_mask[in_one_of_ranges(stars_mask, sg.mask.stars)]
            stars_mask = np.isin(
                np.concatenate([np.arange(start, end) for start, end in sg.mask.stars]),
                stars_mask,
            )
        else:
            stars_mask = null_slice
        if hasattr(cat, "bhlist"):
            black_holes_mask = cat.bhlist
            black_holes_mask = black_holes_mask[
                in_one_of_ranges(black_holes_mask, sg.mask.black_holes)
            ]
            black_holes_mask = np.isin(
                np.concatenate(
                    [np.arange(start, end) for start, end in sg.mask.black_holes]
                ),
                black_holes_mask,
            )
        else:
            black_holes_mask = null_slice
        return MaskCollection(
            gas=gas_mask,
            dark_matter=dark_matter_mask,
            stars=stars_mask,
            black_holes=black_holes_mask,
        )

    @property
    def group_index(self) -> Union[List[int], int]:
        """
        The index (or indices in multi-galaxy mode when no mask is active) of the
        objects of interest in the halo catalogue. This is just the position in the
        arrays stored in the Caesar halo or galaxy lists (according to the ``group_type``
        given at initialization).

        Returns
        -------
        out : :obj:`int` or :obj:`list`
            The index or indices of the object(s) of interest in the halo catalogue.
        """
        # a bit of logic to placate mypy
        index = self._mask_index()
        if index is not None:
            return index
        else:
            raise ValueError("Caesar _index_attr is unset!")

    @property
    def _region_centre(self) -> cosmo_array:
        """
        Centre(s) of the bounding box regions for spatial masking.

        Caesar stores the "default" centre coordinates in the ``pos`` attribute of
        the catalogue, retrieve this for the object(s) of interest.

        Returns
        -------
        out : :class:`~swiftsimio.objects.cosmo_array`
            The coordinates of the centres of the spatial mask regions.
        """
        cats = [self._catalogue] if not self._multi_galaxy else self._catalogue
        assert isinstance(cats, List)  # placate mypy
        if self._multi_galaxy_catalogue_mask is None:
            return cosmo_array(
                [cat.pos.to(u.kpc) for cat in cats],  # maybe comoving, ensure physical
                comoving=False,
                cosmo_factor=cosmo_factor(a**1, self._caesar.simulation.scale_factor),
            ).to_comoving()
        else:
            return cosmo_array(
                cats[self._multi_galaxy_catalogue_mask].pos.to(
                    u.kpc
                ),  # maybe comoving, ensure physical
                comoving=False,
                cosmo_factor=cosmo_factor(a**1, self._caesar.simulation.scale_factor),
            ).to_comoving()

    @property
    def _region_aperture(self) -> cosmo_array:
        """
        Half-length(s) of the bounding box regions for spatial masking.

        Caesar stores the maximum radius of any particle from the ``pos`` centre in
        the ``radii.total_rmax`` attribute of the catalogue, retrieve this for the
        object(s) of interest. In older versions of Caesar this field may not be
        present, in this case warn the user before resorting to reading the entire
        snapshot volume.

        Returns
        -------
        out : :class:`~swiftsimio.objects.cosmo_array`
            The half-length of the bounding box to use to construct the spatial mask
            regions.
        """
        cats = [self._catalogue] if not self._multi_galaxy else self._catalogue
        assert isinstance(cats, List)  # placate mypy
        if "total_rmax" in cats[0].radii.keys():
            # spatial extent information is present
            if self._multi_galaxy_catalogue_mask is None:
                return cosmo_array(
                    [
                        cat.radii["total_rmax"].to(u.kpc) for cat in cats
                    ],  # maybe comoving, ensure physical
                    comoving=False,
                    cosmo_factor=cosmo_factor(
                        a**1, self._caesar.simulation.scale_factor
                    ),
                ).to_comoving()
            else:
                return cosmo_array(
                    cats[self._multi_galaxy_catalogue_mask]
                    .radii["total_rmax"]
                    .to(u.kpc),  # maybe comoving, ensure physical
                    comoving=False,
                    cosmo_factor=cosmo_factor(
                        a**1, self._caesar.simulation.scale_factor
                    ),
                ).to_comoving()
        else:
            # probably an older caesar output file
            raise KeyError(
                "CAESAR catalogue does not contain group extent information, is probably "
                "an old file. See https://github.com/dnarayanan/caesar/issues/92"
            )

    def _get_preload_fields(self, sg: "SWIFTGalaxy") -> Set[str]:
        """
        Preload data needed to evaluate masks when in multi-galaxy mode.

        In Caesar we don't need to preload anything so return an empty set.

        Parameters
        ----------
        sg : :class:`~swiftgalaxy.reader.SWIFTGalaxy`
            The :class:`~swiftgalaxy.reader.SWIFTGalaxy` that this halo finder
            interface is associated to.

        Returns
        -------
        out : :obj:`set`
            The empty set.
        """
        return set()

    def _mask_catalogue(self) -> Union["CaesarHalo", "CaesarGalaxy"]:
        """
        Helper to select an item from the Caesar lists.

        If in multi-galaxy mode and not currently masked this is an error.

        Returns
        -------
        out : :class:`~loader.Halo` or :class:`~loader.Galaxy`
            The item from the caesar list selected by the mask.
        """
        if self._multi_galaxy and self._multi_galaxy_catalogue_mask is not None:
            cat = self._catalogue[self._multi_galaxy_catalogue_mask]
        elif self._multi_galaxy and self._multi_galaxy_catalogue_mask is None:
            raise RuntimeError("Tried to mask catalogue without mask index!")
        else:
            cat = self._catalogue
        return cat

    @property
    def centre(self) -> cosmo_array:
        """
        Obtain the centre specified by the ``centre_type`` from the halo catalogue.

        In multi-galaxy mode if no mask is active return the centres of all objects of
        interest, otherwise return the centre of the (current) object of interest.

        Returns
        -------
        out : :class:`~swiftsimio.objects.cosmo_array`
            The centre(s) of the object(s) of interest.
        """
        centre_attr = {"": "pos", "minpot": "minpotpos"}[self.centre_type]
        if self._multi_galaxy_catalogue_mask is not None:
            return cosmo_array(
                getattr(
                    self._catalogue[self._multi_galaxy_catalogue_mask], centre_attr
                ).to(
                    u.kpc
                ),  # maybe comoving, ensure physical
                comoving=False,
                cosmo_factor=cosmo_factor(a**1, self._caesar.simulation.scale_factor),
            ).to_comoving()
        cat = [self._catalogue] if not self._multi_galaxy else self._catalogue
        centre = cosmo_array(
            [
                getattr(cat_i, centre_attr).to(u.kpc) for cat_i in cat
            ],  # maybe comoving, ensure physical
            comoving=False,
            cosmo_factor=cosmo_factor(a**1, self._caesar.simulation.scale_factor),
        ).to_comoving()
        if not self._multi_galaxy:
            return centre.squeeze()
        return centre

    @property
    def velocity_centre(self) -> cosmo_array:
        """
        Obtain the velocity centre specified by the ``velocity_centre_type`` from the halo
        catalogue.

        Returns
        -------
        velocity_centre : :class:`~swiftsimio.objects.cosmo_array`
            The velocity centre provided by the halo catalogue.
        """
        vcentre_attr = {"": "vel", "minpot": "minpotvel"}[self.centre_type]
        if self._multi_galaxy_catalogue_mask is not None:
            return cosmo_array(
                getattr(
                    self._catalogue[self._multi_galaxy_catalogue_mask], vcentre_attr
                ).to(u.km / u.s),
                comoving=False,
                cosmo_factor=cosmo_factor(a**0, self._caesar.simulation.scale_factor),
            ).to_comoving()
        cat = [self._catalogue] if not self._multi_galaxy else self._catalogue
        vcentre = cosmo_array(
            [getattr(cat_i, vcentre_attr).to(u.km / u.s) for cat_i in cat],
            comoving=False,
            cosmo_factor=cosmo_factor(a**0, self._caesar.simulation.scale_factor),
        ).to_comoving()
        if not self._multi_galaxy:
            return vcentre.squeeze()
        return vcentre

    def __getattr__(self, attr: str) -> Any:
        """
        Exposes the masked halo catalogue.

        Invoked only if the attribute is not found on the interface class (it is then
        assumed to be a request for a halo catalogue property and delegated). If in
        multi-galaxy mode and not currently masked, use a comprehension to return the
        list of properties, Caesar-style.

        Note that :class:`~swiftgalaxy.reader.Caesar`'s ``__getattr__`` overrides
        the ``__getattr__`` from :class:`~swiftgalaxy.reader._HaloCatalogue`.

        Parameters
        ----------
        attr : :obj:`str`
            The name of the requested attribute.

        Returns
        -------
        out : :obj:`object`
            The requested attribute.
        """
        if attr == "_catalogue":  # guard infinite recursion
            try:
                return object.__getattribute__(self, "_catalogue")
            except AttributeError:
                return None
        if self._multi_galaxy_catalogue_mask is not None:
            return getattr(self._catalogue[self._multi_galaxy_catalogue_mask], attr)
        elif self._multi_galaxy and self._multi_galaxy_catalogue_mask is None:
            return [getattr(cat, attr) for cat in self._catalogue]
        else:
            return getattr(self._catalogue, attr)

    def __repr__(self) -> str:
        """
        Expose the catalogue ``__repr__`` for interactive use.

        Delegates creating a string repreesntation to the internal
        :class:`~loader.CAESAR` holding the Caesar data.

        Returns
        -------
        out : :obj:`str`
            The string representation of the catalogue.
        """
        return self._catalogue.__repr__()


class Standalone(_HaloCatalogue):
    """
    A bare-bones tool to initialize a :class:`~swiftgalaxy.reader.SWIFTGalaxy`
    without an associated halo catalogue.

    Provides an interface to specify the minimum required information to
    instantiate a :class:`~swiftgalaxy.reader.SWIFTGalaxy`.

    Parameters
    ----------

    centre : :class:`~swiftsimio.objects.cosmo_array`, default: ``None``
        A value for this parameter is required, and must have units of length.
        Specifies the geometric centre in simulation coordinates. Particle
        coordinates will be shifted such that this position is located at (0, 0, 0)
        and if the boundary is periodic it will be wrapped to place the origin at
        the centre.

    velocity_centre : :class:`~swiftsimio.objects.cosmo_array`, default: ``None``
        A value for this parameter is required, and must have units of speed.
        Specifies the reference velocity relative to the simulation frame. Particle
        velocities will be shifted such that a particle with the specified velocity
        in the simulation frame will have zero velocity in the
        :class:`~swiftgalaxy.reader.SWIFTGalaxy` frame.

    spatial_offsets : :class:`~swiftsimio.objects.cosmo_array`, default: ``None``
        Offsets along each axis to select a spatial region around the ``centre``.
        May be used in conjunction with ``extra_mask``, for example to select all
        simulation particles in an aperture around the object of interest (see
        'Masking' section of documentation for a cookbook example). Provide a pair
        of offsets from the object's centre along each axis to define the region,
        for example for a cube extending +/- 1 Mpc from the centre:
        ``cosmo_array([[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]], u.Mpc)``.

    extra_mask : :obj:`str` or :class:`~swiftgalaxy.masks.MaskCollection` (optional), \
    default: ``None``
        Mask to apply to particles after spatial masking. A user-defined mask
        can be provided as an an object (such as a
        :class:`swiftgalaxy.masks.MaskCollection`) that has attributes with
        names corresponding to present particle names (e.g. gas, dark_matter,
        etc.), each containing a mask.

    Examples
    --------
    Often the most pragmatic way to create a selection of particles using
    :class:`~swiftgalaxy.halo_catalogues.Standalone` is to first select a spatial region
    guaranteed to contain the particles of interest and then create the final mask
    programatically using :class:`~swiftgalaxy.reader.SWIFTGalaxy`'s masking features.
    For example, suppose we know that there is a galaxy with its centre at
    (2, 2, 2) Mpc and we eventually want all particles in a spherical aperture 1 Mpc
    in radius around this point. We start with a cubic spatial mask enclosing this
    region:

    ::

        from swiftgalaxy import SWIFTGalaxy, Standalone, MaskCollection
        from swiftsimio import cosmo_array
        import unyt as u

        sg = SWIFTGalaxy(
            "my_snapshot.hdf5",
            Standalone(
                centre=cosmo_array([2.0, 2.0, 2.0], u.Mpc),
                velocity_centre=cosmo_array([0.0, 0.0, 0.0], u.km / u.s),
                spatial_offsets=cosmo_array(
                    [[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]],
                    u.Mpc,
                ),
                extra_mask=None,  # we'll define the exact set of particles later
            )
        )

    We next define the masks selecting particles in the desired spherical aperture,
    conveniently using :class:`~swiftgalaxy.reader.SWIFTGalaxy`'s spherical coordinates
    feature, and store them in a :class:`~swiftgalaxy.masks.MaskCollection`:

    ::

        mask_collection = MaskCollection(
            gas=sg.gas.spherical_coordinates.r < 1 * u.Mpc,
            dark_matter=sg.dark_matter.spherical_coordinates.r < 1 * u.Mpc,
            stars=sg.stars.spherical_coordinates.r < 1 * u.Mpc,
            black_holes=sg.black_holes.spherical_coordinates.r < 1 * u.Mpc,
        )

    Finally, we apply the mask to the ``sg`` object:

    ::

        sg = sg.mask_particles(mask_collection)

    We're now ready to proceed with analysis of the particles in the 1 Mpc spherical
    aperture using this ``sg`` object.

    .. note::

        :meth:`~swiftgalaxy.reader.SWIFTGalaxy.mask_particles` applies the masks in-place.
        The mask could also be applied with the
        :meth:`~swiftgalaxy.reader.SWIFTGalaxy.__getattr__` method (i.e. in square
        brackets), but this returns a copy of the :class:`~swiftgalaxy.reader.SWIFTGalaxy`
        object. If memory efficiency is a concern, prefer the
        :meth:`~swiftgalaxy.reader.SWIFTGalaxy.mask_particles` approach.
    """

    _index_attr = None

    def __init__(
        self,
        centre: Optional[cosmo_array] = None,
        velocity_centre: Optional[cosmo_array] = None,
        spatial_offsets: Optional[cosmo_array] = None,
        extra_mask: Optional[Union[str, MaskCollection]] = None,
    ) -> None:
        if centre is None:
            raise ValueError("A centre is required.")
        else:
            self._centre = centre
        if velocity_centre is None:
            raise ValueError("A velocity_centre is required.")
        else:
            self._velocity_centre = velocity_centre
        if spatial_offsets is None:
            warn(
                "No spatial_offsets provided. All particles in simulation box will be "
                "read (before masking). This is likely to be slow/inefficient."
            )
        self._user_spatial_offsets = spatial_offsets
        if extra_mask == "bound_only":
            raise ValueError(
                "extra_mask='bound_only' is not supported with Standalone."
            )
        super().__init__(extra_mask=extra_mask)
        if spatial_offsets is None and self._multi_galaxy:
            raise ValueError(
                "To use `Standalone` with multiple galaxies you must initialize with a "
                "`spatial_offsets` argument provided."
            )
        return

    def _load(self) -> None:
        """
        Do non-trivial I/O operations needed at initialization.

        Nothing needed for Standalone, do nothing.
        """
        pass

    def _generate_spatial_mask(self, snapshot_filename: str) -> SWIFTMask:
        """
        Evaluate the spatial mask.

        Returns a mask object to select the particles from the snapshot in the region of
        interest.

        Parameters
        ----------
        snapshot_filename : :obj:`str`
            The location of the SWIFT snapshot file.

        Returns
        -------
        out : :class:`~swiftsimio.masks.SWIFTMask`
            The spatial mask to select particles in the region of interest.
        """
        # if we're here then the user didn't provide a mask, read the whole box
        sm = mask(snapshot_filename, spatial_only=True)
        boxsize = sm.metadata.boxsize
        region = [[0.0 * b, 1.0 * b] for b in boxsize]
        sm.constrain_spatial(region)
        return sm

    def _generate_bound_only_mask(self, sg: "SWIFTGalaxy") -> MaskCollection:
        """
        The :class:`~swiftgalaxy.halo_catalogues.Standalone` class has no intrinsic
        notion of a bound object, so this function is implemented only to be able to
        instantiate an object of this class. It just raises an exception if called.

        Parameters
        ----------
        sg : :class:`~swiftgalaxy.reader.SWIFTGalaxy`
            The :class:`~swiftgalaxy.reader.SWIFTGalaxy` that this halo finder
            interface is associated to.

        Raises
        ------
        NotImplementedError : always raised if this function is called.
        """
        raise NotImplementedError  # guarded in initialization, should not reach here

    @property
    def _region_centre(self) -> cosmo_array:
        """
        Centre(s) of the bounding box regions for spatial masking.

        The :class:`~swiftgalaxy.halo_catalogues.Standalone` class requires the user
        to define a centre (or centres) at initialization, return these.

        Returns
        -------
        out : :class:`~swiftsimio.objects.cosmo_array`
            The coordinates of the centres of the spatial mask regions.
        """
        if self._multi_galaxy_index_mask is None:
            return self._centre
        else:
            return self._centre[self._multi_galaxy_index_mask]

    @property
    def _region_aperture(self) -> cosmo_array:
        """
        Half-length(s) of the bounding box regions for spatial masking.

        The :class:`~swiftgalaxy.halo_catalogues.Standalone` class expects the user to
        define a region of interest at initialization, retrieve this to define the
        spatial masking region if they did (else read the whole snapshot volume).

        Returns
        -------
        out : :class:`~swiftsimio.objects.cosmo_array`
            The half-length of the bounding box to use to construct the spatial mask
            regions.
        """
        if self._user_spatial_offsets is not None:
            if self._multi_galaxy_index_mask is None:
                return np.repeat(
                    np.max(np.abs(self._user_spatial_offsets)), len(self._centre)
                )
            else:
                return np.max(np.abs(self._user_spatial_offsets))
        else:
            # should never reach here (guarded in initialization)
            raise NotImplementedError

    def _get_preload_fields(self, sg: "SWIFTGalaxy") -> Set[str]:
        """
        Preload data needed to evaluate masks when in multi-galaxy mode.

        For Standalone, we don't need any data, so return the empty set.

        Parameters
        ----------
        sg : :class:`~swiftgalaxy.reader.SWIFTGalaxy`
            The :class:`~swiftgalaxy.reader.SWIFTGalaxy` that this halo finder
            interface is associated to.

        Returns
        -------
        out : :obj:`set`
            The empty set.
        """
        return set()

    @property
    def centre(self) -> cosmo_array:
        """
        Obtain the centre specified at initialization.

        In multi-galaxy mode if no mask is active return the centres of all objects of
        interest, otherwise return the centre of the (current) object of interest.

        Returns
        -------
        out : :class:`~swiftsimio.objects.cosmo_array`
            The centre(s) of the object(s) of interest.
        """
        if self._multi_galaxy_index_mask is not None:
            return self._centre[self._multi_galaxy_index_mask]
        return self._centre

    @property
    def velocity_centre(self) -> cosmo_array:
        """
        Obtain the velocity centre specified at initialization.

        In multi-galaxy mode if no mask is active return the centres of all objects of
        interest, otherwise return the centre of the (current) object of interest.

        Returns
        -------
        out : :class:`~swiftsimio.objects.cosmo_array`
            The velocity coordinate origin.
        """
        if self._multi_galaxy_index_mask is not None:
            return self._velocity_centre[self._multi_galaxy_index_mask]
        return self._velocity_centre
