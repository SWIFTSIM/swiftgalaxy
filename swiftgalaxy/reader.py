"""
This module contains wrappers for the parts making up a :mod:`swiftsimio`
dataset to facilitate analyses of individual simulated galaxies.

The top-level wrapper is :class:`SWIFTGalaxy`, which inherits from
:class:`~swiftsimio.reader.SWIFTDataset`. It extends the functionality of a
dataset to select particles belonging to a single galaxy, handle coordinate
transformations while keeping all particles in a consistent frame of reference,
providing spherical and cylindrical coordinates, and more.

Additional wrappers are provided for
:class:`swiftsimio.reader.__SWIFTGroupDataset` and
:class:`swiftsimio.reader.__SWIFTNamedColumnDataset`:
:class:`_SWIFTGroupDatasetHelper` and
:class:`_SWIFTNamedColumnDatasetHelper`, respectively. In general objects of
these types should not be created directly by users, but rather by an object of
the :class:`SWIFTGalaxy` class.
"""

from warnings import warn, catch_warnings, filterwarnings
from copy import deepcopy
import numpy as np
from scipy.spatial.transform import Rotation
import unyt
from swiftsimio import metadata as swiftsimio_metadata
from swiftsimio.reader import (
    SWIFTDataset,
    __SWIFTNamedColumnDataset,
    __SWIFTGroupDataset,
)
from swiftsimio.objects import cosmo_array, cosmo_factor, a
from swiftsimio.masks import SWIFTMask
from swiftgalaxy.halo_catalogues import _HaloCatalogue
from swiftgalaxy.masks import MaskCollection

from typing import Union, Any, Optional, Set


def _apply_box_wrap(coords: cosmo_array, boxsize: Optional[cosmo_array]) -> cosmo_array:
    """
    Wrap coordinates for periodic box.

    Given some coordinates, wrap the box size so that they lie within the periodic volume.

    Parameters
    ----------
    coords : :class:`~swiftsimio.objects.cosmo_array`
        The coordinates to be wrapped.

    boxsize : :class:`~swiftsimio.objects.cosmo_array` or ``None``
        The dimensions of the box to wrap (3 elements).

    Returns
    -------
    out : :class:`~swiftsimio.objects.cosmo_array`
        The coordinates wrapped to lie within the box dimensions.
    """
    if boxsize is None:
        return coords
    # Would like to ensure comoving coordinates here, but metadata gives boxsize as a
    # unyt_array instead of a cosmo_array. Pending swiftsimio issue #128. When
    # implementing, remove cast(s) to cosmo_array in test_coordinate_transformations.
    with catch_warnings():
        filterwarnings(
            "ignore", category=RuntimeWarning, message="Mixing ufunc arguments"
        )
        retval = (coords + boxsize / 2.0) % boxsize - boxsize / 2.0
    retval.comoving = coords.comoving
    retval.cosmo_factor = coords.cosmo_factor
    return retval


def _apply_translation(coords: cosmo_array, offset: cosmo_array) -> cosmo_array:
    """
    Apply a translation to a coordinate array.

    Also warns the user of ambiguity in physical/comoving coordinates.

    Parameters
    ----------
    coords : :class:`~swiftsimio.objects.cosmo_array`
        The coordinate array to be translated.
    offset : :class:`~swiftsimio.objects.cosmo_array`
        The translation vector.

    Returns
    -------
    out : :class:`~swiftsimio.objects.cosmo_array`
        The coordinate array with the translation applied.
    """
    if hasattr(offset, "comoving") and coords.comoving:
        offset = offset.to_comoving()
    elif hasattr(offset, "comoving") and not coords.comoving:
        offset = offset.to_physical()
    elif not hasattr(offset, "comoving"):
        msg = (
            "Translation assumed to be in comoving (not physical) coordinates."
            if coords.comoving
            else "Translation assumed to be in physical (not comoving) coordinates."
        )
        warn(msg, category=UserWarning)
    return coords + offset


def _apply_rotmat(coords: cosmo_array, rotation_matrix: np.ndarray) -> cosmo_array:
    """
    Apply a rotation matrix to a coordinate array.

    Applies a rotation in-place using a view through a :class:`numpy.ndarray`, then
    restores units and metadata of the :class:`~swiftsimio.objects.cosmo_array`.

    Parameters
    ----------
    coords : :class:`~swiftsimio.objects.cosmo_array`
        The coordinate array to be rotated.
    rotation_matrix : :class:`~numpy.ndarray`
        The rotation matrix (3x3).

    Returns
    -------
    out : :class:`~swiftsimio.objects.cosmo_array`
        The coordinate array with rotation applied.
    """
    return cosmo_array(
        coords.view(np.ndarray).dot(rotation_matrix),
        units=coords.units,
        cosmo_factor=coords.cosmo_factor,
        comoving=coords.comoving,
    )


def _apply_4transform(
    coords: cosmo_array, transform: np.ndarray, transform_units: unyt.unyt_quantity
) -> cosmo_array:
    """
    Apply an arbitary coordinate transformation (translation mixed with rotation) to a
    coordinate array.

    An arbitrary coordinate transformation mixing translations and rotations can be
    expressed as a 4x4 matrix. However, such a matrix has mixed units, so we need to
    assume a consistent unit for all transformations and work with bare arrays. We also
    always assume comoving coordinates.

    Parameters
    ----------
    coords : :class:`~swiftsimio.objects.cosmo_array`
        The coordinate array to be transformed.
    transform : :class:`~numpy.ndarray`
        The 4x4 transformation matrix.
    transform_units : :class:`unyt.unyt_quantity`
        The units assumed in the translation portion of the transformation matrix.

    Returns
    -------
    out : :class:`~swiftsimio.objects.cosmo_array`
        The coordinate array with transformation applied.
    """
    retval = cosmo_array(
        np.hstack(
            (
                coords.to_comoving().to_value(transform_units),
                np.ones(coords.shape[0])[:, np.newaxis],
            )
        ).dot(transform)[:, :3],
        units=transform_units,
        comoving=True,
        cosmo_factor=coords.cosmo_factor,
    )
    if coords.comoving:
        return retval.to_comoving()
    else:
        return retval.to_physical()


class _CoordinateHelper(object):
    """
    Container class for coordinates.

    Stores a dictionary of coordinate arrays and names (and aliases) for these,
    and enables accessing the arrays via :meth:`__getattr__` (dot syntax). For
    interactive use, printing a :class:`_CoordinateHelper`
    lists the available coordinate names and aliases.

    Parameters
    ----------
    coordinates : :obj:`dict` or :class:`~swiftsimio.objects.cosmo_array`
        The coordinate array(s) to be stored.

    masks : :class:`dict`
        Available coordinate names and their aliases with corresponding masks
        (or keys) into the coordinate array or dictionary for each.
    """

    def __init__(self, coordinates: Union[dict, cosmo_array], masks: dict) -> None:
        self._coordinates: Union[np.ndarray, dict] = coordinates
        self._masks: dict = masks
        return

    def __getattr__(self, attr: str) -> cosmo_array:
        """
        Get a coordinate array using attribute (dot) syntax.

        Looks up the requested attribute in the internal register of coordinate
        array names and their aliases to retrieve the array corresponding to the
        request.

        Parameters
        ----------
        attr : :obj:`str`
            The name (possibly an alias) of the coordinate array to retrieve.

        Returns
        -------
        out : :class:`~swiftsimio.objects.cosmo_array`
            The requested coordinate array.
        """
        return self._coordinates[self._masks[attr]]

    def __str__(self) -> str:
        """
        Get a string representation of the available coordinate array names.

        Returns
        -------
        out : :obj:`str`
            The string representation.
        """
        keys = ", ".join(self._masks.keys())
        return f"Available coordinates: {keys}."

    def __repr__(self) -> str:
        """
        Get a string representation of the available coordinate array names.

        Returns
        -------
        out : :obj:`str`
            The string representation.
        """
        return self.__str__()


class _SWIFTNamedColumnDatasetHelper(object):
    """
    A wrapper class to enable :class:`SWIFTGalaxy`
    functionality for a :class:`swiftsimio.reader.__SWIFTNamedColumnDataset`.

    Unlike the :class:`SWIFTGalaxy` class that inherits
    directly from :class:`~swiftsimio.reader.SWIFTDataset`, for technical
    reasons this class does *not* inherit
    :class:`swiftsimio.reader.__SWIFTNamedColumnDataset`. It does, however,
    expose the functionality of that class by maintaining an instance
    internally and forwarding any attribute lookups that it does not handle
    itself to its internal named column dataset.

    Like :class:`_SWIFTGroupDatasetHelper`, this class handles the
    transformation and masking of data from calls to :class:`SWIFTGalaxy`
    routines.

    Instances of this helper class should in general not be created separately
    since they require an instance of :class:`SWIFTGalaxy` to function and
    will be created automatically by that class.

    If any datasets contained in a named column dataset should transform like
    particle coordinates or velocities, these can be specified in the arguments
    ``transforms_like_coordinates`` and ``transforms_like_velocities`` to
    :class:`SWIFTGalaxy` as a string containing a dot, e.g. the argument
    ``transforms_like_coordinates={'coordinates',
    'extra_coordinates.an_extra_coordinate'}`` is syntactically valid.

    Parameters
    ----------
    named_column_dataset : :class:`swiftsimio.reader.__SWIFTNamedColumnDataset`
        The named column dataset to be wrapped.

    particle_dataset_helper : :class:`_SWIFTGroupDatasetHelper`
        Used to store a reference to the parent
        :class:`_SWIFTGroupDatasetHelper` object.

    See Also
    --------
    :class:`SWIFTGalaxy`
    :class:`_SWIFTGroupDatasetHelper`
    """

    def __init__(
        self,
        named_column_dataset: "__SWIFTNamedColumnDataset",
        particle_dataset_helper: "_SWIFTGroupDatasetHelper",
    ) -> None:
        self._named_column_dataset: __SWIFTNamedColumnDataset = named_column_dataset
        self._particle_dataset_helper: "_SWIFTGroupDatasetHelper" = (
            particle_dataset_helper
        )
        self._initialised: bool = True
        return

    def __str__(self) -> str:
        """
        Get a string representation of the object.

        Delegated to the underlying :class:`~swiftsimio.reader.__SWIFTNamedColumnDataset`.

        Returns
        -------
        out : :obj:`str`
            The string representation.
        """
        return str(self._named_column_dataset)

    def __repr__(self) -> str:
        """
        Get a string representation of the object.

        Delegated to the underlying :class:`~swiftsimio.reader.__SWIFTNamedColumnDataset`.

        Returns
        -------
        out : :obj:`str`
            The string representation.
        """
        return self.__str__()

    def __getattribute__(self, attr: str) -> Any:
        """
        Get an attribute of the object.

        Redirects requests for datasets to the underlying
        :class:`~swiftsimio.reader.__SWIFTNamedColumnDataset` and then handles
        any transformations and masking when new data are read.

        Parameters
        ----------
        attr : :obj:`str`
            The name of the attribute.

        Returns
        -------
        out : :obj:`object`
            The value of the attribute.
        """
        named_column_dataset = object.__getattribute__(self, "_named_column_dataset")
        particle_dataset_helper = object.__getattribute__(
            self, "_particle_dataset_helper"
        )
        if attr in named_column_dataset.named_columns:
            # we're dealing with one of the named columns
            if getattr(named_column_dataset, f"_{attr}") is None:
                # going to read from file: apply masks, transforms
                if particle_dataset_helper._swiftgalaxy._warn_on_read:
                    warn(
                        f"Reading {attr} from snapshot file, this may be unintended "
                        "(should it be preloaded if using SWIFTGalaxies to iterate over "
                        "objects of interest?",
                        UserWarning,
                    )
                data = getattr(named_column_dataset, attr)  # raw data loaded
                data = particle_dataset_helper._apply_data_mask(data)
                data = particle_dataset_helper._apply_transforms(
                    data, f"{named_column_dataset.name}.{attr}"
                )
                setattr(named_column_dataset, f"_{attr}", data)
            else:
                # just return the data
                pass
        try:
            # beware collisions with SWIFTGroupDataset namespace
            return object.__getattribute__(self, attr)
        except AttributeError:
            # exposes everything else in __dict__
            return getattr(named_column_dataset, attr)

    def __setattr__(self, attr: str, value: Any) -> None:
        """
        Set an attribute of the object.

        Redirects setting attributes to the underlying
        :class:`~swiftsimio.reader.__SWIFTNamedColumnDataset` objects when relevant.

        Parameters
        ----------
        attr : :obj:`str`
            Name of the attribute.
        value : :obj:`object`
            The value to assign to the attribute.
        """
        # pass particle data through to actual SWIFTNamedColumnDataset
        if not hasattr(self, "_initialised"):
            # guard during initialisation
            object.__setattr__(self, attr, value)
            return
        column_names = self._named_column_dataset.named_columns
        if (attr in column_names) or (
            (attr.startswith("_")) and (attr[1:] in column_names)
        ):
            setattr(self._named_column_dataset, attr, value)
            return
        else:
            object.__setattr__(self, attr, value)
            return

    def __getitem__(self, mask: slice) -> "_SWIFTNamedColumnDatasetHelper":
        """
        Apply a mask to the :class:`~swiftgalaxy.reader._SWIFTNamedColumnDatasetHelper`
        with square-bracket notation.

        To ensure internal consistency this requires producing a full ("deep") copy of
        the parent :class:`~swiftgalaxy.reader.SWIFTGalaxy` object and all of its contents
        (but data is masked at copy time to avoid unnecessary memory overhead).

        Parameters
        ----------
        mask : :obj:`slice`
            The mask to apply to the named column data arrays (and all other data arrays
            for particles of the same type).
        """
        return self._data_copy(mask=mask)

    def __copy__(self) -> "_SWIFTNamedColumnDatasetHelper":
        """
        Create a copy of the :class:`~swiftgalaxy.reader._SWIFTNamedColumnDatasetHelper`
        without copying data (a "shallow" copy).

        Returns
        -------
        out : :class:`~swiftgalaxy.reader._SWIFTNamedColumnDatasetHelper`
            The copy of the :class:`~swiftgalaxy.reader._SWIFTNamedColumnDatasetHelper`
            object.
        """
        return getattr(self._particle_dataset_helper.__copy__(), self.name)

    def __deepcopy__(
        self, memo: Optional[dict] = None
    ) -> "_SWIFTNamedColumnDatasetHelper":
        """
        Create a copy of the :class:`~swiftgalaxy.reader._SWIFTNamedColumnDatasetHelper`
        including copying data (a "deep" copy).

        Parameters
        ----------
        memo : :obj:`dict` (optional), default: ``None``
            For the copy operation to keep a record of already copied objects.

        Returns
        -------
        out : :class:`~swiftgalaxy.reader._SWIFTNamedColumnDatasetHelper`
            The copy of the :class:`~swiftgalaxy.reader.SWIFTGalaxy` object.
        """

        return self._data_copy()

    def _data_copy(
        self, mask: Optional[slice] = None
    ) -> "_SWIFTNamedColumnDatasetHelper":
        """
        Create a copy of the :class:`~swiftgalaxy.reader._SWIFTNamedColumnDatasetHelper`
        including copying data (a "deep" copy).

        Parameters
        ----------
        mask : :obj:`slice` (optional), default: ``None``
            Copy only the subset of the data corresponding to the ``mask``.

        Returns
        -------
        out : :class:`~swiftgalaxy.reader._SWIFTNamedColumnDatasetHelper`
            A (possibly masked) copy of the
            :class:`~swiftgalaxy.reader._SWIFTNamedColumnDatasetHelper` object.
        """
        return getattr(self._particle_dataset_helper._data_copy(mask=mask), self.name)


class _SWIFTGroupDatasetHelper(object):
    """
    A wrapper class to enable :class:`SWIFTGalaxy`
    functionality for a :class:`swiftsimio.reader.__SWIFTGroupDatasets`.

    Unlike the :class:`SWIFTGalaxy` class that inherits
    directly from :class:`~swiftsimio.reader.SWIFTDataset`, for technical
    reasons this class does *not* inherit
    :class:`swiftsimio.reader.__SWIFTGroupDatasets`. It does, however,
    expose the functionality of that class by maintaining an instance
    internally and forwarding any attribute lookups that it does not handle
    itself to its internal particle dataset.

    In addition to handling the transformation and masking of data from calls
    to :class:`SWIFTGalaxy` routines, this class provides
    particle coordinates and velocities in cartesian, spherical and cylindrical
    coordinates through the properties:

    + :attr:`cartesian_coordinates`
    + :attr:`cartesian_velocities`
    + :attr:`spherical_coordinates`
    + :attr:`spherical_velocities`
    + :attr:`cylindrical_coordinates`
    + :attr:`cylindrical_velocities`

    These are evaluated lazily and automatically re-calculated if necessary,
    such as after a coordinate rotation.

    Instances of this helper class should in general not be created separately
    since they require an instance of :class:`SWIFTGalaxy` to function and
    will be created automatically by that class.

    Parameters
    ----------
    particle_dataset : :class:`swiftsimio.reader.__SWIFTGroupDatasets`
        The particle dataset to be wrapped.

    swiftgalaxy : :class:`SWIFTGalaxy`
        Used to store a reference to the parent :class:`SWIFTGalaxy`.

    See Also
    --------
    :class:`SWIFTGalaxy`
    :class:`_CoordinateHelper`

    Examples
    --------

    The cartesian, spherical and cylindrical coordinates of gas particles can
    be accessed, for example, by (``mygalaxy`` is a :class:`SWIFTGalaxy`):

    ::

        mygalaxy.gas.cartesian_coordinates.x
        mygalaxy.gas.cartesian_coordinates.y
        mygalaxy.gas.cartesian_coordinates.z
        mygalaxy.gas.cartesian_velocities.x
        mygalaxy.gas.cartesian_velocities.y
        mygalaxy.gas.cartesian_velocities.z
        mygalaxy.gas.spherical_coordinates.r
        mygalaxy.gas.spherical_coordinates.theta
        mygalaxy.gas.spherical_coordinates.phi
        mygalaxy.gas.spherical_velocities.r
        mygalaxy.gas.spherical_velocities.theta
        mygalaxy.gas.spherical_velocities.phi
        mygalaxy.gas.cylindrical_coordinates.rho
        mygalaxy.gas.cylindrical_coordinates.phi
        mygalaxy.gas.cylindrical_coordinates.z
        mygalaxy.gas.cylindrical_velocities.rho
        mygalaxy.gas.cylindrical_velocities.phi
        mygalaxy.gas.cylindrical_velocities.z
    """

    def __init__(
        self, particle_dataset: "__SWIFTGroupDataset", swiftgalaxy: "SWIFTGalaxy"
    ) -> None:
        self._particle_dataset: __SWIFTGroupDataset = particle_dataset
        self._swiftgalaxy: "SWIFTGalaxy" = swiftgalaxy
        self._named_column_dataset_helpers: dict[
            str, _SWIFTNamedColumnDatasetHelper
        ] = dict()
        particle_metadata = getattr(self.metadata, f"{self.group_name}_properties")
        named_columns_names = [
            fn
            for (fn, fp) in zip(
                particle_metadata.field_names, particle_metadata.field_paths
            )
            if particle_metadata.named_columns[fp] is not None
        ]
        for named_columns_name in named_columns_names:
            # This is the named_columns instance to wrap:
            named_columns = getattr(self._particle_dataset, named_columns_name)
            # We'll make a custom type to present a nice name to the user.
            particle_nice_name = swiftsimio_metadata.particle_types.particle_name_class[
                self.group
            ]
            nice_name = (
                f"{particle_nice_name}"
                f"{named_columns.field_path.split('/')[-1]}ColumnsHelper"
            )
            TypeNamedColumnDatasetHelper = type(
                nice_name, (_SWIFTNamedColumnDatasetHelper, object), dict()
            )
            self._named_column_dataset_helpers[named_columns_name] = (
                TypeNamedColumnDatasetHelper(named_columns, self)
            )
        self._spherical_coordinates: Optional[dict] = None
        self._cylindrical_coordinates: Optional[dict] = None
        self._spherical_velocities: Optional[dict] = None
        self._cylindrical_velocities: Optional[dict] = None
        self._initialised: bool = True
        return

    def __str__(self) -> str:
        """
        Get a string representation of the object.

        Delegated to the underlying :class:`~swiftsimio.reader.__SWIFTGroupDataset`.

        Returns
        -------
        out : :obj:`str`
            The string representation.
        """
        return str(self._particle_dataset)

    def __repr__(self) -> str:
        """
        Get a string representation of the object.

        Delegated to the underlying :class:`~swiftsimio.reader.__SWIFTGroupDataset`.

        Returns
        -------
        out : :obj:`str`
            The string representation.
        """
        return self.__str__()

    def __getattribute__(self, attr: str) -> Any:
        """
        Get an attribute of the object.

        Redirects requests for datasets to the underlying
        :class:`~swiftsimio.reader.__SWIFTGroupDataset` and then handles
        any transformations and masking when new data are read.

        Parameters
        ----------
        attr : :obj:`str`
            The name of the attribute.

        Returns
        -------
        out : :obj:`object`
            The value of the attribute.
        """
        particle_name = object.__getattribute__(self, "_particle_dataset").group_name
        metadata = object.__getattribute__(self, "_particle_dataset").metadata
        particle_metadata = getattr(metadata, f"{particle_name}_properties")
        particle_dataset = object.__getattribute__(self, "_particle_dataset")
        if attr in particle_metadata.field_names:
            # check if we're dealing with a named columns field
            if object.__getattribute__(self, "_is_namedcolumns")(attr):
                return object.__getattribute__(self, "_named_column_dataset_helpers")[
                    attr
                ]
            # otherwise we're dealing with a particle data table
            if getattr(particle_dataset, f"_{attr}") is None:
                if object.__getattribute__(self, "_swiftgalaxy")._warn_on_read:
                    warn(
                        f"Reading {attr} from snapshot file, this may be unintended "
                        "(should it be preloaded if using SWIFTGalaxies to iterate over "
                        "objects of interest?",
                        UserWarning,
                    )
                # going to read from file: apply masks, transforms
                data = getattr(particle_dataset, attr)  # raw data loaded
                data = object.__getattribute__(self, "_apply_data_mask")(data)
                data = object.__getattribute__(self, "_apply_transforms")(data, attr)
                setattr(particle_dataset, f"_{attr}", data)
            else:
                # just return the data
                pass
        try:
            # beware collisions with SWIFTDataset namespace
            return object.__getattribute__(self, attr)
        except AttributeError:
            # exposes everything else in __dict__
            return getattr(particle_dataset, attr)

    def __setattr__(self, attr: str, value: Any) -> None:
        """
        Set an attribute of the object.

        Redirects setting attributes to the underlying
        :class:`~swiftsimio.reader.__SWIFTGroupDataset` objects when relevant.

        Parameters
        ----------
        attr : :obj:`str`
            Name of the attribute.
        value : :obj:`object`
            The value to assign to the attribute.
        """
        # pass particle data through to actual SWIFTDataset
        if not hasattr(self, "_initialised"):
            # guard during initialisation
            object.__setattr__(self, attr, value)
            return
        field_names = getattr(
            self._particle_dataset.metadata,
            f"{self._particle_dataset.group_name}_properties",
        ).field_names
        if (attr in field_names) and self._is_namedcolumns(attr):
            self._named_column_dataset_helpers[attr] = value
            return
        elif (attr in field_names) or (
            (attr.startswith("_")) and (attr[1:] in field_names)
        ):
            setattr(self._particle_dataset, attr, value)
            return
        else:
            object.__setattr__(self, attr, value)
            return

    def __getitem__(self, mask: slice) -> "_SWIFTGroupDatasetHelper":
        """
        Apply a mask to the :class:`~swiftgalaxy.reader._SWIFTGroupDatasetHelper`
        with square-bracket notation.

        To ensure internal consistency this requires producing a full ("deep") copy of
        the parent :class:`~swiftgalaxy.reader.SWIFTGalaxy` object and all of its contents
        (but data is masked at copy time to avoid unnecessary memory overhead).

        Parameters
        ----------
        mask : :obj:`slice`
            The mask to apply to the particle data arrays.
        """
        return self._data_copy(mask=mask)

    def __copy__(self) -> "_SWIFTGroupDatasetHelper":
        """
        Create a copy of the :class:`~swiftgalaxy.reader._SWIFTGroupDatasetHelper`
        without copying data (a "shallow" copy).

        Returns
        -------
        out : :class:`~swiftgalaxy.reader._SWIFTGroupDatasetHelper`
            The copy of the :class:`~swiftgalaxy.reader._SWIFTGroupDatasetHelper` object.
        """
        return getattr(self._swiftgalaxy.__copy__(), self.particle_name)

    def __deepcopy__(self, memo: Optional[dict] = None) -> "_SWIFTGroupDatasetHelper":
        """
        Create a copy of the :class:`~swiftgalaxy.reader._SWIFTGroupDatasetHelper`
        including copying data (a "deep" copy).

        Parameters
        ----------
        memo : :obj:`dict` (optional), default: ``None``
            For the copy operation to keep a record of already copied objects.

        Returns
        -------
        out : :class:`~swiftgalaxy.reader._SWIFTGroupDatasetHelper`
            The copy of the :class:`~swiftgalaxy.reader._SWIFTGroupDatasetHelper` object.
        """
        return self._data_copy()

    def _data_copy(self, mask: Optional[slice] = None) -> "_SWIFTGroupDatasetHelper":
        """
        Create a copy of the :class:`~swiftgalaxy.reader._SWIFTGroupDatasetHelper`
        including copying data (a "deep" copy).

        Parameters
        ----------
        mask : :obj:`slice` (optional), default: ``None``
            Copy only the subset of the data corresponding to the ``mask``.

        Returns
        -------
        out : :class:`~swiftgalaxy.reader._SWIFTGroupDatasetHelper`
            A (possibly masked) copy of the
            :class:`~swiftgalaxy.reader._SWIFTGroupDatasetHelper` object.
        """
        mask_collection = MaskCollection(
            **{
                k: None if k != self.group_name else mask
                for k in self.metadata.present_group_names
            }
        )
        return getattr(
            self._swiftgalaxy._data_copy(mask_collection=mask_collection),
            self.group_name,
        )

    def _is_namedcolumns(self, field_name: str) -> bool:
        """
        Checks a string against the metadata to determine if it describes a named column.

        Parameters
        ----------
        field_name : :obj:`str`
            The name of the field to check against the list of named column datasets.

        Returns
        -------
        out : :obj:`bool`
            ``True`` if ``field_name`` describes a named column, else ``False``.
        """
        particle_name = self._particle_dataset.group_name
        particle_metadata = getattr(
            self._particle_dataset.metadata, f"{particle_name}_properties"
        )
        field_path = dict(
            zip(particle_metadata.field_names, particle_metadata.field_paths)
        )[field_name]
        return particle_metadata.named_columns[field_path] is not None

    def _apply_data_mask(self, data: cosmo_array) -> cosmo_array:
        """
        Used internally to apply existing masks on reading new data.

        Parameters
        ----------
        data : :class:`~swiftsimio.objects.cosmo_array`
            The data to mask.

        Returns
        -------
        out : :class:`~swiftsimio.objects.cosmo_array`
            The data with any masks applied.
        """
        if self._swiftgalaxy._extra_mask is not None:
            mask = getattr(
                self._swiftgalaxy._extra_mask, self._particle_dataset.group_name
            )
            if mask is not None:
                return data[mask]
        return data

    def _mask_dataset(self, mask: slice) -> None:
        """
        Apply a mask to this data set.

        Intended for internal use.

        Parameters
        ----------
        mask : :obj:`slice`
            The mask to apply to all data arrays managed by this
            :class:`~swiftgalaxy.reader._SWIFTGroupDatasetHelper`.
        """
        # Users are cautioned against calling this function directly!
        # Use SWIFTGalaxy.mask_particles instead.
        particle_name = self._particle_dataset.group_name
        particle_metadata = getattr(
            self._particle_dataset.metadata, f"{particle_name}_properties"
        )
        for field_name in particle_metadata.field_names:
            if self._is_namedcolumns(field_name):
                for named_column in getattr(self, field_name).named_columns:
                    if (
                        getattr(getattr(self, field_name), f"_{named_column}")
                        is not None
                    ):
                        setattr(
                            getattr(self, field_name),
                            f"_{named_column}",
                            getattr(getattr(self, field_name), f"_{named_column}")[
                                mask
                            ],
                        )
            elif getattr(self, f"_{field_name}") is not None:
                setattr(self, f"_{field_name}", getattr(self, f"_{field_name}")[mask])
        self._mask_derived_coordinates(mask)
        if getattr(self._swiftgalaxy._extra_mask, particle_name) is None:
            setattr(self._swiftgalaxy._extra_mask, particle_name, mask)
        else:
            if self._swiftgalaxy._spatial_mask is None:
                # get a count of particles in the box
                num_part = self._particle_dataset.metadata.num_part[
                    particle_metadata.particle_type
                ]
            else:
                # get a count of particles in the spatial mask region
                num_part = np.sum(
                    self._swiftgalaxy._spatial_mask.get_masked_counts_offsets()[0][
                        particle_name
                    ]
                )
            old_mask = getattr(self._swiftgalaxy._extra_mask, particle_name)
            # need to convert to an integer mask to combine
            # (boolean is insufficient in case of re-ordering masks)
            setattr(
                self._swiftgalaxy._extra_mask,
                particle_name,
                np.arange(num_part, dtype=int)[old_mask][mask],
            )
        return

    def _apply_transforms(self, data: cosmo_array, dataset_name: str) -> cosmo_array:
        """
        Used internally to apply existing coordinate transforms on reading new data.

        Checks whether the input dataset_name is in the list of datasets that need
        to have coordinate transformation (either coordinate-like or velocity-like)
        and applies the transformations as needed.

        Parameters
        ----------
        data : :class:`~swiftsimio.objects.cosmo_array`
            The data to (potentially) transform.
        dataset_name : :obj:`str`
            The name of the dataset contained in ``data``.

        Returns
        -------
        out : :class:`~swiftsimio.objects.cosmo_array`
            The data with any required transformations applied.
        """
        if dataset_name in self._swiftgalaxy.transforms_like_coordinates:
            transform_units = self._swiftgalaxy.metadata.units.length
            transform = self._swiftgalaxy._coordinate_like_transform
        elif dataset_name in self._swiftgalaxy.transforms_like_velocities:
            transform_units = (
                self._swiftgalaxy.metadata.units.length
                / self._swiftgalaxy.metadata.units.time
            )
            transform = self._swiftgalaxy._velocity_like_transform
        else:
            transform = None
        if transform is not None:
            data = _apply_4transform(data, transform, transform_units)
        boxsize = getattr(self._particle_dataset.metadata, "boxsize", None)
        if dataset_name in self._swiftgalaxy.transforms_like_coordinates:
            data = _apply_box_wrap(data, boxsize)
        return data

    @property
    def cartesian_coordinates(self) -> _CoordinateHelper:
        """
        Utility to access the cartesian coordinates of particles.

        Returns a wrapper around the coordinate array which can be accessed using
        attribute syntax. Cartesian coordinates can be accessed separately:

        + ``cartesian_coordinates.x``
        + ``cartesian_coordinates.y``
        + ``cartesian_coordinates.z``

        or as a 2D array:

        + ``cartesian_coordinates.xyz``

        A reference to the coordinates array is obtained each time, so cartesian
        coordinates are automatically updated if the coordinates array is modified (e.g.
        following a rotation or other transformation).

        By default the coorinate array is assumed to be called ``coordinates``,
        but this can be overridden with the ``coordinates_dataset_name``
        argument to :class:`SWIFTGalaxy`.

        Returns
        -------
        coordinate_helper : :class:`_CoordinateHelper`
            Container providing particle cartesian coordinates as attributes.
        """
        return _CoordinateHelper(
            getattr(self, self._swiftgalaxy.coordinates_dataset_name),
            dict(x=np.s_[:, 0], y=np.s_[:, 1], z=np.s_[:, 2], xyz=np.s_[...]),
        )

    @property
    def cartesian_velocities(self) -> _CoordinateHelper:
        """
        Utility to access the cartesian components of particle velocities.

        Returns a wrapper around the velocities array which can be accessed using
        attribute syntax. Cartesian coordinates can be accessed separately:

        + ``cartesian_velocities.x``
        + ``cartesian_velocities.y``
        + ``cartesian_velocities.z``

        or as a 2D array:

        + ``cartesian_velocities.xyz``

        A reference to the velocities array is obtained each time, so cartesian
        velocities are automatically updated if the velocities array is modified (e.g.
        following a rotation or other transformation).

        By default the array of velocities is assumed to be called
        ``velocities``, but this can be overridden with the
        ``velocities_dataset_name`` argument to :class:`SWIFTGalaxy`.

        Returns
        -------
        coordinate_helper : :class:`_CoordinateHelper`
            Container providing particle cartesian velocities as attributes.
        """
        return _CoordinateHelper(
            getattr(self, self._swiftgalaxy.velocities_dataset_name),
            dict(x=np.s_[:, 0], y=np.s_[:, 1], z=np.s_[:, 2], xyz=np.s_[...]),
        )

    @property
    def spherical_coordinates(self) -> _CoordinateHelper:
        """
        Utility to access the spherical coordinates of particles.

        The spherical coordinates of particles are calculated the first time
        this attribute is accessed. If a coordinate transformation (e.g. a
        rotation) or other operation is applied to the :class:`SWIFTGalaxy`
        that would invalidate the derived spherical coordinates, they are
        erased and will be recalculated at the next access of this attribute.
        The coordinates could be transformed when they change instead, but in
        general this requires transforming back through cartesian coordinates,
        so the more efficient "lazy" approach of recalculating on demand is
        used instead.

        The "physics" notation convention, where
        :math:`-\\frac{\\pi}{2} \\leq \\theta \\leq \\frac{\\pi}{2}` is the
        polar angle and :math:`0 < \\phi \\leq 2\\pi` is the azimuthal angle,
        is assumed.

        Several attribute names are supported for each coordinate. They can be
        accessed with the aliases:

        + ``spherical_coordinates.r``:
            + ``spherical_coordinates.radius``
        + ``spherical_coordinates.theta``:
            + ``spherical_coordinates.lat``
            + ``spherical_coordinates.latitude``
            + ``spherical_coordinates.pol``
            + ``spherical_coordinates.polar``
        + ``spherical_coordinates.phi``:
            + ``spherical_coordinates.lon``
            + ``spherical_coordinates.longitude``
            + ``spherical_coordinates.az``
            + ``spherical_coordinates.azimuth``

        By default the coorinate array is assumed to be called ``coordinates``,
        but this can be overridden with the ``coordinates_dataset_name``
        argument to :class:`SWIFTGalaxy`.

        Returns
        -------
        coordinate_helper : :class:`_CoordinateHelper`
            Container providing particle spherical coordinates as attributes.
        """
        if self._spherical_coordinates is None:
            r = np.sqrt(np.sum(np.power(self.cartesian_coordinates.xyz, 2), axis=1))
            theta = cosmo_array(
                np.where(r == 0, 0, np.arcsin(self.cartesian_coordinates.z / r)),
                units=unyt.rad,
                comoving=r.comoving,
                cosmo_factor=cosmo_factor(
                    a**0, scale_factor=r.cosmo_factor.scale_factor
                ),
            )
            if self.cylindrical_coordinates is not None:
                phi = self.cylindrical_coordinates.phi
            else:
                phi = cosmo_array(
                    np.arctan2(
                        self.cartesian_coordinates.y, self.cartesian_coordinates.x
                    ),
                    units=unyt.rad,
                    comoving=r.comoving,
                    cosmo_factor=cosmo_factor(
                        a**0, scale_factor=r.cosmo_factor.scale_factor
                    ),
                )
                phi[phi < 0] = phi[phi < 0] + 2 * np.pi * unyt.rad
            self._spherical_coordinates = dict(_r=r, _theta=theta, _phi=phi)
        return _CoordinateHelper(
            self._spherical_coordinates,
            dict(
                r="_r",
                radius="_r",
                lon="_phi",
                longitude="_phi",
                az="_phi",
                azimuth="_phi",
                phi="_phi",
                lat="_theta",
                latitude="_theta",
                pol="_theta",
                polar="_theta",
                theta="_theta",
            ),
        )

    @property
    def spherical_velocities(self) -> _CoordinateHelper:
        """
        Utility to access the velocities of particles in spherical coordinates.

        The particle velocities in spherical coordinates are calculated the
        first time this attribute is accessed. If a coordinate transformation
        (e.g. a rotation) or other operation is applied to the
        :class:`SWIFTGalaxy` that would invalidate the derived spherical
        velocities, they are erased and will be recalculated at the next access
        of this attribute. The velocities could be transformed when they change
        instead, but in general this requires transforming back through
        cartesian coordinates, so the more efficient "lazy" approach of
        recalculating on demand is used instead.

        The "physics" notation convention, where
        :math:`-\\frac{\\pi}{2} \\leq \\theta \\leq \\frac{\\pi}{2}` is the
        polar angle and :math:`0 < \\phi \\leq 2\\pi` is the azimuthal angle,
        is assumed.

        Several attribute names are supported for each velocity component. They
        can be accessed with the aliases:

        + ``spherical_velocities.r``:
            + ``spherical_velocities.radius``
        + ``spherical_velocities.theta``:
            + ``spherical_velocities.lat``
            + ``spherical_velocities.latitude``
            + ``spherical_velocities.pol``
            + ``spherical_velocities.polar``
        + ``spherical_velocities.phi``:
            + ``spherical_velocities.lon``
            + ``spherical_velocities.longitude``
            + ``spherical_velocities.az``
            + ``spherical_velocities.azimuth``

        By default the array of velocities is assumed to be called
        ``velocities``, but this can be overridden with the
        ``velocities_dataset_name`` argument to :class:`SWIFTGalaxy`.

        Returns
        -------
        coordinate_helper : :class:`_CoordinateHelper`
            Container providing particle velocities in spherical coordinates as
            attributes.
        """
        if self._spherical_coordinates is None:
            self.spherical_coordinates
        if self._spherical_velocities is None:
            _sin_t = np.sin(self.spherical_coordinates.theta)
            _cos_t = np.cos(self.spherical_coordinates.theta)
            _sin_p = np.sin(self.spherical_coordinates.phi)
            _cos_p = np.cos(self.spherical_coordinates.phi)
            v_r = (
                _cos_t * _cos_p * self.cartesian_velocities.x
                + _cos_t * _sin_p * self.cartesian_velocities.y
                + _sin_t * self.cartesian_velocities.z
            )
            v_t = (
                _sin_t * _cos_p * self.cartesian_velocities.x
                + _sin_t * _sin_p * self.cartesian_velocities.y
                - _cos_t * self.cartesian_velocities.z
            )
            v_p = (
                -_sin_p * self.cartesian_velocities.x
                + _cos_p * self.cartesian_velocities.y
            )
            self._spherical_velocities = dict(_v_r=v_r, _v_t=v_t, _v_p=v_p)
        return _CoordinateHelper(
            self._spherical_velocities,
            dict(
                r="_v_r",
                radius="_v_r",
                lon="_v_p",
                longitude="_v_p",
                az="_v_p",
                azimuth="_v_p",
                phi="_v_p",
                lat="_v_t",
                latitude="_v_t",
                pol="_v_t",
                polar="_v_t",
                theta="_v_t",
            ),
        )

    @property
    def cylindrical_coordinates(self) -> _CoordinateHelper:
        """
        Utility to access the cylindrical coordinates of particles.

        The cylindrical coordinates of particles are calculated the first time
        this attribute is accessed. If a coordinate transformation (e.g. a
        rotation) or other operation is applied to the :class:`SWIFTGalaxy`
        that would invalidate the derived cylindrical coordinates, they are
        erased and will be recalculated at the next access of this attribute.
        The coordinates could be transformed when they change instead, but in
        general this requires transforming back through cartesian coordinates,
        so the more efficient "lazy" approach of recalculating on demand is
        used instead.

        The coordinate components are named :math:`(\\rho, \\phi, z)` by
        default, and assume a convention where :math:`0 < \\phi \\leq 2\\pi`.

        Several attribute names are supported for each coordinate. They can be
        accessed with the aliases:

        + ``cylindrical_coordinates.rho``:
            + ``cylindrical_coordinates.R``
            + ``cylindrical_coordinates.radius``
        + ``cylindrical_coordinates.phi``:
            + ``cylindrical_coordinates.lon``
            + ``cylindrical_coordinates.longitude``
            + ``cylindrical_coordinates.az``
            + ``cylindrical_coordinates.azimuth``
        + ``cylindrical_coordinates.z``

        By default the coorinate array is assumed to be called ``coordinates``,
        but this can be overridden with the ``coordinates_dataset_name``
        argument to :class:`SWIFTGalaxy`.

        Returns
        -------
        coordinate_helper : :class:`_CoordinateHelper`
            Container providing particle cylindrical coordinates as attributes.
        """
        if self._cylindrical_coordinates is None:
            rho = np.sqrt(
                np.sum(np.power(self.cartesian_coordinates.xyz[:, :2], 2), axis=1)
            )
            if self._spherical_coordinates is not None:
                phi = self.spherical_coordinates.phi
            else:
                # np.where returns ndarray
                phi = np.arctan2(
                    self.cartesian_coordinates.y, self.cartesian_coordinates.x
                ).view(np.ndarray)
                phi = np.where(phi < 0, phi + 2 * np.pi, phi)
                phi = cosmo_array(
                    phi,
                    units=unyt.rad,
                    comoving=rho.comoving,
                    cosmo_factor=cosmo_factor(
                        a**0, scale_factor=rho.cosmo_factor.scale_factor
                    ),
                )
            z = self.cartesian_coordinates.z
            self._cylindrical_coordinates = dict(_rho=rho, _phi=phi, _z=z)
        return _CoordinateHelper(
            self._cylindrical_coordinates,
            dict(
                R="_rho",
                rho="_rho",
                radius="_rho",
                lon="_phi",
                longitude="_phi",
                az="_phi",
                azimuth="_phi",
                phi="_phi",
                z="_z",
                height="_z",
            ),
        )

    @property
    def cylindrical_velocities(self) -> _CoordinateHelper:
        """
        Utility to access the velocities of particles in cylindrical
        coordinates.

        The particle velocities in cylindrical coordinates are calculated the
        first time this attribute is accessed. If a coordinate transformation
        (e.g. a rotation) or other operation is applied to the
        :class:`SWIFTGalaxy` that would invalidate the derived cylindrical
        velocities, they are erased and will be recalculated at the next access
        of this attribute. The velocities could be transformed when they change
        instead, but in general this requires transforming back through
        cartesian coordinates, so the more efficient "lazy" approach of
        recalculating on demand is used instead.

        The "physics" notation convention, where
        :math:`-\\frac{\\pi}{2} \\leq \\theta \\leq \\frac{\\pi}{2}` is the
        polar angle and :math:`0 < \\phi \\leq 2\\pi` is the azimuthal angle,
        is assumed.

        The coordinate components are named :math:`(\\rho, \\phi, z)` by
        default, and assume a convention where :math:`0 < \\phi \\leq 2\\pi`.

        Several attribute names are supported for each velocity component. They
        can be accessed with the aliases:

        + ``cylindrical_velocities.rho``:
            + ``cylindrical_velocities.R``
            + ``cylindrical_velocities.radius``
        + ``cylindrical_coordinates.phi``:
            + ``cylindrical_velocities.lon``
            + ``cylindrical_velocities.longitude``
            + ``cylindrical_velocities.az``
            + ``cylindrical_velocities.azimuth``
        + ``cylindrical_velocities.z``

        By default the array of velocities is assumed to be called
        ``velocities``, but this can be overridden with the
        ``velocities_dataset_name`` argument to :class:`SWIFTGalaxy`.

        Returns
        -------
        coordinate_helper : :class:`_CoordinateHelper`
            Container providing particle velocities in cylindrical coordinates
            as attributes.
        """
        if self._cylindrical_coordinates is None:
            self.cylindrical_coordinates
        if self._cylindrical_velocities is None:
            _sin_p = np.sin(self.cylindrical_coordinates.phi)
            _cos_p = np.cos(self.cylindrical_coordinates.phi)
            v_rho = (
                _cos_p * self.cartesian_velocities.x
                + _sin_p * self.cartesian_velocities.y
            )
            if self._spherical_velocities is not None:
                v_phi = self.spherical_velocities.phi
            else:
                v_phi = (
                    -_sin_p * self.cartesian_velocities.x
                    + _cos_p * self.cartesian_velocities.y
                )
            v_z = self.cartesian_velocities.z
            self._cylindrical_velocities = dict(_v_rho=v_rho, _v_phi=v_phi, _v_z=v_z)
        return _CoordinateHelper(
            self._cylindrical_velocities,
            dict(
                R="_v_rho",
                rho="_v_rho",
                radius="_v_rho",
                lon="_v_phi",
                longitude="_v_phi",
                az="_v_phi",
                azimuth="_v_phi",
                phi="_v_phi",
                z="_v_z",
                height="_v_z",
            ),
        )

    def _mask_derived_coordinates(self, mask: slice) -> None:
        """
        Apply a mask to the internally maintained derived coordinates.

        Intended for internal use. If the user applies a mask we don't need to
        re-evaluate the derived coordinates but just apply the mask.

        Parameters
        ----------
        mask : :obj:`slice`
            Mask to apply to the derived coordinates maintained by this
            :class:`~swiftgalaxy.reader._SWIFTGroupDatasetHelper`.
        """
        if self._spherical_coordinates is not None:
            for coord in ("r", "theta", "phi"):
                self._spherical_coordinates[f"_{coord}"] = self._spherical_coordinates[
                    f"_{coord}"
                ][mask]
        if self._spherical_velocities is not None:
            for coord in ("v_r", "v_t", "v_p"):
                self._spherical_velocities[f"_{coord}"] = self._spherical_velocities[
                    f"_{coord}"
                ][mask]
        if self._cylindrical_coordinates is not None:
            for coord in ("rho", "phi", "z"):
                self._cylindrical_coordinates[f"_{coord}"] = (
                    self._cylindrical_coordinates[f"_{coord}"][mask]
                )
        if self._cylindrical_velocities is not None:
            for coord in ("v_rho", "v_phi", "v_z"):
                self._cylindrical_velocities[f"_{coord}"] = (
                    self._cylindrical_velocities[f"_{coord}"][mask]
                )
        return

    def _void_derived_coordinates(self) -> None:
        """
        Reset internal references to spherical/cylindrical coordinates to None (e.g.
        because they are no longer valid).
        """
        self._spherical_coordinates = None
        self._cylindrical_coordinates = None
        self._spherical_velocities = None
        self._cylindrical_velocities = None
        return


class SWIFTGalaxy(SWIFTDataset):
    """
    A representation of a simulated galaxy.

    A :class:`SWIFTGalaxy` represents a galaxy from a
    simulation, including both its particles and integrated properties. A halo
    finder catalogue is required to define which particles belong to the galaxy
    and to provide integrated properties. The implementation is an extension of
    the :class:`~swiftsimio.reader.SWIFTDataset` class, so all the
    functionality of such a dataset is also available for a
    :class:`SWIFTGalaxy`. The :class:`swiftsimio.reader.__SWIFTGroupDatasets`
    objects familiar to :mod:`swiftsimio` users (e.g. a ``GasDataset``) are
    wrapped by a :class:`~swiftgalaxy.reader._SWIFTGroupDatasetHelper` class that exposes
    their usual functionality and extends it with new features.
    :class:`swiftsimio.reader.__SWIFTNamedColumnDataset` instances are also
    wrapped, using a :class:`~swiftgalaxy.reader._SWIFTNamedColumnDatasetHelper` class.

    For an overview of available features see the examples below, and the
    narrative documentation pages.

    Parameters
    ----------
    snapshot_filename : :obj:`str`
        Name of file containing snapshot.

    halo_catalogue : :class:`~swiftgalaxy.halo_catalogues._HaloCatalogue` (optional), \
    default: ``None``
        A halo catalogue instance from :mod:`swiftgalaxy.halo_catalogues`, e.g. a
        :class:`swiftgalaxy.halo_catalogues.SOAP` instance.

    auto_recentre : :obj:`bool` (optional), default: ``True``
        If ``True``, the coordinate system will be automatically recentred on
        the position and velocity centres defined by the ``halo_catalogue``.

    transforms_like_coordinates : :obj:`set` (optional), default: ``set()``
        Names of fields that behave as spatial coordinates. It is assumed that
        these exist for all present particle types. When the coordinate system
        is rotated or translated, the associated arrays will be transformed
        accordingly. The ``coordinates`` dataset (or its alternative name given
        in the ``coordinates_dataset_name`` parameter) is implicitly assumed to
        behave as spatial coordinates.

    transforms_like_velocities : :obj:`set` (optional), default: ``set()``
        Names of fields that behave as velocities. It is assumed that these
        exist for all present particle types. When the coordinate system is
        rotated or boosted, the associated arrays will be transformed
        accordingly. The ``velocities`` dataset (or its alternative name given
        in the ``velocities_dataset_name`` parameter) is implicitly assumed to
        behave as velocities.

    id_particle_dataset_name : :obj:`str` (optional), default: ``"particle_ids"``
        Name of the dataset containing the particle IDs, assumed to be the same
        for all present particle types.

    coordinates_dataset_name : :obj:`str` (optional), default: ``"velocities"``
        Name of the dataset containing the particle spatial coordinates,
        assumed to be the same for all present particle types.

    velocities_dataset_name : :obj:`str` (optional), default: ``"velocities"``
        Name of the dataset containing the particle velocities, assumed to be
        the same for all present particle types.

    coordinate_frame_from : :class:`~swiftgalaxy.reader.SWIFTGalaxy` (optional), \
    default: ``None``
        Another :class:`~swiftgalaxy.reader.SWIFTGalaxy` to copy the coordinate frame
        (centre and rotation) and velocity coordinate frame (boost and rotation) from.

    See Also
    --------
    :class:`_SWIFTGroupDatasetHelper`
    :class:`_SWIFTNamedColumnDatasetHelper`
    :mod:`swiftgalaxy.halo_catalogues`

    Examples
    --------

    Assuming we have a snapshot file :file:`{snap}.hdf5`, and velociraptor
    outputs :file:`{halos}.properties`, :file:`{halos}.catalog_groups`, etc.,
    with the default names for coordinates, velocities and particle_ids, we can
    initialise a :class:`SWIFTGalaxy` for the first row (indexed from 0) in the
    halo catalogue very easily:

    ::

        from swiftgalaxy import SWIFTGalaxy, Velociraptor
        mygalaxy = SWIFTGalaxy(
            'snap.hdf5',
            Velociraptor(
                'halos',
                halo_index=0
            )
        )

    Like a :class:`~swiftsimio.reader.SWIFTDataset`, the particle datasets are
    accessed as below, and all data are loaded 'lazily', on demand.

    ::

        mygalaxy.gas.particle_ids
        mygalaxy.dark_matter.coordinates

    However, information from the halo catalogue is used to select only the
    particles identified as bound to this galaxy. The coordinate system is
    centred in both position and velocity on the centre and peculiar velocity
    of the galaxy, as determined by the halo catalogue. The coordinate system can
    be further manipulated, and all particle arrays will stay in a consistent
    reference frame at all times.

    Again like for a :class:`~swiftsimio.reader.SWIFTDataset`, the units and
    metadata are available:

    ::

        mygalaxy.units
        mygalaxy.metadata

    The halo catalogue interface is accessible as shown below. What this interface
    looks like depends on the halo catalogue being used, but will provide values
    for the individual galaxy of interest.

    ::

        mygalaxy.halo_catalogue

    In this case with :class:`~swiftgalaxy.halo_catalogues.Velociraptor`, we can
    get the virial mass like this:

    ::

        mygalaxy.halo_catalogue.masses.mvir

    For a complete description of available features see the narrative
    documentation pages.
    """

    snapshot_filename: str
    halo_catalogue: Optional[_HaloCatalogue]
    transforms_like_coordinates: Set[str]
    transforms_like_velocities: Set[str]
    id_particle_dataset_name: str
    coordinates_dataset_name: str
    velocities_dataset_name: str
    _initialized: bool
    _spatial_mask: SWIFTMask
    _extra_mask: Optional[MaskCollection]
    _warn_on_read: bool

    def __init__(
        self,
        snapshot_filename: str,
        halo_catalogue: Optional[_HaloCatalogue],
        auto_recentre: bool = True,
        transforms_like_coordinates: Set[str] = set(),
        transforms_like_velocities: Set[str] = set(),
        id_particle_dataset_name: str = "particle_ids",
        coordinates_dataset_name: str = "coordinates",
        velocities_dataset_name: str = "velocities",
        coordinate_frame_from: Optional["SWIFTGalaxy"] = None,
    ):
        self._particle_dataset_helpers = dict()
        self.snapshot_filename = snapshot_filename
        self.halo_catalogue = halo_catalogue
        self.transforms_like_coordinates = {coordinates_dataset_name}.union(
            transforms_like_coordinates
        )
        self.transforms_like_velocities = {velocities_dataset_name}.union(
            transforms_like_velocities
        )
        self.id_particle_dataset_name = id_particle_dataset_name
        self.coordinates_dataset_name = coordinates_dataset_name
        self.velocities_dataset_name = velocities_dataset_name
        if not hasattr(self, "_coordinate_like_transform"):
            self._coordinate_like_transform = np.eye(4)
        if not hasattr(self, "_velocity_like_transform"):
            self._velocity_like_transform = np.eye(4)
        if self.halo_catalogue is None:
            # in server mode we don't have a halo_catalogue yet
            pass
        elif self.halo_catalogue._user_spatial_offsets is not None:
            self._spatial_mask = self.halo_catalogue._get_user_spatial_mask(
                self.snapshot_filename
            )
        else:
            self._spatial_mask = self.halo_catalogue._get_spatial_mask(
                self.snapshot_filename
            )
        super().__init__(snapshot_filename, mask=self._spatial_mask)
        if auto_recentre is True and coordinate_frame_from is not None:
            raise ValueError(
                "Cannot use coordinate_frame_from with auto_recentre=True."
            )
        elif coordinate_frame_from is not None:
            if (
                coordinate_frame_from.metadata.units.length
                != self.metadata.units.length
            ) or (
                coordinate_frame_from.metadata.units.time != self.metadata.units.time
            ):
                raise ValueError(
                    "Internal units (length and time) of coordinate_frame_from don't"
                    " match."
                )
            self._coordinate_like_transform = (
                coordinate_frame_from._coordinate_like_transform
            )
            self._velocity_like_transform = (
                coordinate_frame_from._velocity_like_transform
            )
        for particle_name in self.metadata.present_group_names:
            # We'll make a custom type to present a nice name to the user.
            nice_name = swiftsimio_metadata.particle_types.particle_name_class[
                getattr(self.metadata, f"{particle_name}_properties").group
            ]
            TypeDatasetHelper = type(
                f"{nice_name}DatasetHelper",
                (_SWIFTGroupDatasetHelper, object),
                dict(),
            )
            self._particle_dataset_helpers[particle_name] = TypeDatasetHelper(
                super().__getattribute__(particle_name), self
            )

        if not hasattr(self, "_extra_mask"):
            self._extra_mask = None
        if (
            self.halo_catalogue is not None
        ):  # in server mode we don't have a halo_catalogue yet
            self._extra_mask = self.halo_catalogue._get_extra_mask(self)
        if self._extra_mask is not None:
            # need to mask any already loaded data
            for particle_name in self.metadata.present_group_names:
                if getattr(self._extra_mask, particle_name) is None:
                    continue
                particle_metadata = getattr(
                    self.metadata, f"{particle_name}_properties"
                )
                for field_name in particle_metadata.field_names:
                    if getattr(self, particle_name)._is_namedcolumns(field_name):
                        named_columns_dataset = getattr(
                            getattr(self, particle_name), f"{field_name}"
                        )._named_column_dataset
                        for column in named_columns_dataset.named_columns:
                            data = getattr(named_columns_dataset, f"_{column}")
                            if data is None:
                                continue
                            setattr(
                                named_columns_dataset,
                                f"_{column}",
                                data[getattr(self._extra_mask, particle_name)],
                            )
                    else:
                        data = getattr(getattr(self, particle_name), f"_{field_name}")
                        if data is None:
                            continue
                        setattr(
                            # bypass helper:
                            super().__getattribute__(particle_name),
                            f"_{field_name}",
                            data[getattr(self._extra_mask, particle_name)],
                        )
        else:
            self._extra_mask = MaskCollection(
                **{k: None for k in self.metadata.present_group_names}
            )

        if auto_recentre and self.halo_catalogue is not None:
            self.recentre(self.halo_catalogue.centre)
            self.recentre_velocity(self.halo_catalogue.velocity_centre)

        self._warn_on_read = False
        self._initialised = True

        return

    @classmethod
    def _copyinit(
        cls,
        snapshot_filename: str,
        halo_catalogue: Optional[_HaloCatalogue],
        auto_recentre: bool = True,
        transforms_like_coordinates: Set[str] = set(),
        transforms_like_velocities: Set[str] = set(),
        id_particle_dataset_name: str = "particle_ids",
        coordinates_dataset_name: str = "coordinates",
        velocities_dataset_name: str = "velocities",
        coordinate_frame_from: Optional["SWIFTGalaxy"] = None,
        _spatial_mask: Optional[SWIFTMask] = None,
        _extra_mask: Optional[MaskCollection] = None,
        _coordinate_like_transform: Optional[np.ndarray] = None,
        _velocity_like_transform: Optional[np.ndarray] = None,
        _warn_on_read: bool = False,
    ):
        """
        For internal use in copying a :class:`SWIFTGalaxy`.

        An init method with some extra parameters to facilitate copying.

        Parameters
        ----------
        snapshot_filename : :obj:`str`
            Name of file containing snapshot.

        halo_catalogue : :class:`~swiftgalaxy.halo_catalogues._HaloCatalogue` \
        (optional), default: ``None``
            A halo_catalogue instance from :mod:`swiftgalaxy.halo_catalogues`, e.g. a
            :class:`swiftgalaxy.halo_catalogues.SOAP` instance.

        auto_recentre : :obj:`bool`, default: ``True``
            If ``True``, the coordinate system will be automatically recentred on
            the position *and* velocity centres defined by the ``halo_catalogue``.

        transforms_like_coordinates : :obj:`set` containing :obj:`str`s, \
        default: ``set()``
            Names of fields that behave as spatial coordinates. It is assumed that
            these exist for all present particle types. When the coordinate system
            is rotated or translated, the associated arrays will be transformed
            accordingly. The ``coordinates`` dataset (or its alternative name given
            in the ``coordinates_dataset_name`` parameter) is implicitly assumed to
            behave as spatial coordinates.

        transforms_like_velocities : :obj:`set` containing :obj:`str`s, \
        default: ``set()``
            Names of fields that behave as velocities. It is assumed that these
            exist for all present particle types. When the coordinate system is
            rotated or boosted, the associated arrays will be transformed
            accordingly. The ``velocities`` dataset (or its alternative name given
            in the ``velocities_dataset_name`` parameter) is implicitly assumed to
            behave as velocities.

        id_particle_dataset_name : :obj:`str`, default: ``'particle_ids'``
            Name of the dataset containing the particle IDs, assumed to be the same
            for all present particle types.

        coordinates_dataset_name : :obj:`str`, default: ``'velocities'``
            Name of the dataset containing the particle spatial coordinates,
            assumed to be the same for all present particle types.

        velocities_dataset_name : :obj:`str`, default: ``'velocities'``
            Name of the dataset containing the particle velocities, assumed to be
            the same for all present particle types.

        coordinate_frame_from : :class:`~swiftgalaxy.reader.SWIFTGalaxy` (optional), \
        default: ``None``
            Another :class:`~swiftgalaxy.reader.SWIFTGalaxy` to copy the coordinate frame
            (centre and rotation) and velocity coordinate frame (boost and rotation) from.

        _spatial_mask : :class:`~swiftsimio.masks.SWIFTMask` (optional), default: ``None``
            Directly set the spatial mask (intended for internal use only).

        _extra_mask : :class:`~swiftgalaxy.masks.MaskCollection` (optional), \
        default: ``None``
            Directly set the extra mask (intended for internal use only).

        _coordinate_like_transform : :class:`~numpy.ndarray` (optional), default: ``None``
            Directly set the internal representation of the coordinate frame translations
            and rotations (intended for internal use only).

        _velocity_like_transform : :class:`~numpy.ndarray` (optional), default: ``None``
            Directly set the internal representation of the velocity frame boosts and
            rotations (intended for internal use only).

        _warn_on_read : :obj:`bool` (optional), default: ``False``
            If ``True``, warn the user when data is read from disk (e.g. should have been
            included in ``preload`` when using ``SWIFTGalaxies``.
        """
        sg = cls.__new__(cls)
        sg._spatial_mask = _spatial_mask
        sg._extra_mask = _extra_mask
        if _coordinate_like_transform is not None:
            sg._coordinate_like_transform = _coordinate_like_transform
        if _velocity_like_transform is not None:
            sg._velocity_like_transform = _velocity_like_transform
        sg._warn_on_read = _warn_on_read
        SWIFTGalaxy.__init__(
            sg,
            snapshot_filename,
            halo_catalogue,
            auto_recentre=auto_recentre,
            transforms_like_coordinates=transforms_like_coordinates,
            transforms_like_velocities=transforms_like_velocities,
            id_particle_dataset_name=id_particle_dataset_name,
            coordinates_dataset_name=coordinates_dataset_name,
            velocities_dataset_name=velocities_dataset_name,
            coordinate_frame_from=coordinate_frame_from,
        )
        sg._warn_on_read = _warn_on_read  # was set False in __init__ so do this after
        return sg

    def __str__(self) -> str:
        """
        Get a string representation of the object (noting location of the snapshot file).

        Returns
        -------
        out : :obj:`str`
            The string representation.
        """
        return f"SWIFTGalaxy at {self.snapshot_filename}."

    def __repr__(self) -> str:
        """
        Get a string representation of the object (noting location of the snapshot file).

        Returns
        -------
        out : :obj:`str`
            The string representation.
        """
        return self.__str__()

    def __getitem__(self, mask_collection: MaskCollection) -> "SWIFTGalaxy":
        """
        Apply a mask to the :class:`~swiftgalaxy.reader.SWIFTGalaxy` with square-bracket
        notation.

        To ensure internal consistency this requires producing a full ("deep") copy of
        the :class:`~swiftgalaxy.reader.SWIFTGalaxy` object and all of its contents (but
        data is masked at copy time to avoid unnecessary memory overhead).

        Parameters
        ----------
        mask_collection : :class:`swiftgalaxy.masks.MaskCollection`
            The mask to apply to the :class:`~swiftgalaxy.reader.SWIFTGalaxy`.
        """
        return self._data_copy(mask_collection=mask_collection)

    def __copy__(self) -> "SWIFTGalaxy":
        """
        Create a copy of the :class:`~swiftgalaxy.reader.SWIFTGalaxy` without copying
        data (a "shallow" copy).

        Returns
        -------
        out : :class:`~swiftgalaxy.reader.SWIFTGalaxy`
            The copy of the :class:`~swiftgalaxy.reader.SWIFTGalaxy` object.
        """
        sg = self._copyinit(
            self.snapshot_filename,
            self.halo_catalogue,
            auto_recentre=False,  # transforms overwritten below
            transforms_like_coordinates=self.transforms_like_coordinates,
            transforms_like_velocities=self.transforms_like_velocities,
            id_particle_dataset_name=self.id_particle_dataset_name,
            coordinates_dataset_name=self.coordinates_dataset_name,
            velocities_dataset_name=self.velocities_dataset_name,
            _spatial_mask=self._spatial_mask,
            _extra_mask=self._extra_mask,
            _coordinate_like_transform=self._coordinate_like_transform,
            _velocity_like_transform=self._velocity_like_transform,
            _warn_on_read=self._warn_on_read,
        )
        return sg

    def __deepcopy__(self, memo: Optional[dict] = None) -> "SWIFTGalaxy":
        """
        Create a copy of the :class:`~swiftgalaxy.reader.SWIFTGalaxy` including copying
        data (a "deep" copy).

        Parameters
        ----------
        memo : :obj:`dict` (optional), default: ``None``
            For the copy operation to keep a record of already copied objects.

        Returns
        -------
        out : :class:`~swiftgalaxy.reader.SWIFTGalaxy`
            The copy of the :class:`~swiftgalaxy.reader.SWIFTGalaxy` object.
        """
        return self._data_copy()

    def _data_copy(
        self, mask_collection: Optional[MaskCollection] = None
    ) -> "SWIFTGalaxy":
        """
        Create a copy of the :class:`~swiftgalaxy.reader.SWIFTGalaxy` including copying
        data (a "deep" copy).

        Parameters
        ----------
        mask_collection : :class:`~swiftgalaxy.masks.MaskCollection` (optional), \
        default: ``None``
            Copy only the subset of the data corresponding to the ``mask_collection``.

        Returns
        -------
        out : :class:`~swiftgalaxy.reader.SWIFTGalaxy`
            A (possibly masked) copy of the :class:`~swiftgalaxy.reader.SWIFTGalaxy`
            object.
        """
        sg = self._copyinit(
            deepcopy(self.snapshot_filename),
            deepcopy(self.halo_catalogue),
            auto_recentre=False,  # transforms overwritten below
            transforms_like_coordinates=deepcopy(self.transforms_like_coordinates),
            transforms_like_velocities=deepcopy(self.transforms_like_velocities),
            id_particle_dataset_name=deepcopy(self.id_particle_dataset_name),
            coordinates_dataset_name=deepcopy(self.coordinates_dataset_name),
            velocities_dataset_name=deepcopy(self.velocities_dataset_name),
            _spatial_mask=self._spatial_mask,
            _extra_mask=deepcopy(self._extra_mask),
            _coordinate_like_transform=deepcopy(self._coordinate_like_transform),
            _velocity_like_transform=deepcopy(self._velocity_like_transform),
            _warn_on_read=deepcopy(self._warn_on_read),
        )
        for particle_name in sg.metadata.present_group_names:
            particle_metadata = getattr(sg.metadata, f"{particle_name}_properties")
            particle_dataset_helper = getattr(self, particle_name)
            new_particle_dataset_helper = getattr(sg, particle_name)
            if mask_collection is not None:
                mask = getattr(mask_collection, particle_name)
                if mask is None:
                    mask = Ellipsis
            else:
                mask = Ellipsis
            getattr(sg, particle_name)._mask_dataset(mask)
            for field_name in particle_metadata.field_names:
                if particle_dataset_helper._is_namedcolumns(field_name):
                    named_columns_helper = getattr(particle_dataset_helper, field_name)
                    new_named_columns_helper = getattr(
                        new_particle_dataset_helper, field_name
                    )
                    for named_column in named_columns_helper.named_columns:
                        data = getattr(named_columns_helper, f"_{named_column}")
                        if data is not None:
                            setattr(
                                new_named_columns_helper, f"_{named_column}", data[mask]
                            )
                else:
                    data = getattr(particle_dataset_helper, f"_{field_name}")
                    if data is not None:
                        setattr(
                            new_particle_dataset_helper, f"_{field_name}", data[mask]
                        )
            # cartesian_coordinates return a reference to coordinates on-the-fly:
            # no need to initialise here.
            if particle_dataset_helper._spherical_coordinates is not None:
                new_particle_dataset_helper._spherical_coordinates = dict()
                for c in ("_r", "_theta", "_phi"):
                    new_particle_dataset_helper._spherical_coordinates[c] = (
                        particle_dataset_helper._spherical_coordinates[c][mask]
                    )
            if particle_dataset_helper._spherical_velocities is not None:
                new_particle_dataset_helper._spherical_velocities = dict()
                for c in ("_v_r", "_v_t", "_v_p"):
                    new_particle_dataset_helper._spherical_velocities[c] = (
                        particle_dataset_helper._spherical_velocities[c][mask]
                    )
            if particle_dataset_helper._cylindrical_coordinates is not None:
                new_particle_dataset_helper._cylindrical_coordinates = dict()
                for c in ("_rho", "_phi", "_z"):
                    new_particle_dataset_helper._cylindrical_coordinates[c] = (
                        particle_dataset_helper._cylindrical_coordinates[c][mask]
                    )
            if particle_dataset_helper._cylindrical_velocities is not None:
                new_particle_dataset_helper._cylindrical_velocities = dict()
                for c in ("_v_rho", "_v_phi", "_v_z"):
                    new_particle_dataset_helper._cylindrical_velocities[c] = (
                        particle_dataset_helper._cylindrical_velocities[c][mask]
                    )
        return sg

    def __getattribute__(self, attr: str) -> Any:
        """
        Get an attribute of the object.

        Redirects requests for datasets to the corresponding :mod:`swiftgalaxy` helper
        classes.

        Parameters
        ----------
        attr : :obj:`str`
            The name of the attribute.

        Returns
        -------
        out : :obj:`object`
            The value of the attribute.
        """
        # __getattr__ is only checked if the attribute is not found
        # __getattribute__ is checked promptly
        # Note always use super().__getattribute__(...)
        # or object.__getattribute__(self, ...) as appropriate
        # to avoid infinite recursion.
        try:
            metadata = super().__getattribute__("metadata")
        except AttributeError:
            # guard against accessing metadata before it is loaded
            return super().__getattribute__(attr)
        else:
            if attr in metadata.present_group_names:
                # We are entering a <ParticleType>Dataset, return helper.
                return object.__getattribute__(self, "_particle_dataset_helpers")[attr]
            else:
                return super().__getattribute__(attr)

    def __setattr__(self, attr: str, value: Any) -> None:
        """
        Set an attribute of the object.

        Redirects setting attributes to the helper objects when relevant.

        Parameters
        ----------
        attr : :obj:`str`
            Name of the attribute.
        value : :obj:`object`
            The value to assign to the attribute.
        """
        if (not hasattr(self, "_initialised")) or (
            attr not in self.metadata.present_group_names
        ):
            object.__setattr__(self, attr, value)
        else:
            # attr in self.metadata.present_group_names
            self._particle_dataset_helpers[attr] = value
        return

    def rotate(self, rotation: Rotation) -> None:
        """
        Apply a rotation to the particle spatial coordinates.

        The provided rotation is applied to all particle coordinates. All
        datasets specified in the ``transforms_like_coordinates`` and
        ``transforms_like_velocities`` arguments to
        :class:`SWIFTGalaxy` are transformed (by default
        ``coordinates`` and ``velocities`` for all present particle types).

        Parameters
        ----------
        rotation : :class:`scipy.spatial.transform.Rotation`
            The rotation to be applied.
            :class:`~scipy.spatial.transform.Rotation` supports several input
            formats, including axis-angle, rotation matrices, and others.
        """
        rotation_matrix = rotation.as_matrix()
        rotatable = self.transforms_like_coordinates | self.transforms_like_velocities
        for particle_name in self.metadata.present_group_names:
            dataset = getattr(self, particle_name)._particle_dataset
            for field_name in rotatable:
                field_data = getattr(dataset, f"_{field_name}")
                if field_data is not None:
                    field_data = _apply_rotmat(field_data, rotation_matrix)
                    setattr(dataset, f"_{field_name}", field_data)
        rotmat4 = np.eye(4)
        rotmat4[:3, :3] = rotation_matrix
        self._append_to_coordinate_like_transform(rotmat4)
        self._append_to_velocity_like_transform(rotmat4)
        self.wrap_box()
        return

    def _transform(self, transform4: cosmo_array, boost: bool = False) -> None:
        """
        Apply a 4x4 transformation matrix to either the spatial or velocity coordinates.

        For internal use, users should use :meth:`translate`, :meth:`boost` or
        :meth:`rotate` methods as approprirate instead.

        Parameters
        ----------
        transform4 : :class:`~numpy.ndarray`
            The transformation to be applied.
        boost : :obj:`bool`
            If ``True``, translate the velocity coordinates, else translate the spatial
            coordinates.
        """
        # assumes that the input transformation has compatible implicit units, so not
        # intended for users
        transformable = (
            self.transforms_like_velocities
            if boost
            else self.transforms_like_coordinates
        )
        for particle_name in self.metadata.present_group_names:
            dataset = getattr(self, particle_name)._particle_dataset
            for field_name in transformable:
                field_data = getattr(dataset, f"_{field_name}")
                if field_data is not None:
                    field_data = _apply_4transform(
                        field_data, transform4.to_value(), transform4.units
                    )
                    setattr(dataset, f"_{field_name}", field_data)
        if boost:
            self._append_to_velocity_like_transform(transform4)
        else:
            self._append_to_coordinate_like_transform(transform4)
        if not boost:
            self.wrap_box()

    def _translate(self, translation: cosmo_array, boost: bool = False) -> None:
        """
        Apply a translation to either the spatial or velocity coordinates.

        For internal use, users should use :meth:`translate` or :meth:`boost` as
        approprirate instead.

        Parameters
        ----------
        translation : :class:`~swiftsimio.objects.cosmo_array`
            The translation to be applied.
        boost : :obj:`bool`
            If ``True``, translate the velocity coordinates, else translate the spatial
            coordinates.
        """
        translatable = (
            self.transforms_like_velocities
            if boost
            else self.transforms_like_coordinates
        )
        for particle_name in self.metadata.present_group_names:
            dataset = getattr(self, particle_name)._particle_dataset
            for field_name in translatable:
                field_data = getattr(dataset, f"_{field_name}")
                if field_data is not None:
                    field_data = _apply_translation(field_data, translation)
                    setattr(dataset, f"_{field_name}", field_data)
        if boost:
            transform_units = self.metadata.units.length / self.metadata.units.time
        else:
            transform_units = self.metadata.units.length
        transform4 = np.eye(4)
        if hasattr(translation, "comoving"):
            transform4[3, :3] = translation.to_comoving().to_value(transform_units)
        else:
            transform4[3, :3] = translation.to_value(transform_units)
            warn(
                "Translation assumed to be in comoving (not physical) coordinates.",
                category=UserWarning,
            )
        if boost:
            self._append_to_velocity_like_transform(transform4)
        else:
            self._append_to_coordinate_like_transform(transform4)
        if not boost:
            self.wrap_box()
        return

    @property
    def centre(self) -> cosmo_array:
        """
        The current origin of the coordinate reference frame with respect to the native
        simulation coordinate reference frame.

        Returns
        -------
        out : :class:`~swiftsimio.objects.cosmo_array`
            The origin of the coordinate reference frame.
        """
        transform_units = self.metadata.units.length
        transform = np.linalg.inv(self._coordinate_like_transform)
        return _apply_4transform(
            cosmo_array(
                np.zeros((1, 3)),
                units=transform_units,
                comoving=True,
                cosmo_factor=cosmo_factor(
                    a**1, scale_factor=self.metadata.scale_factor
                ),
            ),
            transform,
            transform_units,
        ).squeeze()

    @property
    def velocity_centre(self) -> cosmo_array:
        """
        The current origin of the velocity reference frame with respect to the native
        simulation velocity reference frame.

        Returns
        -------
        out : :class:`~swiftsimio.objects.cosmo_array`
            The origin of the velocity reference frame.
        """
        transform_units = self.metadata.units.length / self.metadata.units.time
        transform = np.linalg.inv(self._velocity_like_transform)
        return _apply_4transform(
            cosmo_array(
                np.zeros((1, 3)),
                units=transform_units,
                comoving=True,
                cosmo_factor=cosmo_factor(
                    a**0, scale_factor=self.metadata.scale_factor
                ),
            ),
            transform,
            transform_units,
        ).squeeze()

    @property
    def rotation(self) -> Rotation:
        """
        The current rotation of the coordinate frame.

        Returns
        -------
        out : :class:`scipy.spatial.transform.Rotation`
            The current rotation.
        """
        return Rotation.from_matrix(self._coordinate_like_transform[:3, :3])

    def translate(self, translation: cosmo_array) -> None:
        """
        Apply a translation to the particle spatial coordinates.

        The provided translation vector is added to all particle coordinates.
        If the new centre position that should be set to zero is known, use
        :meth:`recentre` instead. All datasets
        specified in the ``transforms_like_coordinates`` argument to
        :class:`SWIFTGalaxy` are transformed (by default
        ``coordinates`` for all present particle types).

        Parameters
        ----------
        translation : :class:`~swiftsimio.objects.cosmo_array`
            The vector to translate by.

        See Also
        --------
        :meth:`recentre`
        """
        self._translate(translation)
        return

    def boost(self, boost: cosmo_array) -> None:
        """
        Apply a 'boost' to the velocity coordinates.

        The provided velocity vector is added to all particle velocities. If
        the 'reference velocity' that should be set to zero is known, use
        :meth:`recentre_velocity` instead. All
        datasets specified in the ``transforms_like_velocities`` argument to
        :class:`SWIFTGalaxy` are transformed (by default
        ``velocities`` for all present particle types).

        Parameters
        ----------
        boost : :class:`~swiftsimio.objects.cosmo_array`
            The velocity to boost by.

        See Also
        --------
        :meth:`recentre_velocity`
        """
        self._translate(boost, boost=True)
        return

    def recentre(self, new_centre: cosmo_array) -> None:
        """
        Set a new centre for the particle spatial coordinates.

        The provided (spatial) coordinate centre is set to zero by subtracting
        it from the particle coordinates. Note that this is the new centre in
        the current coordinate system (not e.g. the simulation box
        coordinates). If the coordinate offset to be applied is known, use
        :meth:`translate` instead. All datasets
        specified in the ``transforms_like_coordinates`` argument to
        :class:`SWIFTGalaxy` are transformed (by default
        ``coordinates`` for all present particle types).

        Parameters
        ----------
        new_centre : :class:`~swiftsimio.objects.cosmo_array`
            The new centre for the (spatial) coordinate system.

        See Also
        --------
        :meth:`translate`
        """
        self._translate(-new_centre)
        return

    def recentre_velocity(self, new_centre: cosmo_array) -> None:
        """
        Recentre the velocity coordinates.

        The provided velocity coordinate is set to zero by subtracting it from
        the particle velocities. Note that this is the new velocity centre in
        the current coordinate system (not e.g. the simulation box
        coordinates). If the velocity offset to be applied is known, use
        :meth:`boost` instead. All
        datasets specified in the ``transforms_like_velocities`` argument to
        :class:`SWIFTGalaxy` are transformed (by default
        ``velocities`` for all present particle types).

        Parameters
        ----------
        new_centre : :class:`~swiftsimio.objects.cosmo_array`
            The new centre for the velocity coordinate system.

        See Also
        --------
        :meth:`boost`
        """
        self._translate(-new_centre, boost=True)

    def wrap_box(self) -> None:
        """
        Wrap the particle coordinates in a periodic box.

        Recentres a particle coordinates from a periodic simulation volume such
        that the coordinate (0, 0, 0) is in the centre and the axes of the
        volume are aligned with the coordinate axes.

        .. note::
            This is invoked automatically after any coordinate translations or
            rotations, so manually calling this function should usually not be
            necessary.
        """
        for particle_name in self.metadata.present_group_names:
            dataset = getattr(self, particle_name)._particle_dataset
            for field_name in self.transforms_like_coordinates:
                field_data = getattr(dataset, f"_{field_name}")
                if field_data is not None:
                    field_data = _apply_box_wrap(field_data, self.metadata.boxsize)
                    setattr(dataset, f"_{field_name}", field_data)
        return

    def mask_particles(self, mask_collection: MaskCollection) -> None:
        """
        Select a subset of the currently selected particles.

        The masks to be applied can by in any format accepted by a
        :class:`~swiftsimio.objects.cosmo_array` via
        :meth:`~swiftsimio.objects.cosmo_array.__getitem__` and should be
        collected in a :class:`swiftgalaxy.masks.MaskCollection`. The
        selection is applied permanently to all particle datasets for this
        galaxy. Temporary masks (e.g. for interactive use) can be created by
        using the :meth:`~SWIFTGalaxy.__getitem__` (square brackets) method of
        the :class:`SWIFTGalaxy`, any of its associated
        :class:`_SWIFTGroupDatasetHelper` or
        :class:`_SWIFTNamedColumnDatasetHelper` objects, but
        note that to ensure internal consistency, these return a masked copy of
        the *entire* :class:`SWIFTGalaxy`, and are therefore
        relatively memory-intensive. Masking individual
        :class:`~swiftsimio.objects.cosmo_array` datasets with
        :meth:`~swiftsimio.objects.cosmo_array.__getitem__`
        avoids this: only masked copies of the individual arrays are returned
        in this case.

        Parameters
        ----------
        mask_collection : :class:`swiftgalaxy.masks.MaskCollection`
            Set of masks to be applied to each particle type. Particle types
            may be omitted by setting their mask to None, or simply omitting
            them from the :class:`swiftgalaxy.masks.MaskCollection`.
        """
        for particle_name in self.metadata.present_group_names:
            mask = getattr(mask_collection, particle_name)
            if mask is not None:
                getattr(self, particle_name)._mask_dataset(mask)
        return

    def _append_to_coordinate_like_transform(self, transform: np.ndarray) -> None:
        """
        Add a new transformation to the sequence of transformations for the spatial-like
        coordinates.

        The cumulative transformation is stored as a single 4x4 transformation matrix,
        so we update the current transformation using a dot product. This voids any
        derived (spherical/cylindrical) coordinates.

        Parameters
        ----------
        transform : :class:`~numpy.ndarray`
            The transform to add to the cumulative coordinate transformation.
        """
        self._coordinate_like_transform = self._coordinate_like_transform.dot(transform)
        self._void_derived_coordinates()
        return

    def _append_to_velocity_like_transform(self, transform: np.ndarray) -> None:
        """
        Add a new transformation to the sequence of transformations for the velocity-like
        coordinates.

        The cumulative transformation is stored as a single 4x4 transformation matrix,
        so we update the current transformation using a dot product. This voids any
        derived (spherical/cylindrical) coordinates.

        Parameters
        ----------
        transform : :class:`~numpy.ndarray`
            The transform to add to the cumulative velocity transformation.
        """
        self._velocity_like_transform = self._velocity_like_transform.dot(transform)
        self._void_derived_coordinates()
        return

    def _void_derived_coordinates(self) -> None:
        """
        Reset internal references to spherical/cylindrical coordinates to None (e.g.
        because they are no longer valid).
        """
        # Transforming implies conversion back to cartesian, it's therefore
        # cheaper to just delete any non-cartesian coordinates when a
        # transform occurs and lazily re-calculate them as needed.
        for particle_name in self.metadata.present_group_names:
            getattr(self, particle_name)._void_derived_coordinates()
        return
