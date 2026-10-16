"""
Iterate over :class:`~swiftgalaxy.reader.SWIFTGalaxy` objects efficiently.

Provides the :class:`~swiftgalaxy.iterator.SWIFTGalaxies` class that enables efficient
iteration over :class:`~swiftgalaxy.reader.SWIFTGalaxy` objects for multiple objects of
interest within a single simulation snapshot.

Iteration can be spread across worker processes with the ``nproc`` argument of
:func:`~swiftgalaxy.iterator.SWIFTGalaxies.map`, one region at a time per worker.
"""

import multiprocessing
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import unyt as u
from swiftsimio import mask, cosmo_array
from .reader import SWIFTGalaxy
from .halo_catalogues import _HaloCatalogue
from warnings import warn

from typing import (
    Optional,
    Set,
    Any,
    List,
    Callable,
    Dict,
    Tuple,
    Generator,
    NamedTuple,
    TypedDict,
)
from scipy.spatial.transform import RigidTransform


class _CoordinateFrameSpec(NamedTuple):
    """
    A picklable description of a coordinate frame to be copied onto other galaxies.

    A :class:`~swiftgalaxy.reader.SWIFTGalaxy` cannot be sent to a worker process
    because it holds :mod:`h5py` file handles. Only four small values of the
    ``coordinate_frame_from`` galaxy are ever used, however: its length and time units
    (to check compatibility) and its two rigid transforms. Those all pickle, so this
    carries them instead of the galaxy.

    The units are stored in their string form. A :mod:`unyt` quantity carries a unit
    registry that pickles to tens of kilobytes, which would be paid once per region,
    while the string comparison reproduces the same result at a couple of dozen bytes.

    Attributes
    ----------
    length_unit : :obj:`str`
        Internal length unit of the source galaxy.

    time_unit : :obj:`str`
        Internal time unit of the source galaxy.

    coordinate_transform : :class:`~scipy.spatial.transform.RigidTransform`
        The coordinate-like transform of the source galaxy.

    velocity_transform : :class:`~scipy.spatial.transform.RigidTransform`
        The velocity-like transform of the source galaxy.
    """

    length_unit: str
    time_unit: str
    coordinate_transform: RigidTransform
    velocity_transform: RigidTransform

    @classmethod
    def from_swift_galaxy(cls, sg: "SWIFTGalaxy") -> "_CoordinateFrameSpec":
        """
        Extract the picklable parts of a galaxy's coordinate frame.

        Parameters
        ----------
        sg : :class:`~swiftgalaxy.reader.SWIFTGalaxy`
            The galaxy whose coordinate frame is to be copied.

        Returns
        -------
        :class:`~swiftgalaxy.iterator._CoordinateFrameSpec`
            The extracted coordinate frame.
        """
        return cls(
            length_unit=str(sg.metadata.units.length),
            time_unit=str(sg.metadata.units.time),
            coordinate_transform=sg._coordinate_like_transform,
            velocity_transform=sg._velocity_like_transform,
        )

    def apply_to(self, swift_galaxy: "SWIFTGalaxy") -> None:
        """
        Place a galaxy in this coordinate frame.

        Parameters
        ----------
        swift_galaxy : :class:`~swiftgalaxy.reader.SWIFTGalaxy`
            The galaxy to transform, in place.

        Raises
        ------
        ValueError
            If the internal units of the two galaxies do not match.
        """
        if (self.length_unit != str(swift_galaxy.metadata.units.length)) or (
            self.time_unit != str(swift_galaxy.metadata.units.time)
        ):
            raise ValueError(
                "Internal units (length and time) of coordinate_frame_from don't match."
            )
        swift_galaxy._transform(self.coordinate_transform, boost=False)
        swift_galaxy._transform(self.velocity_transform, boost=True)
        return


class _RegionTask(NamedTuple):
    """
    A self-contained, picklable unit of work: one region and its targets.

    Everything a worker process needs to do its share of the iteration, expressed as
    plain data. No live object (which would hold unpicklable :mod:`h5py` handles)
    appears here.

    Attributes
    ----------
    snapshot_filename : :obj:`str`
        Name of file containing snapshot.

    catalogue_spec : :obj:`tuple`
        A ``(class, kwargs)`` pair rebuilding the halo catalogue for this region's
        targets, from
        :meth:`~swiftgalaxy.halo_catalogues._HaloCatalogue._subset_spec`.

    region : :class:`~swiftsimio.objects.cosmo_array`
        Bounding box of the region to read.

    target_indices : :obj:`list`
        Positions of this region's targets in the user's original target list.

    result_slots : :obj:`list`
        Where each target's result belongs in the output list. Evaluated by the parent
        because it is the halo catalogue that decides: :class:`SOAP` in particular
        overrides
        :meth:`~swiftgalaxy.halo_catalogues._HaloCatalogue._mask_multi_galaxy` to correct
        for :mod:`swiftsimio` reading catalogue rows in sorted order, so a target's
        position in the input list is not necessarily its slot in the results.

    sg_kwargs : :obj:`dict`
        Configuration forwarded to each
        :class:`~swiftgalaxy.reader.SWIFTGalaxy`.

    auto_recentre : :obj:`bool`
        Whether to recentre on the halo catalogue's centres.

    coordinate_frame_spec : :class:`~swiftgalaxy.iterator._CoordinateFrameSpec` \
    (optional)
        Coordinate frame to copy, if any.

    func : callable (optional)
        Function to apply to each galaxy. If ``None`` the galaxies themselves are
        yielded (serial use only).

    args : :obj:`list` (optional)
        Positional arguments for ``func``, one :obj:`tuple` per target.

    kwargs : :obj:`list` (optional)
        Keyword arguments for ``func``, one :obj:`dict` per target.
    """

    snapshot_filename: str
    catalogue_spec: Tuple
    region: cosmo_array
    target_indices: List[int]
    result_slots: List[int]
    sg_kwargs: Dict[str, Any]
    auto_recentre: bool
    coordinate_frame_spec: Optional[_CoordinateFrameSpec] = None
    func: Optional[Callable] = None
    args: Optional[List[Tuple]] = None
    kwargs: Optional[List[Dict]] = None


def _worker_context() -> multiprocessing.context.BaseContext:
    """
    Choose a process start method that is safe to use with HDF5.

    The default on Linux is ``fork``, which copies the parent's memory including any
    open HDF5 file handles. HDF5 is not fork-safe, and python warns that forking a
    multi-threaded process risks deadlocking the child. ``forkserver`` starts workers
    from a clean single-threaded process instead, and ``spawn`` is used where
    ``forkserver`` is unavailable (such as on Windows).

    Returns
    -------
    :class:`multiprocessing.context.BaseContext`
        A context using the preferred start method.
    """
    available = multiprocessing.get_all_start_methods()
    for method in ("forkserver", "spawn"):
        if method in available:
            return multiprocessing.get_context(method)
    return multiprocessing.get_context()


def _make_server(
    snapshot_filename: str, region: cosmo_array, sg_kwargs: Dict[str, Any]
) -> "SWIFTGalaxy":
    """
    Create a "server" :class:`~swiftgalaxy.reader.SWIFTGalaxy` for a region.

    The "server" loads all of the particles in a given region and can be repeatedly
    masked (creating copies of subsets of the particles) to efficiently provide
    :class:`~swiftgalaxy.reader.SWIFTGalaxy`'s for all objects of interest in a common
    region.

    Parameters
    ----------
    snapshot_filename : :obj:`str`
        Name of file containing snapshot.

    region : :class:`~swiftsimio.objects.cosmo_array`
        Bounding box of the region to read.

    sg_kwargs : :obj:`dict`
        Dataset naming and transformation configuration to forward.

    Returns
    -------
    :class:`~swiftgalaxy.reader.SWIFTGalaxy`
        The server for this region.
    """
    region_mask = mask(snapshot_filename)
    region_mask.constrain_spatial(region)
    return SWIFTGalaxy._copyinit(
        snapshot_filename=snapshot_filename,
        halo_catalogue=None,
        auto_recentre=False,
        _spatial_mask=region_mask,
        _extra_mask=None,
        **sg_kwargs,
    )


def _iterate_region(
    task: _RegionTask, halo_catalogue: Optional[_HaloCatalogue] = None
) -> Generator:
    """
    Read one region and yield its galaxies, one at a time.

    This is the single implementation shared by serial iteration and by parallel
    workers, so that the two cannot drift apart. Reading the region once and masking
    copies out of it is what makes grouping targets by region worthwhile.

    Parameters
    ----------
    task : :class:`~swiftgalaxy.iterator._RegionTask`
        The region and targets to iterate.

    halo_catalogue : :class:`~swiftgalaxy.halo_catalogues._HaloCatalogue` (optional), \
    default: ``None``
        An existing catalogue to use. If ``None`` (the case in a worker process) one is
        rebuilt from ``task.catalogue_spec``.

    Yields
    ------
    :obj:`tuple`
        Pairs of the target's position within this task and the corresponding
        :class:`~swiftgalaxy.reader.SWIFTGalaxy`.

    Raises
    ------
    ValueError
        If both ``auto_recentre`` and a coordinate frame to copy are requested.
    """
    if halo_catalogue is None:
        catalogue_class, catalogue_kwargs = task.catalogue_spec
        halo_catalogue = catalogue_class(**catalogue_kwargs)
        # a rebuilt catalogue holds only this region's targets, in order
        local_indices = list(range(len(task.target_indices)))
    else:
        local_indices = list(task.target_indices)
    server = _make_server(task.snapshot_filename, task.region, task.sg_kwargs)
    for position, local_index in enumerate(local_indices):
        halo_catalogue._mask_multi_galaxy(local_index)
        server_mask = halo_catalogue._get_extra_mask(server, mask_loaded=False)
        swift_galaxy = server._data_copy(server_mask, _data_server=server)
        swift_galaxy.halo_catalogue = halo_catalogue
        if task.auto_recentre and task.coordinate_frame_spec is not None:
            raise ValueError(
                "Cannot use coordinate_frame_from with auto_recentre=True."
            )
        elif task.coordinate_frame_spec is not None:
            task.coordinate_frame_spec.apply_to(swift_galaxy)
        elif task.auto_recentre:
            swift_galaxy.recentre(halo_catalogue.centre)
            swift_galaxy.recentre_velocity(halo_catalogue.velocity_centre)
        yield position, swift_galaxy
        halo_catalogue._unmask_multi_galaxy()


def _run_region(task: _RegionTask) -> List[Tuple[int, Any]]:
    """
    Apply a function to every galaxy in one region; the unit of parallel work.

    Runs in a worker process. It rebuilds its own halo catalogue and opens its own file
    handles, so nothing is shared with the parent or with sibling workers and no
    locking is required.

    Parameters
    ----------
    task : :class:`~swiftgalaxy.iterator._RegionTask`
        The region, its targets, and the function to apply.

    Returns
    -------
    :obj:`list`
        ``(result_slot, result)`` pairs, so that the parent can restore the user's
        ordering without relying on the order work completed in.
    """
    assert task.func is not None
    args = task.args if task.args is not None else [tuple()] * len(task.target_indices)
    kwargs = (
        task.kwargs if task.kwargs is not None else [dict()] * len(task.target_indices)
    )
    results = []
    for position, swift_galaxy in _iterate_region(task):
        results.append(
            (
                task.result_slots[position],
                task.func(swift_galaxy, *args[position], **kwargs[position]),
            )
        )
    return results


class _IterationSolution(TypedDict):
    """Type hints for dicts containing a proposed SWIFTGalaxies iteration strategy."""

    regions: np.ndarray
    region_target_indices: list[np.ndarray]
    cost: int


class SWIFTGalaxies(object):
    """
    Facilitates efficiently iterating over many objects of interest from a simulation.

    SWIFT simulation snapshots contain particles grouped by "top-level cells" that
    cover the simulation volume. The minimum number of particles that it makes sense
    to read is therefore those contained in one such top-level cell. If one wants to
    create many :class:`~swiftgalaxy.reader.SWIFTGalaxy` objects from one simulation
    snapshot, there is a risk that the same data are read many times, such as when
    multiple target objects lie within the same top-level cell. This class provides
    a convenient way to iterate over multiple target objects while minimizing the I/O
    overhead by managing the order of iteration to group together target objects that
    occupy common top-level cells and only reading the data once.

    An important consequence to be aware of is that the iteration order is not controlled
    by the user because it must be chosen to group objects in the same top-level cell(s)
    together. The iteration order is available as the :attr:`iteration_order` attribute of
    a :class:`~swiftgalaxy.iterator.SWIFTGalaxies` object. Alternatively, output of a
    function applied to a list of target objects in the same order as the input list can
    be obtained using the :meth:`~swiftgalaxy.iterator.SWIFTGalaxies.map` method.

    The iteration can be parallelized by passing ``nproc`` to
    :meth:`~swiftgalaxy.iterator.SWIFTGalaxies.map`, which hands each region (potentially
    containing several target objects) to a worker process as one becomes free. Each
    worker reads its region and iterates that region's galaxies within the worker, so
    that a set of top-level cells is still read only once. Iterating a
    :class:`~swiftgalaxy.iterator.SWIFTGalaxies` directly is always serial: the
    :class:`~swiftgalaxy.reader.SWIFTGalaxy` objects that it yields hold open file
    handles and cannot be transferred between processes.

    Parameters
    ----------
    snapshot_filename : :obj:`str`
        Name of file containing snapshot.

    halo_catalogue : :class:`~swiftgalaxy.halo_catalogues._HaloCatalogue`
        A halo catalogue instance from :mod:`swiftgalaxy.halo_catalogues`, e.g. a
        :class:`swiftgalaxy.halo_catalogues.SOAP` instance. It should specify more
        than one target object, e.g. by setting its ``soap_index=[0, 123, 456, ...]``.

    auto_recentre : :obj:`bool` (optional), default: ``True``
        If ``True``, the coordinate system will be automatically recentred on the
        position and velocity centres defined by the ``halo_catalogue``.

    preload : set (optional), default: ``set()``
        Deprecated and ignored.

    transforms_like_coordinates : :obj:`set` (optional), default: ``set()``
        Names of fields that behave as velocities. It is assumed that these
        exist for all present particle types. When the coordinate system is
        rotated or boosted, the associated arrays will be transformed
        accordingly. The ``velocities`` dataset (or its alternative name given
        in the ``velocities_dataset_name`` parameter) is implicitly assumed to
        behave as velocities.

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

    coordinates_dataset_name : :obj:`str` (optional), default: ``"coordinates"``
        Name of the dataset containing the particle spatial coordinates,
        assumed to be the same for all present particle types.

    velocities_dataset_name : :obj:`str` (optional), default: ``"velocities"``
        Name of the dataset containing the particle velocities, assumed to be
        the same for all present particle types.

    coordinate_frame_from : :class:`~swiftgalaxy.reader.SWIFTGalaxy` (optional), \
    default: ``None``
        Another :class:`~swiftgalaxy.reader.SWIFTGalaxy` to copy the coordinate frame
        (centre and rotation) and velocity coordinate frame (boost and rotation) from.

    optimize_iteration : :obj:`str` (optional), default: ``"auto"``
        Can be ``"auto"``, ``"dense"`` or ``"sparse"``. See docstrings of methods
        :meth:`~swiftgalaxy.iterator._eval_sparse_optimized_solution` and
        :meth:`~swiftgalaxy.iterator._eval_dense_optimized_solution` for
        explanations of optimization schemes. In most cases leave set to default
        ``"auto"`` to automatically determine optimal solution.

    Examples
    --------
    Using :class:`~swiftgalaxy.iterator.SWIFTGalaxies` is almost the same as using the
    main :class:`~swiftgalaxy.reader.SWIFTGalaxy` class, except that (i) the halo
    catalogue is initialized with multiple target objects and (ii) the
    :class:`~swiftgalaxy.iterator.SWIFTGalaxies` class provides an iteration method
    (``__iter__``), and determines its own iteration order. For example:

    ::

        from swiftgalaxy import SWIFTGalaxies, SOAP
        sgs = SWIFTGalaxies(
            "snapshot.hdf5",
            SOAP(
                "soap.hdf5",
                soap_index=[0, 123, 456],  # multiple target indices
            ),
        )
        iteration_order = sgs.iteration_order  # be aware of the order of iteration
        for sg in sgs:
            # some analysis involving the pre-loaded data fields goes here:
            sg.element_abundances.carbon
            sg.dark_matter.coordinates
            sg.stars.velocities

    Alternatively the :meth:`~swiftgalaxy.iterator.SWIFTGalaxies.map` method can be used
    to apply a function to all of the :class:`~swiftgalaxy.reader.SWIFTGalaxy`'s created
    by this class. For example:

    ::

        from swiftgalaxy import SWIFTGalaxies, SOAP
        sgs = SWIFTGalaxies(
            "snapshot.hdf5",
            SOAP(
                "soap.hdf5",
                soap_index=[0, 123, 456],  # multiple target indices
            ),
        )

        def analysis(sg):
            # this function can also have additional args & kwargs, if needed
            # it should only access the pre-loaded data fields
            sg.element_abundances.carbon
            sg.dark_matter.coordinates
            sg.stars.velocities
            return sg.element_abundances.carbon.mean()

        # map accepts arguments `args` and `kwargs`, passed through to function, if needed
        result = sgs.map(analysis)
    """

    count: int
    halo_catalogue: _HaloCatalogue

    def __init__(
        self,
        snapshot_filename: str,
        halo_catalogue: _HaloCatalogue,
        auto_recentre: bool = True,
        preload: Set[str] = set(),
        transforms_like_coordinates: Set[str] = set(),
        transforms_like_velocities: Set[str] = set(),
        id_particle_dataset_name: str = "particle_ids",
        coordinates_dataset_name: str = "coordinates",
        velocities_dataset_name: str = "velocities",
        coordinate_frame_from: Optional["SWIFTGalaxy"] = None,
        optimize_iteration: str = "auto",
    ) -> None:
        if not halo_catalogue._multi_galaxy:
            raise ValueError(
                "halo_catalogue target list is not iterable, create halo_catalogue with "
                "an iterable list of targets even if there is only one (or use "
                "SWIFTGalaxy instead of SWIFTGalaxies)."
            )
        if len(preload) > 0:
            warn(
                "Preloading is no longer required, `preload` argument will be removed in "
                "a future version.",
                DeprecationWarning,
            )
        self.halo_catalogue = halo_catalogue
        self.snapshot_filename = snapshot_filename
        self.auto_recentre = auto_recentre
        self.transforms_like_coordinates = transforms_like_coordinates
        self.transforms_like_velocities = transforms_like_velocities
        self.id_particle_dataset_name = id_particle_dataset_name
        self.coordinates_dataset_name = coordinates_dataset_name
        self.velocities_dataset_name = velocities_dataset_name
        self.coordinate_frame_from = coordinate_frame_from
        if optimize_iteration not in ("dense", "sparse", "auto"):
            raise ValueError(
                "optimize_iteration must be one of 'dense', 'sparse' or 'auto'."
            )

        if self.halo_catalogue._index_attr is not None:
            num_targets = len(
                getattr(self.halo_catalogue, self.halo_catalogue._index_attr)
            )
        else:
            num_targets = len(self.halo_catalogue._region_centre)
        # before evaluating optimized solutions:
        if num_targets == 0:
            # if we have 0 targets short-circuit
            self._solution = _IterationSolution(
                regions=np.array([]),
                region_target_indices=[np.array([])],
                cost=-1,
            )
            return
        if num_targets == 1:
            # if we have 1 target best strategy is sparse
            optimize_iteration = "sparse"

        if optimize_iteration in ("dense", "auto"):
            self._eval_dense_optimized_solution()
        if optimize_iteration in ("sparse", "auto"):
            self._eval_sparse_optimized_solution()
        if optimize_iteration == "auto":
            # typical cost of the dense solution expected to be average of the minimum
            # and maxiumum costs (one grid aligns with the cell grid), use sparse solution
            # if it costs less than this
            self._solution = (
                self._sparse_optimized_solution
                if self._sparse_optimized_solution["cost"]
                < self._dense_optimized_solution["cost"]
                else self._dense_optimized_solution
            )
        elif optimize_iteration == "dense":
            self._solution = self._dense_optimized_solution
        else:  # optimize_iteration == "sparse"
            self._solution = self._sparse_optimized_solution
        return

    @property
    def iteration_order(self) -> np.ndarray:
        """
        Property holding the order that the target objects will be iterated in.

        The iteration order is likely not the same as the order that the targets are
        provided in because this is probably not an optimal iteration order. This
        property attribute provides the optimized iteration order evaluated by
        :class:`~swiftgalaxy.iterator.SWIFTGalaxies`.

        Returns
        -------
        :class:`numpy.ndarray`
            Array of indices specifying the iteration order.
        """
        return np.concatenate(self._solution["region_target_indices"])

    def _eval_sparse_optimized_solution(self) -> None:
        """
        Evaluate an iteration scheme optimized for targets sparsely distributed in space.

        We assume that targets are sparsely distributed so that the region to be loaded
        for each one usually does not overlap with other regions. We determine the region
        to be read for each object of interest (in increments of entire top-level cells)
        and check if we can group any objects of interest together because they have
        identical regions.

        The result is stored in ``self._sparse_optimized_solution``, a dict with
        keys: ``"regions"`` containing the spatial regions that will define the spatial
        masks; ``"region_target_indices"`` containing the indices of the target objects of
        interest in each of the regions; ``"cost"`` containing the cost (in top-level cell
        read operations) of this iteration scheme.
        """
        target_centres = np.atleast_2d(self.halo_catalogue._region_centre)
        if self.halo_catalogue._user_spatial_offsets is not None:
            target_regions = self.halo_catalogue._user_spatial_offsets[np.newaxis, ...]
        else:
            aperture = self.halo_catalogue._region_aperture
            target_regions = np.vstack(
                (-np.repeat(aperture, 3), np.repeat(aperture, 3))
            ).T.reshape((-1, 3, 2))
        # SWIFTMask gives us a lightweight interface to metadata & cell metadata
        sm = mask(self.snapshot_filename)
        # get the lower cell vertex, probably at origin but not guaranteed
        cell_vertex_origin = sm.centers.min(axis=0) - sm.cell_size / 2
        # align origin with the cell grid, we allow going out the upper bounds of the
        # box since SWIFTMask handles wrapping for us (but we want a common grid
        # to easily group shared regions)
        target_centres = np.where(
            target_centres < cell_vertex_origin,
            target_centres + sm.metadata.boxsize,
            target_centres,
        )
        target_region_indices = (
            (
                (
                    target_centres[:, :, np.newaxis]
                    + target_regions
                    - cell_vertex_origin[np.newaxis, :, np.newaxis]
                )
                // sm.cell_size[np.newaxis, :, np.newaxis]
            )
            .to_comoving_value(u.dimensionless)
            .astype(int)
        )
        unique_region_indices, inv = np.unique(
            target_region_indices, axis=0, return_inverse=True
        )
        unique_regions = (
            unique_region_indices + np.array([0.01, 0.99])
        ) * sm.cell_size[np.newaxis, :, np.newaxis] + cell_vertex_origin[
            np.newaxis, :, np.newaxis
        ]
        sorter = np.argsort(inv)
        # in the following dict
        # regions are the bboxes that can be passed directly to spatial mask
        # (each one an entry along the 0th axis)
        # region_target_indices contains an array of indices into the targets array
        # for each region
        # cost is the integer number of cells that will be read during this iteration
        self._sparse_optimized_solution = _IterationSolution(
            regions=unique_regions,
            region_target_indices=np.split(
                np.arange(inv.size)[sorter],
                np.unique(inv[sorter], return_index=True)[1][1:],
            ),
            cost=np.sum(
                np.prod(
                    np.diff(unique_region_indices + np.array([0, 1]), axis=2), axis=1
                )
            ),
        )

    def _eval_dense_optimized_solution(self) -> None:
        """
        Evaluate an iteration scheme optimized for targets densely distributed in space.

        We assume that targets are densely distributed so that the region to be loaded
        for each one usually overlaps with (possibly several) other regions. We determine
        the minimum size of a region containing the largest target object and conceptually
        tile the volume with a grid with cells of this size. Each object of interest is
        assigned to the cell in which its centre lies. This groups targets into spatial
        associations. For each group of targets, the bounding box containing all of the
        targets is evaluated, resulting in a set of (probably partially overlapping)
        regions with targets assigned to each.

        The result is stored in ``self._dense_optimized_solution``, a dict with
        keys: ``"regions"`` containing the spatial regions that will define the spatial
        masks; ``"region_target_indices"`` containing the indices of the target objects of
        interest in each of the regions; ``"cost"`` containing the estimated cost (average
        of minimum and maximum cost in top-level cell read operations) of this iteration
        scheme. The actual cost depends on how each region aligns with the cell grid. For
        example if the region has a side lengths of 1.8 cells, this could touch either 2
        or 3 cells, depending on where in the grid the vertex lands.
        """
        target_centres = np.atleast_2d(self.halo_catalogue._region_centre)
        if self.halo_catalogue._user_spatial_offsets is not None:
            target_sizes = np.diff(self.halo_catalogue._user_spatial_offsets).T
        else:
            aperture = self.halo_catalogue._region_aperture
            target_sizes = np.diff(
                np.vstack((-np.repeat(aperture, 3), np.repeat(aperture, 3))).T.reshape(
                    (-1, 3, 2)
                )
            ).squeeze(2)
        # SWIFTMask gives us a lightweight interface to metadata & cell metadata
        sm = mask(self.snapshot_filename)
        # grid should be at least 1 cell in size so that we efficiently group targets
        # in the same grid location
        grid_element_dim = (
            np.max(target_sizes // sm.cell_size[np.newaxis], axis=0)
            .to_comoving_value(u.dimensionless)
            .astype(int)
            + 1
        )
        # If the grid doesn't fully cover the box need to add one more grid cell to
        # cover "too much".
        target_grid_offsets, target_grid_indices = np.modf(
            (target_centres / (grid_element_dim * sm.cell_size)).to_comoving_value(
                u.dimensionless
            )
        )
        target_grid_indices = target_grid_indices.astype(int)
        # unique_regions each correspond to a "server" SWIFTGalaxy that will read a region
        unique_grid_regions, inv = np.unique(
            target_grid_indices, axis=0, return_inverse=True
        )
        unique_regions = list()
        for unique_region_index in range(unique_grid_regions.shape[0]):
            umask = unique_region_index == inv
            masked_target_sizes = (
                target_sizes[umask]
                if self.halo_catalogue._user_spatial_offsets is None
                else target_sizes
            )
            unique_regions.append(
                np.vstack(
                    (
                        np.min(target_centres[umask] - masked_target_sizes, axis=0),
                        np.max(target_centres[umask] + masked_target_sizes, axis=0),
                    )
                ).T
            )
        unique_regions = cosmo_array(unique_regions)
        # in the following dict
        # regions are the bboxes that can be passed directly to spatial mask
        # (each one an entry along the 0th axis)
        # region_target_indices contains an array of indices into the targets array
        # for each region
        # cost_min is an integer number of cells that will be read during this iteration
        # in the best case (assuming the regions are near aligned with the cell grid)
        # cost_max is an integer number of cells that will be read during this iteration
        # in the worst case (assuming the regions are near mis-aligned with the cell grid)
        sorter = np.argsort(inv)
        cost_min = int(
            np.sum(
                np.prod(
                    np.ceil(
                        np.diff(unique_regions, axis=2).squeeze(axis=2) / sm.cell_size
                    ).to_comoving_value(u.dimensionless),
                    axis=1,
                )
            )
        )
        cost_max = int(
            np.sum(
                np.prod(
                    np.ceil(
                        np.diff(unique_regions, axis=2).squeeze(axis=2) / sm.cell_size
                    ).to_comoving_value(u.dimensionless)
                    + 1,
                    axis=1,
                )
            )
        )
        self._dense_optimized_solution = _IterationSolution(
            regions=unique_regions,
            region_target_indices=np.split(
                np.arange(inv.size)[sorter],
                np.unique(inv[sorter], return_index=True)[1][1:],
            ),
            cost=int(np.ceil(0.5 * (cost_min + cost_max))),  # estimate as the average
        )

    @property
    def _sg_kwargs(self) -> Dict[str, Any]:
        """
        Configuration forwarded to every :class:`~swiftgalaxy.reader.SWIFTGalaxy`.

        Returns
        -------
        :obj:`dict`
            The dataset naming and transformation settings.
        """
        return dict(
            transforms_like_coordinates=self.transforms_like_coordinates,
            transforms_like_velocities=self.transforms_like_velocities,
            id_particle_dataset_name=self.id_particle_dataset_name,
            coordinates_dataset_name=self.coordinates_dataset_name,
            velocities_dataset_name=self.velocities_dataset_name,
        )

    def _region_tasks(
        self,
        func: Optional[Callable] = None,
        args: Optional[List[Tuple]] = None,
        kwargs: Optional[List[Dict]] = None,
    ) -> List[_RegionTask]:
        """
        Express the iteration solution as a list of picklable units of work.

        Parameters
        ----------
        func : callable (optional), default: ``None``
            Function to apply to each galaxy.

        args : :obj:`list` (optional), default: ``None``
            Positional arguments for ``func``, one :obj:`tuple` per target.

        kwargs : :obj:`list` (optional), default: ``None``
            Keyword arguments for ``func``, one :obj:`dict` per target.

        Returns
        -------
        :obj:`list`
            One :class:`~swiftgalaxy.iterator._RegionTask` per region.
        """
        frame_spec = (
            _CoordinateFrameSpec.from_swift_galaxy(self.coordinate_frame_from)
            if self.coordinate_frame_from is not None
            else None
        )
        sg_kwargs = self._sg_kwargs
        tasks = []
        for region, target_indices in zip(
            self._solution["regions"], self._solution["region_target_indices"]
        ):
            indices = [int(i) for i in target_indices]
            # Ask the catalogue where each result belongs. SOAP corrects for
            # swiftsimio reading rows in sorted order, so this is not always the
            # target's position in the input list.
            result_slots = []
            for index in indices:
                self.halo_catalogue._mask_multi_galaxy(index)
                slot = self.halo_catalogue._multi_galaxy_catalogue_mask
                # masking always sets this; the Optional is for the unmasked state
                assert slot is not None
                result_slots.append(int(slot))
                self.halo_catalogue._unmask_multi_galaxy()
            tasks.append(
                _RegionTask(
                    snapshot_filename=self.snapshot_filename,
                    catalogue_spec=self.halo_catalogue._subset_spec(indices),
                    region=region,
                    target_indices=indices,
                    result_slots=result_slots,
                    sg_kwargs=sg_kwargs,
                    auto_recentre=self.auto_recentre,
                    coordinate_frame_spec=frame_spec,
                    func=func,
                    args=[args[i] for i in indices] if args is not None else None,
                    kwargs=[kwargs[i] for i in indices] if kwargs is not None else None,
                )
            )
        return tasks

    def __iter__(self) -> Generator:
        """
        Iterate over the objects of interest defined by the halo catalogue.

        A :class:`~swiftgalaxy.iterator.SWIFTGalaxies` object is iterable, and on each
        iteration will provide one :class:`~swiftgalaxy.reader.SWIFTGalaxy` object from
        the set of objects of interest defined by the halo catalogue. The order of
        iteration is not the order that the objects are listed in the halo catalogue,
        but an optimized order determined by the
        :class:`~swiftgalaxy.iterator.SWIFTGalaxies` class. The order of iteration is
        available from :func:`~swiftgalaxy.iterator.SWIFTGalaxies.iteration_order`.
        Ordered results of applying a function to each item in the iterable set can be
        obtained using :func:`~swiftgalaxy.iterator.SWIFTGalaxies.map`.

        Yields
        ------
        :class:`~swiftgalaxy.reader.SWIFTGalaxy`
            Each subsequent :class:`~swiftgalaxy.reader.SWIFTGalaxy` object to be
            iterated over.

        See Also
        --------
        iteration_order
        map
        """
        for task in self._region_tasks():
            for _, swift_galaxy in _iterate_region(
                task, halo_catalogue=self.halo_catalogue
            ):
                yield swift_galaxy

    def map(
        self,
        func: Callable,
        args: Optional[List[Tuple]] = None,
        kwargs: Optional[List[Dict]] = None,
        nproc: int = 1,
    ) -> List[Any]:
        """
        Apply a function to each object of interest and return a list of results.

        The iteration order of :class:`~swiftgalaxy.iterator.SWIFTGalaxies` is not
        necessarily the order that the objects of interest are provided by the user
        because the class determined an efficient iteration order to minimize I/O
        operations. This method applies a provided function to each object of interest
        in an efficient order then returns the results in a list ordered in the same
        order that the objects of interest were input.

        The function to be evaluated should expect a
        :class:`~swiftgalaxy.reader.SWIFTGalaxy` (from those to be iterated over) as its
        first argument. It may accept lists of additional arguments and/or keyword
        arguments (with each element corresponding to one entry in the list of target
        objects) that can be passed to map as a :obj:`tuple` of arguments and a
        :obj:`dict` of keyword arguments.

        Set ``nproc`` greater than 1 to evaluate the function in parallel, with one
        worker process per region. Because each worker opens its own files and rebuilds
        its own halo catalogue, nothing is shared between workers. With ``nproc=1``
        (the default) execution is serial and unchanged.

        Parameters
        ----------
        func : callable
            The function to be evaluated.
        args : :obj:`list` (optional), default: ``None``
            List of additional arguments to the function to be evaluated (the first
            argument is always the current :class:`~swiftgalaxy.reader.SWIFTGalaxy` in the
            iteration). Each item in the list should be a :obj:`tuple` of arguments, with
            one :obj:`tuple` for each galaxy being iterated over. See examples section for
            further details.
        kwargs : :obj:`list` (optional), default: ``None``
            List of additional keyword arguments to pass to the function to be evaluated.
            Each item in the list should be a :obj:`dict` of keyword arguments, with one
            :obj:`dict` for each galaxy being iterated over. Dictionary keys are
            the names of the keyword arguments and the corresponding dictionary values are
            the values of the keyword arguments. See examples section for further details.

        nproc : :obj:`int` (optional), default: ``1``
            Number of worker processes to use, one region at a time each. The default of
            ``1`` evaluates serially in the current process. Values greater than ``1``
            require that ``func`` is defined at module level (so that it can be pickled)
            and that its return values are picklable and small: returning particle-sized
            arrays from many regions can cost more in inter-process communication than
            the parallelism saves.

        Returns
        -------
        :obj:`list`
            A list containing the return value(s) of the function applied to each object
            of interest, in the same order as the objects of interest were passed to the
            halo finder interface.

        Raises
        ------
        ValueError
            If ``nproc`` is less than ``1``.

        Examples
        --------
        A simple example that applies a function ``dm_median_position`` to each galaxy in
        a list of targets ``[11, 22, 33]``:

        ::

            from swiftgalaxy import SWIFTGalaxies, SOAP

            # define the function that we will apply to each SWIFTGalaxy object:
            def dm_median_position(sg):
                return np.median(sg.dark_matter.coordinates, axis=0)

            sgs = SWIFTGalaxies(
                "my_snapshot.hdf5",
                SOAP(
                    "my_soap.hdf5",
                    soap_index=[11, 22, 33],
                ),
            )
            my_result = sgs.map(dm_median_position)

        The result stored in ``my_result`` contains the result of the function for the
        galaxies at index ``11``, ``22`` and ``33``, in the same order as they are given
        in the ``soap_index`` list.

        This second example shows how to pass extra arguments and/or keyword arguments to
        the function given to ``map``:

        ::

            from swiftgalaxy import SWIFTGalaxies, SOAP

            # define the function that we will apply to each SWIFTGalaxy object:
            def dm_median_position(
                sg,  # the first argument is always a SWIFTGalaxy from the iteration
                extra_argument_1,
                extra_argument_2,
                extra_kwarg_1=None,
                extra_kwarg_2=None,
            ):
                # presumably make use of the extra arguments and/or kwargs here...
                return np.median(sg.dark_matter.coordinates, axis=0)

            sgs = SWIFTGalaxies(
                "my_snapshot.hdf5",
                SOAP("my_soap.hdf5",
                soap_index=[11, 22, 33]),
            )
            my_result = sg.map(
                dm_median_position,
                args=[
                    (my_extra_arg_1_for_galaxy_11, my_extra_arg_2_for_galaxy_11),
                    (my_extra_arg_1_for_galaxy_22, my_extra_arg_2_for_galaxy_22),
                    (my_extra_arg_1_for_galaxy_33, my_extra_arg_2_for_galaxy_33),
                ],
                kwargs=[
                    dict(
                        extra_kwarg_1=my_extra_kwarg_1_for_galaxy_11,
                        extra_kwarg_2=my_extra_kwarg_2_for_galaxy_11,
                    ),
                    dict(
                        extra_kwarg_1=my_extra_kwarg_1_for_galaxy_22,
                        extra_kwarg_2=my_extra_kwarg_2_for_galaxy_22,
                    ),
                    dict(
                        extra_kwarg_1=my_extra_kwarg_1_for_galaxy_33,
                        extra_kwarg_2=my_extra_kwarg_2_for_galaxy_33,
                    ),
                ]
            )

        Note that if you have only a single extra argument it must still be packaged as
        a tuple, for instance:

        ::

            args=[
                (my_extra_arg_for_galaxy_11, ),
                (my_extra_arg_for_galaxy_22, ),
                (my_extra_arg_for_galaxy_33, ),
            ]

        The commas inside the parentheses are not optional!
        """
        if nproc < 1:
            raise ValueError("nproc must be at least 1.")
        ntargets = len(self.iteration_order)
        result: List[Any] = [None] * ntargets
        if args is None:
            args = [tuple()] * ntargets
        if kwargs is None:
            kwargs = [dict()] * ntargets
        tasks = self._region_tasks(func=func, args=args, kwargs=kwargs)
        if nproc == 1:
            for task in tasks:
                for position, swift_galaxy in _iterate_region(
                    task, halo_catalogue=self.halo_catalogue
                ):
                    target_index = task.target_indices[position]
                    result[task.result_slots[position]] = func(
                        swift_galaxy, *args[target_index], **kwargs[target_index]
                    )
            return result
        # Largest regions first: finishing on a straggler wastes the most time, and
        # region costs vary a great deal in practice.
        tasks.sort(key=lambda task: len(task.target_indices), reverse=True)
        with ProcessPoolExecutor(
            max_workers=nproc, mp_context=_worker_context()
        ) as executor:
            for region_results in executor.map(_run_region, tasks):
                for result_slot, value in region_results:
                    result[result_slot] = value
        return result
