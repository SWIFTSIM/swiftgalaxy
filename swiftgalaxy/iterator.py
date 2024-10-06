"""
Provides the :class:`~swiftgalaxy.iterator.SWIFTGalaxies` class that enables efficient
iteration over :class:`~swiftgalaxy.reader.SWIFTGalaxy` objects for multiple objects of
interest within a single simulation snapshot.

Parallelization is not yet implemented but is prioritized for future release.
"""

import numpy as np
import unyt as u
from swiftsimio import mask, cosmo_array
from .reader import SWIFTGalaxy
from .halo_catalogues import _HaloCatalogue

from typing import Optional, Set, Any, List, Callable, Dict, Tuple, Generator
from swiftsimio.masks import SWIFTMask


class SWIFTGalaxies:
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

    There are two important consequences to be aware of:
     - The iteration order is not controlled by the user because it must be chosen to
       group objects in the same top-level cell(s) together. The iteration order is
       available as the :attr:`iteration_order` attribute of a
       :class:`~swiftgalaxy.iterator.SWIFTGalaxies` object. Alternatively, output of
       a function applied to a list of target objects in the same order as the input
       list can be obtained using the :meth:`~swiftgalaxy.iterator.SWIFTGalaxies.map`
       method.
     - To avoid duplicating I/O, the :class:`~swiftgalaxy.iterator.SWIFTGalaxies` class
       needs to be aware of what particle data fields will be accessed during analysis
       so that it can read them in for all target objects in a group before iterating
       over them. The set of fields to read should be specified with the ``preload``
       initialization argument. Omitting fields in this list that are accessed during
       the iteration over the target objects largely defeats the purpose of using this
       efficient iteration class in the first place.

    There is an obvious opportunity to parallelize the iteration process by passing each
    region (potentially each containing multiple target objects) to worker processes as
    they become available, for example. This current initial version of the
    :class:`~swiftgalaxy.iterator.SWIFTGalaxies` class does not yet support parallel
    iteration, instead prioritizing the release of a working serial implementation.
    Support for parallelization will be added later as a high priority.

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

    preload : :obj:`set` (optional), default: ``set()``
        A :obj:`set` containing strings specifying fields that will be accessed while
        the :class:`~swiftgalaxy.reader.SWIFTGalaxy`'s managed by this class are being
        iterated over. For example:
        ``{"gas.element_abundances.carbon", "dark_matter.coordinates, stars.velocities}``.

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
    catalogue is initialized with multiple target objects, (ii) the data to be used needs
    to be specified with the ``preload`` argument and (iii) the
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
            preload={
                        "gas.element_abundances.carbon",
                        "dark_matter.coordinates",
                        "stars.velocities",
                    },
        )
        iteration_order = sgs.iteration_order  # be aware of the order of iteration
        for sg in sgs:
            # some analysis involving the pre-loaded data fields goes here:
            sg.element_abundances.carbon
            sg.dark_matter.coordinates
            sg.stars.velocities

    Alternatively the ``map`` method can be used to apply a function to all of the
    :class:`~swiftgalaxy.reader.SWIFTGalaxy`'s created by this class. For example:

    ::
        from swiftgalaxy import SWIFTGalaxies, SOAP
        sgs = SWIFTGalaxies(
            "snapshot.hdf5",
            SOAP(
                "soap.hdf5",
                soap_index=[0, 123, 456],  # multiple target indices
            ),
            preload={
                        "gas.element_abundances.carbon",
                        "dark_matter.coordinates",
                        "stars.velocities",
                    },
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
    ):
        if not halo_catalogue._multi_galaxy:
            raise ValueError(
                "halo_catalogue target list is not iterable, create halo_catalogue with "
                "an iterable list of targets even if there is only one (or use "
                "SWIFTGalaxy instead of SWIFTGalaxies)."
            )
        self.halo_catalogue = halo_catalogue
        self.snapshot_filename = snapshot_filename
        self.auto_recentre = auto_recentre
        self.preload = preload
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
                < 0.5
                * (
                    self._dense_optimized_solution["cost_min"]
                    + self._dense_optimized_solution["cost_max"]
                )
                else self._dense_optimized_solution
            )
        elif optimize_iteration == "dense":
            self._solution = self._dense_optimized_solution
        elif optimize_iteration == "sparse":
            self._solution = self._sparse_optimized_solution

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
        out : :class:`numpy.ndarray`
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
        target_centres = self.halo_catalogue._region_centre
        if self.halo_catalogue._user_spatial_offsets is not None:
            target_regions = self.halo_catalogue._user_spatial_offsets[np.newaxis, ...]
        else:
            aperture = self.halo_catalogue._region_aperture
            target_regions = np.vstack(
                (-np.repeat(aperture, 3), np.repeat(aperture, 3))
            ).T.reshape((-1, 3, 2))
        # SWIFTMask gives us a lightweight interface to metadata & cell metadata
        sm = mask(self.snapshot_filename, spatial_only=True)
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
            .to_value(u.dimensionless)
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
        self._sparse_optimized_solution = dict(
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

    def _eval_dense_optimized_solution(self):
        """
        Evaluate an iteration scheme optimized for targets densely distributed in space.

        We assume that targets are densely distributed so that the region to be loaded
        for each one usually overlaps with (possibly several) other regions. We determine
        the minimum size of a region containing the largest target object and conceptually
        tile the volume with two overlapping grids with cells of this size offset by half
        a grid spacing along each axis. Each object of interest is assigned to the cell in
        either of the grids that minimizes the distance between its centre and the cell
        centre, guaranteeing that it fits in the region. Any grid cells containing no
        target objects of interest are not included in the iteration.

        The result is stored in ``self._dense_optimized_solution``, a dict with
        keys: ``"regions"`` containing the spatial regions that will define the spatial
        masks; ``"region_target_indices"`` containing the indices of the target objects of
        interest in each of the regions; ``"cost_min"`` and ``"cost_max"`` containing the
        minimum and maximum cost (in top-level cell read operations) of this iteration
        scheme. The actual cost depends on the size of the grid regions and whether none,
        one or both of the grids are aligned with the top-level cell grid.
        """
        target_centres = self.halo_catalogue._region_centre
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
        sm = mask(self.snapshot_filename, spatial_only=True)
        # grid should be at least 1 cell in size so that we efficiently group targets
        # in the same grid location
        grid_element_dim = (
            np.max(
                target_sizes // sm.cell_size[np.newaxis] + 1,
                axis=0,
            )
            .to_value(u.dimensionless)
            .astype(int)
        )
        # Cells are not guaranteed to have a vertex at box coordinate (0, 0, 0)
        # but usually are. At least one of our grids is misaligned with the cells
        # anyway, so we'll align one grid to the box. If the cells happen to align
        # to the box we'll get slightly lower i/o cost "for free".
        cells_dim = (
            (sm.metadata.boxsize // sm.cell_size).to_value(u.dimensionless).astype(int)
        )
        # If the grid doesn't fully cover the box need to add one more grid cell to
        # cover "too much".
        grid_dim = cells_dim // grid_element_dim + (
            cells_dim % grid_element_dim > 0
        ).astype(int)
        target_grid_offsets_aligned, target_grid_indices_aligned = np.modf(
            target_centres / (grid_element_dim * sm.cell_size)
        )
        target_grid_offsets_offset, target_grid_indices_offset = np.modf(
            (target_centres + 0.5 * grid_element_dim * sm.cell_size)
            / (grid_element_dim * sm.cell_size)
            - 1
        )
        target_grid_indices_aligned = target_grid_indices_aligned.to_value(
            u.dimensionless
        ).astype(int)
        # need a "box wrap" of targets to the far side of the grid for the offset grid:
        target_grid_indices_offset = np.where(
            np.logical_and(
                target_grid_offsets_offset < 0, target_grid_indices_offset == 0
            ),
            grid_dim - 1,
            target_grid_indices_offset,
        )  # complains about missing cosmo_array info
        target_grid_indices_offset = target_grid_indices_offset.to_value(
            u.dimensionless
        ).astype(int)
        target_grid_distances_aligned = np.sqrt(
            np.sum(
                np.power(
                    0.5 * np.ones(sm.metadata.dimension) - target_grid_offsets_aligned,
                    2,
                ),
                axis=1,
            )
        )
        target_grid_distances_offset = np.sqrt(
            np.sum(
                np.power(
                    0.5 * np.ones(sm.metadata.dimension) - target_grid_offsets_offset, 2
                ),
                axis=1,
            )
        )
        use_offset_grid = target_grid_distances_offset < target_grid_distances_aligned
        # target_regions columns: (1) flag which grid; (2,3,4) grid i, j, k
        target_regions = np.concatenate(
            (
                use_offset_grid[..., np.newaxis],
                np.where(
                    use_offset_grid[..., np.newaxis],
                    target_grid_indices_offset,
                    target_grid_indices_aligned,
                ),
            ),
            axis=1,
        )
        # unique_regions each correspond to a "server" SWIFTGalaxy that will read a region
        unique_grid_regions, inv = np.unique(
            target_regions, axis=0, return_inverse=True
        )
        # `centres_[aligned|offset]` and `grid_element_dim * sm.cell_size` define regions
        centres_aligned = (
            np.array(
                np.meshgrid(*[np.arange(gdi) + 0.5 for gdi in grid_dim], indexing="ij")
            )
            * sm.cell_size[:, np.newaxis, np.newaxis, np.newaxis]
        )
        centres_offset = (
            np.array(
                np.meshgrid(*[np.arange(gdi) + 1.0 for gdi in grid_dim], indexing="ij")
            )
            * sm.cell_size[:, np.newaxis, np.newaxis, np.newaxis]
        )
        unique_region_centres = cosmo_array(
            (centres_aligned, centres_offset)
        ).transpose((0, 2, 3, 4, 1))[tuple(unique_grid_regions.T)]
        unique_regions = cosmo_array(
            (
                unique_region_centres - 0.499 * grid_element_dim * sm.cell_size,
                unique_region_centres + 0.499 * grid_element_dim * sm.cell_size,
            )
        ).transpose((1, 2, 0))
        # in the following dict
        # regions are the bboxes that can be passed directly to spatial mask
        # (each one an entry along the 0th axis)
        # region_target_indices contains an array of indices into the targets array
        # for each region
        # cost_min is an integer number of cells that will be read during this iteration
        # in the best case (assuming both grids are aligned with the cell grid)
        # cost_max is an integer number of cells that will be read during this iteration
        # in the worst case (assuming both grids are mis-aligned with the cell grid)
        sorter = np.argsort(inv)
        self._dense_optimized_solution = dict(
            regions=unique_regions,
            region_target_indices=np.split(
                np.arange(inv.size)[sorter],
                np.unique(inv[sorter], return_index=True)[1][1:],
            ),
            cost_min=(
                np.prod(grid_element_dim)  # cost per grid element (optimistic)
                * unique_regions.shape[0]  # number of grid elements
            ),
            cost_max=(
                np.prod(grid_element_dim + 1)  # cost per grid element (pessimistic)
                * unique_regions.shape[0]  # number of grid elements
            ),
        )

    def _start_server(self, region_mask: SWIFTMask) -> None:
        """
        Create a "server" :class:`~swiftgalaxy.reader.SWIFTGalaxy`.

        The "server" loads all of the particles in a given region and can be repeatedly
        masked (creating copies of subsets of the particles) to efficiently provide
        :class:`~swiftgalaxy.reader.SWIFTGalaxy`'s for all objects of interest in a common
        region.

        Parameters
        ----------
        region_mask : :class:`~swiftsimio.masks.SWIFTMask`
            The spatial mask object for the server.
        """
        self._server = SWIFTGalaxy._copyinit(
            snapshot_filename=self.snapshot_filename,
            halo_catalogue=None,
            auto_recentre=False,
            transforms_like_coordinates=self.transforms_like_coordinates,
            transforms_like_velocities=self.transforms_like_velocities,
            id_particle_dataset_name=self.id_particle_dataset_name,
            coordinates_dataset_name=self.coordinates_dataset_name,
            velocities_dataset_name=self.velocities_dataset_name,
            _spatial_mask=region_mask,
            _extra_mask=None,
        )
        return

    def _preload(self) -> None:
        """
        Pre-load any data that a user will want to access for each object of interest.

        To avoid repeating I/O operations, :class:`~swiftgalaxy.iterator.SWIFTGalaxies`
        objects must be initialized with a list of fields to be "pre-loaded". This
        function carries out these read operations.
        """
        for preload_field in self.preload | self.halo_catalogue._get_preload_fields(
            self._server
        ):
            obj = self._server
            for attr in preload_field.split("."):
                obj = getattr(obj, attr)
        return

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
        out : :class:`~swiftgalaxy.reader.SWIFTGalaxy`
            Each subsequent :class:`~swiftgalaxy.reader.SWIFTGalaxy` object to be
            iterated over.

        See Also
        --------
        iteration_order
        map
        """
        region_mask = mask(self.snapshot_filename)
        for region, target_indices in zip(
            self._solution["regions"], self._solution["region_target_indices"]
        ):
            region_mask.constrain_spatial(region)
            self._start_server(region_mask)
            self._preload()
            for igalaxy in target_indices:
                self.halo_catalogue._mask_multi_galaxy(igalaxy)
                swift_galaxy = self._server[
                    self.halo_catalogue._get_extra_mask(self._server)
                ]
                swift_galaxy.halo_catalogue = self.halo_catalogue
                if self.auto_recentre and self.coordinate_frame_from is not None:
                    raise ValueError(
                        "Cannot use coordinate_frame_from with auto_recentre=True."
                    )
                elif self.coordinate_frame_from is not None:
                    if (
                        self.coordinate_frame_from.metadata.units.length
                        != swift_galaxy.metadata.units.length
                    ) or (
                        self.coordinate_frame_from.metadata.units.time
                        != swift_galaxy.metadata.units.time
                    ):
                        raise ValueError(
                            "Internal units (length and time) of coordinate_frame_from"
                            " don't match."
                        )
                    swift_galaxy._transform(
                        self.coordinate_frame_from._coordinate_like_transform,
                        boost=False,
                    )
                    swift_galaxy._transform(
                        self.coordinate_frame_from._velocity_like_transform,
                        boost=True,
                    )
                elif self.auto_recentre:
                    swift_galaxy.recentre(self.halo_catalogue.centre)
                    swift_galaxy.recentre_velocity(self.halo_catalogue.velocity_centre)
                swift_galaxy._initialised = True
                yield swift_galaxy
                self.halo_catalogue._unmask_multi_galaxy()

    def map(
        self, func: Callable, args: Tuple = tuple(), kwargs: Dict = dict()
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
        first argument. It may accept additional arguments and/or keyword arguments
        that can be passed

        Currently this function only executes serially but adding a parallel execution
        option, and further support for parallelization in analysis, is a high priority.

        Parameters
        ----------
        func : callable
            The function to be evaluated.
        args : :obj:`tuple`
            Additional arguments to the function to be evaluated (the first argument is
            always the current :class:`~swiftgalaxy.reader.SWIFTGalaxy` in the iteration).
        kwargs : :obj:`dict`
            Keyword arguments to pass to the function to be evaluated.

        Returns
        -------
        out : :obj:`list`
            A list containing the return value(s) of the function applied to each object
            of interest, in the same order as the objects of interest were passed to the
            halo finder interface.
        """
        result = list()
        for sg in self:
            result.append(func(sg, *args, **kwargs))
        result[:] = [result[i] for i in self.iteration_order]
        return result
