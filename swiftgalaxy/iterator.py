import numpy as np
import unyt as u
from swiftsimio import mask, cosmo_array
from .reader import SWIFTGalaxy
from .halo_catalogues import _HaloCatalogue

from typing import Optional, Set


class SWIFTGalaxies:

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
        self._init_args = dict(
            snapshot_filename=snapshot_filename,
            auto_recentre=auto_recentre,
            preload=set(preload),
            transforms_like_coordinates=transforms_like_coordinates,
            transforms_like_velocities=transforms_like_velocities,
            id_particle_dataset_name=id_particle_dataset_name,
            coordinates_dataset_name=coordinates_dataset_name,
            velocities_dataset_name=velocities_dataset_name,
            coordinate_frame_from=coordinate_frame_from,
        )
        # want 3 optimization modes:
        # sparse targets:
        #  determine region for each galaxy, group any that share the same region
        # dense targets:
        #  two overlapping region grids, assign each galaxy to nearest centre in either
        # auto
        #  evaluate both and then choose the one that minimizes i/o
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
    def iteration_order(self):
        return np.concatenate(self._solution["region_target_indices"])

    def _eval_sparse_optimized_solution(self):
        target_centres = self.halo_catalogue._region_centre
        if self.halo_catalogue._user_spatial_offsets is not None:
            target_regions = self.halo_catalogue._user_spatial_offsets[np.newaxis, ...]
        else:
            aperture = self.halo_catalogue._region_aperture
            target_regions = np.vstack(
                (-np.repeat(aperture, 3), np.repeat(aperture, 3))
            ).T.reshape((-1, 3, 2))
        # SWIFTMask gives us a lightweight interface to metadata & cell metadata
        sm = mask(self._init_args["snapshot_filename"], spatial_only=True)
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
        sm = mask(self._init_args["snapshot_filename"], spatial_only=True)
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
                unique_region_centres - 0.5 * grid_element_dim * sm.cell_size,
                unique_region_centres + 0.5 * grid_element_dim * sm.cell_size,
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

    def __iter__(self):
        region_mask = mask(self._init_args["snapshot_filename"])
        for region, target_indices in zip(
            self._solution["regions"], self._solution["region_target_indices"]
        ):
            region_mask.constrain_spatial(region)
            self._server = SWIFTGalaxy(
                snapshot_filename=self._init_args["snapshot_filename"],
                halo_catalogue=None,
                auto_recentre=False,
                transforms_like_coordinates=self._init_args[
                    "transforms_like_coordinates"
                ],
                transforms_like_velocities=self._init_args[
                    "transforms_like_velocities"
                ],
                id_particle_dataset_name=self._init_args["id_particle_dataset_name"],
                coordinates_dataset_name=self._init_args["coordinates_dataset_name"],
                velocities_dataset_name=self._init_args["velocities_dataset_name"],
                _spatial_mask=region_mask,
                _extra_mask=None,
            )
            for preload_field in self._init_args[
                "preload"
            ] | self.halo_catalogue._get_preload_fields(self._server):
                obj = self._server
                for attr in preload_field.split("."):
                    obj = getattr(obj, attr)
            for igalaxy in target_indices:
                self.halo_catalogue._mask_multi_galaxy(igalaxy)
                swift_galaxy = self._server[
                    self.halo_catalogue._get_extra_mask(self._server)
                ]
                swift_galaxy.halo_catalogue = self.halo_catalogue
                if (
                    self._init_args["auto_recentre"]
                    and self._init_args["coordinate_frame_from"] is not None
                ):
                    raise ValueError(
                        "Cannot use coordinate_frame_from with auto_recentre=True."
                    )
                elif self._init_args["coordinate_frame_from"] is not None:
                    if (
                        self._init_args["coordinate_frame_from"].metadata.units.length
                        != swift_galaxy.metadata.units.length
                    ) or (
                        self._init_args["coordinate_frame_from"].metadata.units.time
                        != swift_galaxy.metadata.units.time
                    ):
                        raise ValueError(
                            "Internal units (length and time) of coordinate_frame_from"
                            " don't match."
                        )
                    swift_galaxy._transform(
                        self._init_args[
                            "coordinate_frame_from"
                        ]._coordinate_like_transform,
                        boost=False,
                    )
                    swift_galaxy._transform(
                        self._init_args[
                            "coordinate_frame_from"
                        ]._velocity_like_transform,
                        boost=True,
                    )
                elif self._init_args["auto_recentre"]:
                    swift_galaxy.recentre(self.halo_catalogue.centre)
                    swift_galaxy.recentre_velocity(self.halo_catalogue.velocity_centre)
                swift_galaxy._initialised = True
                yield swift_galaxy
                self.halo_catalogue._unmask_multi_galaxy()
