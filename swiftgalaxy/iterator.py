import numpy as np
import unyt as u
from swiftsimio import mask, cosmo_array
from swiftsimio.masks import SWIFTMask
from .reader import SWIFTGalaxy
from .halo_catalogues import _HaloCatalogue

from typing import Optional, Set


class _SWIFTGalaxyServer(SWIFTGalaxy):

    def __init__(
        self,
        snapshot_filename: str,
        transforms_like_coordinates: Set[str] = set(),
        transforms_like_velocities: Set[str] = set(),
        id_particle_dataset_name: str = "particle_ids",
        coordinates_dataset_name: str = "coordinates",
        velocities_dataset_name: str = "velocities",
        spatial_mask: Optional[SWIFTMask] = None,
    ):
        super().__init__(
            snapshot_filename=snapshot_filename,
            transforms_like_coordinates=transforms_like_coordinates,
            transforms_like_velocities=transforms_like_velocities,
            id_particle_dataset_name=id_particle_dataset_name,
            coordinates_dataset_name=coordinates_dataset_name,
            velocities_dataset_name=velocities_dataset_name,
            _spatial_mask=spatial_mask,
        )


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
            preload=preload,
            transforms_like_coordinates=transforms_like_coordinates,
            transforms_like_velocities=transforms_like_velocities,
            id_particle_dataset_name=id_particle_dataset_name,
            coordinates_dataset_name=coordinates_dataset_name,
            velocities_dataset_name=velocities_dataset_name,
            coordinate_frame_from=coordinate_frame_from,
        )
        self._triage_regions()

    def _triage_regions(self):
        # want 2 modes:
        # optimize_for_sparse:
        #  determine region for each galaxy, group any that share the same region
        # optimize_for_dense:
        #  two overlapping region grids, assign each galaxy to nearest centre in either
        # relatively cheap to evaluate both and then choose the one that minimizes i/o
        target_centres = (
            self.halo_catalogue._bound_centre
        )  # NOT NECESSARILY IN THE BOX?? NEED TO WRAP??
        target_sizes = (
            self.halo_catalogue._bound_aperture
        )  # also handle custom_spatial_offset here
        # SWIFTMask gives us a lightweight interface to metadata & cell metadata
        sm = mask(self._init_args["snapshot_filename"], spatial_only=True)
        # grid should be at least 1 cell in size so that we efficiently group targets
        # in the same grid location
        grid_element_dim = (
            np.max(
                target_sizes[..., np.newaxis] // sm.cell_size[np.newaxis] + 1,
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
        )
        target_grid_indices_aligned = target_grid_indices_aligned.to_value(
            u.dimensionless
        ).astype(int)
        # need a "box wrap" of targets to the far side of the grid for the offset grid:
        # I think there's a bug here? check it...
        # might always force all 3 indices to max instead of only the wrapped one
        target_grid_indices_offset = np.where(
            target_grid_offsets_offset < 0.5,
            grid_dim - 1,
            target_grid_indices_offset,
        )  # complains about missing cosmo_array info
        target_grid_indices_offset = target_grid_indices_offset.to_value(
            u.dimensionless
        ).astype(int)
        # should replace grid_dim.size with dimension from metadata
        target_grid_distances_aligned = np.sqrt(
            np.sum(
                np.power(0.5 * np.ones(grid_dim.size) - target_grid_offsets_aligned, 2),
                axis=1,
            )
        )
        target_grid_distances_offset = np.sqrt(
            np.sum(
                np.power(0.5 * np.ones(grid_dim.size) - target_grid_offsets_offset, 2),
                axis=1,
            )
        )
        assert all(target_grid_distances_aligned < np.sqrt(grid_dim.size))
        assert all(target_grid_distances_offset < np.sqrt(grid_dim.size))
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
        unique_grid_regions = np.unique(target_regions, axis=0)
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
        self._dense_optimized_solution = dict(
            regions=unique_regions,
            region_target_indices=[
                np.argwhere(
                    (target_regions == unique_grid_region).all(axis=1)
                ).squeeze()
                for unique_grid_region in unique_grid_regions
            ],
            cost_min=(
                np.prod(grid_element_dim)  # cost per grid element (optimistic)
                * unique_regions.shape[0]  # number of grid elements
            ),
            cost_max=(
                np.prod(grid_element_dim + 1)  # cost per grid element (pessimistic)
                * unique_regions.shape[0]  # number of grid elements
            ),
        )
        print(
            self._dense_optimized_solution["cost_min"],
            self._dense_optimized_solution["cost_max"],
        )

    def __iter__(self):
        # server creation will later be moved here
        for igalaxy in range(self.halo_catalogue.count):
            # ----------- SERVER START -------------
            # for now we inefficiently create a server for every iteration
            # later the server should be shared by galaxies in a common region
            self.halo_catalogue._mask_multi_galaxy(igalaxy)
            server = SWIFTGalaxy(
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
                _spatial_mask=self.halo_catalogue._get_spatial_mask(
                    self._init_args["snapshot_filename"]
                ),  # replace with common region!
                _extra_mask=None,
            )
            for preload_field in self._init_args["preload"]:
                obj = server
                for attr in preload_field.split("."):
                    obj = getattr(obj, attr)
            # ----------- SERVER END -------------
            swift_galaxy = server[self.halo_catalogue._get_extra_mask(server)]
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
                        "Internal units (length and time) of coordinate_frame_from don't"
                        " match."
                    )
                swift_galaxy._transform(
                    self._init_args["coordinate_frame_from"]._coordinate_like_transform,
                    boost=False,
                )
                swift_galaxy._transform(
                    self._init_args["coordinate_frame_from"]._velocity_like_transform,
                    boost=True,
                )
            elif self._init_args["auto_recentre"]:
                swift_galaxy.recentre(self.halo_catalogue.centre)
                swift_galaxy.recentre_velocity(self.halo_catalogue.velocity_centre)
            swift_galaxy._initialised = True
            yield swift_galaxy
            self.halo_catalogue._unmask_multi_galaxy()
