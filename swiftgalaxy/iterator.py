from swiftsimio import mask
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
        optimize_for_sparse: bool = False,
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
        self.optimize_for_sparse = optimize_for_sparse
        self._triage_regions()

    def _triage_regions(self):
        # want 2 modes:
        # optimize_for_sparse:
        #  determine region for each galaxy, group any that share the same region
        # not optimize_for_sparse:
        #  two overlapping region grids, assign each galaxy to nearest centre in either
        target_centres = self.halo_catalogue._bound_centre
        target_sizes = self.halo_catalogue._bound_aperture
        # SWIFTMask gives us a lightweight helper to cell metadata
        sm = mask(self._init_args["snapshot_filename"], spatial_only=True)
        sm.centers
        sm.cell_size

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
