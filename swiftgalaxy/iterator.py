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
            halo_catalogue=None,
            auto_recentre=False,
            transforms_like_coordinates=transforms_like_coordinates,
            transforms_like_velocities=transforms_like_velocities,
            id_particle_dataset_name=id_particle_dataset_name,
            coordinates_dataset_name=coordinates_dataset_name,
            velocities_dataset_name=velocities_dataset_name,
            _spatial_mask=spatial_mask,
            _extra_mask=None,
        )


class SWIFTGalaxies:

    count: int
    halo_catalogue: _HaloCatalogue

    def __init__(
        self,
        snapshot_filename: str,
        halo_catalogue: _HaloCatalogue,
        auto_recentre: bool = True,
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
            transforms_like_coordinates=transforms_like_coordinates,
            transforms_like_velocities=transforms_like_velocities,
            id_particle_dataset_name=id_particle_dataset_name,
            coordinates_dataset_name=coordinates_dataset_name,
            velocities_dataset_name=velocities_dataset_name,
            coordinate_frame_from=coordinate_frame_from,
        )
        # can inspect halo_catalogue._bound_centre and halo_catalogue._bound_aperture
        # to determine spatial mask regions and triage targets into those regions

    def __iter__(self):
        # server creation will later be moved here
        # data pre-loading will later happend here
        for igalaxy in range(self.halo_catalogue.count):
            # for now we inefficiently create a server for every iteration
            # later the server should be shared by galaxies in a common region
            self.halo_catalogue._mask_multi_galaxy(igalaxy)
            server = _SWIFTGalaxyServer(
                snapshot_filename=self._init_args["snapshot_filename"],
                transforms_like_coordinates=self._init_args[
                    "transforms_like_coordinates"
                ],
                transforms_like_velocities=self._init_args[
                    "transforms_like_velocities"
                ],
                id_particle_dataset_name=self._init_args["id_particle_dataset_name"],
                coordinates_dataset_name=self._init_args["coordinates_dataset_name"],
                velocities_dataset_name=self._init_args["velocities_dataset_name"],
                spatial_mask=self.halo_catalogue._get_spatial_mask(
                    self._init_args["snapshot_filename"]
                ),  # replace with common region!
            )
            swift_galaxy = server[self.halo_catalogue._get_extra_mask(server)]
            swift_galaxy.halo_catalogue = self.halo_catalogue
            # in addition to auto_recentre want to support coordinate_frame_from here:
            if self._init_args["auto_recentre"]:
                swift_galaxy.recentre(self.halo_catalogue.centre)
                swift_galaxy.recentre_velocity(self.halo_catalogue.velocity_centre)
            swift_galaxy._initialised = True
            yield swift_galaxy
            self.halo_catalogue._unmask_multi_galaxy()
