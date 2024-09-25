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
from collections.abc import Collection
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

    def __init__(self, data, mask):
        self._mask_helper_data = data
        self._mask_helper_mask = mask

    def __getattr__(self, attr: str) -> Any:
        return getattr(self._mask_helper_data, attr)[self._mask_helper_mask]

    def __getitem__(self, key: str) -> Any:
        return self.__getattr__(key)

    def __repr__(self):
        return self._mask_helper_data.__repr__()


class _HaloCatalogue(ABC):
    _user_spatial_offsets: Optional[cosmo_array] = None
    _multi_galaxy: bool = False
    _multi_galaxy_mask_index: Optional[int] = None
    _multi_count: int
    _index_attr: Optional[str]

    def __init__(
        self, extra_mask: Optional[Union[str, MaskCollection]] = "bound_only"
    ) -> None:
        self.extra_mask = extra_mask
        self._check_multi()
        self._load()
        return

    def _mask_multi_galaxy(self, index):
        self._multi_galaxy_mask_index = index

    def _unmask_multi_galaxy(self):
        self._multi_galaxy_mask_index = None

    @abstractmethod
    def _load(self) -> None:
        pass

    @abstractmethod
    def _generate_spatial_mask(self, snapshot_filename: str) -> SWIFTMask:
        # return spatial_mask
        pass

    def _get_user_spatial_mask(self, snapshot_filename: str) -> SWIFTMask:
        sm = mask(snapshot_filename, spatial_only=True)
        region = self._user_spatial_offsets
        if region is not None:
            for ax in range(3):
                region[ax] = (
                    [
                        self.centre[ax] + region[ax][0],
                        self.centre[ax] + region[ax][1],
                    ]
                    if region[ax] is not None
                    else None
                )
            sm.constrain_spatial(region)
        return sm

    def _get_spatial_mask(self, snapshot_filename: str) -> MaskCollection:
        if self._multi_galaxy and self._multi_galaxy_mask_index is None:
            raise RuntimeError(
                "Halo catalogue has multiple galaxies and is not currently masked."
            )
        return self._generate_spatial_mask(snapshot_filename)

    @property
    def count(self) -> int:
        if self._multi_galaxy and self._multi_galaxy_mask_index is None:
            return self._multi_count
        else:
            return 1

    @abstractmethod
    def _generate_bound_only_mask(self, SG: "SWIFTGalaxy") -> MaskCollection:
        # return _extra_mask
        pass

    def _get_extra_mask(self, SG: "SWIFTGalaxy") -> MaskCollection:
        if self.extra_mask == "bound_only":
            if self._multi_galaxy and self._multi_galaxy_mask_index is None:
                raise RuntimeError(
                    "Halo catalogue has multiple galaxies and is not currently masked."
                )
            return self._generate_bound_only_mask(SG)
        elif self.extra_mask is None:
            return MaskCollection(**{k: None for k in SG.metadata.present_group_names})
        else:
            # Keep user provided mask. If no mask provided for a particle type
            # use None (no mask).
            return MaskCollection(
                **{
                    name: getattr(self.extra_mask, name, None)
                    for name in SG.metadata.present_group_names
                }
            )

    @property
    @abstractmethod
    def centre(self) -> cosmo_array:
        # return halo centre
        pass

    @property
    @abstractmethod
    def velocity_centre(self) -> cosmo_array:
        # return halo velocity centre
        pass

    @property
    @abstractmethod
    def _region_centre(self) -> cosmo_array:
        # return a centre for the spatial region
        pass

    @property
    @abstractmethod
    def _region_aperture(self) -> cosmo_array:
        # return a size for the spatial region
        pass

    @abstractmethod
    def _get_preload_fields(self, SG: "SWIFTGalaxy") -> Set[str]:
        # define fields that need preloading to compute masks
        pass

    def _mask_index(self):
        index = getattr(self, self._index_attr)
        return (
            index[self._multi_galaxy_mask_index]
            if self._multi_galaxy_mask_index is not None
            else index
        )

    def _check_multi(self):
        if self._index_attr is None:
            # for now this means we're in Standalone
            if self._centre.ndim > 1:
                self._multi_galaxy = True
                self._multi_count = len(self._centre)
            else:
                self._multi_galaxy = False
                self._multi_count = 1
        else:
            index = getattr(self, self._index_attr)
            if isinstance(index, Collection):
                self._multi_galaxy = True
                if not isinstance(index, int):  # placate mypy
                    self._multi_count = len(index)
            else:
                self._multi_galaxy = False
                self._multi_count = 1
        if self._multi_galaxy:
            assert self.extra_mask in (None, "bound_only")

    def __getattr__(self, attr: str) -> Any:
        # Invoked if attribute not found.
        # Use to expose the masked catalogue.
        if attr == "_catalogue":  # guard infinite recursion
            try:
                return object.__getattribute__(self, "_catalogue")
            except AttributeError:
                return None
        obj = getattr(self._catalogue, attr)
        if self._multi_galaxy_mask_index is not None:
            return _MaskHelper(obj, self._multi_galaxy_mask_index)
        else:
            return obj


class SOAP(_HaloCatalogue):
    """
    Interface to SOAP halo catalogues for use with :mod:`swiftgalaxy`.

    Takes a set of ``SOAP`` output files and configuration options and provides an
    interface that :mod:`swiftgalaxy` understands. Also exposes the galaxy properties
    computed by ``SOAP`` for a single object of interest through the :mod:`swiftsimio`
    interface.

    Parameters
    ----------

    soap_file: ``Optional[str]``, default: ``None``
        The filename of a SOAP catalogue file, possibly including the path.
    soap_index: ``Optional[int]``, default: ``None``
        The position (row) in the SOAP catalogue corresponding to the object of interest.
    extra_mask: ``Union[str, MaskCollection]``, default: ``"bound_only"``
        Mask to apply to particles after spatial masking. If ``"bound_only"``,
        then the galaxy is masked to include only the gravitationally bound
        particles as determined by ``SOAP``. A user-defined mask can also be provided
        as an an object (such as a :class:`swiftgalaxy.masks.MaskCollection`) that has
        attributes with names corresponding to present particle names (e.g. gas,
        dark_matter, etc.), each containing a mask.
    centre_type: ``str``, default: ``"input_halos.halo_centre"``
        Type of centre, chosen from those provided by ``SOAP``. This should be
        expressed as a string analogous to what would be written in
        :mod:`swiftsimio` code (or :mod:`swiftgalaxy`) to access that property in the
        SOAP catalogue. The default takes the ``"input_halos.halo_centre"`` (usually
        the centre of potential, e.g. the HBT+ halo finder defines it in this way;
        another option amongst many more is ``"bound_subhalo.centre_of_mass"``.
    velocity_centre_type: ``str``, default: ``"bound_subhalo.centre_of_mass_velocity"``
        Type of velocity centre, chosen from those provided by ``SOAP``. This should be
        expressed as a string analogous to what would be written in
        :mod:`swiftsimio` code (or :mod:`swiftgalaxy`) to access that property in the
        SOAP catalogue. The default takes the ``"bound_subhalo.centre_of_mass_velocity"``;
        note that there is no velocity corresponding to the centre of potential
        (``"input_halos.halo_centre_velocity"`` is not defined). Another useful option
        could be ``"exclusive_sphere_1kpc.centre_of_mass_velocity"`` to choose the
        velocity of bound particles in the central 1 kpc.
    custom_spatial_offsets: ``Optional[cosmo_array]``, default: ``None``
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
    _soap_index: Union[int, Collection[int]]
    centre_type: str
    velocity_centre_type: str
    _catalogue: SWIFTDataset
    _index_attr = "_soap_index"

    def __init__(
        self,
        soap_file: Optional[str] = None,
        soap_index: Optional[Union[int, Collection[int]]] = None,
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
        sm = mask(self.soap_file, spatial_only=not self._multi_galaxy)
        if self._multi_galaxy:
            sm.constrain_indices(self._soap_index)
        else:
            sm.constrain_index(self._soap_index)
        self._catalogue = SWIFTDataset(
            self.soap_file,
            mask=sm,
        )

    @property
    def soap_index(self):
        return self._mask_index()

    @property
    def _region_centre(self) -> cosmo_array:
        # should not need to box wrap here but there's a bug upstream
        boxsize = self._catalogue.metadata.boxsize
        coords = self.bound_subhalo.centre_of_mass.squeeze()
        return coords % boxsize

    @property
    def _region_aperture(self) -> cosmo_array:
        return self.bound_subhalo.enclose_radius.squeeze()

    def _get_preload_fields(self, SG: "SWIFTGalaxy") -> Set[str]:
        if self.extra_mask == "bound_only":
            return {
                f"{group_name}.group_nr_bound"
                for group_name in SG.metadata.present_group_names
            }
        else:
            return set()

    def _generate_spatial_mask(self, snapshot_filename: str) -> SWIFTMask:
        pos, rmax = (self._region_centre, self._region_aperture)
        sm = mask(snapshot_filename, spatial_only=True)
        load_region = cosmo_array([pos - rmax, pos + rmax]).T
        sm.constrain_spatial(load_region)
        return sm

    def _generate_bound_only_mask(self, SG: "SWIFTGalaxy") -> MaskCollection:
        # The halo_catalogue_index is the index into the full (HBT+ not SOAP) catalogue;
        # this is what group_nr_bound matches against.
        masks = MaskCollection(
            **{
                group_name: getattr(
                    SG, group_name
                )._particle_dataset.group_nr_bound.to_value(u.dimensionless)
                == self.input_halos.halo_catalogue_index.to_value(u.dimensionless)
                for group_name in SG.metadata.present_group_names
            }
        )
        if not self._multi_galaxy:
            for group_name in SG.metadata.present_group_names:
                del getattr(SG, group_name)._particle_dataset.group_nr_bound
        return masks

    @property
    def centre(self) -> cosmo_array:
        obj = self._catalogue
        for attr in self.centre_type.split("."):
            obj = getattr(obj, attr)
        if self._multi_galaxy_mask_index is not None:
            return obj[self._multi_galaxy_mask_index]
        return obj.squeeze()

    @property
    def velocity_centre(self) -> cosmo_array:
        obj = self._catalogue
        for attr in self.velocity_centre_type.split("."):
            obj = getattr(obj, attr)
        if self._multi_galaxy_mask_index is not None:
            return obj[self._multi_galaxy_mask_index]
        return obj.squeeze()

    def __repr__(self) -> str:
        # Expose the catalogue __repr__ for interactive use.
        return self._catalogue.__repr__()


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

    velociraptor_filebase: ``str``
        The initial part of the velociraptor filenames (possibly including
        path), e.g. if there is a :file:`{halos}.properties` file, pass
        ``halos`` as this argument. Provide this or `velociraptor_files`,
        not both.

    velociraptor_files: ``dict[str]``
        A dictionary containing the names of the velociraptor files (possibly
        including paths). There should be two entries, with keys `properties`
        and `catalog_groups` containing locations of the `{halos}.properties`
        and `{halos}.catalog_groups` files, respectively. Provide this or
        `velociraptor_filebase`, not both.

    halo_index: ``int``
        Position of the object of interest in the catalogue arrays.

    extra_mask: ``Union[str, MaskCollection]``, default: ``"bound_only"``
        Mask to apply to particles after spatial masking. If ``"bound_only"``,
        then the galaxy is masked to include only the gravitationally bound
        particles as determined by :mod:`velociraptor`. A user-defined mask
        can also be provided as an an object (such as a
        :class:`swiftgalaxy.masks.MaskCollection`) that has attributes with
        names corresponding to present particle names (e.g. gas, dark_matter,
        etc.), each containing a mask.

    centre_type: ``str``, default: ``"minpot"``
        Type of centre, chosen from those provided by :mod:`velociraptor`.
        Default is the position of the particle with the minimum potential,
        ``"minpot"``; other possibilities may include ``""``, ``"_gas"``,
        ``"_star"``, ``"mbp"`` (most bound particle).

    custom_spatial_offsets: ``Optional[cosmo_array]``, default: ``None``
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
    _halo_index: Union[int, Collection[int]]
    centre_type: str
    velocity_centre_type: str
    _catalogue: "VelociraptorCatalogue"
    _index_attr = "_halo_index"

    def __init__(
        self,
        velociraptor_filebase: Optional[str] = None,
        velociraptor_files: Optional[dict] = None,
        halo_index: Optional[Union[int, Collection[int]]] = None,
        extra_mask: Union[str, MaskCollection] = "bound_only",
        centre_type: str = "minpot",  # _gas _star mbp minpot
        velociraptor_suffix: str = "",
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
        import h5py
        from velociraptor.catalogue.catalogue import Catalogue as VelociraptorCatalogue
        from velociraptor import load as load_catalogue
        from velociraptor.particles import load_groups

        with h5py.File(self.velociraptor_files["properties"]) as propfile:
            self.scale_factor = (
                float(propfile["SimulationInfo"].attrs["ScaleFactor"])
                if propfile["SimulationInfo"].attrs["Cosmological_Sim"]
                else 1.0
            )

        self._catalogue: "VelociraptorCatalogue" = load_catalogue(
            self.velociraptor_files["properties"], mask=self._halo_index
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
        from velociraptor.swift.swift import generate_spatial_mask

        # super()._get_spatial_mask guards getting here in multi-galaxy mode
        # without a self._multi_galaxy_mask_index
        return generate_spatial_mask(
            (
                self._particles[self._multi_galaxy_mask_index]
                if self._multi_galaxy_mask_index is not None
                else self._particles
            ),
            snapshot_filename,
        )

    def _generate_bound_only_mask(self, SG: "SWIFTGalaxy") -> MaskCollection:
        from velociraptor.swift.swift import generate_bound_mask

        return MaskCollection(
            **generate_bound_mask(
                SG,
                (
                    self._particles[self._multi_galaxy_mask_index]
                    if self._multi_galaxy_mask_index is not None
                    else self._particles
                ),
            )._asdict()
        )

    @property
    def halo_index(self):
        return self._mask_index()

    @property
    def _region_centre(self) -> cosmo_array:
        length_factor = (
            self._particles[0].groups_instance.catalogue.units.a
            if not self._particles[0].groups_instance.catalogue.units.comoving
            else 1.0
        )
        if self._multi_galaxy_mask_index is None:
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
                    self._particles[self._multi_galaxy_mask_index].x.to_value(u.Mpc)
                    / length_factor,
                    self._particles[self._multi_galaxy_mask_index].y.to_value(u.Mpc)
                    / length_factor,
                    self._particles[self._multi_galaxy_mask_index].z.to_value(u.Mpc)
                    / length_factor,
                ],
                u.Mpc,
                comoving=True,
                cosmo_factor=cosmo_factor(a**1, length_factor),
            )

    @property
    def _region_aperture(self) -> cosmo_array:
        length_factor = (
            self._particles[0].groups_instance.catalogue.units.a
            if not self._particles[0].groups_instance.catalogue.units.comoving
            else 1.0
        )
        if self._multi_galaxy_mask_index is None:
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
                self._particles[self._multi_galaxy_mask_index].r_size.to_value(u.Mpc)
                / length_factor,
                u.Mpc,
                comoving=True,
                cosmo_factor=cosmo_factor(a**1, length_factor),
            )

    def _get_preload_fields(self, SG: "SWIFTGalaxy") -> Set[str]:
        if self.extra_mask == "bound_only":
            return {
                f"{group_name}.particle_ids"
                for group_name in SG.metadata.present_group_names
            }
        else:
            return set()

    @property
    def centre(self) -> cosmo_array:
        """
        Obtain the centre specified by the ``centre_type`` from the halo catalogue.

        Returns
        -------
        centre: :class:`~swiftsimio.objects.cosmo_array`
            The centre provided by the halo catalogue.
        """
        # According to Velociraptor documentation:
        if self.centre_type in ("_gas", "_stars"):
            # {XYZ}c_gas and {XYZ}c_stars are relative to {XYZ}c
            relative_to = np.hstack(
                [getattr(self._catalogue.positions, "{:s}c".format(c)) for c in "xyz"]
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
                        getattr(
                            self._catalogue.positions,
                            "{:s}c{:s}".format(c, self.centre_type),
                        )
                        for c in "xyz"
                    ]
                ).T
            ).to_value(u.Mpc),
            u.Mpc,
            comoving=False,  # velociraptor gives physical centres!
            cosmo_factor=cosmo_factor(a**1, self.scale_factor),
        ).to_comoving()
        if self._multi_galaxy and self._multi_galaxy_mask_index is None:
            return centre
        elif self._multi_galaxy and self._multi_galaxy_mask_index is not None:
            return centre[self._multi_galaxy_mask_index]
        else:
            return centre.squeeze()

    @property
    def velocity_centre(self) -> cosmo_array:
        """
        Obtain the velocity centre specified by the ``centre_type`` from the halo
        catalogue.

        Returns
        -------
        velocity_centre: :class:`~swiftsimio.objects.cosmo_array`
            The velocity centre provided by the halo catalogue.
        """
        # According to Velociraptor documentation:
        if self.centre_type in ("_gas", "_stars"):
            # V{XYZ}c_gas and V{XYZ}c_stars are relative to {XYZ}c
            relative_to = np.hstack(
                [getattr(self._catalogue.velocities, "v{:s}c".format(c)) for c in "xyz"]
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
                        getattr(
                            self._catalogue.velocities,
                            "v{:s}c{:s}".format(c, self.centre_type),
                        )
                        for c in "xyz"
                    ]
                ).T
            ).to_value(u.km / u.s),
            u.km / u.s,
            comoving=False,
            cosmo_factor=cosmo_factor(a**0, self.scale_factor),
        ).to_comoving()
        if self._multi_galaxy and self._multi_galaxy_mask_index is None:
            return vcentre
        elif self._multi_galaxy and self._multi_galaxy_mask_index is not None:
            return vcentre[self._multi_galaxy_mask_index]
        else:
            return vcentre.squeeze()

    def __repr__(self) -> str:
        # Expose the catalogue __repr__ for interactive use.
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

    caesar_file: ``str``
        The catalogue file (hdf5 format) output by caesar.

    group_type: ``str``
        The category of the object of interest, either ``"halo"`` or ``"galaxy"``.

    group_index: ``int``
        Position of the object of interest in the catalogue arrays.

    extra_mask: ``Union[str, MaskCollection]``, default: ``"bound_only"``
        Mask to apply to particles after spatial masking. If ``"bound_only"``,
        then the galaxy is masked to include only the gravitationally bound
        particles as provided by :mod:`caesar`. A user-defined mask can also be
        as an an object (such as a :class:`swiftgalaxy.masks.MaskCollection`) that has
        attributes with names corresponding to present particle names (e.g. gas,
        dark_matter, etc.), each containing a mask.

    centre_type: ``str``, default: ``"minpot"``
        Type of centre, chosen from those provided by :mod:`caesar`.
        Default is the position of the particle with the minimum potential,
        ``"minpot"``, alternatively ``""`` can be used for the centre of mass.

    custom_spatial_offsets: ``Optional[cosmo_array]``, default: ``None``
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
        :mod:`loader.CAESAR` only supports index access to catalogue lists, not
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
    _group_index: Union[int, Collection[int]]
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
        group_index: Optional[Union[int, Collection[int]]] = None,
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
        # any non-trivial io/calculation at initialisation time goes here
        pass

    def _generate_spatial_mask(self, snapshot_filename: str) -> SWIFTMask:
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

    def _generate_bound_only_mask(self, SG: "SWIFTGalaxy") -> MaskCollection:
        def in_one_of_ranges(
            ints: NDArray[np.int_],
            int_ranges: NDArray[np.int_],
        ) -> NDArray[np.bool_]:
            """
            Produces a boolean mask corresponding to `ints`. For each element in `ints`,
            the mask is `True` if the value is between (at least) one of the pairs of
            integers in `int_ranges`. This is potentially memory intensive with a
            footprint proportional to ints.size * int_ranges.size.
            """
            return np.logical_and(
                ints >= int_ranges[:, 0, np.newaxis],
                ints < int_ranges[:, 1, np.newaxis],
            ).any(axis=0)

        cat = self._mask_catalogue()
        null_slice = np.s_[:0]  # mask that selects no particles
        if hasattr(cat, "glist"):
            gas_mask = cat.glist
            gas_mask = gas_mask[in_one_of_ranges(gas_mask, SG.mask.gas)]
            gas_mask = np.isin(
                np.concatenate([np.arange(start, end) for start, end in SG.mask.gas]),
                gas_mask,
            )
        else:
            gas_mask = null_slice
        # seems like name could be dmlist or dlist?
        if hasattr(cat, "dlist") or hasattr(cat, "dmlist"):
            dark_matter_mask = getattr(cat, "dlist", cat.dmlist)
            dark_matter_mask = dark_matter_mask[
                in_one_of_ranges(dark_matter_mask, SG.mask.dark_matter)
            ]
            dark_matter_mask = np.isin(
                np.concatenate(
                    [np.arange(start, end) for start, end in SG.mask.dark_matter]
                ),
                dark_matter_mask,
            )
        else:
            dark_matter_mask = null_slice

        if hasattr(cat, "slist"):
            stars_mask = cat.slist
            stars_mask = stars_mask[in_one_of_ranges(stars_mask, SG.mask.stars)]
            stars_mask = np.isin(
                np.concatenate([np.arange(start, end) for start, end in SG.mask.stars]),
                stars_mask,
            )
        else:
            stars_mask = null_slice
        if hasattr(cat, "bhlist"):
            black_holes_mask = cat.bhlist
            black_holes_mask = black_holes_mask[
                in_one_of_ranges(black_holes_mask, SG.mask.black_holes)
            ]
            black_holes_mask = np.isin(
                np.concatenate(
                    [np.arange(start, end) for start, end in SG.mask.black_holes]
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
    def group_index(self):
        return self._mask_index()

    @property
    def _region_centre(self) -> cosmo_array:
        # return a centre for the spatial region
        cats = [self._catalogue] if not self._multi_galaxy else self._catalogue
        assert isinstance(cats, List)  # placate mypy
        if self._multi_galaxy_mask_index is None:
            return cosmo_array(
                [cat.pos.to(u.kpc) for cat in cats],  # maybe comoving, ensure physical
                comoving=False,
                cosmo_factor=cosmo_factor(a**1, self._caesar.simulation.scale_factor),
            ).to_comoving()
        else:
            return cosmo_array(
                cats[self._multi_galaxy_mask_index].pos.to(
                    u.kpc
                ),  # maybe comoving, ensure physical
                comoving=False,
                cosmo_factor=cosmo_factor(a**1, self._caesar.simulation.scale_factor),
            ).to_comoving()

    @property
    def _region_aperture(self) -> cosmo_array:
        cats = [self._catalogue] if not self._multi_galaxy else self._catalogue
        assert isinstance(cats, List)  # placate mypy
        # return a size for the spatial region
        if "total_rmax" in cats[0].radii.keys():
            # spatial extent information is present
            if self._multi_galaxy_mask_index is None:
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
                    cats[self._multi_galaxy_mask_index]
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

    def _get_preload_fields(self, SG: "SWIFTGalaxy") -> Set[str]:
        # define fields that need preloading to compute masks
        return set()

    def _mask_catalogue(self) -> Union["CaesarHalo", "CaesarGalaxy"]:
        if self._multi_galaxy and self._multi_galaxy_mask_index is not None:
            cat = self._catalogue[self._multi_galaxy_mask_index]
        elif self._multi_galaxy and self._multi_galaxy_mask_index is None:
            raise RuntimeError("Tried to mask catalogue without mask index!")
        else:
            cat = self._catalogue
        return cat

    @property
    def centre(self) -> cosmo_array:
        """
        Obtain the centre specified by the ``centre_type`` from the halo catalogue.

        Returns
        -------
        centre: :class:`~swiftsimio.objects.cosmo_array`
            The centre provided by the halo catalogue.
        """
        centre_attr = {"": "pos", "minpot": "minpotpos"}[self.centre_type]
        if self._multi_galaxy_mask_index is not None:
            return cosmo_array(
                getattr(self._catalogue[self._multi_galaxy_mask_index], centre_attr).to(
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
        Obtain the velocity centre specified by the ``centre_type`` from the halo
        catalogue.

        Returns
        -------
        velocity_centre: :class:`~swiftsimio.objects.cosmo_array`
            The velocity centre provided by the halo catalogue.
        """
        vcentre_attr = {"": "vel", "minpot": "minpotvel"}[self.centre_type]
        if self._multi_galaxy_mask_index is not None:
            return cosmo_array(
                getattr(
                    self._catalogue[self._multi_galaxy_mask_index], vcentre_attr
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
        # Invoked if attribute not found.
        # Use to expose the masked catalogue.
        if attr == "_catalogue":  # guard infinite recursion
            try:
                return object.__getattribute__(self, "_catalogue")
            except AttributeError:
                return None
        if self._multi_galaxy_mask_index is not None:
            return getattr(self._catalogue[self._multi_galaxy_mask_index], attr)
        elif self._multi_galaxy and self._multi_galaxy_mask_index is None:
            return [getattr(cat, attr) for cat in self._catalogue]
        else:
            return getattr(self._catalogue, attr)

    def __repr__(self) -> str:
        return self._catalogue.__repr__()


class Standalone(_HaloCatalogue):
    """
    A bare-bones tool to initialize a :class:`~swiftgalaxy.reader.SWIFTGalaxy`
    without an associated halo catalogue.

    Provides an interface to specify the minimum required information to
    instantiate a :class:`~swiftgalaxy.reader.SWIFTGalaxy`.

    Parameters
    ----------

    centre: ``Optional[cosmo_array]``, default: ``None``
        A value for this parameter is required, and must have units of length.
        Specifies the geometric centre in simulation coordinates. Particle
        coordinates will be shifted such that this position is located at (0, 0, 0)
        and if the boundary is periodic it will be wrapped to place the origin at
        the centre.

    velocity_centre: ``Optional[cosmo_array]``, default: ``None``
        A value for this parameter is required, and must have units of speed.
        Specifies the reference velocity relative to the simulation frame. Particle
        velocities will be shifted such that a particle with the specified velocity
        in the simulation frame will have zero velocity in the
        :class:`~swiftgalaxy.reader.SWIFTGalaxy` frame.

    spatial_offsets: ``Optional[cosmo_array]``, default: ``None``
        Offsets along each axis to select a spatial region around the ``centre``.
        May be used in conjunction with ``extra_mask``, for example to select all
        simulation particles in an aperture around the object of interest (see
        'Masking' section of documentation for a cookbook example). Provide a pair
        of offsets from the object's centre along each axis to define the region,
        for example for a cube extending +/- 1 Mpc from the centre:
        ``cosmo_array([[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]], u.Mpc)``.

    extra_mask: ``Optional[Union[str, MaskCollection]]``, default: ``None``
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
        pass

    def _generate_spatial_mask(self, snapshot_filename: str) -> SWIFTMask:
        # if we're here then the user didn't provide a mask, read the whole box
        sm = mask(snapshot_filename, spatial_only=True)
        boxsize = sm.metadata.boxsize
        region = [[0.0 * b, 1.0 * b] for b in boxsize]
        sm.constrain_spatial(region)
        return sm

    def _generate_bound_only_mask(self, SG: "SWIFTGalaxy") -> MaskCollection:
        raise NotImplementedError  # guarded in initialisation, should not reach here

    @property
    def _region_centre(self) -> cosmo_array:
        # return a centre for the spatial region
        if self._multi_galaxy_mask_index is None:
            return self._centre
        else:
            return self._centre[self._multi_galaxy_mask_index]

    @property
    def _region_aperture(self) -> cosmo_array:
        # return a size for the spatial region
        if self._user_spatial_offsets is not None:
            if self._multi_galaxy_mask_index is None:
                return np.repeat(
                    np.max(np.abs(self._user_spatial_offsets)), len(self._centre)
                )
            else:
                return np.max(np.abs(self._user_spatial_offsets))
        else:
            # should never reach here (guarded in initialization)
            raise NotImplementedError

    def _get_preload_fields(self, SG: "SWIFTGalaxy") -> Set[str]:
        # define fields that need preloading to compute masks
        return set()

    @property
    def centre(self) -> cosmo_array:
        """
        Obtain the centre specified at initialisation.
        """
        if self._multi_galaxy_mask_index is not None:
            return self._centre[self._multi_galaxy_mask_index]
        return self._centre

    @property
    def velocity_centre(self) -> cosmo_array:
        """
        Obtain the velocity centre specified at initialisation.
        """
        if self._multi_galaxy_mask_index is not None:
            return self._velocity_centre[self._multi_galaxy_mask_index]
        return self._velocity_centre
