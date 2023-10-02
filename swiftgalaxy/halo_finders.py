"""
This module contains classes providing interfaces to halo finders used with
SWIFT so that :mod:`swiftgalaxy` can obtain the information it requires in a
streamlined way. Currently only the Velociraptor_ halo finder is supported, but
support for other halo finders (e.g. `SOAP`, `HBT+`_) is planned.

.. _Velociraptor: https://ui.adsabs.harvard.edu/abs/2019PASA...36...21E/\
abstract
.. _HBT+: https://ui.adsabs.harvard.edu/abs/2018MNRAS.474..604H/abstract
"""

from abc import ABC, abstractmethod
import numpy as np
import unyt as u
from swiftsimio import SWIFTMetadata, SWIFTUnits, SWIFTMask
from swiftgalaxy.masks import MaskCollection
from swiftsimio.objects import cosmo_array, cosmo_factor, a

from typing import Any, Union, Optional, TYPE_CHECKING
from numpy.typing import NDArray

if TYPE_CHECKING:
    from swiftgalaxy.reader import SWIFTGalaxy


class _HaloFinder(ABC):
    def __init__(self, extra_mask: Union[str, MaskCollection] = "bound_only") -> None:
        self.extra_mask = extra_mask
        self._load()
        return

    @abstractmethod
    def _load(self) -> None:
        pass

    @abstractmethod
    def _get_spatial_mask(self, snapshot_filename: str) -> SWIFTMask:
        # return _spatial_mask
        pass

    @abstractmethod
    def _generate_bound_only_mask(self, SG: "SWIFTGalaxy") -> MaskCollection:
        # return _extra_mask
        pass

    def _get_extra_mask(self, SG: "SWIFTGalaxy") -> MaskCollection:
        if self.extra_mask == "bound_only":
            return self._generate_bound_only_mask(SG)
        elif self.extra_mask is None:
            return MaskCollection(
                **{k: None for k in SG.metadata.present_particle_names}
            )
        else:
            # Keep user provided mask. If no mask provided for a particle type
            # use None (no mask).
            return MaskCollection(
                **{
                    name: getattr(self.extra_mask, name, None)
                    for name in SG.metadata.present_particle_names
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

    # In addition, it is recommended to expose the properties computed
    # by the halo finder through this object, masked to the values
    # corresponding to the object of interest. It probably makes sense
    # to match the syntax used to the usual syntax for the halo finder
    # in question? See e.g. implementation in __getattr__ in Velociraptor
    # subclass below.


class Velociraptor(_HaloFinder):

    """
    Interface to velociraptor halo catalogues for use with :mod:`swiftgalaxy`.

    Takes a set of :mod:`velociraptor` output files and configuration options
    and provides an interface that :mod:`swiftgalaxy` understands. Also exposes
    the halo/galaxy properties computed by :mod:`velociraptor` for a single
    object of interest with the same interface_ provided by the
    :mod:`velociraptor` python package. Reading of properties is done
    on-the-fly, and only rows corresponding to the object of interest are read
    from disk.

    .. _interface: https://velociraptor-python.readthedocs.io/en/latest/

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

    def __init__(
        self,
        velociraptor_filebase: Optional[str] = None,
        velociraptor_files: Optional[dict] = None,
        halo_index: Optional[int] = None,
        extra_mask: Union[str, MaskCollection] = "bound_only",
        centre_type: str = "minpot",  # _gas _star mbp minpot
        velociraptor_suffix: str = "",
    ) -> None:
        from velociraptor.catalogue.catalogue import Catalogue

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
            self.halo_index: int = halo_index
        self.centre_type: str = centre_type
        super().__init__(extra_mask=extra_mask)
        # currently velociraptor_python works with a halo index, not halo_id
        # self.catalogue_mask = (catalogue.ids.id == halo_id).nonzero()
        return

    def _load(self) -> None:
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
            self.velociraptor_files["properties"], mask=self.halo_index
        )
        groups = load_groups(
            self.velociraptor_files["catalog_groups"],
            catalogue=load_catalogue(self.velociraptor_files["properties"]),
        )
        self._particles, unbound_particles = groups.extract_halo(
            halo_index=self.halo_index
        )
        return

    def _get_spatial_mask(self, snapshot_filename: str) -> SWIFTMask:
        from velociraptor.swift.swift import generate_spatial_mask

        return generate_spatial_mask(self._particles, snapshot_filename)

    def _generate_bound_only_mask(self, SG: "SWIFTGalaxy") -> MaskCollection:
        from velociraptor.swift.swift import generate_bound_mask

        return MaskCollection(**generate_bound_mask(SG, self._particles)._asdict())

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
            relative_to = u.uhstack(
                [getattr(self._catalogue.positions, "{:s}c".format(c)) for c in "xyz"]
            )
        else:
            # {XYZ}cmbp, {XYZ}cminpot and {XYZ}c are absolute
            relative_to = cosmo_array([0.0, 0.0, 0.0], u.Mpc)
        return cosmo_array(
            relative_to
            + u.uhstack(
                [
                    getattr(
                        self._catalogue.positions,
                        "{:s}c{:s}".format(c, self.centre_type),
                    )
                    for c in "xyz"
                ]
            ),
            comoving=False,  # velociraptor gives physical centres!
            cosmo_factor=cosmo_factor(a**1, self.scale_factor),
        ).to_comoving()

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
            relative_to = u.uhstack(
                [getattr(self._catalogue.velocities, "v{:s}c".format(c)) for c in "xyz"]
            )
        else:
            # V{XYZ}cmbp, V{XYZ}cminpot and V{XYZ}c are absolute
            relative_to = cosmo_array([0.0, 0.0, 0.0], u.km / u.s)
        return cosmo_array(
            relative_to
            + u.uhstack(
                [
                    getattr(
                        self._catalogue.velocities,
                        "v{:s}c{:s}".format(c, self.centre_type),
                    )
                    for c in "xyz"
                ]
            ),
            comoving=False,
            cosmo_factor=cosmo_factor(a**0, self.scale_factor),
        ).to_comoving()

    def __getattr__(self, attr: str) -> Any:
        # Invoked if attribute not found.
        # Use to expose the masked catalogue.
        if attr == "_catalogue":  # guard infinite recursion
            return object.__getattribute__(self, "_catalogue")
        return getattr(self._catalogue, attr)

    def __repr__(self) -> str:
        # Expose the catalogue __repr__ for interactive use.
        return self._catalogue.__repr__()


class Caesar(_HaloFinder):

    """
    Interface to caesar halo catalogues for use with :mod:`swiftgalaxy`.

    Takes a :mod:`caesar` output file and configuration options and provides
    an interface that :mod:`swiftgalaxy` understands. Also exposes the halo/galaxy
    properties computed by :mod:`velociraptor` for a single object of interest with
    the same interface_ provided by the :mod:`caesar` python package. Reading of
    properties is done on-the-fly, and only rows corresponding to the object of
    interest are read from disk.

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

    Notes
    -----

    .. note::
        :mod:`caesar` only supports index access to catalogue arrays, not
        identifier access. This means that the ``group_index`` is simply the
        position of the object of interest in the catalogue arrays.

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

    def __init__(
        self,
        caesar_file: Optional[str] = None,
        group_type: Optional[str] = None,  # halos galaxies
        group_index: Optional[int] = None,
        centre_type: str = "minpot",  # "" "minpot"
        extra_mask: Union[str, MaskCollection] = "bound_only",
    ) -> None:
        import caesar
        import logging
        from yt.utilities import logger as yt_logger

        log_level = logging.getLogger("yt").level  # cache the log level before we start
        yt_logger.set_log_level("warning")  # disable INFO log messages
        self._caesar = caesar.load(caesar_file)
        yt_logger.set_log_level(log_level)  # restore old log level

        valid_group_types = dict(halo="halos", galaxy="galaxies")
        if group_type in valid_group_types:
            self._group = getattr(self._caesar, valid_group_types[group_type])[
                group_index
            ]
        else:
            raise ValueError(
                "group_type required, valid values are 'halo' or 'galaxy'."
            )
        self.group_type: str = group_type
        if group_index is None:
            raise ValueError("group_index (int) required.")
        else:
            self.group_index: int = group_index

        self.centre_type = centre_type

        super().__init__(extra_mask=extra_mask)
        return

    def _load(self) -> None:
        # any non-trivial io/calculation at initialisation time goes here
        pass

    def _get_spatial_mask(self, snapshot_filename: str) -> SWIFTMask:
        sm = SWIFTMask(
            SWIFTMetadata(snapshot_filename, SWIFTUnits(snapshot_filename)),
            spatial_only=True,
        )
        # no guaranteed way to get sub-region containing all group particles
        # from information in caesar outputs - requested a new property with
        # maximum particle radius, in the meantime we just the whole box:
        boxsize = sm.metadata.boxsize
        # presumably max radius will be from standard centre, so should add offset between
        # minpot centre and normal centre if using minpot centre to be conservative
        # load_region = [[0.0 * b, 1.0 * b] for b in boxsize]
        load_region = [
            [0.0 * boxsize[0], 1.0 * boxsize[0]],
            [0.0 * boxsize[1], 1.0 * boxsize[1]],
            [0.0 * boxsize[2], 1.0 * boxsize[2]],
        ]
        sm.constrain_spatial(load_region)
        return sm

    def _generate_bound_only_mask(self, SG: "SWIFTGalaxy") -> MaskCollection:
        def in_one_of_ranges(
            ints: NDArray[np.int_],
            int_ranges: NDArray[np.int_],
        ):
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

        gas_mask = getattr(self._group, "glist", None)
        if gas_mask is not None:
            gas_mask = gas_mask[in_one_of_ranges(gas_mask, SG.mask.gas)]
            gas_mask = np.isin(
                np.concatenate([np.arange(start, end) for start, end in SG.mask.gas]),
                gas_mask,
            )
        # seems like name could be dmlist or dlist?
        if hasattr(self._group, "dlist"):
            dark_matter_mask = self._group.dlist
        elif hasattr(self._group, "dmlist"):
            dark_matter_mask = self._group.dmlist
        else:
            dark_matter_mask = np.array([])
        if dark_matter_mask is not None:
            dark_matter_mask = dark_matter_mask[
                in_one_of_ranges(dark_matter_mask, SG.mask.dark_matter)
            ]
            dark_matter_mask = np.isin(
                np.concatenate(
                    [np.arange(start, end) for start, end in SG.mask.dark_matter]
                ),
                dark_matter_mask,
            )
        stars_mask = getattr(self._group, "slist", None)
        if stars_mask is not None:
            stars_mask = stars_mask[in_one_of_ranges(stars_mask, SG.mask.stars)]
            stars_mask = np.isin(
                np.concatenate([np.arange(start, end) for start, end in SG.mask.stars]),
                stars_mask,
            )
        black_holes_mask = getattr(self._group, "bhlist", None)
        if black_holes_mask is not None:
            black_holes_mask = black_holes_mask[
                in_one_of_ranges(black_holes_mask, SG.mask.black_holes)
            ]
            black_holes_mask = np.isin(
                np.concatenate(
                    [np.arange(start, end) for start, end in SG.mask.black_holes]
                ),
                black_holes_mask,
            )
        return MaskCollection(
            gas=gas_mask,
            dark_matter=dark_matter_mask,
            stars=stars_mask,
            black_holes=black_holes_mask,
        )

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
        return cosmo_array(
            getattr(self._group, centre_attr).to(
                u.kpc
            ),  # maybe comoving, ensure physical
            comoving=False,
            cosmo_factor=cosmo_factor(a**1, self._caesar.simulation.scale_factor),
        ).to_comoving()

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
        return cosmo_array(
            getattr(self._group, vcentre_attr).to(u.km / u.s),
            comoving=False,
            cosmo_factor=cosmo_factor(a**0, self._caesar.simulation.scale_factor),
        ).to_comoving()

    def __getattr__(self, attr: str) -> Any:
        # Invoked if attribute not found.
        # Use to expose the masked catalogue.
        if attr == "_group":  # guard infinite recursion
            return object.__getattribute__(self, "_group")
        return getattr(self._group, attr)

    def __repr__(self) -> str:
        return self._group.__repr__()
