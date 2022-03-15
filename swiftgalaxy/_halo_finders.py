from abc import ABC, abstractmethod
import unyt as u
from _swiftgalaxy import MaskCollection


class _HaloFinder(ABC):

    def __init__(self, extra_mask='bound_only'):
        self.extra_mask = extra_mask
        self._load()
        return

    @abstractmethod
    def _load(self):
        pass

    @abstractmethod
    def _init_spatial_mask(self, SG):
        # define SG._spatial_mask
        pass

    @abstractmethod
    def _init_extra_mask(self, SG):
        # define SG._extra_mask
        pass

    @abstractmethod
    def _centre(self):
        # return halo centre
        pass

    @abstractmethod
    def _vcentre(self):
        # return halo velocity centre
        pass

    # In addition, it is recommended to expose the properties computed
    # by the halo finder through this object, masked to the values
    # corresponding to the object of interest. It probably makes sense
    # to match the syntax used to the usual syntax for the halo finder
    # in question? See e.g. implementation in __getattr__ in Velociraptor
    # subclass below.


class Velociraptor(_HaloFinder):

    def __init__(
            self,
            velociraptor_filebase,
            halo_index=None,
            extra_mask='bound_only',
            centre_type='minpot'  # _gas _star mbp minpot
    ):
        self.velociraptor_filebase = velociraptor_filebase
        self.halo_index = halo_index
        self.centre_type = centre_type
        super().__init__()
        # currently velociraptor_python works with a halo index, not halo_id!
        # self.catalogue_mask = (catalogue.ids.id == halo_id).nonzero()
        return

    def _load(self):
        from velociraptor import load as load_catalogue
        self._catalogue = load_catalogue(
            f'{self.velociraptor_filebase}.properties', mask=self.halo_index)
        return

    def _init_spatial_mask(self, SG):
        from velociraptor import load as load_catalogue
        from velociraptor.particles import load_groups
        from velociraptor.swift.swift import generate_spatial_mask
        groups = load_groups(f'{self.velociraptor_filebase}.catalog_groups',
                             catalogue=load_catalogue(
                                 f'{self.velociraptor_filebase}.properties'))
        # extract_halo requests a "halo_id", but actually wants an index!
        self._particles, unbound_particles = \
            groups.extract_halo(halo_id=self.halo_index)
        SG._spatial_mask = generate_spatial_mask(self._particles,
                                                 SG.snapshot_filename)
        return

    def _init_extra_mask(self, SG):
        from velociraptor.swift.swift import generate_bound_mask
        if self.extra_mask == 'bound_only':
            SG._extra_mask = MaskCollection(
                **generate_bound_mask(SG, self._particles)._asdict())
        elif self.extra_mask is None:
            SG._extra_mask = MaskCollection(
                **{k: None
                   for k in SG.metadata.present_particle_names})
        else:
            # Keep user provided mask.
            assert all([
                hasattr(self.extra_mask, name)
                for name in SG.metadata.present_particle_names
            ])
            SG._extra_mask = MaskCollection(
                **{
                    name: getattr(self.extra_mask, name)
                    for name in SG.metadata.present_particle_names
                })

    def _centre(self):
        return u.uhstack([
            getattr(self._catalogue.positions,
                    '{:s}c{:s}'.format(c, self.centre_type)) for c in 'xyz'
        ])

    def _vcentre(self):
        return u.uhstack([
            getattr(self._catalogue.velocities,
                    'v{:s}c{:s}'.format(c, self.centre_type)) for c in 'xyz'
        ])

    def __getattr__(self, attr):
        # Invoked if attribute not found.
        # Use to expose the masked catalogue.
        if attr == '_catalogue':
            return object.__getattribute__(self, '_catalogue')
        return getattr(self._catalogue, attr)

    def __repr__(self):
        # Expose the catalogue __repr__ for interactive use.
        return self._catalogue.__repr__()
