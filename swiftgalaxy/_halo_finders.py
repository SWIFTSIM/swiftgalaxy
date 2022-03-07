from abc import ABC, abstractmethod
import unyt as u


class _HaloFinder(ABC):

    catalogue_mask = None

    def __init__(self, extra_mask='bound_only'):
        self.extra_mask = None
        self.received_extra_mask = extra_mask
        self.load()
        return

    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def init_spatial_mask(self, SG):
        # define self.spatial_mask
        pass

    @abstractmethod
    def init_extra_mask(self, SG):
        # define self.extra_mask
        pass

    @abstractmethod
    def centre(self):
        # return halo centre
        pass

    @abstractmethod
    def vcentre(self):
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
            halo_id=None,
            extra_mask='bound_only',
            centre_type='minpot'  # _gas _star mbp minpot
    ):
        self.velociraptor_filebase = velociraptor_filebase
        self.halo_id = halo_id
        self.centre_type = centre_type
        super().__init__()
        # currently halo_id is actually the index, not the id!
        # self._catalogue_mask = (catalogue.ids.id == halo_id).nonzero()
        self.catalogue_mask = self.halo_id
        return

    def load(self):
        from velociraptor import load as load_catalogue
        from velociraptor.particles import load_groups
        self.catalogue = load_catalogue(
            f'{self.velociraptor_filebase}.properties'
        )
        # Need to streamline to use a single (masked) catalogue.
        # Much more efficient to avoid reading entire catalogue arrays!
        # Currently groups barfs on receiving scalar values though.
        self.mcatalogue = load_catalogue(
            f'{self.velociraptor_filebase}.properties',
            mask=self.halo_id
        )
        self.groups = load_groups(
            f'{self.velociraptor_filebase}.catalog_groups',
            catalogue=self.catalogue
        )
        return

    def init_spatial_mask(self, SG):
        from velociraptor.swift.swift import generate_spatial_mask
        self.particles, self.unbound_particles = \
            self.groups.extract_halo(halo_id=self.halo_id)  # probably need to set this to 0 once catalogue is masked?
        self.spatial_mask = generate_spatial_mask(
            self.particles,
            SG.snapshot_filename
        )
        return

    def init_extra_mask(self, SG):
        from velociraptor.swift.swift import generate_bound_mask
        if self.received_extra_mask == 'bound_only':
            self.extra_mask = generate_bound_mask(SG, self.particles)
        else:
            pass  # Keep user provided mask, or None.
            # We should guard against applying None as a mask later.
            # Would be nice to check here that this looks like a mask
            # to avoid a typo'd string waiting until after an expensive
            # read to raise an exception.

    def centre(self):
        return u.uhstack(
            [getattr(
                self.catalogue.positions,
                '{:s}c{:s}'.format(c, self.centre_type)
            )[self.catalogue_mask] for c in 'xyz']
        )

    def vcentre(self):
        return u.uhstack(
            [getattr(
                self.catalogue.velocities,
                'v{:s}c{:s}'.format(c, self.centre_type)
            )[self.catalogue_mask] for c in 'xyz']
        )

    def __getattr__(self, attr):
        # Invoked if attribute not found.
        # Use to expose the masked catalogue.
        return getattr(self.mcatalogue, attr)
