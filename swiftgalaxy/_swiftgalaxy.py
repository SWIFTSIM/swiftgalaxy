import unyt as u
from astropy.coordinates.matrix_utilities import rotation_matrix
from astropy.coordinates import CartesianRepresentation, \
    SphericalRepresentation, CylindricalRepresentation, \
    CartesianDifferential, SphericalDifferential, CylindricalDifferential
from swiftsimio import metadata as swiftsimio_metadata
from swiftsimio.reader import SWIFTDataset
from velociraptor import load as load_catalogue
from velociraptor.particles import load_groups
from velociraptor.swift.swift import generate_spatial_mask, generate_bound_mask


def _apply_box_wrap(coords, boxsize):
    for axis in range(3):
        too_high = coords[:, axis] > boxsize[axis] / 2.
        while too_high.any():
            coords[too_high, axis] -= boxsize[axis]
            too_high = coords[:, axis] > boxsize[axis] / 2.
        too_low = coords[:, axis] <= -boxsize[axis] / 2.
        while too_low.any():
            coords[too_low, axis] += boxsize[axis]
            too_low = coords[:, axis] <= -boxsize[axis] / 2.
    return coords


def _apply_translation(coords, offset):
    coords += offset
    return coords


def _apply_rotmat(coords, rotmat):
    coords = coords.dot(rotmat)
    return coords


def _apply_transform_stack(
        data,
        transform_stack,
        is_translatable=None,
        is_rotatable=None,
        is_boostable=None,
        boxsize=None
):
    for transform_type, transform in transform_stack:
        if (transform_type == 'T') and is_translatable:
            data = _apply_translation(data, transform)
        elif (transform_type == 'B') and is_boostable:
            data = _apply_translation(data, transform)
        elif (transform_type == 'R') and is_rotatable:
            data = _apply_rotmat(data, transform)
        if is_translatable and boxsize is not None:
            # this is a position-like dataset, so wrap box,
            # either translation or rotation can in principle
            # require a wrap
            # for a non-periodic box lbox=None should be passed
            data = _apply_box_wrap(data, boxsize)
    return data


class _SWIFTParticleDatasetHelper(object):

    def __init__(
            self,
            ptype,  # can determine by introspection instead?
            particle_dataset,
            extra_mask=None,
            translatable=tuple(),
            boostable=tuple(),
            rotatable=tuple(),
            transform_stack=list()
    ):
        self._ptype = ptype
        self._particle_dataset = particle_dataset
        self._extra_mask = getattr(extra_mask, ptype) \
            if extra_mask is not None else None
        self._translatable = translatable
        self._boostable = boostable
        self._rotatable = rotatable
        self._transform_stack = transform_stack
        self._particle_dataset._cartesian_representation = None
        self._initialised = True
        return

    def __getattribute__(self, attr):
        ptype = object.__getattribute__(self, '_ptype')
        metadata = object.__getattribute__(self, '_particle_dataset').metadata
        field_names = \
            getattr(metadata, '{:s}_properties'.format(ptype)).field_names
        transform_stack = object.__getattribute__(self, '_transform_stack')
        particle_dataset = object.__getattribute__(self, '_particle_dataset')
        if attr in field_names:
            # we're dealing with a particle data table
            # TODO: named columns
            if particle_dataset.__dict__.get('_{:s}'.format(attr)) is None:
                # going to read from file: apply masks, transforms
                data = getattr(particle_dataset, attr)  # raw data loaded
                data = object.__getattribute__(self, '_apply_mask')(data)
                translatable = object.__getattribute__(self, '_translatable')
                rotatable = object.__getattribute__(self, '_rotatable')
                boostable = object.__getattribute__(self, '_boostable')
                try:
                    boxsize = metadata.boxsize
                except AttributeError:
                    boxsize = None
                data = _apply_transform_stack(
                    data,
                    transform_stack,
                    is_translatable=attr in translatable,
                    is_rotatable=attr in rotatable,
                    is_boostable=attr in boostable,
                    boxsize=boxsize
                )
                setattr(
                    particle_dataset,
                    '_{:s}'.format(attr),
                    data
                )
            else:
                # just return the data
                pass
        if attr == 'cartesian_coordinates':
            return object.__getattribute__(self, '_cartesian_coordinates')()
        try:
            # beware collisions with SWIFTDataset namespace
            return object.__getattribute__(self, attr)
        except AttributeError:
            # exposes everything else in __dict__
            return getattr(particle_dataset, attr)

    def __setattr__(self, attr, value):
        # pass particle data through to actual SWIFTDataset
        if not hasattr(self, '_initialised'):
            # guard during initialisation
            object.__setattr__(self, attr, value)
            return
        field_names = getattr(
            self._particle_dataset.metadata,
            '{:s}_properties'.format(self._ptype)
        ).field_names
        if (attr in field_names) or \
           ((attr[0] == '_') and (attr[1:] in field_names)):
            setattr(
                self._particle_dataset,
                attr,
                value
            )
            return
        object.__setattr__(self, attr, value)
        return

    def _apply_mask(self, data):
        if self._extra_mask is not None:
            return data[self._extra_mask]
        else:
            return data

    def _cartesian_coordinates(self):
        if self._particle_dataset._cartresian_representation is None:
            # lose the extra cosmo array attributes here
            self._particle_dataset._cartesian_representation = \
                CartesianRepresentation(
                    self._particle_dataset.coordinates.ndarray_view(),
                    unit=str(self._particle_dataset.coordinates.units),
                    xyz_axis=1
                )
        self._cartesian_coordinates = \
            self._particle_dataset._cartesian_representation
        return self._cartesian_coordinates

    # @property
    # def spherical_coordinates(self):
    #     if self._particle_dataset._cartresian_representation is None:
    #         # lose the extra cosmo array attributes here
    #         self._particle_dataset._cartesian_representation = \
    #             CartesianRepresentation(
    #                 self._particle_dataset.coordinates.ndarray_view(),
    #                 unit=str(self._particle_dataset.coordinates.units),
    #                 xyz_axis=1
    #             )


class SWIFTGalaxy(SWIFTDataset):

    def __init__(
            self,
            snapshot_filename,
            velociraptor_filebase,
            halo_id,
            extra_mask=None,
            centre_type='minpot',  # _gas _star mbp minpot
            auto_recentre=True,
            translatable=('coordinates', ),
            boostable=('velocities', ),
            rotatable=('coordinates', 'velocities'),
            id_particle_dataset_name='particle_ids'
    ):
        self.extra_mask = None  # needed for initialisation, overwritten below
        self.rotatable = rotatable
        self.translatable = translatable
        self.boostable = boostable
        self._transform_stack = list()
        self._particle_dataset_helpers = dict()
        catalogue = load_catalogue(f'{velociraptor_filebase}.properties')
        # currently halo_id is actually the index, not the id!
        # self._catalogue_mask = (catalogue.ids.id == halo_id).nonzero()
        self._catalogue_mask = halo_id
        groups = load_groups(
            f'{velociraptor_filebase}.catalog_groups',
            catalogue=catalogue
        )
        particles, unbound_particles = groups.extract_halo(halo_id=halo_id)
        swift_mask = generate_spatial_mask(particles, snapshot_filename)
        super().__init__(snapshot_filename, mask=swift_mask)
        if extra_mask == 'bound_only':
            self.extra_mask = generate_bound_mask(self, particles)
        else:
            self.extra_mask = extra_mask  # user can provide mask
            # would be nice to check here that this looks like a mask
            # to avoid a typo'd string waiting until after an expensive
            # read to raise an exception
            # Note this will also cover the default None case,
            # we should guard against applying None as a mask later.
        if self.extra_mask is not None:
            # only particle ids should be loaded so far, need to mask these
            for ptype in self.metadata.present_particle_names:
                particle_ids = getattr(
                    getattr(self, ptype),
                    '_{:s}'.format(id_particle_dataset_name)
                )
                setattr(
                    super().__getattribute__(ptype),  # bypass our helper
                    '_{:s}'.format(id_particle_dataset_name),
                    particle_ids[getattr(self.extra_mask, ptype)]
                )
        if auto_recentre:
            centre = u.uhstack(
                [getattr(
                    catalogue.positions,
                    '{:s}c{:s}'.format(c, centre_type)
                )[self._catalogue_mask] for c in 'xyz']
            )
            self.recentre(centre)
            vcentre = u.uhstack(
                [getattr(
                    catalogue.velocities,
                    'v{:s}c{:s}'.format(c, centre_type)
                )[self._catalogue_mask] for c in 'xyz']
            )
            self.recentre(vcentre, velocity=True)
        return

    def __getattribute__(self, attr):
        # __getattr__ is only checked if the attribute is not found
        # __getattribute__ is checked promptly
        # Note always use super().__getattribute__(...)
        # or object.__getattribute__(self, ...) as appropriate
        # to avoid infinite recursion.
        try:
            metadata = super().__getattribute__('metadata')
        except AttributeError:
            # guard against accessing metadata before it is loaded
            return super().__getattribute__(attr)
        else:
            if attr in metadata.present_particle_names:
                # We are entering a <ParticleType>Dataset:
                # intercept this and wrap it in a class that we
                # can use to manipulate it.
                if attr not in object.__getattribute__(
                        self,
                        '_particle_dataset_helpers'
                ).keys():
                    # We'll make a custom type to present a nice name to the user.
                    nice_name = \
                        swiftsimio_metadata.particle_types.particle_name_class[
                            getattr(
                                metadata,
                                '{:s}_properties'.format(attr)
                            ).particle_type
                        ]
                    TypeDatasetHelper = type(
                        '{:s}DatasetHelper'.format(nice_name),
                        (_SWIFTParticleDatasetHelper, object),
                        dict()
                    )
                    object.__getattribute__(
                        self, '_particle_dataset_helpers'
                    )[attr] = TypeDatasetHelper(
                        attr,
                        super().__getattribute__(attr),
                        extra_mask=object.__getattribute__(self, 'extra_mask'),
                        translatable=object.__getattribute__(
                            self, 'translatable'
                        ),
                        boostable=object.__getattribute__(self, 'boostable'),
                        rotatable=object.__getattribute__(self, 'rotatable'),
                        transform_stack=object.__getattribute__(
                            self, '_transform_stack'
                        )
                    )
                return object.__getattribute__(
                    self,
                    '_particle_dataset_helpers'
                )[attr]
            else:
                return super().__getattribute__(attr)

    def rotate(self, angle_axis=None, rotmat=None):
        if (angle_axis is not None) and (rotmat is not None):
            raise ValueError('Provide angle_axis or rotmat to rotate,'
                             ' not both.')
        if angle_axis is not None:
            rotmat = rotation_matrix(*angle_axis)
        for ptype in self.metadata.present_particle_names:
            dataset = getattr(self, ptype)
            for field_name in self.rotatable:
                field_data = getattr(dataset, '_{:s}'.format(field_name))
                if field_data is not None:
                    field_data = _apply_rotmat(field_data, rotmat)
                    setattr(
                        dataset,
                        '_{:s}'.format(field_name),
                        field_data
                    )
        self._transform_stack.append(('R', rotmat))
        self.wrap_box()
        return

    def translate(self, translation, velocity=False):
        do_fields = self.boostable if velocity else self.translatable
        for ptype in self.metadata.present_particle_names:
            dataset = getattr(self, ptype)
            for field_name in do_fields:
                field_data = getattr(dataset, '_{:s}'.format(field_name))
                if field_data is not None:
                    field_data = _apply_translation(field_data, translation)
                    setattr(
                        dataset,
                        '_{:s}'.format(field_name),
                        field_data
                    )
        self._transform_stack.append(
            ({True: 'B', False: 'T'}[velocity], translation)
        )
        if not velocity:
            self.wrap_box()
        return

    def recentre(self, new_centre, velocity=False):
        self.translate(-new_centre, velocity=velocity)
        return

    def wrap_box(self):
        for ptype in self.metadata.present_particle_names:
            dataset = getattr(self, ptype)
            for field_name in self.translatable:
                field_data = getattr(dataset, '_{:s}'.format(field_name))
                if field_data is not None:
                    field_data = _apply_box_wrap(
                        field_data,
                        self.metadata.boxsize
                    )
                    setattr(
                        dataset,
                        '_{:s}'.format(field_name),
                        field_data
                    )
        return
