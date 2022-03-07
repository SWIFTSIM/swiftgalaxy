import unyt as u
from sympy import Rational
from astropy.coordinates.matrix_utilities import rotation_matrix
from astropy.coordinates import CartesianRepresentation, \
    SphericalRepresentation, CylindricalRepresentation, \
    CartesianDifferential, SphericalDifferential, CylindricalDifferential
from swiftsimio import metadata as swiftsimio_metadata
from swiftsimio.reader import SWIFTDataset
from swiftsimio.objects import cosmo_array


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


class _AstropyRepresentationOrDifferentialHelper(object):
    def __init__(
            self,
            representation_or_differential,
            name_map
    ):
        self._representation_or_differential = representation_or_differential
        self._name_map = name_map

    def __getattr__(self, attr):
        astropy_quantity = getattr(
            self._representation_or_differential,
            self._name_map[attr]
        )
        # borrow some code from unyt_array.from_astropy for conversion
        units = astropy_quantity.unit
        ap_units = []
        for base, exponent in zip(units.bases, units.powers):
            unit_str = base.to_string()
            # we have to do this because AstroPy is silly and defines
            # hour as 'h'
            if unit_str == 'h':
                unit_str = 'hr'
            ap_units.append(
                "%s**(%s)" % (unit_str, Rational(exponent))
            )
        ap_units = '*'.join(ap_units)
        # explicitly avoid copying by using a view
        return_array = \
            astropy_quantity.view(type=cosmo_array) * u.Unit(ap_units)
        # could restore the cosmo_array attributes here, if we still knew
        # what they were (also, comoving angular coordinates is pretty
        # non-sensical, avoid that)
        return return_array

    # def __str__(self):
        # should probably print something nicer


class _SWIFTParticleDatasetHelper(object):

    def __init__(
            self,
            particle_dataset,
            swiftgalaxy
    ):
        self._particle_dataset = particle_dataset
        self._swiftgalaxy = swiftgalaxy
        self._cartesian_representation = None
        self._spherical_representation = None
        self._initialised = True
        return

    def __getattribute__(self, attr):
        particle_name = \
            object.__getattribute__(self, '_particle_dataset').particle_name
        metadata = object.__getattribute__(self, '_particle_dataset').metadata
        field_names = getattr(
            metadata,
            '{:s}_properties'.format(particle_name)
        ).field_names
        swiftgalaxy = object.__getattribute__(self, '_swiftgalaxy')
        particle_dataset = object.__getattribute__(self, '_particle_dataset')
        if attr in field_names:
            # we're dealing with a particle data table
            # TODO: named columns
            if particle_dataset.__dict__.get('_{:s}'.format(attr)) is None:
                # going to read from file: apply masks, transforms
                data = getattr(particle_dataset, attr)  # raw data loaded
                data = object.__getattribute__(self, '_apply_mask')(data)
                translatable = swiftgalaxy.translatable
                rotatable = swiftgalaxy.rotatable
                boostable = swiftgalaxy.boostable
                try:
                    boxsize = metadata.boxsize
                except AttributeError:
                    boxsize = None
                data = _apply_transform_stack(
                    data,
                    swiftgalaxy._transform_stack,
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
        if attr in (
                'cartesian_coordinates',
                'spherical_coordinates',
                'cylindrical_coordinates',
                'cartesian_velocities',
                'spherical_velocities',
                'cylindrical_velocities'
        ):
            return object.__getattribute__(self, '_{:s}'.format(attr))()
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
            '{:s}_properties'.format(self._particle_dataset.particle_name)
        ).field_names
        if (attr in field_names) or \
           ((attr[0] == '_') and (attr[1:] in field_names)):
            setattr(
                self._particle_dataset,
                attr,
                value
            )
            return
        else:
            object.__setattr__(self, attr, value)
            return

    def _apply_mask(self, data):
        if self._swiftgalaxy.halo_finder._extra_mask is not None:
            mask = self._swiftgalaxy.halo_finder._extra_mask.__getattribute__(
                self._particle_dataset.particle_name
            )
            if mask is not None:
                return data[mask]
        return data

    def _cartesian_coordinates(self):
        if self._cartesian_representation is None:
            # lose the extra cosmo array attributes here
            self._cartesian_representation = \
                CartesianRepresentation(
                    self.coordinates.ndarray_view(),
                    # Tempting to use unyt's to/from astropy,
                    # but that forces a copy.
                    # unyt relies on str representation in
                    # to_astropy, so should be safe to do same.
                    unit=str(self.coordinates.units),
                    xyz_axis=1,
                    copy=False
                )
        return _AstropyRepresentationOrDifferentialHelper(
            self._cartesian_representation,
            dict(x='x', y='y', z='z', xyz='xyz')
        )

    def _cartesian_velocities(self):
        if self._cartesian_representation is None:
            self._cartesian_coordinates()
        if 's' not in self._cartesian_representation.differentials.keys():
            self._cartesian_representation.differentials['s'] = \
                CartesianDifferential(
                    self.velocities.ndarray_view(),
                    unit=str(self.velocities.units),
                    xyz_axis=1,
                    copy=False
                )
        return _AstropyRepresentationOrDifferentialHelper(
            self._cartesian_representation.differentials['s'],
            dict(x='d_x', y='d_y', z='d_z', xyz='xyz')
        )

    def _spherical_coordinates(self):
        if self._cartesian_representation is None:
            self._cartesian_coordinates()
        if self._spherical_representation is None:
            # lose the extra cosmo array attributes here
            self._spherical_representation = \
                SphericalRepresentation.from_cartesian(
                    self._cartesian_representation
                )
        return _AstropyRepresentationOrDifferentialHelper(
            self._spherical_representation,
            dict(lon='lon', lat='lat', r='distance')
        )

    def _spherical_velocities(self):
        # possible to be slightly more efficient by initialising
        # both the rep and the diff together if neither exist?
        if self._spherical_representation is None:
            self._spherical_representation()
        # existence of self._cartesian_representation guaranteed by
        # initialisation of self._spherical_representation immediately above
        if 's' not in self._cartesian_representation.differentials.keys():
            self._cartesian_velocities()
        if 's' not in self._spherical_representation.differentials.keys():
            self._spherical_representation.differentials['s'] = \
                self._cartesian_representation.differentials['s'].represent_as(
                    SphericalDifferential,
                    self._spherical_representation
                )
        return _AstropyRepresentationOrDifferentialHelper(
            self._spherical_representation.differentials['s'],
            dict(lon='d_lon', lat='d_lat', r='d_distance')
        )

    def _cylindrical_coordinates(self):
        if self._cartesian_representation is None:
            self._cartesian_coordinates()
        if self._cylindrical_representation is None:
            # lose the extra cosmo array attributes here
            self._cylindrical_representation = \
                CylindricalRepresentation.from_cartesian(
                    self._cartesian_representation
                )
        return _AstropyRepresentationOrDifferentialHelper(
            self._cylindrical_representation,
            dict(R='rho', rho='rho', phi='phi', lon='phi', z='z')
        )

    def _cylindrical_velocities(self):
        # possible to be slightly more efficient by initialising
        # both the rep and the diff together if neither exist?
        if self._cylindrical_representation is None:
            self._cylindrical_representation()
        # existence of self._cartesian_representation guaranteed by
        # initialisation of self._cylindrical_representation immediately above
        if 's' not in self._cartesian_representation.differentials.keys():
            self._cartesian_velocities()
        if 's' not in self._cylindrical_representation.differentials.keys():
            self._cylindrical_representation.differentials['s'] = \
                self._cartesian_representation.differentials['s'].represent_as(
                    CylindricalDifferential,
                    self._cylindrical_representation
                )
        return _AstropyRepresentationOrDifferentialHelper(
            self._cylindrical_representation.differentials['s'],
            dict(R='d_rho', rho='d_rho', phi='d_phi', lon='d_phi', z='d_z')
        )


class SWIFTGalaxy(SWIFTDataset):

    def __init__(
            self,
            snapshot_filename,
            halo_finder,
            auto_recentre=True,
            translatable=('coordinates', ),
            boostable=('velocities', ),
            rotatable=('coordinates', 'velocities'),
            id_particle_dataset_name='particle_ids'
    ):
        self.snapshot_filename = snapshot_filename
        self.halo_finder = halo_finder
        self.halo_finder._init_spatial_mask(self)
        self.rotatable = rotatable
        self.translatable = translatable
        self.boostable = boostable
        self._transform_stack = list()
        super().__init__(snapshot_filename, mask=self.halo_finder._spatial_mask)
        self._particle_dataset_helpers = dict()
        for particle_name in self.metadata.present_particle_names:
            # We'll make a custom type to present a nice name to the user.
            nice_name = \
                swiftsimio_metadata.particle_types.particle_name_class[
                    getattr(
                        self.metadata,
                        '{:s}_properties'.format(particle_name)
                    ).particle_type
                ]
            TypeDatasetHelper = type(
                '{:s}DatasetHelper'.format(nice_name),
                (_SWIFTParticleDatasetHelper, object),
                dict()
            )
            self._particle_dataset_helpers[particle_name] = TypeDatasetHelper(
                super().__getattribute__(particle_name),
                self
            )

        self.halo_finder._init_extra_mask(self)
        if self.halo_finder._extra_mask is not None:
            # only particle ids should be loaded so far, need to mask these
            for particle_name in self.metadata.present_particle_names:
                particle_ids = getattr(
                    getattr(self, particle_name),
                    '_{:s}'.format(id_particle_dataset_name)
                )
                setattr(
                    super().__getattribute__(particle_name),  # bypass helper
                    '_{:s}'.format(id_particle_dataset_name),
                    particle_ids[
                        getattr(self.halo_finder._extra_mask, particle_name)
                    ]
                )
        if auto_recentre:
            self.recentre(self.halo_finder._centre())
            self.recentre(self.halo_finder._vcentre(), velocity=True)
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
        for particle_name in self.metadata.present_particle_names:
            dataset = getattr(self, particle_name)
            for field_name in self.rotatable:
                field_data = getattr(dataset, '_{:s}'.format(field_name))
                if field_data is not None:
                    field_data = _apply_rotmat(field_data, rotmat)
                    setattr(
                        dataset,
                        '_{:s}'.format(field_name),
                        field_data
                    )
        self._append_to_transform_stack(('R', rotmat))
        self.wrap_box()
        return

    def translate(self, translation, velocity=False):
        do_fields = self.boostable if velocity else self.translatable
        for particle_name in self.metadata.present_particle_names:
            dataset = getattr(self, particle_name)
            for field_name in do_fields:
                field_data = getattr(dataset, '_{:s}'.format(field_name))
                if field_data is not None:
                    field_data = _apply_translation(field_data, translation)
                    setattr(
                        dataset,
                        '_{:s}'.format(field_name),
                        field_data
                    )
        self._append_to_transform_stack(
            ({True: 'B', False: 'T'}[velocity], translation)
        )
        if not velocity:
            self.wrap_box()
        return

    def recentre(self, new_centre, velocity=False):
        self.translate(-new_centre, velocity=velocity)
        return

    def wrap_box(self):
        for particle_name in self.metadata.present_particle_names:
            dataset = getattr(self, particle_name)
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

    def _append_to_transform_stack(self, transform):
        self._transform_stack.append(transform)
        self._void_derived_representations()
        return

    def _void_derived_representations(self):
        # Transforming representations actually converts back to cartesian,
        # transforms, and then converts forward to the representation in
        # question. It's therefore cheaper to just delete any non-cartesian
        # representations when a transform occurs and lazily re-calculate
        # them as needed. Note any attached differentials get chopped at the
        # same time.
        for particle_name in self.metadata.present_particle_names:
            getattr(self, particle_name)._spherical_representation = None
            getattr(self, particle_name)._cylindrical_representation = None
        return
