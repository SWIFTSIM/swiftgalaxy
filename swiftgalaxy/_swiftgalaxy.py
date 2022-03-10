import numpy as np
import unyt as u
from swiftsimio import metadata as swiftsimio_metadata
from swiftsimio.reader import SWIFTDataset
from swiftsimio.objects import cosmo_array


def _apply_box_wrap(coords, boxsize):
    retval = coords
    if boxsize is None:
        return retval
    for axis in range(3):
        too_high = retval[:, axis] > boxsize[axis] / 2.
        while too_high.any():
            retval[too_high, axis] -= boxsize[axis]
            too_high = retval[:, axis] > boxsize[axis] / 2.
        too_low = retval[:, axis] <= -boxsize[axis] / 2.
        while too_low.any():
            retval[too_low, axis] += boxsize[axis]
            too_low = retval[:, axis] <= -boxsize[axis] / 2.
    return retval


def _apply_translation(coords, offset):
    return coords + offset


def _apply_rotmat(coords, rotation_matrix):
    return coords.dot(rotation_matrix)


def _apply_4transform(coords, transform, transform_units):
    # A 4x4 transformation matrix has mixed units, so need to
    # assume a consistent unit for all transformations and
    # work with raw arrays.
    return np.hstack((
        coords.to_value(transform_units),
        np.ones(coords.shape[0])[:, np.newaxis]
    )).dot(transform)[:, :3] * transform_units


class _CoordinateHelper(object):

    def __init__(self, coordinates, masks):
        self._coordinates = coordinates
        self._masks = masks
        return

    def __getattr__(self, attr):
        return self._coordinates[self._masks[attr]]

    def __repr__(self):
        return 'Available coordinates: {:s}.'.format(
            ', '.join(self._masks.keys())
        )


class _SWIFTParticleDatasetHelper(object):

    def __init__(
            self,
            particle_dataset,
            swiftgalaxy
    ):
        self._particle_dataset = particle_dataset
        self._swiftgalaxy = swiftgalaxy
        self._cartesian_coordinates = None
        self._spherical_coordinates = None
        self._cylindrical_coordinates = None
        self._cartesian_velocities = None
        self._spherical_velocities = None
        self._cylindrical_velocities = None
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
                if attr in swiftgalaxy.transforms_like_coordinates:
                    transform_units = swiftgalaxy.metadata.units.length
                    transform = swiftgalaxy._coordinate_like_transform
                elif attr in swiftgalaxy.transforms_like_velocities:
                    transform_units = swiftgalaxy.metadata.units.length \
                        / swiftgalaxy.metadata.units.time
                    transform = swiftgalaxy._velocity_like_transform
                else:
                    transform = None
                if transform is not None:
                    data = _apply_4transform(data, transform, transform_units)
                try:
                    boxsize = metadata.boxsize
                except AttributeError:
                    boxsize = None
                if attr in swiftgalaxy.transforms_like_coordinates:
                    data = _apply_box_wrap(data, boxsize)
                setattr(
                    particle_dataset,
                    '_{:s}'.format(attr),
                    data
                )
            else:
                # just return the data
                pass
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
           ((attr.startswith('_')) and (attr[1:] in field_names)):
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

    @property
    def cartesian_coordinates(self):
        if self._cartesian_coordinates is None:
            self._cartesian_coordinates = \
                getattr(
                    self,
                    self._swiftgalaxy.coordinates_dataset_name
                ).view()
        return _CoordinateHelper(
            self._cartesian_coordinates,
            dict(
                x=np.s_[:, 0],
                y=np.s_[:, 1],
                z=np.s_[:, 2],
                xyz=np.s_[...]
            )
        )

    @property
    def cartesian_velocities(self):
        if self._cartesian_coordinates is None:
            self.cartesian_coordinates
        if self._cartesian_velocities is None:
            self._cartesian_velocities = \
                getattr(
                    self,
                    self._swiftgalaxy.velocities_dataset_name
                ).view()
        return _CoordinateHelper(
            self._cartesian_velocities,
            dict(
                x=np.s_[:, 0],
                y=np.s_[:, 1],
                z=np.s_[:, 2],
                xyz=np.s_[...]
            )
        )

    @property
    def spherical_coordinates(self):
        if self._cartesian_coordinates is None:
            self.cartesian_coordinates
        if self._spherical_coordinates is None:
            r = np.sqrt(
                np.sum(
                    np.power(self.cartesian_coordinates.xyz, 2),
                    axis=1
                )
            )
            theta = np.arcsin(self.cartesian_coordinates.z / r)
            theta = cosmo_array(theta, units=u.rad)
            if self.cylindrical_coordinates is not None:
                phi = self.cylindrical_coordinates.phi
            else:
                phi = np.arctan2(
                    self.cartesian_coordinates.y,
                    self.cartesian_coordinates.x
                )
                phi = np.where(phi < 0, phi + 2 * np.pi, phi)
                phi = cosmo_array(phi, units=u.rad)
            self._spherical_coordinates = dict(
                _r=r,
                _theta=theta,
                _phi=phi
            )
        return _CoordinateHelper(
            self._spherical_coordinates,
            dict(
                r='_r',
                radius='_r',
                lon='_phi',
                longitude='_phi',
                az='_phi',
                azimuth='_phi',
                phi='_phi',
                lat='_theta',
                latitude='_theta',
                pol='_theta',
                polar='_theta',
                theta='_theta'
            )
        )

    @property
    def spherical_velocities(self):
        if self._spherical_coordinates is None:
            self.spherical_coordinates
        # existence of self.cartesian_coordinates guaranteed by
        # initialisation of self.spherical_coordinates immediately above
        if self._cartesian_velocities is None:
            self.cartesian_velocities
        if self._spherical_velocities is None:
            _sin_t = np.sin(self.spherical_coordinates.theta)
            _cos_t = np.cos(self.spherical_coordinates.theta)
            _sin_p = np.sin(self.spherical_coordinates.phi)
            _cos_p = np.cos(self.spherical_coordinates.phi)
            v_r = _cos_t * _cos_p * self.cartesian_velocities.x \
                + _cos_t * _sin_p * self.cartesian_velocities.y \
                + _sin_t * self.cartesian_velocities.z
            v_t = _sin_t * _cos_p * self.cartesian_velocities.x \
                + _sin_t * _sin_p * self.cartesian_velocities.y \
                - _cos_t * self.cartesian_velocities.z
            v_p = -_sin_p * self.cartesian_velocities.x \
                + _cos_p * self.cartesian_velocities.y
            self._spherical_velocities = dict(
                _v_r=v_r,
                _v_t=v_t,
                _v_p=v_p
            )
        return _CoordinateHelper(
            self._spherical_velocities,
            dict(
                r='_v_r',
                radius='_v_r',
                lon='_v_p',
                longitude='_v_p',
                az='_v_p',
                azimuth='_v_p',
                phi='_v_p',
                lat='_v_t',
                latitude='_v_t',
                pol='_v_t',
                polar='_v_t',
                theta='_v_t'
            )
        )

    @property
    def cylindrical_coordinates(self):
        if self._cartesian_coordinates is None:
            self.cartesian_coordinates
        if self._cylindrical_coordinates is None:
            rho = np.sqrt(
                np.sum(
                    np.power(self.cartesian_coordinates.xyz[:, :2], 2),
                    axis=1
                )
            )
            if self._spherical_coordinates is not None:
                phi = self.spherical_coordinates.phi
            else:
                phi = np.arctan2(
                    self.cartesian_coordinates.y,
                    self.cartesian_coordinates.x
                )
                phi = np.where(phi < 0, phi + 2 * np.pi, phi)
                phi = cosmo_array(phi, units=u.rad)
            z = self.cartesian_coordinates.z
            self._cylindrical_coordinates = dict(
                _rho=rho,
                _phi=phi,
                _z=z
            )
        return _CoordinateHelper(
            self._cylindrical_coordinates,
            dict(
                R='_rho',
                rho='_rho',
                radius='_rho',
                lon='_phi',
                longitude='_phi',
                az='_phi',
                azimuth='_phi',
                phi='_phi',
                z='_z'
            )
        )

    @property
    def cylindrical_velocities(self):
        if self._cylindrical_coordinates is None:
            self.cylindrical_coordinates
        # existence of self.cartesian_coordinates guaranteed by
        # initialisation of self.cylindrical_coordinates immediately above
        if self._cartesian_velocities is None:
            self.cartesian_velocities
        if self._cylindrical_velocities is None:
            _sin_p = np.sin(self.cylindrical_coordinates.phi)
            _cos_p = np.cos(self.cylindrical_coordinates.phi)
            v_rho = _cos_p * self.cartesian_velocities.x \
                + _sin_p * self.cartesian_velocities.y
            if self._spherical_velocities is not None:
                v_phi = self.spherical_velocities.phi
            else:
                v_phi = -_sin_p * self.cartesian_velocities.x \
                    + _cos_p * self.cartesian_velocities.y
            v_z = self.cartesian_velocities.z
            self._cylindrical_velocities = dict(
                _v_rho=v_rho,
                _v_phi=v_phi,
                _v_z=v_z
            )
        return _CoordinateHelper(
            self._cylindrical_velocities,
            dict(
                R='_v_rho',
                rho='_v_rho',
                radius='_v_rho',
                lon='_v_phi',
                longitude='_v_phi',
                az='_v_phi',
                azimuth='_v_phi',
                phi='_v_phi',
                z='_v_z'
            )
        )


class SWIFTGalaxy(SWIFTDataset):

    def __init__(
            self,
            snapshot_filename,
            halo_finder,
            auto_recentre=True,
            transforms_like_coordinates={'coordinates', },
            transforms_like_velocities={'velocities', },
            id_particle_dataset_name='particle_ids',
            coordinates_dataset_name='coordinates',
            velocities_dataset_name='velocities'
    ):
        self.snapshot_filename = snapshot_filename
        self.halo_finder = halo_finder
        self.halo_finder._init_spatial_mask(self)
        self.transforms_like_coordinates = transforms_like_coordinates
        self.transforms_like_velocities = transforms_like_velocities
        self.id_particle_dataset_name = id_particle_dataset_name
        self.coordinates_dataset_name = coordinates_dataset_name
        self.velocities_dataset_name = velocities_dataset_name
        self._coordinate_like_transform = np.eye(4)
        self._velocity_like_transform = np.eye(4)
        super().__init__(
            snapshot_filename,
            mask=self.halo_finder._spatial_mask
        )
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
                    '_{:s}'.format(self.id_particle_dataset_name)
                )
                setattr(
                    super().__getattribute__(particle_name),  # bypass helper
                    '_{:s}'.format(self.id_particle_dataset_name),
                    particle_ids[
                        getattr(self.halo_finder._extra_mask, particle_name)
                    ]
                )
        if auto_recentre:
            self.recentre(self.halo_finder._centre())
            self.recentre_velocity(self.halo_finder._vcentre())
        return

    # # Could implement:
    # def __getitem__(self, mask):
    #     # If mask is a MaskCollection, could return a copy suitably masked.
    #     # Would need to be careful to preserve loaded data, rotations, etc.
    #     return SWIFTGalaxy(...)

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

    def rotate(self, rotation):
        # expect a scipy.spatial.transform.Rotation
        rotation_matrix = rotation.as_matrix()
        rotatable = (
            self.transforms_like_coordinates
            | self.transforms_like_velocities
        )
        for particle_name in self.metadata.present_particle_names:
            dataset = getattr(self, particle_name)
            for field_name in rotatable:
                field_data = getattr(dataset, '_{:s}'.format(field_name))
                if field_data is not None:
                    field_data = _apply_rotmat(field_data, rotation_matrix)
                    setattr(
                        dataset,
                        '_{:s}'.format(field_name),
                        field_data
                    )
        rotmat4 = np.eye(4)
        rotmat4[:3, :3] = rotation_matrix
        self._append_to_coordinate_like_transform(rotmat4)
        self._append_to_velocity_like_transform(rotmat4)
        self.wrap_box()
        return

    def _translate(self, translation, boost=False):
        translatable = self.transforms_like_velocities if boost \
            else self.transforms_like_coordinates
        for particle_name in self.metadata.present_particle_names:
            dataset = getattr(self, particle_name)
            for field_name in translatable:
                field_data = getattr(dataset, '_{:s}'.format(field_name))
                if field_data is not None:
                    field_data = _apply_translation(field_data, translation)
                    setattr(
                        dataset,
                        '_{:s}'.format(field_name),
                        field_data
                    )
        if boost:
            transform_units = self.metadata.units.length \
                / self.metadata.units.time
        else:
            transform_units = self.metadata.units.length
        translation4 = np.eye(4)
        translation4[3, :3] = translation.to_value(transform_units)
        if boost:
            self._append_to_velocity_like_transform(translation4)
        else:
            self._append_to_coordinate_like_transform(translation4)
        if not boost:
            self.wrap_box()
        return

    def translate(self, translation):
        self._translate(translation)

    def boost(self, boost):
        self._translate(boost, boost=True)
        return

    def recentre(self, new_centre):
        self._translate(-new_centre)
        return

    def recentre_velocity(self, new_centre):
        self._translate(-new_centre, boost=True)

    def wrap_box(self):
        for particle_name in self.metadata.present_particle_names:
            dataset = getattr(self, particle_name)
            for field_name in self.transforms_like_coordinates:
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

    def _append_to_coordinate_like_transform(self, transform):
        self._coordinate_like_transform = \
            self._coordinate_like_transform.dot(transform)
        self._void_derived_coordinates()
        return

    def _append_to_velocity_like_transform(self, transform):
        self._velocity_like_transform = \
            self._velocity_like_transform.dot(transform)
        self._void_derived_coordinates()
        return

    def _void_derived_coordinates(self):
        # Transforming implies conversion back to cartesian, it's therefore
        # cheaper to just delete any non-cartesian coordinates when a
        # transform occurs and lazily re-calculate them as needed.
        for particle_name in self.metadata.present_particle_names:
            getattr(self, particle_name)._spherical_coordinates = None
            getattr(self, particle_name)._cylindrical_coordinates = None
            getattr(self, particle_name)._spherical_velocities = None
            getattr(self, particle_name)._cylindrical_velocities = None
        return
