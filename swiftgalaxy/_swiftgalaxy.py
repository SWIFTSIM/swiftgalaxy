import numpy as np
import unyt as u
from swiftsimio import metadata as swiftsimio_metadata
from swiftsimio.reader import SWIFTDataset
from swiftsimio.objects import cosmo_array


def _getattr_with_dots(obj, attr):
    attrs = attr.split('.')
    retval = getattr(obj, attrs[0], None)
    for attr in attrs[1:]:
        retval = getattr(retval, attr, None)
    return retval


def _setattr_with_dots(obj, attr, value):
    attrs = attr.split('.')
    if len(attrs) == 1:
        setattr(obj, attr, value)
    else:
        target = getattr(obj, attrs[0])
        for attr in attrs[1:-1]:
            target = getattr(target, attr)
        setattr(target, attrs[-1], value)
    return


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


class MaskCollection(object):

    # Could use dataclasses module, but requires python 3.7+

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        return


class _CoordinateHelper(object):

    def __init__(self, coordinates, masks):
        self._coordinates = coordinates
        self._masks = masks
        return

    def __getattr__(self, attr):
        return self._coordinates[self._masks[attr]]

    def __repr__(self):
        keys = ', '.join(self._masks.keys())
        return f'Available coordinates: {keys}.'


class _SWIFTNamedColumnDatasetHelper(object):

    def __init__(
            self,
            named_column_dataset,
            particle_dataset_helper
    ):
        self._named_column_dataset = named_column_dataset
        self._particle_dataset_helper = particle_dataset_helper
        self._initialised = True
        return

    def __getattribute__(self, attr):
        named_column_dataset = \
            object.__getattribute__(self, '_named_column_dataset')
        particle_dataset_helper = \
            object.__getattribute__(self, '_particle_dataset_helper')
        if attr in named_column_dataset.named_columns:
            # we're dealing with one of the named columns
            if getattr(named_column_dataset, f'_{attr}') is None:
                # going to read from file: apply masks, transforms
                data = getattr(named_column_dataset, attr)  # raw data loaded
                data = particle_dataset_helper._apply_data_mask(
                    data
                )
                data = particle_dataset_helper._apply_transforms(
                    data,
                    f'{named_column_dataset.name}.{attr}'
                )
                setattr(
                    named_column_dataset,
                    f'_{attr}',
                    data
                )
            else:
                # just return the data
                pass
        try:
            # beware collisions with SWIFTParticleDataset namespace
            return object.__getattribute__(self, attr)
        except AttributeError:
            # exposes everything else in __dict__
            return getattr(named_column_dataset, attr)

    def __setattr__(self, attr, value):
        # pass particle data through to actual SWIFTNamedColumnDataset
        if not hasattr(self, '_initialised'):
            # guard during initialisation
            object.__setattr__(self, attr, value)
            return
        column_names = self._named_column_dataset.named_columns
        if (attr in column_names) or \
           ((attr.startswith('_')) and (attr[1:] in column_names)):
            setattr(
                self._named_column_dataset,
                attr,
                value
            )
            return
        else:
            object.__setattr__(self, attr, value)
            return

    def __getitem__(self, mask):
        return self._data_copy(mask=mask)

    def __copy__(self):
        return getattr(
            self._particle_dataset_helper.__copy__(),
            self.name
        )

    def __deepcopy__(self, memo=None):
        return self._data_copy()

    def _data_copy(self, mask=None):
        return getattr(
            self._particle_dataset_helper._data_copy(mask=mask),
            self.name
        )


class _SWIFTParticleDatasetHelper(object):

    def __init__(
            self,
            particle_dataset,
            swiftgalaxy
    ):
        self._particle_dataset = particle_dataset
        self._swiftgalaxy = swiftgalaxy
        self._named_column_dataset_helpers = dict()
        particle_metadata = getattr(
            self.metadata,
            f'{self.particle_name}_properties'
        )
        named_columns_names = [
            fn
            for (fn, fp)
            in zip(
                particle_metadata.field_names,
                particle_metadata.field_paths
            )
            if particle_metadata.named_columns[fp] is not None
        ]
        for named_columns_name in named_columns_names:
            # This is the named_columns instance to wrap:
            named_columns = getattr(self._particle_dataset, named_columns_name)
            # We'll make a custom type to present a nice name to the user.
            particle_nice_name = \
                swiftsimio_metadata.particle_types.particle_name_class[
                    getattr(
                        self.metadata,
                        f'{self.particle_name}_properties'
                    ).particle_type
                ]
            nice_name = f"{particle_nice_name}"\
                f"{named_columns.field_path.split('/')[-1]}ColumnsHelper"
            TypeNamedColumnDatasetHelper = type(
                nice_name,
                (_SWIFTNamedColumnDatasetHelper, object),
                dict()
            )
            self._named_column_dataset_helpers[named_columns_name] = \
                TypeNamedColumnDatasetHelper(
                    named_columns,
                    self
                )
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
        particle_metadata = getattr(metadata, f'{particle_name}_properties')
        particle_dataset = object.__getattribute__(self, '_particle_dataset')
        if attr in particle_metadata.field_names:
            # check if we're dealing with a named columns field
            if object.__getattribute__(self, '_is_namedcolumns')(attr):
                return object.__getattribute__(
                    self,
                    '_named_column_dataset_helpers'
                )[attr]
            # otherwise we're dealing with a particle data table
            if getattr(particle_dataset, f'_{attr}') is None:
                # going to read from file: apply masks, transforms
                data = getattr(particle_dataset, attr)  # raw data loaded
                data = object.__getattribute__(self, '_apply_data_mask')(
                    data
                )
                data = object.__getattribute__(self, '_apply_transforms')(
                    data,
                    attr
                )
                setattr(
                    particle_dataset,
                    f'_{attr}',
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
            f'{self._particle_dataset.particle_name}_properties'
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

    def __getitem__(self, mask):
        return self._data_copy(mask=mask)

    def __copy__(self):
        return getattr(
            self._swiftgalaxy.__copy__(),
            self.particle_name
        )

    def __deepcopy__(self, memo=None):
        return self._data_copy()

    def _data_copy(self, mask=None):
        mask_collection = MaskCollection(
            **{
                k: None if k != self.particle_name else mask
                for k in self.metadata.present_particle_names
            }
        )
        return getattr(
            self._swiftgalaxy._data_copy(mask_collection=mask_collection),
            self.particle_name
        )

    def _is_namedcolumns(self, field_name):
        particle_name = self._particle_dataset.particle_name
        particle_metadata = getattr(
            self._particle_dataset.metadata,
            f'{particle_name}_properties'
        )
        field_path = dict(zip(
            particle_metadata.field_names,
            particle_metadata.field_paths
        ))[field_name]
        return particle_metadata.named_columns[field_path] is not None

    def _apply_data_mask(self, data):
        if self._swiftgalaxy._extra_mask is not None:
            mask = self._swiftgalaxy._extra_mask.__getattribute__(
                self._particle_dataset.particle_name
            )
            if mask is not None:
                return data[mask]
        return data

    def _mask_dataset(self, mask):  # SHOULD EXPOSE TO USERS FROM SWIFTGALAXY LEVEL?
        particle_name = self._particle_dataset.particle_name
        particle_metadata = getattr(
            self._particle_dataset.metadata,
            f'{particle_name}_properties'
        )
        for field_name in particle_metadata.field_names:
            if self._is_namedcolumns(field_name):
                for named_column in getattr(self, field_name).named_columns:
                    if getattr(
                            getattr(self, field_name),
                            f'_{named_column}'
                    ) is not None:
                        setattr(
                            getattr(self, field_name),
                            f'_{named_column}',
                            getattr(
                                getattr(self, field_name),
                                f'_{named_column}'
                            )[mask]
                        )
            elif getattr(self, f'_{field_name}') is not None:
                setattr(
                    self,
                    f'_{field_name}',
                    getattr(self, f'_{field_name}')[mask]
                )
        self._mask_derived_coordinates(mask)
        if getattr(
            self._swiftgalaxy._extra_mask,
            particle_name
        ) is None:
            setattr(
                self._swiftgalaxy._extra_mask,
                particle_name,
                mask
            )
        else:
            realised_mask = np.zeros(
                getattr(
                    self._swiftgalaxy._extra_mask,
                    particle_name
                ).sum(),
                dtype=bool
            )
            realised_mask[mask] = True
            getattr(
                self._swiftgalaxy._extra_mask,
                particle_name
            )[
                getattr(
                    self._swiftgalaxy._extra_mask,
                    particle_name
                )
            ] = realised_mask
        return

    def _apply_transforms(self, data, dataset_name):
        if dataset_name in self._swiftgalaxy.transforms_like_coordinates:
            transform_units = self._swiftgalaxy.metadata.units.length
            transform = self._swiftgalaxy._coordinate_like_transform
        elif dataset_name in self._swiftgalaxy.transforms_like_velocities:
            transform_units = self._swiftgalaxy.metadata.units.length \
                / self._swiftgalaxy.metadata.units.time
            transform = self._swiftgalaxy._velocity_like_transform
        else:
            transform = None
        if transform is not None:
            data = _apply_4transform(data, transform, transform_units)
        boxsize = getattr(self._particle_dataset.metadata, 'boxsize', None)
        if dataset_name in self._swiftgalaxy.transforms_like_coordinates:
            data = _apply_box_wrap(data, boxsize)
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

    def _mask_derived_coordinates(self, mask):
        if self._cartesian_coordinates is not None:
            self._cartesian_coordinates = self._cartesian_coordinates[mask]
        if self._cartesian_velocities is not None:
            self._cartesian_velocities = self._cartesian_velocities[mask]
        if self._spherical_coordinates is not None:
            for coord in ('r', 'theta', 'phi'):
                self._spherical_coordinates[f'_{coord}'] = \
                    self._spherical_coordinates[f'_{coord}'][mask]
        if self._spherical_velocities is not None:
            for coord in ('v_r', 'v_t', 'v_p'):
                self._spherical_velocities[f'_{coord}'] = \
                    self._spherical_velocities[f'_{coord}'][mask]
        if self._cylindrical_coordinates is not None:
            for coord in ('rho', 'phi', 'z'):
                self._cylindrical_coordinates[f'_{coord}'] = \
                    self._cylindrical_coordinates[f'_{coord}'][mask]
        if self._cylindrical_velocities is not None:
            for coord in ('v_rho', 'v_phi', 'v_z'):
                self._cylindrical_velocities[f'_{coord}'] = \
                    self._cylindrical_velocities[f'_{coord}'][mask]
        return

    def _void_derived_coordinates(self):
        self._spherical_coordinates = None
        self._cylindrical_coordinates = None
        self._spherical_velocities = None
        self._cylindrical_velocities = None
        return


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
            velocities_dataset_name='velocities',
            _spatial_mask=None,
            _extra_mask=None,
            _coordinate_like_transform=None,
            _velocity_like_transform=None
    ):
        self.snapshot_filename = snapshot_filename
        self.halo_finder = halo_finder
        self.auto_recentre = auto_recentre
        if _spatial_mask is not None:
            self._spatial_mask = _spatial_mask
        else:
            self.halo_finder._init_spatial_mask(self)
        self.transforms_like_coordinates = transforms_like_coordinates
        self.transforms_like_velocities = transforms_like_velocities
        self.id_particle_dataset_name = id_particle_dataset_name
        self.coordinates_dataset_name = coordinates_dataset_name
        self.velocities_dataset_name = velocities_dataset_name
        if _coordinate_like_transform is not None:
            self._coordinate_like_transform = _coordinate_like_transform
        else:
            self._coordinate_like_transform = np.eye(4)
        if _velocity_like_transform is not None:
            self._velocity_like_transform = _velocity_like_transform
        else:
            self._velocity_like_transform = np.eye(4)
        super().__init__(
            snapshot_filename,
            mask=self._spatial_mask
        )
        self._particle_dataset_helpers = dict()
        for particle_name in self.metadata.present_particle_names:
            # We'll make a custom type to present a nice name to the user.
            nice_name = \
                swiftsimio_metadata.particle_types.particle_name_class[
                    getattr(
                        self.metadata,
                        f'{particle_name}_properties'
                    ).particle_type
                ]
            TypeDatasetHelper = type(
                f'{nice_name}DatasetHelper',
                (_SWIFTParticleDatasetHelper, object),
                dict()
            )
            self._particle_dataset_helpers[particle_name] = TypeDatasetHelper(
                super().__getattribute__(particle_name),
                self
            )

        if _extra_mask is not None:
            self._extra_mask = _extra_mask
        else:
            self._extra_mask = None
            self.halo_finder._init_extra_mask(self)
            if self._extra_mask is not None:
                # only particle ids should be loaded so far, need to mask these
                for particle_name in self.metadata.present_particle_names:
                    particle_ids = getattr(
                        getattr(self, particle_name),
                        f'_{self.id_particle_dataset_name}'
                    )
                    if getattr(self._extra_mask, particle_name) is not None:
                        setattr(
                            # bypass helper:
                            super().__getattribute__(particle_name),
                            f'_{self.id_particle_dataset_name}',
                            particle_ids[
                                getattr(self._extra_mask, particle_name)
                            ]
                        )
            else:
                self._extra_mask = MaskCollection(
                    **{k: None for k in self.metadata.present_particle_names}
                )

        if auto_recentre:
            self.recentre(self.halo_finder._centre())
            self.recentre_velocity(self.halo_finder._vcentre())

        return

    def __getitem__(self, mask_collection):
        return self._data_copy(mask_collection=mask_collection)

    def __copy__(self):
        SG = SWIFTGalaxy(
            self.snapshot_filename,
            self.halo_finder,
            auto_recentre=False,  # transforms overwritten below
            transforms_like_coordinates=self.transforms_like_coordinates,
            transforms_like_velocities=self.transforms_like_velocities,
            id_particle_dataset_name=self.id_particle_dataset_name,
            coordinates_dataset_name=self.coordinates_dataset_name,
            velocities_dataset_name=self.velocities_dataset_name,
            _spatial_mask=self._spatial_mask,
            _extra_mask=self._extra_mask,
            _coordinate_like_transform=self._coordinate_like_transform,
            _velocity_like_transform=self._velocity_like_transform
        )
        return SG

    def __deepcopy__(self, memo=None):
        return self._data_copy()

    def _data_copy(self, mask_collection=None):
        SG = self.__copy__()
        for particle_name in SG.metadata.present_particle_names:
            particle_metadata = getattr(
                SG.metadata,
                f'{particle_name}_properties'
            )
            particle_dataset_helper = getattr(self, particle_name)
            new_particle_dataset_helper = getattr(SG, particle_name)
            if mask_collection is not None:
                mask = getattr(mask_collection, particle_name)
                if mask is None:
                    mask = Ellipsis
            else:
                mask = Ellipsis
            getattr(SG, particle_name)._mask_dataset(mask)
            for field_name in particle_metadata.field_names:
                if particle_dataset_helper._is_namedcolumns(field_name):
                    named_columns_helper = getattr(
                        particle_dataset_helper,
                        field_name
                    )
                    new_named_columns_helper = getattr(
                        new_particle_dataset_helper,
                        field_name
                    )
                    for named_column in named_columns_helper.named_columns:
                        data = getattr(
                            named_columns_helper,
                            f'_{named_column}'
                        )
                        if data is not None:
                            setattr(
                                new_named_columns_helper,
                                f'_{named_column}',
                                data[mask]
                            )
                else:
                    data = getattr(
                        particle_dataset_helper,
                        f'_{field_name}'
                    )
                    if data is not None:
                        setattr(
                            new_particle_dataset_helper,
                            f'_{field_name}',
                            data[mask]
                        )
            # Don't link across objects with a view!
            if particle_dataset_helper._cartesian_coordinates is not None:
                new_particle_dataset_helper.cartesian_coordinates  # initialise
            if particle_dataset_helper._cartesian_velocities is not None:
                new_particle_dataset_helper.cartesian_velocities  # initialise
            if particle_dataset_helper._spherical_coordinates is not None:
                new_particle_dataset_helper._spherical_coordinates = dict()
                for c in ('_r', '_theta', '_phi'):
                    new_particle_dataset_helper._spherical_coordinates[c] = \
                        particle_dataset_helper._spherical_coordinates[c][mask]
            if particle_dataset_helper._spherical_velocities is not None:
                new_particle_dataset_helper._spherical_velocities = dict()
                for c in ('_v_r', '_v_t', '_v_p'):
                    new_particle_dataset_helper._spherical_velocities[c] = \
                        particle_dataset_helper._spherical_velocities[c][mask]
            if particle_dataset_helper._cylindrical_coordinates is not None:
                new_particle_dataset_helper._cylindrical_coordinates = dict()
                for c in ('_rho', '_phi', '_z'):
                    new_particle_dataset_helper._cylindrical_coordinates[c] = \
                        particle_dataset_helper._cylindrical_coordinates[c][
                            mask
                        ]
            if particle_dataset_helper._cylindrical_velocities is not None:
                new_particle_dataset_helper._cylindrical_velocities = dict()
                for c in ('_v_rho', '_v_phi', '_v_z'):
                    new_particle_dataset_helper._cylindrical_velocities[c] = \
                        particle_dataset_helper._cylindrical_velocities[c][
                            mask
                        ]
        return SG

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
                # We are entering a <ParticleType>Dataset, return helper.
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
            dataset = getattr(self, particle_name)._particle_dataset
            for field_name in rotatable:
                field_location = f"{'._'.join(field_name.rsplit('.', 1))}" \
                    if '.' in field_name else f'_{field_name}'
                field_data = _getattr_with_dots(
                    dataset,
                    field_location
                )
                if field_data is not None:
                    field_data = _apply_rotmat(field_data, rotation_matrix)
                    setattr(
                        dataset,
                        field_location,
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
            dataset = getattr(self, particle_name)._particle_dataset
            for field_name in translatable:
                field_location = f"{'._'.join(field_name.rsplit('.', 1))}" \
                    if '.' in field_name else f'_{field_name}'
                field_data = _getattr_with_dots(
                    dataset,
                    field_location
                )
                if field_data is not None:
                    field_data = _apply_translation(field_data, translation)
                    _setattr_with_dots(
                        dataset,
                        field_location,
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
            dataset = getattr(self, particle_name)._particle_dataset
            for field_name in self.transforms_like_coordinates:
                field_location = f"{'._'.join(field_name.rsplit('.', 1))}" \
                    if '.' in field_name else f'_{field_name}'
                field_data = _getattr_with_dots(
                    dataset,
                    field_location
                )
                if field_data is not None:
                    field_data = _apply_box_wrap(
                        field_data,
                        self.metadata.boxsize
                    )
                    setattr(
                        dataset,
                        field_location,
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
            getattr(self, particle_name)._void_derived_coordinates()
        return
