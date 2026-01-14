Coordinate transformations
==========================

A :class:`~swiftgalaxy.reader.SWIFTGalaxy` enforces a consistent coordinate system for all particle types and coordinate and velocity arrays (accelerations are not supported, yet). Any relevant particle arrays already in memory are modified when a new transformation is applied, and any relevant particle arrays loaded later will be automatically transformed to the current frame.

The current centre and velocity centre of the coordinate system (in simulation coordinates) are available as properties of a :class:`~swiftgalaxy.reader.SWIFTGalaxy`. The current rotation (about the origin of the simulation coordinates) is also available:

.. code-block:: python

    sg = SWIFTGalaxy(...)
    sg.centre
    sg.velocity_centre
    sg.rotation

It may be that along with the basic coordinates and velocities arrays, you have additional arrays that you wish to live in the same coordinate system. We could imagine, for instance, a property called ``wind_velocity`` attached to star particles and containing a vector giving the velocity of a (directional) stellar wind in simulation box coordinates. We can use the ``transforms_like_velocities`` to specify that this property should have any coordinate transformations that would be relevant to the ``velocities`` of particles applied to these as well:

.. code-block:: python

    SWIFTGalaxy(
        ...,
        transforms_like_coordinates=set(),
        transforms_like_velocities={'wind_velocity'},
    )

The equivalent for coordinate-like arrays is the ``transforms_like_coordinates`` argument. The ``coordinates`` and ``velocities`` arrays are always assumed to transform as coordinates and velocities, respectively. Note that it is not necessary that a name provided in one of these arguments be present for all particle types.

By default, :class:`~swiftgalaxy.reader.SWIFTGalaxy` assumes that particle coordinates are stored in a dataset (one for each particle type) called ``coordinates``, and velocities in datasets called ``velocities``. If your data use non-standard naming of the coordinate and velocity arrays, you can provide alternative names (replace the defaults in the example snippet below with your custom names). Note that the datasets named in these arguments are implicitly added to the sets ``transforms_like_coordinates`` and ``transforms_like_velocities``, respectively.

.. code-block:: python

    SWIFTGalaxy(
        ...,
        coordinates_dataset_name='coordinates',
        velocities_dataset_name='velocities',
    )


Recentering
-----------

Automatic recentering
^^^^^^^^^^^^^^^^^^^^^

By default, a :class:`~swiftgalaxy.reader.SWIFTGalaxy` is recentered on the centre of the galaxy (as defined by the halo catalogue), and its velocity is shifted to follow the bulk velocity of the galaxy (again as defined by the halo catalogue). This behaviour can be disabled by setting ``SWIFTGalaxy(..., auto_recentre=False)``. Note that this recentering only applies to particles -- no coordinate transformations of any kind are every applied to entries in the halo catalogues, so for instance querying the position of the galaxy in the halo catalogue will always give this in the simulation box frame.

Some halo catalogues define more than one centre and/or bulk velocity. The one used is the one returned by :attr:`swiftgalaxy.halo_catalogues._HaloCatalogue.centre` (for the centre) and :attr:`swiftgalaxy.halo_catalogues._HaloCatalogue.velocity_centre` (for the bulk velocity). The centre to be used is configurable if the halo catalogue defines multiple centres. In the case of a :class:`~swiftgalaxy.halo_catalogues.SOAP` halo catalogue, for example, the centre and bulk velocity used can be manipulated by the ``centre_type`` and ``velocity_centre_type`` arguments. By default :class:`~swiftgalaxy.halo_catalogues.SOAP` uses the ``input_halos.halo_centre`` for the position and ``bound_subhalo.centre_of_mass_velocity`` for the bulk velocity. The centre in position could be changed to also use the centre of mass like this:

.. code-block:: python

    SWIFTGalaxy(
        ...,
        SOAP(
            ...,
            centre_type='bound_subhalo.centre_of_mass',
        ),
    )

Manual recentering
^^^^^^^^^^^^^^^^^^

You may always choose a new coordinate centre or bulk velocity by providing the new centre (or bulk velocity) *in the current coordinate frame* to the appropriate function:

+ :meth:`~swiftgalaxy.reader.SWIFTGalaxy.recentre`
+ :meth:`~swiftgalaxy.reader.SWIFTGalaxy.recentre_velocity`

Recall that :mod:`swiftgalaxy` is unit-aware, so the centres must come with units -- these can be any compatible unit; conversions are handled internally. For example, for a Milky Way-like galaxy already centred on the galactic centre and rotated to lie in the :math:`x-y` plane, switching to a heliocentric frame could be achieved with something like:

.. code-block:: python

    import unyt as u
    sg = SWIFTGalaxy(...)
    ...  # presumably need to perform a rotation to align the plane
    sg.recentre((8, 0, 0) * u.kpc)
    sg.recentre_velocity((220, 0, 0) * u.km * u.s**-1)

Translations
------------

Very similarly to manually recentering the coordinate or velocity frame, functions are provided to apply a translation to the particle coordinate or velocity arrays. Note that velocity translations are referred to as *boosts*:

+ :meth:`~swiftgalaxy.reader.SWIFTGalaxy.translate`
+ :meth:`~swiftgalaxy.reader.SWIFTGalaxy.boost`

The only difference is that these are more convenient when you know the vector to translate by, instead of the vector pointing to the new centre. Keep in mind that the translation vector is interpreted *in the current frame of reference*.

Rotations
---------

Rotations of the coordinate frame apply to both particle coordinates and velocities and are therefore applied to datasets specified by both ``transforms_like_coordinates`` and ``transforms_like_velocities`` (see above). As with all coordinate transformations, a rotation is always interpreted *in the current frame of reference*.

Flexible encoding of rotations (e.g. from a rotation matrix, or Euler angles, or quaternions, etc.) are enabled via the :class:`~scipy.spatial.transform.Rotation` class. For example, if the rotation matrix is known it can be provided as (here with an identity rotation):

.. code-block:: python

    from scipy.spatial.transform import Rotation
    sg = SWIFTGalaxy(...)
    sg.rotate(
        Rotation.from_matrix(
            [[1, 0, 0],
             [0, 1, 0],
             [0, 0, 1]]
        )
    )

For the full list of encodings supported by :class:`~scipy.spatial.transform.Rotation`, see its documentation.

Box wrapping
------------

For a periodic simulation box, the spatial coordinates will automatically be wrapped as necessary to ensure that their absolute values remain less than half a box length. If for any reason you wish to force a box wrapping operation simply call :meth:`~swiftgalaxy.reader.SWIFTGalaxy.wrap_box`.

Copying a coordinate system
---------------------------

A second :class:`~swiftgalaxy.reader.SWIFTGalaxy` can be created using the (current) coordinate system of an existing instance:

.. code-block:: python

    sg1 = SWIFTGalaxy(...)
    sg1.rotate(...)
    sg1.translate(...)
    sg2 = SWIFTGalaxy(..., auto_recentre=False, coordinate_frame_from=sg1)

This is only possible at creation time, and ``auto_recentre`` must be set to ``False``. Also note that this does not lock the coordinate systems together. For instance, if one galaxy is subsequently rotated, the second will not rotate with it.
