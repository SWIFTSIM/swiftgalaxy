Additional coordinates
======================

When analysing a galaxy it is often convenient to work in spherical or cylindrical coordinates. A :class:`~swiftgalaxy.reader.SWIFTGalaxy` will compute these (and the velocities) on the fly if requested. The poles of the spherical/cylindrical coordinate frame are assumed to lie along the cartesian :math:`z` axis of the :doc:`current coordinate frame <../coordinate_transformations/index>`, and the coordinate system is assumed to be right-handed.

Any computed coordinates (or velocities) are cached to speed up subsequent requests, but note that :doc:`coordinate transformations <../coordinate_transformations/index>` in general would require a conversion back to cartesian coordinates, application of the transformation, and then re-calculation of the spherical/cylindrical coordinates. Therefore, in the interest of efficiency, if a coordinate transformation occurs the spherical/cylindrical coordinates are simply discarded, and re-computed if they are subsequently requested.

Spherical coordinates
---------------------

.. note::
   
   A "physics" notation convention is assumed, with coordinate components named :math:`(r, \theta, \phi)`, where :math:`-\frac{\pi}{2} \leq \theta \leq \frac{\pi}{2}` is the polar angle and :math:`0 < \phi \leq 2\pi` is the azimuthal angle.

The spherical coordinates can be accessed through the :attr:`~swiftgalaxy.reader._SWIFTParticleDatasetHelper.spherical_coordinates` property of a particle dataset. Supposing that we are interested in the ``gas`` particles:

.. code-block:: python

    sg = SWIFTGalaxy(...)
    sg.gas.spherical_coordinates.r
    sg.gas.spherical_coordinates.theta
    sg.gas.spherical_coordinates.phi

For convenience and/or readability, some common (?) aliases to the coordinates are also supported:

+ ``spherical_coordinates.r``:
    + ``spherical_coordinates.radius``
+ ``spherical_coordinates.theta``:
    + ``spherical_coordinates.lat``
    + ``spherical_coordinates.latitude``
    + ``spherical_coordinates.pol``
    + ``spherical_coordinates.polar``
+ ``spherical_coordinates.phi``:
    + ``spherical_coordinates.lon``
    + ``spherical_coordinates.longitude``
    + ``spherical_coordinates.az``
    + ``spherical_coordinates.azimuth``

Spherical velocities
^^^^^^^^^^^^^^^^^^^^

The velocity components in the directions of the spherical unit vectors -- :math:`(v_r, v_\theta, v_\phi)` -- can be accessed with the following syntax. Again, some aliases are provided:

+ ``spherical_velocities.r``:
    + ``spherical_velocities.radius``
+ ``spherical_velocities.theta``:
    + ``spherical_velocities.lat``
    + ``spherical_velocities.latitude``
    + ``spherical_velocities.pol``
    + ``spherical_velocities.polar``
+ ``spherical_velocities.phi``:
    + ``spherical_velocities.lon``
    + ``spherical_velocities.longitude``
    + ``spherical_velocities.az``
    + ``spherical_velocities.azimuth``

Cylindrical coordinates
-----------------------

.. note::
   
   The coordinate components are named :math:`(\rho, \phi, z)` by default, and assume a convention where :math:`0 < \phi \leq 2\pi`.

Similarly to the spherical coordinates, the cylindrical coordinates can be accessed through the :attr:`~swiftgalaxy.reader._SWIFTParticleDatasetHelper.cylindrical_coordinates` property of a particle dataset. Supposing again that we are interested in the ``gas`` particles:

.. code-block:: python

    sg = SWIFTGalaxy(...)
    sg.gas.cylindrical_coordinates.rho
    sg.gas.cylindrical_coordinates.phi
    sg.gas.cylindrical_coordinates.z

With aliases:

+ ``cylindrical_coordinates.rho``:
    + ``cylindrical_coordinates.R``
    + ``cylindrical_coordinates.radius``
+ ``cylindrical_coordinates.phi``:
    + ``cylindrical_coordinates.lon``
    + ``cylindrical_coordinates.longitude``
    + ``cylindrical_coordinates.az``
    + ``cylindrical_coordinates.azimuth``
+ ``cylindrical_coordinates.z``

Cylindrical velocities
^^^^^^^^^^^^^^^^^^^^^^

The velocity components in the directions of the cylindrical unit vectors -- :math:`(v_\rho, v_\phi, v_z)` -- can be accessed with the following syntax. Again, some aliases are provided:

+ ``cylindrical_velocities.rho``:
    + ``cylindrical_velocities.R``
    + ``cylindrical_velocities.radius``
+ ``cylindrical_coordinates.phi``:
    + ``cylindrical_velocities.lon``
    + ``cylindrical_velocities.longitude``
    + ``cylindrical_velocities.az``
    + ``cylindrical_velocities.azimuth``
+ ``cylindrical_velocities.z``

Cartesian coordinates
---------------------

Cartesian velocities
^^^^^^^^^^^^^^^^^^^^