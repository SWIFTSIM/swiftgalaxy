Halo finders
============

:mod:`swiftgalaxy` uses a helper class to create a uniform interface to outputs from different halo finders. Provided a python library to read the halo finder outputs already exists, this helper class is usually lightweight and easy to create. Currently, the Velociraptor halo finder and Caesar catalogue format have built-in support, and support for SOAP is planned. Other halo finders may be supported on request -- pull requests to the repository are also welcome.

The second argument to create a :class:`~swiftgalaxy.reader.SWIFTGalaxy` is an instance of a class derived from the base helper class :class:`~swiftgalaxy.halo_finders._HaloFinder`, such as :class:`~swiftgalaxy.halo_finders.Velociraptor`. This object has multiple roles. It will be aware of:

  + the location of the halo finder output files;
  + how to extract the properties of a galaxy of interest from those files;
  + how to create a mask specifying the particles belonging to a galaxy of interest.

SOAP
----

Future support for `SOAP`_ is planned as soon as a stable python interface to SOAP catalogues becomes available. Such an interface is currently in development.

.. _SOAP: https://github.com/SWIFTSIM/SOAP

Caesar
------

The Caesar catalogue format is popular in the Simba_ simulation community and lives within the yt_ ecosystem. The :class:`~swiftgalaxy.halo_finders.Caesar` helper class relies on the :class:`~loader.Group` interface to the halo finder outputs. Setting up an instance of the helper class is straightforward. We'll assume a Caesar output called :file:`caesar_catalogue.hdf5`. There are two types of groups compatible with :class:`~swiftgalaxy.reader.SWIFTGalaxy` that are defined in these catalogues: halos and galaxies (refer to the Caesar documentation for details). The type of object is specified in a ``group_type`` argument (valid values are ``"halo:`` or ``"galaxy"``). The position of the object of interest in the corresponding Caesar list is specified in the ``group_index`` argument. For example, to choose the 4th entry in the halo list (``halo_index=3``, since the list is indexed from 0):

.. _Simba: http://simba.roe.ac.uk/
.. _yt: https://yt-project.org/doc/index.html

.. code-block:: python

    cat = Caesar(
        caesar_file="caesar_catalogue.hdf5",
	group_type="halo",
	group_index=3,
    )

The first argument could also include a path if needed, e.g. :file:`"/path/to/caesar_catalogue.hdf5"`.

The properties of the object of interest are made conveniently available with the :meth:`~swiftgalaxy.halo_finders.Caesar.__getattr__` (dot) syntax, which exposes the interface provided by the :class:`~loader.Group` class. For example, the :meth:`~loader.Group.info` function familiar to Caesar users (e.g. using the caesar tools ``load("caesar_catalogue.hdf5").galaxies[3].info()``) is available as:

.. code-block:: python

    cat.info()

This lists available integrated properties of the object of interest, for example the virial mass (if available) would be accessed as:

.. code-block:: python

    cat.virial_quantities["m200c"]

Caesar is compatible with yt and returns values with units specified with yt that :mod:`unyt` understands by default.

.. warning ::

    Caesar defines its own unit registry that specifies how some customised units convert to units provided by yt. For example, a `Mpccm` (co-moving Mpc) unit is defined. Because :mod:`swiftsimio` provides its own custom implementation of co-moving units that is not explicitly aware of the :class:`~main.CAESAR` implementation, but both are compatible with yt, some issues can arise. The :class:`~swiftsimio.objects.cosmo_array` provided by :mod:`swiftsimio` does produce a warning when potentially ambiguous calculations are attempted (e.g. where its doesn't know that both argument are co-moving, or that both are physical), and this will trigger on calculations mixing incompatible :class:`~main.CAESAR`-style and :class:`~swiftsimio.objects.cosmo_array` units. However, occasionally :mod:`swiftsimio` uses bare :mod:`unyt` quantities or arrays, and if a :class:`~main.CAESAR`-style quantity collides with one of these in a calculation silent and incorrect conversion from comoving to physical units (or any other redshift-dependent units) can occur. It is therefore recommended that users convert :class:`~main.CAESAR`-style quantities to use :class:`~swiftsimio.objects.cosmo_array` before they are passed to :mod:`swiftsimio` or :mod:`swiftgalaxy` functions. For example:

    .. code-block:: python

        import unyt as u
        from swiftsimio.objects import cosmo_array, cosmo_factor
	scale_factor = ...  # retrieve scale factor from snapshot or catalogue file
        cosmo_array(
	    cat.virial_quantities["r200c"].to(u.kpc),  # ensures physical units
	    comoving=False,
	    cosmo_factor=cosmo_factor(a**1, scale_factor=scale_factor)
	).to_comoving()  # or leave in physical units if desired

Usually the :class:`~swiftgalaxy.halo_finders.Caesar` object is used to create a :class:`~swiftgalaxy.reader.SWIFTGalaxy` object. In this case the interface is exposed through the ``halo_finder`` attribute, for example:

.. code-block:: python

    sg = SWIFTGalaxy(
        ...,
	Caesar(...),
    )
    sg.halo_finder.info()
    sg.halo_finder.virial_quantities["m200c"]

By default, the :class:`~swiftgalaxy.halo_finders.Caesar` class will identify the particles that the halo finder deems bound to the object as belonging to the galaxy. This is controlled by the argument:

.. code-block:: python

    Caesar(
        ...,
	extra_mask="bound_only"
    )

This behaviour can be adjusted. If ``None`` is passed instead, then only the spatial masking provided by :meth:`~swiftgalaxy.halo_finders.Caesar._get_spatial_mask` is used. This means that all particles in the set of (probably cubic) subvolumes of the simulation that overlap with the region of interest will be read in. Alternatively, a :class:`~swiftgalaxy.masks.MaskCollection` can be provided. This will be used to select particles from those already selected spatially.

.. warning::

   Older :class:`~main.CAESAR` outputs (prior to updates to the package in October 2023) do not contain enough information to define a spatial sub-region to take advantage of :mod:`swiftsimio`'s spatial masking. :mod:`swiftgalaxy` is still compatible with these older output files but properties of all particles in the box will be read and then masked down to the object of interest, which is very inefficient. When :mod:`swiftgalaxy` doesn't find the information needed for spatial masking in a :class:`~main.CAESAR` output file, it will produce a warning at runtime before proceeding (very inefficiently).

If a different subset of particles is desired, often the most practical option is to first set up the :class:`~swiftgalaxy.reader.SWIFTGalaxy` with either ``extra_mask="bound_only"`` or ``extra_mask=None`` and then use the loaded particles to :doc:`compute a new mask that can then be applied <../masking/index>`, perhaps permanently. Since all particles within the spatially masked region will always be read in any case, this does not imply any loss of efficiency.

The Caesar catalogue lists two centres for halos and galaxies. By default, the location of the gravitational potential minimum is assumed as the centre of the objet (and will be used to :doc:`set the coordinate system <../coordinate_transformations/index>`, unless the argument ``auto_recentre=False`` is passed to :class:`~swiftgalaxy.reader.SWIFTGalaxy`). Usually the available centring options are:

  + ``"minpot"`` -- potential minimum
  + ``""`` -- centre of mass

These can be used as, for example:

.. code-block:: python

    Caesar(
        ...,
	centre_type="",  # centre of mass (no suffix in Caesar catalogue)
    )


Velociraptor
------------

Velociraptor_ is a widely-used halo finder. Some SWIFT-based simulations projects have used it, but are largely moving to a model where particles are assigned to halos with Velociraptor (or another finder) and a catalogue is produced with the `SOAP`_ tool. The Velociraptor catalogue format is therefore falling somewhat out of fashion in the SWIFT community. It is supported for use with :class:`~swiftgalaxy.reader.SWIFTGalaxy`, but is unlikely to be further developed or maintained. The :class:`~swiftgalaxy.halo_finders.Velociraptor` helper class relies on the :mod:`velociraptor` interface to the halo finder outputs. Setting up an instance of the helper class is straightforward. If the halo finder outputs are named, for example, :file:`{halos}.properties`, :file:`{halos}.catalog_groups`, etc., and the galaxy of interest occupies the 4th row in the catalogue (``halo_index=3``, since rows are indexed from 0), then:

.. _Velociraptor: https://ui.adsabs.harvard.edu/abs/2019PASA...36...21E/abstract
.. _SOAP: https://github.com/SWIFTSIM/SOAP

.. code-block:: python

    cat = Velociraptor(
        "halos",
	halo_index=3
    )

The first argument could also include a path if needed, e.g. :file:`"/path/to/{halos}"`.

.. warning ::

    Currently the :mod:`velociraptor` module does not support selecting galaxies by a unique identifier, e.g. ``cat.ids.id``. Users are advised to check this identifier for their selected galaxy to ensure that they obtain the object that they expected.

The properties of the galaxy of interest as calculated by Velociraptor are made conveniently available with the :meth:`~swiftgalaxy.halo_finders.Velociraptor.__getattr__` (dot) syntax, which exposes the interface provided by the :mod:`velociraptor` module. For example, the virial mass can be accessed as ``cat.masses.mvir``. Lists of available properties can be printed interactively using ``print(cat)`` (or simply ``cat`` at the python prompt), or ``print(cat.masses)``, etc. When a :class:`~swiftgalaxy.halo_finders.Velociraptor` instance is used to initialize a :class:`~swiftgalaxy.reader.SWIFTGalaxy`, it is made available through the ``halo_finder`` attribute. For example, to access the virial mass:

.. code-block:: python

    sg = SWIFTGalaxy(
        ...,
	Velociraptor(
	    ...
	)
    )
    sg.halo_finder.masses.mvir
    
By default, the :class:`~swiftgalaxy.halo_finders.Velociraptor` class will identify the particles that the halo finder deems bound to the object as belonging to the galaxy. This is controlled by the argument:

.. code-block:: python

    Velociraptor(
        ...,
	extra_mask="bound_only"
    )

This behaviour can be adjusted. If ``None`` is passed instead, then only the spatial masking provided by :func:`velociraptor.swift.swift.generate_spatial_mask` is used. This means that all particles in the set of (probably cubic) subvolumes of the simulation that overlap with the region of interest will be read in. Alternatively, a :class:`~swiftgalaxy.masks.MaskCollection` can be provided. This will be used to select particles from those already selected using :func:`~velociraptor.swift.swift.generate_spatial_mask`.

If a different subset of particles is desired, often the most practical option is to first set up the :class:`~swiftgalaxy.reader.SWIFTGalaxy` with either ``extra_mask="bound_only"`` or ``extra_mask=None`` and then use the loaded particles to :doc:`compute a new mask that can then be applied <../masking/index>`, perhaps permanently. Since all particles in the spatial region defined by :func:`~velociraptor.swift.swift.generate_spatial_mask` will always be read in any case, this does not imply any loss of efficiency.

The Velociraptor halo finder computes several centres for halos. By default, the location of the gravitational potential minimum is assumed as the centre of the galaxy (and will be used to :doc:`set the coordinate system <../coordinate_transformations/index>`, unless the argument ``auto_recentre=False`` is passed to :class:`~swiftgalaxy.reader.SWIFTGalaxy`). Usually the available centring options are:

  + ``"minpot"`` -- potential minimum
  + ``""`` -- centre of mass (?)
  + ``"_gas"`` -- gas centre of mass (?)
  + ``"_star"`` -- stellar centre of mass (?)
  + ``"mbp"`` -- most bound particle

These can be used as, for example:

.. code-block:: python

    Velociraptor(
        ...,
	centre_type="mbp"
    )

Other halo finders
------------------

Support for other halo finders will be considered on request.

Entrepreneurial users may also create their own helper class inheriting from :class:`swiftgalaxy.halo_finders._HaloFinder`. In this case, the following methods should be implemented:

  + :meth:`~swiftgalaxy.halo_finders._HaloFinder._load`: called during :meth:`~swiftgalaxy.halo_finders._HaloFinder.__init__`, implement any initialisation tasks here.
  + :meth:`~swiftgalaxy.halo_finders._HaloFinder._get_spatial_mask`: return a :class:`~swiftsimio.masks.SWIFTMask` defining the spatial region to be loaded for the galaxy of interest.
  + :meth:`~swiftgalaxy.halo_finders._HaloFinder._get_extra_mask`: return a :class:`~swiftgalaxy.masks.MaskCollection` defining the subset of particles from the loaded spatial region that belong to the galaxy of interest.
  + :meth:`~swiftgalaxy.halo_finders._HaloFinder.centre`: return the coordinates (as a :class:`~swiftsimio.objects.cosmo_array`) to be used as the centre of the galaxy of interest (implemented with the `@property` decorator).
  + :meth:`~swiftgalaxy.halo_finders._HaloFinder.velocity_centre`: return the coordinates (as a :class:`~swiftsimio.objects.cosmo_array`) to be used as the bulk velocity of the galaxy of interest (implemented with the `@property` decorator).

In addition, it is recommended to expose the properties computed by the halo finder, masked to the values corresponding to the object of interest. To make this intuitive for users, the syntax to access attributes of the galaxy of interest should preferably match the syntax used for the library conventionally used to read outputs of that halo finder, if it exists. For instance, for Velociraptor this is implemented via ``__getattr__`` (dot syntax), which simply exposes the usual interface (with a mask to pick out the galaxy of interest).
