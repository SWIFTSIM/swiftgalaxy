Halo finders
============

:mod:`swiftgalaxy` uses a helper class to create a uniform interface to outputs from different halo finders. Provided a python library to read the halo finder outputs already exists, this helper class is usually lightweight and easy to create. Currently, the Velociraptor halo finder has built-in support, and support for SOAP is planned. Other halo finders may be supported on request -- pull requests to the repository are also welcome.

The second argument to create a :class:`~swiftgalaxy.reader.SWIFTGalaxy` is an instance of a class derived from the base helper class :class:`~swiftgalaxy.halo_finders._HaloFinder`, such as :class:`~swiftgalaxy.halo_finders.Velociraptor`. This object has multiple roles. It will be aware of:

  + the location of the halo finder output files;
  + how to extract the properties of a galaxy of interest from those files;
  + how to create a mask specifying the particles belonging to a galaxy of interest.

Velociraptor
------------

The Velociraptor_ halo finder is used in several flagship simulation projects, such as Colibre. The :class:`~swiftgalaxy.halo_finders.Velociraptor` helper class relies on the :mod:`velociraptor` interface to the halo finder outputs. Setting up an instance of the helper class is straightforward. If the halo finder outputs are named, for example, :file:`{halos}.properties`, :file:`{halos}.catalog_groups`, etc., and the galaxy of interest occupies the 4th row in the catalogue (``halo_index=3``, since rows are indexed from 0), then:

.. _Velociraptor: https://ui.adsabs.harvard.edu/abs/2019PASA...36...21E/abstract

.. code-block:: python

    cat = Velociraptor(
        'halos',
	halo_index=3
    )

The first argument could also include a path if needed, e.g. :file:`'/path/to/{halos}'`.

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
	extra_mask='bound_only'
    )

This behaviour can be adjusted. If ``None`` is passed instead, then only the spatial masking provided by :func:`velociraptor.swift.swift.generate_spatial_mask` is used. This means that all particles the set of (probably cubic) subvolumes of the simulation that overlap with the region of interest will be read in. Alternatively, a :class:`~swiftgalaxy.masks.MaskCollection` can be provided. This will be used to select particles from those already selected using :func:`~velociraptor.swift.swift.generate_spatial_mask`.

If a different subset of particles is desired, often the most practical option is to first set up the :class:`~swiftgalaxy.reader.SWIFTGalaxy` with either ``extra_mask='bound_only'`` or ``extra_mask=None`` and then use the loaded particles to :doc:`compute a new mask that can then be applied <../masking/index>`, perhaps permanently. Since all particles in the spatial region defined by :func:`~velociraptor.swift.swift.generate_spatial_mask` will always be read in any case, this does not imply any loss of efficiency.

The Velociraptor halo finder computes several centres for halos. By default, the location of the gravitational potential minimum is assumed as the centre of the galaxy (and will be used to :doc:`set the coordinate system <../coordinate_transformations/index>`, unless the argument ``auto_recentre=False`` is passed to :class:`~swiftgalaxy.reader.SWIFTGalaxy`). Usually the available centring options are:

  + ``'minpot'`` -- potential minimum
  + ``''`` -- centre of mass (?)
  + ``'_gas'`` -- gas centre of mass (?)
  + ``'_star'`` -- stellar centre of mass (?)
  + ``'mbp'`` -- most bound particle

These can be used as, for example:

.. code-block:: python

    Velociraptor(
        ...,
	centre_type='mbp'
    )

SOAP
----

Future support for `SOAP` is planned.

Other halo finders
------------------

Support for other halo finders will be considered on request.

Entrepreneurial users may also create their own helper class inheriting from :class:`swiftgalaxy.halo_finders._HaloFinder`. In this case, the following methods should be implemented:

  + :meth:`~swiftgalaxy.halo_finders._HaloFinder._load`: called during :meth:`~swiftgalaxy.halo_finders._HaloFinder.__init__`, implement any initialisation tasks here.
  + :meth:`~swiftgalaxy.halo_finders._HaloFinder._get_spatial_mask`: return a :class:`~swiftsimio.masks.SWIFTMask` defining the spatial region to be loaded for the galaxy of interest.
  + :meth:`~swiftgalaxy.halo_finders._HaloFinder._get_extra_mask`: return a :class:`~swiftgalaxy.masks.MaskCollection` defining the subset of particles from the loaded spatial region that belong to the galaxy of interest.
  + :meth:`~swiftgalaxy.halo_finders._HaloFinder._centre`: return the coordinates (as a :class:`~swiftsimio.objects.cosmo_array`) to be used as the centre of the galaxy of interest.
  + :meth:`~swiftgalaxy.halo_finders._HaloFinder._vcentre`: return the coordinates (as a :class:`~swiftsimio.objects.cosmo_array`) to be used as the bulk velocity of the galaxy of interest.

In addition, it is recommended to expose the properties computed by the halo finder, masked to the values corresponding to the object of interest. To make this intuitive for users, the syntax to access attributes of the galaxy of interest should preferably match the syntax used for the library conventionally used to read outputs of that halo finder, if it exists. For instance, for Velociraptor this is implemented via ``__getattr__`` (dot syntax), which simply exposes the usual interface (with a mask to pick out the galaxy of interest).
