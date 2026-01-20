Halo catalogues
===============

:mod:`swiftgalaxy` uses a helper class to create a uniform interface to outputs from
different halo finders. Provided a python library to read the halo catalogues already
exists, this helper class is usually lightweight and easy to create. Currently, the SOAP,
Caesar and Velociraptor catalogue formats have built-in support. Other halo finders may be
supported on request -- pull requests to the repository are also welcome.

The second argument to create a :class:`~swiftgalaxy.reader.SWIFTGalaxy` is an instance of
a class derived from the base helper class
:class:`~swiftgalaxy.halo_finders._HaloCatalogue`, such as
:class:`~swiftgalaxy.halo_catalogues.SOAP`. This object has multiple roles. It will be
aware of:

  + the location of the halo catalogue files;
  + how to extract the properties of a galaxy of interest from those files;
  + how to create a mask specifying the particles belonging to a galaxy of interest.

SOAP
----

The `SOAP`_ catalogue format is the preferred catalogue format of the `SWIFT community`_
and is supported with i/o tools in :mod:`swiftsimio`. The
:class:`~swiftgalaxy.halo_catalogues.SOAP` helper class uses :mod:`swiftsimio` to read the
information that it needs from the SOAP catalogues.

.. _SWIFT community: https://github.com/SWIFTSIM
.. _SOAP: https://github.com/SWIFTSIM/SOAP

Setting up an instance of the helper class is straightforward. We'll assume a SOAP
catalogue called :file:`halo_properties_0123.hdf5`, and suppose that we're interested in
the first object in the catalogue (row ``0``):

.. code-block:: python

    soap = SOAP(
        soap_file="halo_properties_0123.hdf5",
        soap_index=0,
    )

The first argument could also include a path if needed, e.g.
:file:`/path/to/halo_properties_0123.hdf5`.

The properties of the object of interest are made conveniently available with the
:meth:`~swiftgalaxy.halo_catalogues.SOAP.__getattr__` (dot) syntax, which exposes the
interface provided by the :class:`~swiftsimio.reader.SWIFTDataset` object handling the
SOAP catalogue. For example, the virial mass (M200crit) of our object of interest can be
retrieved in the same way as if we were using :mod:`swiftsimio` to read the SOAP
catalogue, except that in this case we only get the row corresponding to our chosen
object:

.. code-block:: python

    soap.spherical_overdensity_200_crit.total_mass.to(u.Msun)

SOAP calculates well over a hundred integrated quantities for each halo in the catalogue.
All are available for the object of interest using similar syntax. Refer to the SOAP and
:mod:`swiftsimio` documentation for further details of what quantities are available.

Usually the :class:`~swiftgalaxy.halo_catalogues.SOAP` object is used to create a
:class:`~swiftgalaxy.reader.SWIFTGalaxy` object. Assuming that we have a simulation
snapshot file :file:`snapshot_0123.hdf5` that goes with the catalogue file
:file:`halo_properties_0123.hdf5` the object is created as:

.. code-block:: python

    sg = SWIFTGalaxy(
        "snapshot_0123.hdf5",
        SOAP(
            "halo_properties_0123.hdf5",
            soap_index=0,
        )
    )

.. note::

   SOAP records which particles belong to each individual halo in a set of "membership"
   files, usually found alongside the halo catalogue (e.g.
   :file:`halo_properties_0123.hdf5`) in a subdirectory, e.g.
   :file:`membership_0123/membership_0123.X.hdf5` (where ``X`` is replaced by integers).
   :mod:`swiftgalaxy` expects to find the information contained in these files directly in
   the (single, monolithic) simulation snapshot file. Some simulations (including Colibre)
   provide a snapshot that includes the membership information already. If such a file is
   not available, the SOAP `code distribution`_ comes with a script
   ``make_virtual_snapshot.py`` that can create the necessary snapshot file containing the
   particle membership information. The file is "virtual" in the sense that it doesn't
   directly store (i.e. copy) the data in the snapshot and membership files but instead
   contains hyperlinks to the existing data files, providing a single file interface to
   all of the relevant information. The script help information is available with
   ``python make_virtual_snapshot.py --help``. In our example we could create the
   "virtual" snapshot file as:

   .. code-block:: bash

       python make_virtual_snapshot.py \
       --absolute-paths \
       # input virtual snapshot without membership information:
       'snapshot_{snap_nr:04}.hdf5' \
       # input membership files:
       'membership_{snap_nr:04}/membership_{snap_nr:04}.{file_nr}.hdf5' \
       # output virtual snapshot with membership information:
       'snapshot_{snap_nr:04}.hdf5' \
       # snapshot number:
       123

   Notice that this script wants a virtual snapshot file as input. This file is copied, so
   while the script will (probably) work on a non-virtual input snapshot, this will result
   in data duplication on disk. The ``{snap_nr:04)`` is the pattern replaced with the
   snapshot number provided as the last argument. The ``{file_nr}`` is replaced with the
   number of each file. Attempting to use :mod:`swiftgalaxy` with a snapshot file that
   does not contain the particle membership information will result in an error similar
   to ``AttributeError: 'GasDataset' object has no attribute 'group_nr_bound'``.

.. _code distribution: https://github.com/SWIFTSIM/SOAP

When working with a :class:`~swiftgalaxy.reader.SWIFTGalaxy` object the interface to the
integrated properties is exposed through the ``halo_catalogue`` attribute, for example:

.. code-block:: python

    sg.halo_catalogue.spherical_overdensity_200_crit.total_mass.to(u.Msun)

By default, the :class:`~swiftgalaxy.halo_catalogues.SOAP` class will identify the
particles that the halo finder deems bound to the object as belonging to the galaxy. This
is controlled by the argument:

.. code-block:: python

    SOAP(
        ...,
        extra_mask="bound_only"
    )

This behaviour can be adjusted. If ``None`` is passed instead, then only the spatial
masking (provided internally by
:meth:`~swiftgalaxy.halo_catalogues.SOAP._get_spatial_mask`) is used. This means that all
particles in the set of (probably cubic) subvolumes of the simulation that overlap with
the region of interest will be read in. Alternatively, a
:class:`~swiftgalaxy.masks.MaskCollection` can be provided for finer control of the
particle selection. This will be used to select particles from those already selected
spatially.

If a different subset of particles is desired, often the most practical option is to first
set up the :class:`~swiftgalaxy.reader.SWIFTGalaxy` with either
``extra_mask="bound_only"`` or ``extra_mask=None`` and then use the loaded particles to
:doc:`compute a new mask that can then be applied <../masking/index>`, perhaps
permanently. Since all particles within the spatially masked region will always be read in
any case, this does not imply any loss of efficiency.

SOAP catalogues lists many centres for halos. :mod:`swiftgalaxy` uses the "input halo
centre" (for the HBT+ halo finder this is the centre of potential), and the mass-weighted
average velocity of bound particles in the catalogue, as the
:doc:`default coordinate origin <../coordinate_transformations/index>` (unless the
argument ``auto_recentre=False`` is passed to :class:`~swiftgalaxy.reader.SWIFTGalaxy`).
Any centre and/or reference velocity from a SOAP catalogue can be used, referring to them
(in a string) using the same syntax as would be used to access them in :mod:`swiftsimio`,
for example:

.. code-block:: python

    SOAP(
        ...,
        # centre of mass of particles in R500crit:
        centre_type="spherical_overdensity_500_crit.centre_of_mass",
        # mass-weighted mean velocity of particles in central 1kpc:
        velocity_centre_type="exclusive_sphere_1kpc.centre_of_mass_velocity",
    )

The centre and reference velocity
:doc:`can also be shifted (and rotated) <../coordinate_transformations/index>` to an
arbitrary coordinate frame after the :class:`~swiftgalaxy.reader.SWIFTGalaxy` has been
created.

To select *all* particles (not only bound particles) in an aperture around the halo of
interest, see the :ref:`example below <aperture-example>`.

Caesar
------

The Caesar catalogue format is popular in the Simba_ simulation community and lives within
the yt_ ecosystem. The :class:`~swiftgalaxy.halo_catalogues.Caesar` helper class relies on
the :class:`~loader.Group` interface to the halo catalogues. Setting up an instance of the
helper class is straightforward. We'll assume a Caesar output called
:file:`caesar_catalogue.hdf5`. There are two types of groups compatible with
:class:`~swiftgalaxy.reader.SWIFTGalaxy` that are defined in these catalogues: halos and
galaxies (refer to the Caesar documentation for details). The type of object is specified
in a ``group_type`` argument (valid values are ``"halo:`` or ``"galaxy"``). The position
of the object of interest in the corresponding Caesar list is specified in the
``group_index`` argument. For example, to choose the 4th entry in the halo list
(``halo_index=3``, since the list is indexed from 0):

.. _Simba: http://simba.roe.ac.uk/
.. _yt: https://yt-project.org/doc/index.html

.. code-block:: python

    cat = Caesar(
        caesar_file="caesar_catalogue.hdf5",
        group_type="halo",
        group_index=3,
    )

The first argument could also include a path if needed, e.g.
:file:`"/path/to/caesar_catalogue.hdf5"`.

The properties of the object of interest are made conveniently available with the
:meth:`~swiftgalaxy.halo_catalogues.Caesar.__getattr__` (dot) syntax, which exposes the
interface provided by the :class:`~loader.Group` class. For example, the
:meth:`~loader.Group.info` function familiar to Caesar users (e.g. using the caesar tools
``load("caesar_catalogue.hdf5").galaxies[3].info()``) is available as:

.. code-block:: python

    cat.info()

This lists available integrated properties of the object of interest, for example the
virial mass (if available) would be accessed as:

.. code-block:: python

    cat.virial_quantities["m200c"]

Caesar is compatible with yt and returns values with units specified with yt that
:mod:`unyt` understands by default.

.. warning ::

    Caesar defines its own unit registry that specifies how some customised units convert
    to units provided by yt. For example, a `Mpccm` (co-moving Mpc) unit is defined.
    Because :mod:`swiftsimio` provides its own custom implementation of co-moving units
    that is not explicitly aware of the :class:`~main.CAESAR` implementation, but both
    are compatible with yt, some issues can arise. The
    :class:`~swiftsimio.objects.cosmo_array` provided by :mod:`swiftsimio` does produce a
    warning when potentially ambiguous calculations are attempted (e.g. where its doesn't
    know that both argument are co-moving, or that both are physical), and this will
    trigger on calculations mixing incompatible :class:`~main.CAESAR`-style and
    :class:`~swiftsimio.objects.cosmo_array` units. However, occasionally
    :mod:`swiftsimio` uses bare :mod:`unyt` quantities or arrays, and if a
    :class:`~main.CAESAR`-style quantity collides with one of these in a calculation
    silent and incorrect conversion from comoving to physical units (or any other
    redshift-dependent units) can occur. It is therefore recommended that users convert
    :class:`~main.CAESAR`-style quantities to use :class:`~swiftsimio.objects.cosmo_array`
    before they are passed to :mod:`swiftsimio` or :mod:`swiftgalaxy` functions. For
    example:

    .. code-block:: python

        import unyt as u
        from swiftsimio.objects import cosmo_array, cosmo_factor
        scale_factor = ...  # retrieve scale factor from snapshot or catalogue file
        cosmo_array(
            cat.virial_quantities["r200c"].to(u.kpc),  # ensures physical units
            comoving=False,
            cosmo_factor=cosmo_factor(a**1, scale_factor=scale_factor)
        ).to_comoving()  # or leave in physical units if desired

Usually the :class:`~swiftgalaxy.halo_catalogues.Caesar` object is used to create a
:class:`~swiftgalaxy.reader.SWIFTGalaxy` object. In this case the interface is exposed
through the ``halo_catalogue`` attribute, for example:

.. code-block:: python

    sg = SWIFTGalaxy(
        ...,
        Caesar(...),
    )
    sg.halo_catalogue.info()
    sg.halo_catalogue.virial_quantities["m200c"]

By default, the :class:`~swiftgalaxy.halo_catalogues.Caesar` class will identify the
particles that the halo finder deems bound to the object as belonging to the galaxy. This
is controlled by the argument:

.. code-block:: python

    Caesar(
        ...,
        extra_mask="bound_only"
    )

This behaviour can be adjusted. If ``None`` is passed instead, then only the spatial
masking provided by :meth:`~swiftgalaxy.halo_catalogues.Caesar._get_spatial_mask` is used.
This means that all particles in the set of (probably cubic) subvolumes of the simulation
that overlap with the region of interest will be read in. Alternatively, a
:class:`~swiftgalaxy.masks.MaskCollection` can be provided. This will be used to select
particles from those already selected spatially.

.. warning::

   Older :class:`~main.CAESAR` outputs (prior to updates to the package in October 2023)
   do not contain enough information to define a spatial sub-region to take advantage of
   :mod:`swiftsimio`'s spatial masking. :mod:`swiftgalaxy` is still compatible with these
   older output files but properties of all particles in the box will be read and then
   masked down to the object of interest, which is very inefficient. When
   :mod:`swiftgalaxy` doesn't find the information needed for spatial masking in a
   :class:`~main.CAESAR` output file, it will produce a warning at runtime before
   proceeding (very inefficiently).

If a different subset of particles is desired, often the most practical option is to first
set up the :class:`~swiftgalaxy.reader.SWIFTGalaxy` with either
``extra_mask="bound_only"`` or ``extra_mask=None`` and then use the loaded particles to
:doc:`compute a new mask that can then be applied <../masking/index>`, perhaps
permanently. Since all particles within the spatially masked region will always be read in
any case, this does not imply any loss of efficiency.

The Caesar catalogue lists two centres for halos and galaxies. By default, the location of
the gravitational potential minimum is assumed as the centre of the objet (and will be
used to :doc:`set the coordinate system <../coordinate_transformations/index>`, unless the
argument ``auto_recentre=False`` is passed to :class:`~swiftgalaxy.reader.SWIFTGalaxy`).
Usually the available centring options are:

  + ``"minpot"`` -- potential minimum
  + ``""`` -- centre of mass

These can be used as, for example:

.. code-block:: python

    Caesar(
        ...,
        centre_type="",  # centre of mass (no suffix in Caesar catalogue)
    )

To select *all* particles (not only bound particles) in an aperture around the halo of
interest, see the :ref:`example below <aperture-example>`.

Velociraptor
------------

Velociraptor_ is a widely-used halo finder. Some SWIFT-based simulations projects have
used it, but are largely moving to a model where particles are assigned to halos with
Velociraptor (or another finder) and a catalogue is produced with the `SOAP`_ tool. The
Velociraptor catalogue format is therefore falling somewhat out of fashion in the SWIFT
community. It is supported for use with :class:`~swiftgalaxy.reader.SWIFTGalaxy`, but is
unlikely to be further developed or maintained. The
:class:`~swiftgalaxy.halo_catalogues.Velociraptor` helper class relies on the
:mod:`velociraptor` interface to the halo catalogues. Setting up an instance of the helper
class is straightforward. If the halo catalogues are named, for example,
:file:`{halos}.properties`, :file:`{halos}.catalog_groups`, etc., and the galaxy of
interest occupies the 4th row in the catalogue (``halo_index=3``, since rows are indexed
from 0), then:

.. _Velociraptor: https://ui.adsabs.harvard.edu/abs/2019PASA...36...21E/abstract
.. _SOAP: https://github.com/SWIFTSIM/SOAP

.. code-block:: python

    cat = Velociraptor(
        "halos",
        halo_index=3
    )

The first argument could also include a path if needed, e.g. :file:`"/path/to/{halos}"`.

.. warning ::

    Currently the :mod:`velociraptor` module does not support selecting galaxies by a
    unique identifier, e.g. ``cat.ids.id``. Users are advised to check this identifier for
    their selected galaxy to ensure that they obtain the object that they expected.

The properties of the galaxy of interest as calculated by Velociraptor are made
conveniently available with the
:meth:`~swiftgalaxy.halo_catalogues.Velociraptor.__getattr__` (dot) syntax, which exposes
the interface provided by the :mod:`velociraptor` module. For example, the virial mass can
be accessed as ``cat.masses.mvir``. Lists of available properties can be printed
interactively using ``print(cat)`` (or simply ``cat`` at the python prompt), or
``print(cat.masses)``, etc. When a :class:`~swiftgalaxy.halo_catalogues.Velociraptor`
instance is used to initialize a :class:`~swiftgalaxy.reader.SWIFTGalaxy`, it is made
available through the ``halo_catalogue`` attribute. For example, to access the virial
mass:

.. code-block:: python

    sg = SWIFTGalaxy(
        ...,
        Velociraptor(
            ...
        )
    )
    sg.halo_catalogue.masses.mvir

By default, the :class:`~swiftgalaxy.halo_catalogues.Velociraptor` class will identify the
particles that the halo finder deems bound to the object as belonging to the galaxy. This
is controlled by the argument:

.. code-block:: python

    Velociraptor(
        ...,
        extra_mask="bound_only"
    )

This behaviour can be adjusted. If ``None`` is passed instead, then only the spatial
masking provided by :func:`velociraptor.swift.swift.generate_spatial_mask` is used. This
means that all particles in the set of (probably cubic) subvolumes of the simulation that
overlap with the region of interest will be read in. Alternatively, a
:class:`~swiftgalaxy.masks.MaskCollection` can be provided. This will be used to select
particles from those already selected using
:func:`~velociraptor.swift.swift.generate_spatial_mask`.

If a different subset of particles is desired, often the most practical option is to first
set up the :class:`~swiftgalaxy.reader.SWIFTGalaxy` with either
``extra_mask="bound_only"`` or ``extra_mask=None`` and then use the loaded particles to
:doc:`compute a new mask that can then be applied <../masking/index>`, perhaps
permanently. Since all particles in the spatial region defined by
:func:`~velociraptor.swift.swift.generate_spatial_mask` will always be read in any case,
this does not imply any loss of efficiency.

The Velociraptor halo finder computes several centres for halos. By default, the location
of the gravitational potential minimum is assumed as the centre of the galaxy (and will be
used to :doc:`set the coordinate system <../coordinate_transformations/index>`, unless the
argument ``auto_recentre=False`` is passed to :class:`~swiftgalaxy.reader.SWIFTGalaxy`).
Usually the available centring options are:

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

To select *all* particles (not only bound particles) in an aperture around the halo of
interest, see the :ref:`example below <aperture-example>`.

Other halo catalogues
---------------------

Support for other halo catalogue formats will be considered on request.

Entrepreneurial users may also create their own helper class inheriting from
:class:`swiftgalaxy.halo_catalogues._HaloCatalogue`. In this case, the following methods
should be implemented:

  + :meth:`~swiftgalaxy.halo_catalogues._HaloCatalogue._load`: called during
    :meth:`~swiftgalaxy.halo_catalogues._HaloCatalogue.__init__`, implement any
    initialisation tasks here.
  + :meth:`~swiftgalaxy.halo_catalogues._HaloCatalogue._generate_spatial_mask`: return a
    :class:`~swiftsimio.masks.SWIFTMask` defining the spatial region to be loaded for the
    galaxy of interest.
  + :meth:`~swiftgalaxy.halo_catalogues._HaloCatalogue._generate_bound_only_mask`: return
    a :class:`~swiftgalaxy.masks.MaskCollection` defining the subset of particles from the
    loaded spatial region that belong to the galaxy of interest.
  + :meth:`~swiftgalaxy.halo_catalogues._HaloCatalogue.centre`: return the coordinates (as
    a :class:`~swiftsimio.objects.cosmo_array`) to be used as the centre of the galaxy of
    interest (implemented with the ``@property`` decorator).
  + :meth:`~swiftgalaxy.halo_catalogues._HaloCatalogue.velocity_centre`: return the
    coordinates (as a :class:`~swiftsimio.objects.cosmo_array`) to be used as the bulk
    velocity of the galaxy of interest (implemented with the ``@property`` decorator).
  + :meth:`~swiftgalaxy.halo_catalogues._HaloCatalogue._region_centre`: return the
    coordinates (as a :class:`~swiftsimio.objects.cosmo_array`) to be used as the centre
    of a bounding box guaranteed to contain all particles belonging to the galaxy of
    interest (implemented with the ``@property`` decorator).
  + :meth:`~swiftgalaxy.halo_catalogues._HaloCatalogue._region_aperture`: return the
    half-lengths (as a :class:`~swiftsimio.objects.cosmo_array`) to be used to construct a
    bounding box guaranteed to contain all particles belonging to the galaxy of interest
    (implemented with the ``@property`` decorator).

In addition, it is recommended to expose the properties computed by the halo finder,
masked to the values corresponding to the object of interest. To make this intuitive for
users, the syntax to access attributes of the galaxy of interest should preferably match
the syntax used for the library conventionally used to read outputs of that halo finder,
if it exists. For instance, for Velociraptor this is implemented via ``__getattr__`` (dot
syntax), which simply exposes the usual interface (with a mask to pick out the galaxy of
interest).

Using swiftgalaxy without a halo catalogue
------------------------------------------

A helper class called :class:`swiftgalaxy.halo_catalogues.Standalone` is provided so that
the features of :mod:`swiftgalaxy` that aren't directly tied to a halo catalogue (e.g.
spherical and cylindrical coordinates, consistent coordinate frame, etc.) can be used when
no supported halo catalogue is available.

Often the most pragmatic way to create a selection of particles using
:class:`~swiftgalaxy.halo_catalogues.Standalone` is to first select a spatial region
guaranteed to contain the particles of interest and then create the final mask
programatically using :class:`~swiftgalaxy.reader.SWIFTGalaxy`'s masking features. For
example, suppose that you know that there is a galaxy with its centre at (2, 2, 2) Mpc and
that you eventually want to select all particles in a spherical aperture 1 Mpc in radius
around this point. Start with a cubic spatial mask enclosing this region:

.. code-block:: python

    from swiftgalaxy import SWIFTGalaxy, Standalone, MaskCollection
    from swiftsimio import cosmo_array
    import unyt as u

    sg = SWIFTGalaxy(
        "my_snapshot.hdf5",
        Standalone(
            centre=cosmo_array([2.0, 2.0, 2.0], u.Mpc),
            velocity_centre=cosmo_array([0.0, 0.0, 0.0], u.km / u.s),
            spatial_offsets=cosmo_array([[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]], u.Mpc),
            extra_mask=None,  # we'll define the exact set of particles later
        )
    )

You can next define the masks selecting particles in your desired spherical aperture,
using :class:`~swiftgalaxy.reader.SWIFTGalaxy`'s convenient spherical coordinates feature,
and store them in a :class:`~swiftgalaxy.masks.MaskCollection`:

.. code-block:: python

    mask_collection = MaskCollection(
        gas=sg.gas.spherical_coordinates.r < 1 * u.Mpc,
        dark_matter=sg.dark_matter.spherical_coordinates.r < 1 * u.Mpc,
        stars=sg.stars.spherical_coordinates.r < 1 * u.Mpc,
        black_holes=sg.black_holes.spherical_coordinates.r < 1 * u.Mpc,
    )

Finally, apply the mask to the ``sg`` object:

.. code-block:: python

    sg.mask_particles(mask_collection)

You're now ready to proceed with analysis of the particles in the 1 Mpc spherical aperture
using this ``sg`` object.

.. note::

   :meth:`~swiftgalaxy.reader.SWIFTGalaxy.mask_particles` applies the masks in-place. The
   mask could also be applied with the :meth:`~swiftgalaxy.reader.SWIFTGalaxy.__getattr__`
   method (i.e. in square brackets), but this returns a copy of the
   :class:`~swiftgalaxy.reader.SWIFTGalaxy` object. If memory efficiency is a concern,
   prefer the :meth:`~swiftgalaxy.reader.SWIFTGalaxy.mask_particles` approach.

.. _aperture-example:

Selecting particles within an aperture
--------------------------------------

The workflow to select all particles within a given aperture (e.g. 1 Mpc) also works when
starting from a halo catalogue object. For instance, using SOAP you could do the
following:

.. code-block:: python

    sg = SWIFTGalaxy(
        "my_snapshot.hdf5",
        SOAP(
            "my_soap_file.hdf5",
            soap_index=0,
            # disable selecting only particles flagged as bound by the halo finder:
            extra_mask=None,
            custom_spatial_offsets=cosmo_array(
                [[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]], u.Mpc
            ),
        )
    )
    mask_collection = MaskCollection(
        gas=sg.gas.spherical_coordinates.r < 1 * u.Mpc,
        dark_matter=sg.dark_matter.spherical_coordinates.r < 1 * u.Mpc,
        stars=sg.stars.spherical_coordinates.r < 1 * u.Mpc,
        black_holes=sg.black_holes.spherical_coordinates.r < 1 * u.Mpc,
    )
    sg.mask_particles(mask_collection)

The ``sg`` object is now ready for further analysis. The same approach works with any halo
catalogue interface by setting the ``extra_mask`` and ``custom_spatial_offsets`` arguments
appropriately.
