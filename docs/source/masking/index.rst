Particle masking
================

The selection of subsets of particles belonging to a
:class:`~swiftgalaxy.reader.SWIFTGalaxy` is intended to be as intuitive as possible while
also ensuring consistency between the various particle arrays and accounting for the
practicalities of implementation. Since some masking operations involve (possibly
memory-intensive) copy operations, details to guide the resource-conscious user are
provided below.

Masking a SWIFTGalaxy
---------------------

There are two ways to manipulate the :class:`~swiftgalaxy.reader.SWIFTGalaxy` object
itself to select a subset of particles. The first is with the
:meth:`~swiftgalaxy.reader.SWIFTGalaxy.mask_particles` method. This method expects to
receive an instance of :class:`swiftgalaxy.masks.MaskCollection`.

.. note::

   The :mod:`velociraptor.swift.swift` module makes some use of a :obj:`namedtuple` called
   ``MaskCollection``. These objects are not valid where :mod:`swiftgalaxy` functions
   expect a :obj:`MaskCollection` because :obj:`namedtuple` objects are immutable.

The :class:`~swiftgalaxy.masks.MaskCollection` is initialised with keyword arguments
specifying masks for each particle type to be masked. Particle types that are not to be
masked can either be omitted, or explicitly given a "null" mask, such as a literal
``Ellipsis`` (``...``). Any python object that could be given in square brackets as a mask
for a :class:`~swiftsimio.objects.cosmo_array` (or :class:`~numpy.ndarray`) will be
accepted.

.. warning::

   There is one exception to arguments being interpreted as they would be in the square
   brackets of an :class:`~numpy.ndarray`: ``None``. In :mod:`numpy`, the name
   :obj:`numpy.newaxis` is an alias for ``None``, and extends the dimension of an array by
   one at the position where it is inserted. In a
   :class:`~swiftgalaxy.masks.MaskCollection`, however, ``None`` (and therefore also
   :obj:`numpy.newaxis`) is instead interpreted as "no mask".

Some illustrative examples:

.. code-block:: python

    import unyt as u
    from swiftgalaxy import SWIFTGalaxy, MaskCollection
    sg = SWIFTGalaxy(...)
    mask = MaskCollection(
        gas=sg.gas.temperatures > 1e6 * u.K,  # boolean mask
        dark_matter=np.s_[:10],  # first 10 particles
        stars=...,  # Ellipsis, equivalent to keeping previous mask
        # black_holes omitted, equivalent to keeping previous mask
    )
    sg.mask_particles(mask)

Notice the use of the :obj:`numpy.s_` "index tricks" tool.

.. note::

   Using the :meth:`~swiftgalaxy.reader.SWIFTGalaxy.mask_particles` method is *the only
   way* to select a subset of particles in-place, i.e. without copying any data. After
   calling this function, any particle arrays in memory will be permanently masked, and
   any new particle arrays loaded will be trimmed according to the new mask.

The second way to select a subset of particles is using the
:class:`~swiftgalaxy.reader.SWIFTGalaxy.__getattr__` method (square brackets). Although
the result is similar to that obtained with the first method, the implementation differs:
this method returns a masked copy of *the entire* :class:`~swiftgalaxy.reader.SWIFTGalaxy`
object. Using the same mask as above:

.. code-block:: python

    masked_sg = sg[mask]  # copy operation!

Therefore, if attempting to minimize memory usage, keep in mind:

.. code-block:: python

    sg.mask_particles(mask)  # memory-efficient
    sg = sg[mask]  # equivalent result, but memory-inefficient

Masking individual SWIFTParticleDatasets
----------------------------------------

Passing masks to the :class:`~swiftgalaxy.reader.SWIFTGalaxy` directly (possibly via the
:meth:`~swiftgalaxy.reader.SWIFTGalaxy.mask_particles` method) is usually the best way to
select subsets of particles. However, it is also possible to apply masks to the particle
datasets (e.g. ``sg.gas``) -- this is intended mostly for interactive use-cases. To do
this, simply provide the mask to the dataset's
:meth:`~swiftgalaxy.reader.SWIFTParticleDatasetHelper.__getattr__` method (i.e. in square
brackets). For example:

.. code-block:: python

    gasmask = sg.gas.temperatures > 1e6 * u.K
    sg.gas[gasmask]

.. warning::

   To ensure internal consistency, this operation creates a copy of the *entire*
   :class:`~swiftgalaxy.reader.SWIFTGalaxy`, and may therefore be memory-intensive if many
   particle arrays have been loaded (including other particle types). Furthermore, while
   ``sg.gas = sg.gas[gasmask]`` will initially work as expected, because the masked
   dataset "belongs" to a different :class:`~swiftgalaxy.reader.SWIFTGalaxy` (the copy)
   than it is being assigned to, this can lead to subtle and difficult-to-debug issues
   down the line and is therefore *not* recommended. Assigning to another name (e.g.
   ``masked_gas = sg.gas[gasmask]``) does not pose any obvious problems, other than being
   somewhat memory-inefficient.

Masking and SWIFTNamedColumnDatasets
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Named column datasets can be masked just like particle datasets, using the named column
dataset's :meth:`~swiftgalaxy.reader.SWIFTNamedColumnDatasetHelper.__getattr__` method.
For example:

.. code-block:: python

    # example for a gas property from the Colibre model:
    sg.gas.element_mass_fractions[gasmask]

.. warning::

   As for particle datasets, the operation
   ``sg.gas.element_mass_fractions = sg.gas.element_mass_fractions[gasmask]`` is not
   recommended -- see warning above for rationale. In addition, this operation would
   result in a named column dataset whose constituent particle arrays have shapes
   different from the rest of the particle arrays in the particle dataset hosting the
   named column dataset -- this is probably undesirable!

Masking particle arrays
-----------------------

As may be intuitively expected, individual particle arrays can be masked on-the-fly as
usual:

.. code-block:: python

    sg.gas.masses[gasmask]

This returns a masked copy of the individual particle array, so does not imply the same
level of potentially expensive copy operations discussed above. While it is possible to
assign the result back to the particle array (e.g.
``sg.gas.masses = sg.gas.masses[gasmask]``), this is inadvisable since it will break the
consistency between the shapes of the particle arrays for that particle type. After doing
this, some operations, such as attempting to mask the
:class:`~swiftgalaxy.reader.SWIFTGalaxy` again, are then likely to raise an exception.

Cookbook: all particles in a spherical aperture
-----------------------------------------------

One of :mod:`swiftgalaxy`'s features is that it can conveniently provide a set of
particles that a halo finder has identified as belonging to a galaxy (or other object).
However, in some cases this might not be the selection of particles that you want. When
the desired particles include some that are not identified as members by the halo finder,
a modified approach is needed. For example, let's suppose that you want to select *all*
simulation particles within a 1 Mpc aperture of a galaxy's centre, regardless of their
membership status according to the halo catalogue. For illustration we'll take a galaxy
picked from a :class:`~swiftgalaxy.halo_finders.SOAP` catalogue. The first step is to
override the default ``extra_mask="bound_only"`` behaviour with ``extra_mask=None``. We
also need to override the default spatial selection from the simulation, because the 1 Mpc
spherical region of interest might extend beyond the region occupied by member particles
as defined by the halo finder, which is all that the default spatial selection is
guaranteed to enclose:

.. code-block:: python

    sg = SWIFTGalaxy(
        "my_snapshot.hdf5",
        SOAP(
            "my_soap.hdf5",  # name of catalogue file
            soap_index=3,  # pick the 4th galaxy (indexed from 0) in the catalogue array
            extra_mask=None,  # select all particles in the spatial region (for now)
            custom_spatial_offsets=cosmo_array(
                [[-1, 1], [-1, 1], [-1, 1]], u.Mpc  # relative to centre
            ),
        ),
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
