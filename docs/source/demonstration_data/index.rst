Example data
============

:mod:`swiftgalaxy` comes with utilities to generate some schematic example data on the fly, and to download some more realistic example datasets. This is implemented via two helper objects in the :mod:`~swiftgalaxy.demo_data` module, ``generated_examples`` and ``web_examples``. You can view the available example data sets with:

.. code-block:: python

   from swiftgalaxy.demo_data import generated_examples, web_examples

   print(generated_examples)
   print(web_examples)

Each example can be accessed as an attribute, for example ``generated_examples.snapshot`` creates an example snapshot file. The helper class is set up so that the path to the example file(s) is returned when the attribute is accessed. This means that we can pass the examples directly to :mod:`swiftgalaxy` classes that expect file paths as arguments - the needed input files will be created/downloaded (unless they already exist) and their location passed automatically. The example data is stored in a directory :file:`demo_data/` in the current working directory, and this directory is created if it does not exist.

The example data generated on the fly is very simplistic: the snapshot consists of a cubic volume with uniform randomly distributed dark matter and gas particles. Two "galaxies" are superimposed on this by placing additional particles forming a sphere of dark matter and a disc of stars and gas for each. The halo catalogue examples contain two objects (the two galaxies) but their properties are mostly written in "by hand" rather than calculated from the particle distributions. These examples are useful for demonstrating or verifying the functionality of :mod:`swiftgalaxy`, but not much more.

The example data available for download (about 700 MB for the entire set) is more realistic. The sample snapshot is taken from a run of the ``EAGLE_6`` example simulation `included with SWIFT`_. The halo catalogues come from running Velociraptor, SOAP (with HBT-Herons backend) and Caesar on the snapshot.

.. _included with SWIFT: https://github.com/SWIFTSIM/SWIFT/tree/master/examples/EAGLE_ICs

Usage
-----

The following code snippets set up a :class:`~swiftgalaxy.reader.SWIFTGalaxy` object with example data generated on the fly:

.. code-block:: python

   # a synthetically created snapshot with a minimalist implementation of a halo catalogue

   from swiftgalaxy import SWIFTGalaxy
   from swiftgalaxy.demo_data import generated_examples, ToyHF
   
   SWIFTGalaxy(
       generated_examples.snapshot,
       ToyHF(index=0),
   )

.. code-block:: python

   # a synthetically created snapshot with a synthetically created SOAP catalogue

   from swiftgalaxy.demo_data import generated_examples
   from swiftgalaxy import SWIFTGalaxy, SOAP
   
   SWIFTGalaxy(
       generated_examples.virtual_snapshot,  # notice virtual_snapshot, not snapshot
       SOAP(generated_examples.soap, soap_index=0),
   )

.. code-block:: python

   # a synthetically created snapshot with a synthetically created Velociraptor catalogue

   from swiftgalaxy.demo_data import generated_examples
   from swiftgalaxy import SWIFTGalaxy, Velociraptor
   
   SWIFTGalaxy(
       generated_examples.snapshot,
       Velociraptor(generated_examples.velociraptor, halo_index=0),
   )

.. code-block:: python

   # a synthetically created snapshot with a synthetically created Caesar catalogue

   from swiftgalaxy.demo_data import generated_examples
   from swiftgalaxy import SWIFTGalaxy, Caesar
   
   SWIFTGalaxy(
       generated_examples.snapshot,
       Caesar(generated_examples.caesar, group_type="galaxy", group_index=0),  # or group_type="halo"
   )

The following code snippets set up a :class:`~swiftgalaxy.reader.SWIFTGalaxy` object with downloaded example data:

.. code-block:: python

   # a small EAGLE snapshot with a SOAP catalogue

   from swiftgalaxy.demo_data import web_examples
   from swiftgalaxy import SWIFTGalaxy, SOAP
   
   SWIFTGalaxy(
       web_examples.virtual_snapshot,  # notice virtual_snapshot, not snapshot
       SOAP(web_examples.soap, soap_index=0),
   )

.. code-block:: python

   # a small EAGLE snapshot with a Velociraptor catalogue

   from swiftgalaxy.demo_data import web_examples
   from swiftgalaxy import SWIFTGalaxy, Velociraptor
   
   SWIFTGalaxy(
       web_examples.snapshot,
       Velociraptor(web_examples.velociraptor, halo_index=0),
   )

.. code-block:: python

   # a small EAGLE snapshot with a Caesar catalogue

   from swiftgalaxy.demo_data import web_examples
   from swiftgalaxy import SWIFTGalaxy, Caesar
   
   SWIFTGalaxy(
       web_examples.snapshot,
       Caesar(web_examples.caesar, group_type="galaxy", group_index=0),  # or group_type="halo"
   )

The example data can be removed with ``generated_examples.remove()`` and ``web_examples.remove()``.
