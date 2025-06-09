import os
import pytest
import numpy as np
import unyt as u
from swiftsimio import cosmo_array, mask
from swiftgalaxy import SWIFTGalaxy, Velociraptor, Caesar, SOAP, Standalone
from swiftgalaxy.demo_data import (
    web_examples,
    generated_examples,
    ToyHF,
    _toysnap_filename,
    _toyvr_filebase,
    _toysoap_filename,
    _toysoap_membership_filebase,
    _toysoap_virtual_snapshot_filename,
    _toycaesar_filename,
    _centre2,
    _vcentre2,
)


class TestWebExampleData:

    def test_str(self):
        """
        Check that string representation lists available examples.
        """
        assert "snapshot" in str(web_examples)

    def test_snapshot(self):
        """
        Check that we can create a swiftgalaxy, retrieving a sample snapshot file.
        """
        sm = mask(web_examples.snapshot)  # just to get an interface to metadata
        SWIFTGalaxy(
            web_examples.snapshot,
            Standalone(
                centre=cosmo_array(
                    [2, 2, 2],
                    u.Mpc,
                    comoving=True,
                    scale_factor=sm.metadata.scale_factor,
                    scale_exponent=1,
                ),
                velocity_centre=cosmo_array(
                    [200, 200, 200],
                    u.km / u.s,
                    comoving=True,
                    scale_factor=sm.metadata.scale_factor,
                    scale_exponent=0,
                ),
                spatial_offsets=cosmo_array(
                    [[-1, 1], [-1, 1], [-1, 1]],
                    u.Mpc,
                    comoving=True,
                    scale_factor=sm.metadata.scale_factor,
                    scale_exponent=1,
                ),
            ),
        )

    @pytest.mark.parametrize("group_type", ["halo", "galaxy"])
    def test_caesar(self, group_type):
        """
        Check that we can create a swiftgalaxy, retrieving a sample snapshot file and
        caesar catalogue.
        """
        SWIFTGalaxy(
            web_examples.snapshot,
            Caesar(web_examples.caesar, group_type=group_type, group_index=0),
        )

    def test_soap(self):
        """
        Check that we can create a swiftgalaxy, retrieving a sample virtual snapshot
        file and a soap catalogue.
        """
        SWIFTGalaxy(
            web_examples.virtual_snapshot, SOAP(web_examples.soap, soap_index=0)
        )

    def test_vr(self):
        """
        Check that we can create a swiftgalaxy, retrieving a sample snapshot file
        and velociraptor catalogue.
        """
        SWIFTGalaxy(
            web_examples.snapshot, Velociraptor(web_examples.velociraptor, halo_index=0)
        )

    def test_remove(self):
        """
        Check that example data files get cleaned up on request.
        """
        # download all the example data (if not present)
        for example in web_examples.available_examples.keys():
            getattr(web_examples, example)
        web_examples.remove()
        for absent_files in web_examples.available_examples.values():
            for absent_file in absent_files:
                assert not os.path.isfile(absent_file)


class TestGeneratedExampleData:

    def test_str(self):
        """
        Check that string representation lists available examples.
        """
        assert "snapshot" in str(web_examples)

    def test_snapshot(self):
        """
        Check that we can create a swiftgalaxy using the helper for a generated snapshot.
        """
        sg = SWIFTGalaxy(
            generated_examples.snapshot,
            ToyHF(index=1),
        )
        assert np.allclose(
            sg.centre.to_physical_value(u.Mpc),
            np.ones(3) * _centre2,
        )
        assert np.allclose(
            sg.velocity_centre.to_physical_value(u.Mpc),
            np.ones(3) * _vcentre2,
        )

    def test_velociraptor(self):
        """
        Check that we can create a swiftgalaxy using the helper for a generated
        velociraptor catalogue.
        """
        SWIFTGalaxy(
            generated_examples.snapshot,
            Velociraptor(generated_examples.velociraptor, halo_index=0),
        )

    @pytest.mark.parametrize("group_type", ["halo", "galaxy"])
    def test_caesar(self, group_type):
        """
        Check that we can create a swiftgalaxy using the helper for a generated caesar
        catalogue.
        """
        SWIFTGalaxy(
            generated_examples.snapshot,
            Caesar(generated_examples.caesar, group_type=group_type, group_index=0),
        )

    def test_soap(self):
        """
        Check that we can create a swiftgalaxy using the helper for a generated soap
        catalogue.
        """
        SWIFTGalaxy(
            generated_examples.virtual_snapshot,
            SOAP(generated_examples.soap, soap_index=0),
        )

    def test_remove(self):
        """
        Check that examples get cleaned up on request.
        """
        # create all the example data (if not present)
        for example in generated_examples.available_examples:
            getattr(generated_examples, example)
        generated_examples.remove()
        absent_files = (
            _toysnap_filename,
            f"{_toyvr_filebase}.properties",
            f"{_toyvr_filebase}.catalog_groups",
            f"{_toyvr_filebase}.catalog_particles",
            f"{_toyvr_filebase}.catalog_particles.unbound",
            f"{_toyvr_filebase}.catalog_parttypes",
            f"{_toyvr_filebase}.catalog_parttypes.unbound",
            _toysoap_filename,
            f"{_toysoap_membership_filebase}.0.hdf5",
            _toysoap_virtual_snapshot_filename,
            _toycaesar_filename,
        )
        for absent_file in absent_files:
            assert not os.path.isfile(absent_file)


class TestExampleNotebooks:

    def test_generated_example_notebook(self):
        """
        Check that the example notebook with data generated on the fly runs without error.
        """
        pytest.importorskip(
            "nbmake", reason="nbmake (optional dependency) not available"
        )
        from nbmake.nb_run import NotebookRun
        from nbmake.pytest_items import NotebookFailedException
        import pathlib

        generated_examples.remove()
        nbr = NotebookRun(pathlib.Path("examples/SWIFTGalaxy_demo.ipynb"), 300)
        try:
            result = nbr.execute()
            if result.error is not None:
                raise NotebookFailedException(result)
        finally:
            generated_examples.remove()
