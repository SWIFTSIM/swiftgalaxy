from pathlib import Path
import pytest
import unyt as u
from swiftsimio import cosmo_array, mask
from swiftgalaxy import SWIFTGalaxy, Velociraptor, Caesar, SOAP, Standalone
from swiftgalaxy.demo_data import (
    ToyHF,
    _toysnap_filename,
    _toyvr_filebase,
    _toysoap_filename,
    _toysoap_membership_filebase,
    _toysoap_virtual_snapshot_filename,
    _toycaesar_filename,
)


class TestWebExampleData:

    def test_str(self, web_examples_tmpdir):
        """
        Check that string representation lists available examples.
        """
        assert "snapshot" in str(web_examples_tmpdir)

    def test_snapshot(self, web_examples_tmpdir):
        """
        Check that we can create a swiftgalaxy, retrieving a sample snapshot file.
        """
        sm = mask(web_examples_tmpdir.snapshot)  # just to get an interface to metadata
        SWIFTGalaxy(
            web_examples_tmpdir.snapshot,
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
    def test_caesar(self, web_examples_tmpdir, group_type):
        """
        Check that we can create a swiftgalaxy, retrieving a sample snapshot file and
        caesar catalogue.
        """
        SWIFTGalaxy(
            web_examples_tmpdir.snapshot,
            Caesar(web_examples_tmpdir.caesar, group_type=group_type, group_index=0),
        )

    def test_soap(self, web_examples_tmpdir):
        """
        Check that we can create a swiftgalaxy, retrieving a sample virtual snapshot
        file and a soap catalogue.
        """
        SWIFTGalaxy(
            web_examples_tmpdir.virtual_snapshot,
            SOAP(web_examples_tmpdir.soap, soap_index=0),
        )

    def test_vr(self, web_examples_tmpdir):
        """
        Check that we can create a swiftgalaxy, retrieving a sample snapshot file
        and velociraptor catalogue.
        """
        SWIFTGalaxy(
            web_examples_tmpdir.snapshot,
            Velociraptor(web_examples_tmpdir.velociraptor, halo_index=0),
        )

    def test_remove(self, web_examples_tmpdir):
        """
        Check that example data files get cleaned up on request.
        """
        # download all the example data (if not present)
        for example in web_examples_tmpdir.available_examples.keys():
            getattr(web_examples_tmpdir, example)
        web_examples_tmpdir.remove()
        for absent_files in web_examples_tmpdir.available_examples.values():
            for absent_file in absent_files:
                assert not Path(absent_file).is_file()


class TestGeneratedExampleData:

    def test_str(self, generated_examples_tmpdir):
        """
        Check that string representation lists available examples.
        """
        assert "snapshot" in str(generated_examples_tmpdir)

    def test_snapshot(self, generated_examples_tmpdir):
        """
        Check that we can create a swiftgalaxy using the helper for a generated snapshot.
        """
        SWIFTGalaxy(
            generated_examples_tmpdir.snapshot,
            ToyHF(snapfile=generated_examples_tmpdir.snapshot, index=0),
        )

    def test_velociraptor(self, generated_examples_tmpdir):
        """
        Check that we can create a swiftgalaxy using the helper for a generated
        velociraptor catalogue.
        """
        SWIFTGalaxy(
            generated_examples_tmpdir.snapshot,
            Velociraptor(generated_examples_tmpdir.velociraptor, halo_index=0),
        )

    @pytest.mark.parametrize("group_type", ["halo", "galaxy"])
    def test_caesar(self, generated_examples_tmpdir, group_type):
        """
        Check that we can create a swiftgalaxy using the helper for a generated caesar
        catalogue.
        """
        SWIFTGalaxy(
            generated_examples_tmpdir.snapshot,
            Caesar(
                generated_examples_tmpdir.caesar, group_type=group_type, group_index=0
            ),
        )

    def test_soap(self, generated_examples_tmpdir):
        """
        Check that we can create a swiftgalaxy using the helper for a generated soap
        catalogue.
        """
        SWIFTGalaxy(
            generated_examples_tmpdir.virtual_snapshot,
            SOAP(generated_examples_tmpdir.soap, soap_index=0),
        )

    def test_remove(self, generated_examples_tmpdir):
        """
        Check that examples get cleaned up on request.
        """
        # create all the example data (if not present)
        for example in generated_examples_tmpdir.available_examples:
            getattr(generated_examples_tmpdir, example)
        generated_examples_tmpdir.remove()
        tp = generated_examples_tmpdir._demo_data_dir
        absent_files = (
            tp / _toysnap_filename.name,
            tp / f"{_toyvr_filebase.name}.properties",
            tp / f"{_toyvr_filebase.name}.catalog_groups",
            tp / f"{_toyvr_filebase.name}.catalog_particles",
            tp / f"{_toyvr_filebase.name}.catalog_particles.unbound",
            tp / f"{_toyvr_filebase.name}.catalog_parttypes",
            tp / f"{_toyvr_filebase.name}.catalog_parttypes.unbound",
            tp / _toysoap_filename.name,
            tp / f"{_toysoap_membership_filebase.name}.0.hdf5",
            tp / _toysoap_virtual_snapshot_filename.name,
            tp / _toycaesar_filename.name,
        )
        for absent_file in absent_files:
            assert not Path(absent_file).is_file()


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

        nbr = NotebookRun(Path("examples/SWIFTGalaxy_demo.ipynb"), 300)
        result = nbr.execute()
        if result.error is not None:
            raise NotebookFailedException(result)
