from causalitygame.repository._base import get_scm_overview
from causalitygame.scm import SCM
import logging
import json
import pytest


# define stream handler
ch = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
ch.setLevel(logging.DEBUG)

# configure logger for tester
logger = logging.getLogger("tester")
logger.handlers.clear()
logger.addHandler(ch)
logger.setLevel(logging.DEBUG)

@pytest.mark.parametrize(
    "name, filename",
    get_scm_overview(folder="causalitygame/data/scm")[["name", "filename"]].values,
)
def test_usability_of_shipped_scms(name, filename):

    logger.info(f"Trying to load SCM {name} at file {filename}")
    with open(filename) as f:
        scm_as_dict = json.load(f)
        scm = SCM.from_dict(scm_as_dict)

        # test that we can draw samples from this SCM and that all values are in the domains
        sample = scm.generate_samples(num_samples=10)
        assert len(sample) == 10
        for obs in sample:
            for var, val in obs.items():
                assert val >= scm.nodes[var].domain[0]
                assert val <= scm.nodes[var].domain[1]

        # test that we can intervene at least one variable
        assert len(scm.controllable_vars) >= 1, "The SCM has no controllable variables!"

@pytest.mark.parametrize(
    "name, filename",
    get_scm_overview(folder="causalitygame/data/scm")[["name", "filename"]].values,
)
def test_playability_of_shipped_scms(name, filename):

    logger.info(f"Trying to load SCM {name} at file {filename}")
    with open(filename) as f:
        scm_as_dict = json.load(f)
        scm = SCM.from_dict(scm_as_dict)

        # test that we can draw samples from this SCM and that all values are in the domains
        sample = scm.generate_samples(num_samples=10)
        assert len(sample) == 10
        for obs in sample:
            for var, val in obs.items():
                assert val >= scm.nodes[var].domain[0]
                assert val <= scm.nodes[var].domain[1]

        # test that we can intervene at least one variable
        assert len(scm.controllable_vars) >= 1, "The SCM has no controllable variables!"