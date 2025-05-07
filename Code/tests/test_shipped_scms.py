from causalitygame.repository._base import get_scm_overview
from causalitygame.scm import SCM
import logging
import json


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

def test_deserializability_of_shipped_scms():

    # get overview of SCMs
    overview = get_scm_overview(folder="causalitygame/data/scm")

    # try to load all of them
    for name, filename in overview[["name", "filename"]].values:
        logger.info(f"Trying to load SCM {name} at file {filename}")
        with open(filename) as f:
            scm_as_dict = json.load(f)
            scm = SCM.from_dict(scm_as_dict)

            sample = scm.generate_samples(num_samples=10)
            assert len(sample) == 10
            for obs in sample:
                for var, val in obs.items():
                    assert val >= scm.nodes[var].domain[0]
                    assert val <= scm.nodes[var].domain[1]

