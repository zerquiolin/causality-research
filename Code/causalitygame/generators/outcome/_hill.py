from causalitygame.generators.outcome.base import OutcomeGenerator
from scipy.stats import Normal
import numpy as np

class HillOutcomeGenerator(OutcomeGenerator):

    def __init__(
            self,
            index_of_treatment_variable,
            coefficient_domain=[0, 1, 2, 3, 4],
            coefficient_probabilities=[0.5, 0.2, 0.15, 0.1, 0.05],
            noise_std=1,
            random_state=None
        ):

        super().__init__(random_state=random_state)

        # configuration
        self.index_of_treatment_variable = index_of_treatment_variable
        self.coefficient_domain = coefficient_domain
        self.coefficient_probabilities = coefficient_probabilities
        self.noise_std = noise_std

        # state
        self.beta = None

    def fit(self, x, y):
        
        # define beta (can only be done now since the number of coefficients is not know before)
        self.beta = self.random_state.choice(self.coefficient_domain, size=x.shape[1] - 1, p=self.coefficient_probabilities, replace=True)
    
    def generate(self, x, random_state=None):
        assert isinstance(x, np.ndarray), f"x must be an numpy array but is {type(x)}"
        assert len(x.shape) == 2, f"x must be an 2 dimensional array but is of shape {x.shape}"
        _x = np.delete(x, self.index_of_treatment_variable, axis=1)
        t = x[:, self.index_of_treatment_variable].reshape(-1)
        assert _x.shape[1] == len(self.beta), f"Trained beta has {len(self.beta)} entries, but given data has {_x.shape[1]} attributes."
        mu = self._get_mu(_x, t)

        if random_state is None:
            random_state = self.random_state

        return random_state.normal(loc=mu, scale=self.noise_std)
    
    def _to_dict(self):
        return {
            "index_of_treatment_variable": self.index_of_treatment_variable,
            "coefficient_domain": self.coefficient_domain,
            "coefficient_probabilities": self.coefficient_probabilities,
            "noise_std": self.noise_std,
            "beta": [int(v) for v in self.beta] if self.beta is not None else None
        }


class SetupAOutcomeGenerator(HillOutcomeGenerator):

    def __init__(
            self,
            index_of_treatment_variable,
            coefficient_domain=[0, 1, 2, 3, 4],
            coefficient_probabilities=[0.5, 0.2, 0.15, 0.1, 0.05],
            noise_std=1,
            offset=4,
            random_state=None
        ):
        super().__init__(
            index_of_treatment_variable,
            coefficient_domain=coefficient_domain,
            coefficient_probabilities=coefficient_probabilities,
            noise_std=noise_std,
            random_state=random_state
        )

        self.offset = offset

    def _get_mu(self, x, t):
        base_val = x @ self.beta
        treatment_effect = (self.offset * t)
        return base_val + treatment_effect
    
    def _to_dict(self):
        d = super()._to_dict()
        d.update({
            "offset": self.offset
        })
        return d
    
    @classmethod
    def _from_dict(cls, data):
        beta = None
        if "beta" in data:
            beta = data.pop("beta")
        obj = cls(**data)
        if beta is not None:
            obj.beta = np.array(beta)
        return obj

class SetupBOutcomeGenerator(HillOutcomeGenerator):

    def __init__(
            self,
            index_of_treatment_variable,
            coefficient_domain=[0, 1, 2, 3, 4],
            coefficient_probabilities=[0.6, 0.1, 0.1, 0.1, 0.1],
            noise_std=1,
            offset=4,
            omega=0,
            random_state=None
        ):
        super().__init__(
            index_of_treatment_variable=index_of_treatment_variable,
            coefficient_domain=coefficient_domain,
            coefficient_probabilities=coefficient_probabilities,
            noise_std=noise_std,
            random_state=random_state
        )

        self.offset = offset
        self.omega = omega

    def _get_mu(self, x, t):
        mu_1 = np.exp((x + self.offset) @ self.beta)
        mu_0 = x @ self.beta - self.omega
        _mu =  t * mu_1 + (1 - t) * mu_0
        return _mu

    def _to_dict(self):
        d = super()._to_dict()
        d.update({
            "offset": self.offset,
            "omega": self.omega,
        })
        return d

    @classmethod
    def _from_dict(cls, data):
        beta = None
        if "beta" in data:
            beta = data.pop("beta")
        obj = cls(**data)
        if beta is not None:
            obj.beta = np.array(beta)
        return obj
