# Math
import numpy as np
import sympy as sp
# Utils
import logging
class SCMNode:
 """
 Represents a single node in the Structural Causal Model (SCM).
 This class handles both numerical and categorical nodes.
 For numerical nodes, the `equation` is used for generating values - parent variables that are categorical
 are converted via label encoding using the provided parent mappings.
 For categorical nodes, a separate equation is generated for each possible class.
 A CDF is computed (from at least 1000 datapoints) for each equation and stored in cdf_mappings.
 Attributes:
 name (str): The node name.
 equation: For numerical nodes, a symbolic equation; for categorical nodes, a dict mapping each category to its equation.
 domain (tuple or list): The domain of values.
 var_type (str): "numerical" or "categorical".
 cdf_mappings (dict): For categorical nodes, maps each category to a CDF function.
 parent_mappings (dict): Maps parent node names (if categorical) to a label encoding dictionary.
 """
 def __init__(
 self,
 name: str,
 equation,
 domain,
 var_type: str,
 cdf_mappings: dict,
 parent_mappings: dict,
 random_state: np.random.RandomState = np.random,
 ):
 """
 Initializes an SCMNode instance.
 Args:
 name (str): The name of the node.
 equation: The symbolic equation for the node (or a dict for categorical nodes).
 domain (tuple or list): The range of values for numerical nodes or list of possible categories for categorical nodes.
 var_type (str): The type of the variable ('numerical' or 'categorical').
 cdf_mappings (dict, optional): For categorical nodes, a mapping from each category to its CDF function.
 parent_mappings (dict, optional): A mapping of parent names to their label encoding dictionaries.
 random_state (np.random.RandomState, optional): Random state for reproducibility.
 """
 self.name = name
 self.equation = equation
 self.domain = domain
 self.var_type = var_type
 self.cdf_mappings = cdf_mappings
 self.parent_mappings = parent_mappings
 self.random_state = random_state
 def generate_value(
 self, parent_values: dict, random_state: np.random.RandomState
 ) -> float:
 """
 Generates a value for the node based on its type and parent values.
 For numerical nodes, substitutes parent values into the symbolic equation.
 If a parent value is categorical, its numeric (label encoded) version is obtained from parent_mappings.
 For categorical nodes, this method is expected to use the stored CDF mappings (not shown here).
 Args:
 parent_values (dict): A dictionary of parent values.
 random_state (np.random.RandomState): Random state for generating random values.
 Returns:
 float: The generated value (or, for categorical nodes, the chosen category).
 """
 
 has_parent_values = len(parent_values) != 0
 # Determine the random state to use.
 rs = random_state or self.random_state
 # Iterate over the free symbols in the equation.
 if self.var_type == "numerical":
 symbols = self.equation.free_symbols
 else:
 symbols = set()
 for eq in self.equation.values():
 symbols.update(eq.free_symbols)
 
 # draw random noise
 if not has_parent_values:
 if isinstance(self.domain, tuple):
 min_val, max_val = self.domain
 return rs.uniform(min_val, max_val) # TODO: replace by the noise distribution self.noise_distribution in the object
 else:
 # For categorical nodes without parents, randomly select a category from the domain.
 return rs.choice(self.domain, p=None) # TODO: replace the p by the distribution specified in the object self.noise_distribution
 
 
 # replace values in the RHS of the equation by the actual values of the parents
 subs_dict = {​​​​​​}​​​​​​
 for parent_name, parent_val in parent_values.items():
 # If the parent's value is a string (categorical), convert it using parent_mappings.
 if isinstance(parent_val, str) and parent_name in self.parent_mappings:
 parent_val = self.parent_mappings[parent_name][parent_val]
 
 # Add the value to the substitution dictionary.
 subs_dict[parent_name] = parent_val
 
 # Warn if any symbols remain unresolved.
 for var in symbols:
 if str(var) not in subs_dict:
 logging.warning(f"Unresolved symbol in {​​​​​​self.name}​​​​​​ ->", var)
 # compute value for the variable using the values of parent variables and noise
 if self.var_type == "numerical":
 # draw noise term
 noise = rs.uniform(min_val, max_val) # TODO: replace by the noise distribution self.noise_distribution in the object
 # TODO: Use the noise term for the equation evaluation!!
 # For numerical nodes, substitute the values into the equation.
 eval_equation = self.equation.subs(subs_dict).evalf()
 if isinstance(eval_equation, sp.Basic) and not eval_equation.is_number:
 logging.warning(
 f"Equation for {​​​​​​self.name}​​​​​​ did not fully resolve:",
 eval_equation,
 )
 # just in case the value is a complex number, only extract the real component
 result = (
 float(eval_equation.as_real_imag()[0])
 if not eval_equation.is_real
 else float(eval_equation)
 )
 # make sure that the variable value is in the permitted bounds
 if isinstance(self.domain, tuple):
 min_val, max_val = self.domain
 result = max(min_val, min(max_val, result))
 return result
 else:
 # if also all parents are categorical, don't use an equation but the look-up-table
 if all parents categorical:
 distribution = ... #
 
 else:
 # determine score for each possible category
 cat_values = {​​​​​​}​​​​​​
 for possible_cat_of_this_var, eq in self.equation.items():
 cat_eq = eq.subs(subs_dict).evalf() # Substitutes the parent values into the equation for each category.
 if isinstance(cat_eq, sp.Basic) and not cat_eq.is_number:
 logging.warning(f"Equation for {​​​​​​self.name}​​​​​​ -> {​​​​​​possible_cat_of_this_var}​​​​​​ did not fully resolve:", cat_eq)
 cat_values[possible_cat_of_this_var] = self.cdf_mappings[possible_cat_of_this_var](cat_eq)
 noise_for_category = rs.uniform(min_val, max_val) # TODO: replace by the noise distribution self.noise_distribution in the object
 # TODO: Use the noise term for the equation evaluation!!
 # normalize CDF values into a probability vector (well defined since cat_values are all strictly positive)
 cdf_values = np.array(list(cat_values.values()))
 distribution = cdf_values / np.sum(cdf_values)
 assert np.isclose(np.sum(distribution), 1.0), f"CDF values for {​​​​​​self.name}​​​​​​ do not sum to 1: {​​​​​​distribution}​​​​​​"
 # Sample a category based on the CDF values.
 sampled_cat = rs.choice(list(cat_values.keys()), p=distribution)
 # Check if the sampled category is in the domain.
 if str(sampled_cat) not in [str(d) for d in self.domain]:
 raise ValueError(
 f"Sampled category {​​​​​​sampled_cat}​​​​​​ not in domain {​​​​​​self.domain}​​​​​​."
 )
 return sampled_cat
 def _extract_step_points(self, cdf_lambda) -> list:
 """
 Extracts step points from a given CDF lambda function by sampling.
 Args:
 cdf_lambda: The lambda function representing the CDF.
 Returns:
 list: A list of step points extracted from the CDF.
 """
 x_values = np.linspace(0, 1, 100)
 cdf_values = np.array([cdf_lambda(x) for x in x_values])
 return cdf_values.tolist()
 @staticmethod
 def _create_cdf_lambda(step_points: np.ndarray) -> callable:
 """
 Reconstructs a lambda function from step points.
 Args:
 step_points (np.ndarray): The step points used to create the CDF.
 Returns:
 callable: A lambda function representing the CDF.
 """
 return lambda x: np.searchsorted(step_points, x, side="right") / len(
 step_points
 )
 def to_dict(self) -> dict:
 """
 Serializes the node to a dictionary, converting any non-serializable objects as needed.
 Returns:
 dict: A dictionary representation of the node.
 """
 # Serialize the random state.
 state = self.random_state.get_state()
 return {​​​​​​
 "name": self.name,
 "equation": (
 sp.srepr(self.equation)
 if self.var_type == "numerical"
 else {​​​​​​k: sp.srepr(v) for k, v in self.equation.items()}​​​​​​
 ),
 "domain": self.domain,
 "var_type": self.var_type,
 "cdf_mappings": {​​​​​​
 # todo: check if this is correct
 # cat: self._extract_step_points(self.cdf_mappings[cat])
 cat: self.cdf_mappings[cat].to_list()
 for cat in self.cdf_mappings
 }​​​​​​,
 "parent_mappings": self.parent_mappings,
 "random_state": {​​​​​​
 "state": state[0],
 "keys": state[1].tolist(),
 "pos": state[2],
 "has_gauss": state[3],
 "cached_gaussian": state[4],
 }​​​​​​,
 }​​​​​​
 @classmethod
 def from_dict(cls, data: dict) -> "SCMNode":
 """
 Deserializes an SCMNode instance from a dictionary.
 Args:
 data (dict): The dictionary containing node data.
 Returns:
 SCMNode: An instance of SCMNode.
 """
 # Reconstruct the equation from the serialized string.
 safe_dict = {​​​​​​
 "Symbol": sp.Symbol,
 "Integer": sp.Integer,
 "Float": sp.Float,
 "Add": sp.Add,
 "Mul": sp.Mul,
 "Pow": sp.Pow,
 "Max": sp.Max,
 "Min": sp.Min,
 "re": sp.re,
 "np": np,
 "im": sp.im, # todo: check if this is correct
 }​​​​​​
 if data["var_type"] == "numerical":
 equation = eval(data["equation"], safe_dict)
 cdf_mappings = {​​​​​​}​​​​​​
 else:
 # For categorical nodes, reconstruct the equation dictionary.
 equation = {​​​​​​k: eval(v, safe_dict) for k, v in data["equation"].items()}​​​​​​
 # Reconstruct the CDF mappings from step points.
 cdf_mappings = {​​​​​​
 # todo: check if this is correct
 cat: SerializableCDF.from_list(points)
 # cat: cls._create_cdf_lambda(np.array(points))
 for cat, points in data["cdf_mappings"].items()
 }​​​​​​
 # Reconstruct the random state.
 state_tuple = (
 str(data["random_state"]["state"]),
 np.array(data["random_state"]["keys"], dtype=np.uint32),
 int(data["random_state"]["pos"]),
 int(data["random_state"]["has_gauss"]),
 float(data["random_state"]["cached_gaussian"]),
 )
 random_state = np.random.RandomState()
 random_state.set_state(state_tuple)
 # Create the SCMNode instance.
 node = cls(
 name=data["name"],
 equation=equation,
 domain=data["domain"],
 var_type=data["var_type"],
 cdf_mappings=cdf_mappings,
 parent_mappings=data.get("parent_mappings", {​​​​​​}​​​​​​),
 random_state=random_state,
 )
 return node
class SerializableCDF:
 def __init__(self, sorted_samples):
 self.sorted_samples = np.array(sorted_samples)
 def __call__(self, x):
 return np.searchsorted(self.sorted_samples, x, side="right") / len(
 self.sorted_samples
 )
 def to_list(self):
 return self.sorted_samples.tolist()
 @classmethod
 def from_list(cls, data):
 return cls(np.array(data))
 