from pymoo.algorithms.nsga2 import NSGA2
from pymoo.factory import get_problem
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from pymoo.model.problem import Problem
from ent_purification import probability, fidelity, m1, m2, rho_xy
from pymoo.factory import get_sampling, get_crossover, get_mutation, get_termination
import numpy as np


class EntPurif(Problem):

    def __init__(self, x, y, phi):
        super().__init__(n_var=4, n_obj=2, n_constr=2, xl=0, xu=1, elementwise_evaluation=True)
        self.m_op = m1
        self.rho, x, y = rho_xy(x, y, phi)
        self.fid = lambda x: fidelity(self.rho, self.m_op, x)
        self.prob = lambda x: 1 - probability(self.rho, self.m_op, x)

    def _evaluate(self, x, out, *args, **kwargs):
        # print(x.shape)
        f1 = self.fid(x)
        f2 = self.prob(x)

        g1 = self.prob(x) - 1
        g2 = self.fid(x) - 1

        out["F"] = [f1, f2]
        out["G"] = [g1, g2]

        return out


x = 0.3
y = np.sqrt(1 - x ** 2)
problem = EntPurif(x, y, 0.1)
algorithm = NSGA2(pop_size=50)

res = minimize(problem,
               algorithm,
               ('n_gen', 200),
               seed=1,
               verbose=True)

print(res.X, res.F, res.G, res.CV)
plot = Scatter()
plot.add(problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
plot.add(res.F, color="red")
plot.show()
