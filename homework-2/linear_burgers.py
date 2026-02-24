import jax.numpy as jnp
import numpy as np
import jax
from jax import jit
import matplotlib.pyplot as plt

class linear_solver(object):
    def __init__(self, delta_x, delta_t, pec_num=50):
        self.pec_num = pec_num
        self.delta_x = delta_x
        self.delta_t = delta_t
        self.cfl = self.delta_t / self.delta_x
        self.r = self.delta_t / (self.pec_num * self.delta_x**2)

    def solve(
            self,
            init: jnp.ndarray,
            num_steps: int = 100,
            scheme_num : int = 0,
    ):

        @jit
        def central_update(solution, xs):
            new_solution = solution.at[1:-1].set(
                solution[:-2] * (self.r + self.cfl/2)+
                solution[1:-1] * (1 - 2*self.r) +
                solution[2:] * (self.r - self.cfl/2)
            )
            return new_solution, None
        
        @jit
        def upwind_update_1st(solution, xs):
            new_solution = solution.at[1:-1].set(
                solution[:-2] * (self.r + self.cfl) +
                solution[1:-1] * (1 - 2*self.r -self.cfl) +
                solution[2:] * self.r
            )
            return new_solution, None
        
        @jit
        def upwind_update_2nd(solution, xs):
            new_solution = solution.at[1].set(
                solution[0] * (self.r + self.cfl) +
                solution[1] * (1 - 2*self.r -self.cfl) +
                solution[2] * self.r
            )

            new_solution = new_solution.at[2:-1].set(
                solution[:-3] * (- self.cfl/2) +
                solution[1:-2] * (self.r + 2*self.cfl) +
                solution[2:-1] * (1 - 2*self.r - 3/2 *self.cfl) +
                solution[3:] * self.r
            )

            return new_solution, None

        @jit
        def quick_update(solution, xs):
            new_solution = solution.at[1].set(
                solution[0] * (self.r + self.cfl) +
                solution[1] * (1 - 2*self.r -self.cfl) +
                solution[2] * self.r
            )

            new_solution = new_solution.at[2:-1].set(
                solution[:-3] * (- self.cfl/8) +
                solution[1:-2] * (self.r + 7/8*self.cfl) +
                solution[2:-1] * (1 - 2*self.r - 3/8 *self.cfl) +
                solution[3:] * (self.r - 3/8*self.cfl)
            )

            return new_solution, None

        assert scheme_num in [0,1,2,3]

        if scheme_num == 0:
            solution = jax.lax.scan(central_update, init, None, length=num_steps)[0]

        elif scheme_num == 1:
            solution = jax.lax.scan(upwind_update_1st, init, None, length=num_steps)[0]

        elif scheme_num == 2:
            solution = jax.lax.scan(upwind_update_2nd, init, None, length=num_steps)[0]
        
        elif scheme_num == 3:
            solution = jax.lax.scan(quick_update, init, None, length=num_steps)[0]

        return solution
    
def get_exact(x, Pec_num=50):
    return (jnp.exp(Pec_num * x) - 1 ) / (jnp.exp(Pec_num) - 1)
    
if __name__ == "__main__":

    delta_t = 1e-3
    delta_x = 1 / 50
    x_space = jnp.arange(0, 1+delta_x/2, delta_x)
    init = jnp.zeros_like(x_space).at[-1].set(1)

    linear_solver = linear_solver(delta_t = delta_t, delta_x = delta_x)
    solution = linear_solver.solve(init, num_steps=10_000_000, scheme_num=3)

    plt.plot(x_space, solution, label="Numerical")
    plt.plot(x_space, get_exact(x_space), label="Exact")
    # plt.plot(x_space, get_exact(x_space, Pec_num=1/(1/50+delta_x**2/3*50-delta_x**3/4*50**2)), label="predicted")
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.title("Burgers' Equation Solution")
    plt.grid()
    plt.legend()
    plt.savefig("burgers_solution.png", dpi = 600)
    plt.show()

