from functools import partial
import jax.numpy as jnp
import jax
from jax import jit
from matplotlib.colors import Normalize
import numpy as np
import matplotlib.pyplot as plt
from jax_tqdm import scan_tqdm

class burgers_solver(object):
    def __init__(self, delta_x, delta_y, delta_t, mu):
        self.delta_x = delta_x
        self.delta_y = delta_y
        self.delta_t = delta_t
        self.mu = mu
        self.reset_params()

    def reset_params(self):
        self.r_x = self.mu*self.delta_t/self.delta_x**2
        self.r_y = self.mu*self.delta_t/self.delta_y**2
        self.cfl_x = self.delta_t/self.delta_x
        self.cfl_y = self.delta_t/self.delta_y

        self.delta_k = 1. / (1+4*self.r_x+4*self.r_y)

    def solve(self, init, num_steps, show_progress=True, bdry_type="dirichlet"):
        # i,j=0,1,2,...,N
        if bdry_type == "dirichlet":
            jacobi = self.jacobi_dirichlet
        elif bdry_type == 'neumann':
            jacobi = self.jacobi_neumann
        else:
            raise NotImplementedError("Boundary condition type not implemented")
        
        def update_u(self, solution): # only handles dirichlet boundary condition
            u = solution[...,0]
            v = solution[...,1]
            
            rhs = (
                u[1:-1,1:-1] +
                u[1:-1,1:-1]*self.cfl_x/2 * u[:-2,1:-1] +
                v[1:-1,1:-1]*self.cfl_y/2 * u[1:-1,:-2] -
                u[1:-1,1:-1]*self.cfl_x/2 * u[2:,1:-1] -
                v[1:-1,1:-1]*self.cfl_y/2 * u[1:-1,2:]
            )

            new_u = jacobi(u, rhs)

            return new_u

        def update_v(self, solution): # only handles dirichlet boundary condition
            u = solution[...,0]
            v = solution[...,1]
            
            rhs = (
                v[1:-1,1:-1] +
                u[1:-1,1:-1]*self.cfl_x/2 * v[:-2,1:-1] +
                v[1:-1,1:-1]*self.cfl_y/2 * v[1:-1,:-2] -
                u[1:-1,1:-1]*self.cfl_x/2 * v[2:,1:-1] -
                v[1:-1,1:-1]*self.cfl_y/2 * v[1:-1,2:] 
                )

            new_v = jacobi(v, rhs)

            return new_v
        

        def step(solution, xs):
            new_u = update_u(self, solution)
            new_v = update_v(self, solution)

            return jnp.stack([new_u, new_v], axis=-1), None
        
        step = jit(step)
        step = scan_tqdm(n=num_steps, print_rate=1)(step) if show_progress else step
        solution, _ = jax.lax.scan(step, init, jnp.arange(num_steps))

        return solution
    
    def jacobi_dirichlet(self, init, rhs, tol=1e-7, max_iter=2000):

        delta_k = self.delta_k
        
        def cond_fn(state):
            solution, residual_norm, iteration = state
            return (residual_norm > tol) & (iteration < max_iter)
        
        def body_fn(state):
            solution, _, iteration = state
            
            conved = (
                (1 + 2*self.r_x + 2*self.r_y) * solution[1:-1,1:-1] # u_i,j
                - self.r_x * solution[:-2, 1:-1] # u_i-1,j
                - self.r_x * solution[2:, 1:-1] # u_i+1,j
                - self.r_y * solution[1:-1, :-2] # ui,j-1
                - self.r_y * solution[1:-1, 2:] # u_i,j+1
            )
            residual = rhs - conved
            
            solution = solution.at[1:-1,1:-1].set(
                solution[1:-1,1:-1] + delta_k * residual
            )
            
            residual_norm = jnp.linalg.norm(residual) / residual.shape[0]
            
            return solution, residual_norm, iteration + 1
        
        # Initialize
        init_residual = 1.0
        init_state = (init, init_residual, 0)
        
        solution, final_residual, iterations = jax.lax.while_loop(cond_fn, body_fn, init_state)
        
        return solution
    
    def jacobi_neumann(self, init, rhs, tol=1e-7, max_iter=2000):
        
        delta_k = self.delta_k
        
        def cond_fn(state):
            solution, residual_norm, iteration = state
            return (residual_norm > tol) & (iteration < max_iter)
        
        def body_fn(state):
            solution, _, iteration = state
            
            conved = (
                (1 + 2*self.r_x + 2*self.r_y) * solution[1:-1,1:-1] # u_i,j
                - self.r_x * solution[:-2, 1:-1] # u_i-1,j
                - self.r_x * solution[2:, 1:-1] # u_i+1,j
                - self.r_y * solution[1:-1, :-2] # ui,j-1
                - self.r_y * solution[1:-1, 2:] # u_i,j+1
            )
            residual = rhs - conved
            
            solution = solution.at[1:-1,1:-1].set(
                solution[1:-1,1:-1] + delta_k * residual
            )
            solution = solution.at[-1, :].set(solution[-2, :])
            
            residual_norm = jnp.linalg.norm(residual) / residual.shape[0]
            
            return solution, residual_norm, iteration + 1
        
        # Initialize
        init_residual = 1.0
        init_state = (init, init_residual, 0)
        
        solution, final_residual, iterations = jax.lax.while_loop(cond_fn, body_fn, init_state)
        
        return solution
    
def get_init(grid):
    Vt = 0.25
    x0 = 0.5
    y0 = 0.5
    r0 = 0.1
    x, y = grid[...,0], grid[...,1]
    r = jnp.sqrt((x-x0)**2 + (y-y0)**2)
    u = 1 - Vt*(y-y0) * jnp.exp( (1-(r/r0)**2) / 2 )
    v = Vt*(x-x0) * jnp.exp( (1-(r/r0)**2) / 2 )
    return jnp.stack([u,v],axis=-1)

def get_vorticity(solution, delta_x, delta_y):
    u = solution[...,0]
    v = solution[...,1]

    u_y = u.at[:,1:-1].set(
        (u[:, 2:] - u[:, :-2]) / (2*delta_y)
    )
    u_y = u_y.at[:,0].set(
        (u[:,1] - u[:,0]) / (2*delta_y)
    )
    u_y = u_y.at[:,-1].set(
        (u[:,-1] - u[:,-2]) / (2*delta_y)
    )
    v_x = v.at[1:-1,:].set(
        (v[2:, :] - v[:-2, :]) / (2*delta_x)
    )
    v_x = v_x.at[0,:].set(
        (v[1,:] - v[0,:]) / (2*delta_x)
    )
    v_x = v_x.at[-1,:].set(
        (v[-1,:] - v[-2,:]) / (2*delta_x)
    )

    return v_x-u_y

if __name__ == "__main__":

    plt.rcParams.update({'font.size': 14})

    delta_t = 1e-3
    delta_x = 5e-3
    delta_y = 5e-3
    x_lim = 4
    y_lim = 1
    mu = 1e-2
    num_steps = 3000

    gridx, gridy = jnp.meshgrid(jnp.arange(0, x_lim+delta_x/2, delta_x), jnp.arange(0, y_lim+delta_y/2, delta_y), indexing='ij')
    grid = jnp.stack([gridx, gridy], axis=-1)

    init_solution = get_init(grid)
    init_solution = init_solution.at[0,:,:].set(jnp.array([1.0, 0.0]))
    init_solution = init_solution.at[-1,:,:].set(jnp.array([1.0, 0.0]))
    init_solution = init_solution.at[:,0,:].set(jnp.array([1.0, 0.0]))
    init_solution = init_solution.at[:,-1,:].set(jnp.array([1.0, 0.0]))
    
    init_vorticity = get_vorticity(init_solution, delta_x, delta_y)

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8,12), layout='constrained')
    im = axes[0].imshow(init_solution[...,0].T, origin='lower', cmap='plasma', extent=(0,x_lim,0,y_lim))
    axes[0].set_title("u")
    plt.colorbar(im)
    im = axes[1].imshow(init_solution[...,1].T, origin='lower', cmap='plasma', extent=(0,x_lim,0,y_lim))
    axes[1].set_title("v")
    plt.colorbar(im)
    im = axes[2].imshow(init_vorticity.T, origin='lower', cmap='plasma', extent=(0,x_lim,0,y_lim))
    axes[2].set_title(r"$\nabla \times\mathbf{u}$")
    plt.colorbar(im)
    
    plt.savefig("prelim_tests/burgers_init.png", dpi = 600)
    plt.close()

    solver = burgers_solver(delta_x, delta_y, delta_t, mu)
    solution = solver.solve(init_solution, num_steps, show_progress=True)
    vorticity = get_vorticity(solution, delta_x, delta_y)

    u_norm = Normalize(vmin=init_solution[...,0].min(), vmax=init_solution[...,0].max())
    v_norm = Normalize(vmin=init_solution[...,1].min(), vmax=init_solution[...,1].max())
    vort_norm = Normalize(vmin=init_vorticity.min(), vmax=init_vorticity.max())
    
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8, 12), layout='constrained')
    im = axes[0].imshow(solution[...,0].T, origin='lower', extent=(0,x_lim,0,y_lim), cmap='plasma')
    axes[0].set_title("u")
    axes[0].set_aspect('equal')
    plt.colorbar(im)

    im = axes[1].imshow(solution[...,1].T, origin='lower', extent=(0,x_lim,0,y_lim), cmap='plasma')
    axes[1].set_title("v")
    axes[1].set_aspect('equal')
    plt.colorbar(im)

    im = axes[2].imshow(vorticity.T, origin='lower', extent=(0,x_lim,0,y_lim), cmap='plasma')
    axes[2].set_title(r"$\nabla \times\mathbf{u}$")
    axes[2].set_aspect('equal')
    plt.colorbar(im)
    
    plt.savefig("prelim_tests/dirichlet_solution.png", dpi = 600)
    plt.show()


