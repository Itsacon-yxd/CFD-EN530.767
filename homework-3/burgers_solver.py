from functools import partial
from arrow import get
import jax.numpy as jnp
import jax
from jax import jit
import numpy as np
import matplotlib.pyplot as plt
from jax_tqdm import scan_tqdm
from pyparsing import nums

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
        


    def build_lhs(self, Nx, Ny):

        lhs_a = 1 + 2*self.r_x+ 2*self.r_y
        lhs_b = -self.r_x
        lhs_c = -self.r_y
        lhs_d = -self.r_x
        lhs_e = -self.r_y

        lhs_A = np.zeros(((Nx-1)*(Ny-1), (Ny-1)*(Nx-1)))
        lhs_block = np.zeros((Ny-1,Ny-1))
        np.fill_diagonal(lhs_block, lhs_a)
        np.fill_diagonal(lhs_block[1:,:-1], lhs_c)
        np.fill_diagonal(lhs_block[:-1,1:], lhs_e)

        for i in range(Nx-1):
            lhs_A[i*(Ny-1):(i+1)*(Ny-1), i*(Ny-1):(i+1)*(Ny-1)] = lhs_block

        np.fill_diagonal(lhs_A[:-(Ny-1),Ny-1:], lhs_d)
        np.fill_diagonal(lhs_A[Ny-1:,:-(Ny-1)], lhs_b)

        lhs_A = jnp.array(lhs_A)

        return lhs_A
    
    def solve(self, init, bdry_cond, num_steps, penta_solver):
        # i,j=0,1,2,...,N
        Nx, Ny, C = init.shape
        assert C == 2
        Nx = Nx - 1
        Ny = Ny - 1
        self.reset_params()
        
        lhs = self.build_lhs(Nx, Ny)
        
        r_x = self.mu * self.delta_t / self.delta_x**2
        r_y = self.mu * self.delta_t / self.delta_y**2

        g_u = bdry_cond[..., 0]   # (Ny, Nx) boundary values, interior can be 0
        g_v = bdry_cond[..., 1]

        bc_u = jnp.zeros((Nx-1, Ny-1))
        bc_v = jnp.zeros((Nx-1, Ny-1))

        # left/right boundaries (i=0 and i=-1) contribute to interior i=1 and i=N-1
        bc_u = bc_u.at[0,  :].add(r_x * g_u[0,  1:-1])
        bc_u = bc_u.at[-1, :].add(r_x * g_u[-1, 1:-1])
        bc_v = bc_v.at[0,  :].add(r_x * g_v[0,  1:-1])
        bc_v = bc_v.at[-1, :].add(r_x * g_v[-1, 1:-1])

        # bottom/top boundaries (j=0 and j=-1) contribute to interior j=1 and j=N-1
        bc_u = bc_u.at[:, 0 ].add(r_y * g_u[1:-1, 0 ])
        bc_u = bc_u.at[:, -1].add(r_y * g_u[1:-1, -1])
        bc_v = bc_v.at[:, 0 ].add(r_y * g_v[1:-1, 0 ])
        bc_v = bc_v.at[:, -1].add(r_y * g_v[1:-1, -1])

        bdry_cond = jnp.stack([bc_u, bc_v], axis=-1)
        
        def update_u(self, solution, lhs_A, bdry_cond,penta_solver): # only handles dirichlet boundary condition
            u = solution[...,0]
            v = solution[...,1]
            
            rhs = (
                u[1:-1,1:-1] +
                u[1:-1,1:-1]*self.cfl_x/2 * u[:-2,1:-1] +
                v[1:-1,1:-1]*self.cfl_y/2 * u[1:-1,:-2] -
                u[1:-1,1:-1]*self.cfl_x/2 * u[2:,1:-1] -
                v[1:-1,1:-1]*self.cfl_y/2 * u[1:-1,2:]
            )
            rhs = rhs.flatten() + bdry_cond.flatten()

            new_u = u.at[1:-1,1:-1].set(
                penta_solver(lhs_A, rhs).reshape(u[1:-1,1:-1].shape)
            )

            return new_u

        def update_v(self, solution, lhs_A, bdry_cond, penta_solver): # only handles dirichlet boundary condition
            u = solution[...,0]
            v = solution[...,1]
            
            rhs = (
                v[1:-1,1:-1] +
                u[1:-1,1:-1]*self.cfl_x/2 * v[:-2,1:-1] +
                v[1:-1,1:-1]*self.cfl_y/2 * v[1:-1,:-2] -
                u[1:-1,1:-1]*self.cfl_x/2 * v[2:,1:-1] -
                v[1:-1,1:-1]*self.cfl_y/2 * v[1:-1,2:] 
                )
            rhs = rhs.flatten() + bdry_cond.flatten()

            new_v = v.at[1:-1,1:-1].set(
                penta_solver(lhs_A, rhs).reshape(v[1:-1,1:-1].shape)
            )

            return new_v
        

        def step(solution, xs, lhs_A, bdry_cond, penta_solver):
            new_u = update_u(self, solution, lhs_A, bdry_cond[...,0], penta_solver)
            new_v = update_v(self, solution, lhs_A, bdry_cond[...,1], penta_solver)

            return jnp.stack([new_u, new_v], axis=-1), None
        
        step = jit(partial(step, lhs_A=lhs, bdry_cond=bdry_cond, penta_solver=penta_solver))
        step = scan_tqdm(n=num_steps)(step)
        solution = jax.lax.scan(step, init, jnp.arange(num_steps))[0]

        return solution
    
def penta_solver(A, b):
    # to be done: implement ADI method for solving pentadiagonal system Ax=b
    return jnp.linalg.solve(A, b)
    
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

if __name__ == "__main__":
    delta_t = 1e-4
    delta_x = 2e-2
    delta_y = 2e-2
    mu = 1e-3
    num_steps = 20000
    gridx, gridy = jnp.meshgrid(jnp.arange(0,2+delta_x/2,delta_x), jnp.arange(0,1+delta_y/2,delta_y), indexing='ij')
    grid = jnp.stack([gridx, gridy], axis=-1)

    bdry_cond = jnp.zeros_like(grid)
    bdry_cond = bdry_cond.at[0,:,:].set(jnp.array([1.0, 0.0]))
    bdry_cond = bdry_cond.at[-1,:,:].set(jnp.array([1.0, 0.0]))
    bdry_cond = bdry_cond.at[:,0,:].set(jnp.array([1.0, 0.0]))
    bdry_cond = bdry_cond.at[:,-1,:].set(jnp.array([1.0, 0.0]))
    
    init = bdry_cond.at[1:-1,1:-1].set(get_init(grid[1:-1,1:-1]))

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12,6))
    im = axes[0].imshow(init[...,0].T, origin='lower')
    plt.colorbar(im)
    im = axes[1].imshow(init[...,1].T, origin='lower')
    plt.colorbar(im)
    plt.close()

    solver = burgers_solver(delta_x, delta_y, delta_t, mu)
    solution = solver.solve(init, bdry_cond, num_steps, penta_solver)
    plt.imshow(solution[...,0].T, origin='lower', extent=(0,2,0,1), cmap='plasma')
    plt.colorbar()
    plt.show()


