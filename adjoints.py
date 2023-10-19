import torch
from tqdm import tqdm
from mdsim.common.utils import process_gradient

def get_adjoints(simulator, pos_traj, grad_outputs, force_fn):
    '''
    pos_traj: [N_ics, T, N_atoms, 3]
    grad_outputs: [N_ics, T, N_atoms, 3]
    '''
    with torch.no_grad():
        a_dt = simulator.dt*1 #save frequency = 1 for now
        M = simulator.masses
        adjoints = []
        adjoint_norms = []
        testR = []
        #initial adjoints
        #grad_outputs *= a_dt**2/M #pre-multiply the grad outputs to make norms closer to naive backprop (still not sure why this is needed)
        a = grad_outputs[:, -1]
        for i in tqdm(range(simulator.vacf_window)):
            #work backwards from the final state
            R = pos_traj[:, -i -1].detach().to(simulator.device)
            
            testR.append(R.detach())
            adjoints.append(a.detach())
            adjoint_norms.append(a.norm(dim = (-2, -1)).detach())
            #compute VJP between adjoint (a) and df/dR which is the time-derivative of the adjoint state
            #verified that this does the same thing as torch.autograd.grad(forces, R, a, create_graph = True)
            #where forces = force_fn(R)
            _, vjp_a = torch.autograd.functional.vjp(force_fn, R, a)
            #update adjoint state
            a = a + a_dt**2 * vjp_a /M - a_dt*simulator.gamma * a
            #adjust in the direction of the next grad outputs
            if i != simulator.vacf_window -1:   
                a = a + grad_outputs[:, -i - 2]
            
        adjoints = torch.stack(adjoints, axis=1)
        adjoint_norms = torch.stack(adjoint_norms)
        testR = torch.stack(testR, axis=1)
    return adjoints, adjoint_norms, testR

#calcuate final model gradients using adjoints
def get_model_grads(adjoints, radii, simulator):
    '''
    radii: [T, N_atoms, 3]
    adjoints: [T, N_atoms, 3]
    '''
    with torch.enable_grad():
        radii.requires_grad=True
        _, forces = simulator.force_calc(radii, retain_grad = True)
    #compute gradient of force w.r.t model params
    grads = torch.autograd.grad(forces, simulator.model.parameters(), \
                                adjoints, create_graph = True, allow_unused = True)
    processed_grads = process_gradient(simulator.model.parameters(), grads, simulator.device)
    return processed_grads
    