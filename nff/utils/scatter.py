from itertools import repeat
from torch.autograd import grad


def compute_grad(inputs, output, create_graph=True, retain_graph=True):
    """Compute gradient of the scalar output with respect to inputs.
    
    Args:
        inputs (torch.Tensor): torch tensor, requires_grad=True
        output (torch.Tensor): scalar output 
    
    Returns:
        torch.Tensor: gradients with respect to each input component 
    """
    try:
        assert inputs.requires_grad
    except AttributeError:
        pass
    
    

    # gradspred, = grad(output, inputs, grad_outputs=output.data.new(output.shape).fill_(1),
    #             create_graph=create_graph, retain_graph=retain_graph)
    gradspred = grad(output, inputs, create_graph=create_graph, retain_graph=retain_graph)
    return gradspred[0] if len(gradspred) == 1 else gradspred


def gen(src, index, dim=-1, out=None, dim_size=None, fill_value=0):
    dim = range(src.dim())[dim]  # Get real dim value.

    # Automatically expand index tensor to the right dimensions.
    if index.dim() == 1:
        index_size = list(repeat(1, src.dim()))
        index_size[dim] = src.size(dim)
        index = index.view(index_size).expand_as(src)

    # Generate output tensor if not given.
    if out is None:
        dim_size = index.max().item() + 1 if dim_size is None else dim_size
        out_size = list(src.size())
        out_size[dim] = dim_size
        out = src.new_full(out_size, fill_value)

    return src, out, index, dim


def scatter_add(src, index, dim=-1, out=None, dim_size=None, fill_value=0):
    src, out, index, dim = gen(src, index, dim, out, dim_size, fill_value)
    return out.scatter_add_(dim, index, src)
