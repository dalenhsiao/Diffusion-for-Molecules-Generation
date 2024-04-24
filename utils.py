import matplotlib.pyplot as plt
import numpy as np
import torch

def extract(a, t, x_shape):
    """
    The extract function is used to fetch specific 
    timestep values from the coefficients based on the
    timestep t and reshape them to match the dimensions of x_t:

    Args:
        a (_type_): _description_
        t (_type_): _description_
        x_shape (_type_): _description_

    Returns:
        _type_: _description_
    """
    b, *_ = t.shape  # b is the batch size; * captures the remaining dimensions if any (though typically this might be empty)
    out = a.gather(-1, t)  # Retrieves values from 'a' at indices specified by 't' along the last dimension of 'a'
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))  # Reshapes 'out' to have a batch size of 'b' and the rest are singular dimensions


# def save_samples(samples, fname, nrow=6, title='Samples'):
#     grid_img = make_grid(samples, nrow=nrow)
   
#     plt.title(title)
#     plt.imshow(grid_img.permute(1, 2, 0))
#     plt.tight_layout()
#     plt.savefig(fname)

# helpers functions for diffusion
def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

# normalization functions
def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

def min_max_scale(x):
    # Compute the min and max values of the tensor along the feature dimension
    min_val = torch.min(x, dim=0, keepdim=True).values
    max_val = torch.max(x, dim=0, keepdim=True).values
    
    # Compute the range and add a small epsilon to prevent division by zero
    range_val = max_val - min_val + 1e-10
    
    # Scale x to the range [0, 1]
    x_scaled = (x - min_val) / range_val
    return x_scaled