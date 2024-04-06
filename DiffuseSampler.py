import torch
import torch.nn as nn 
import torch.nn.functional as F

class DiffuseSampler():
    def __init__(self):
        pass
         
            
    def beta_scheduler(self,timesteps, start=0.0001, end=0.02, mode="linear"): # Beta for forward pass sampling
        """

        Args:
            timesteps (_type_): total timesteps for diffusion process 
            start (float, optional): scheduler start timestep. Defaults to 0.0001.
            end (float, optional): scheduler end timestep. Defaults to 0.02.
            mode (str, optional): linear or cosine scheduler. Defaults to "linear".

        Returns:
            _type_: _description_
        """
        mode = mode.lower()
        steps = torch.linspace(start, end, timesteps)
        if mode == "linear":
            return steps # linear schedule 
        elif mode == "cosine": 
            # Calculate the cosine values, scaled to fit the range between start and end
            values = (torch.cos(steps) + 1) / 2  # Rescale cosine to go from 0 to 1
            schedule = values * (end - start) + start  # Scale and shift to fit [start, end]
            
            return schedule
        else:
            raise ValueError("Available modes are \"linear\" & \"cosine\" scheduler")
    


     
    def get_index_from_list(self, vals, t, x_shape): # helper function for getting index
        """
        Returns a specific index t of a passed list of values vals
        while considering the batch dimension.
        """
        batch_size = t.shape[0]
        out = vals.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)     
            
    @staticmethod
    def sample_forward_diffuse_training(x_0, total_timestep, t,device, sampling_mode="linear"):
            """
            Takes a sample and a timestep as input and
            returns the noisy version of it

            x0 (torch Tensor) : original molecules (n_batch, n_feat)
            total_timesteps (int): total sampled timestep 
            t(torch Tensor) : sampled timesteps (n_timestep, )
            

            """
            ########### pre-calculated terms
            sampler = DiffuseSampler()
            betas = sampler.beta_scheduler(timesteps=total_timestep, mode=sampling_mode)
            alphas = 1. - betas
            alphas_cumprod = torch.cumprod(alphas, axis=0)
            alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
            sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
            sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
            sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
            posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
            ############
            noise = torch.randn_like(x_0) # sampling noise
            sqrt_alphas_cumprod_t = sampler.get_index_from_list(sqrt_alphas_cumprod, t, x_0.shape) # sqrt_alphas (pre-calculated noises)
            sqrt_one_minus_alphas_cumprod_t = sampler.get_index_from_list(
                sqrt_one_minus_alphas_cumprod, t, x_0.shape
            )
            # mean + variance
            return sqrt_alphas_cumprod_t.to(device) * x_0.to(device) \
            + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device), noise.to(device) # noisy version of the image