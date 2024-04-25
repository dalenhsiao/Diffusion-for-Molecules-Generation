import torch
import torch.nn as nn 
import torch.nn.functional as F
from tqdm import tqdm
from utils import * 

class DiffusionModel(nn.Module):
    def __init__(
        self, 
        model, 
        timesteps=1000, 
        sampling_timesteps=None, 
        noise_schedule="cosine"
        ):
        super(DiffusionModel, self).__init__()
        self.model = model 
        self.device = torch.cuda.current_device()
        self.timestep = timesteps
        ########### pre-calculated terms
        # sampler = DiffusionModel()
        self.betas = self.beta_scheduler(timesteps=timesteps, mode=noise_schedule).to(self.device)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0).to(self.device)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0).to(self.device)
        # num timestep for diffusion sampling
        self.num_timesteps = self.betas.shape[0]
        
        ###############################
        # Precompute values needed for the diffusion forward processs
        ###############################
        # This is the coefficient of x_t when predicting x_0
        self.x_0_pred_coef_1 = 1 / torch.sqrt(self.alphas_cumprod)
        # This is the coefficient of pred_noise when predicting x_0
        self.x_0_pred_coef_2 = torch.sqrt(1 - self.alphas_cumprod)
        
        ##################################################################
        # Compute the coefficients for the mean.
        ##################################################################
        # This is coefficient of x_0 in the DDPM section
        self.posterior_mean_coef1 = torch.sqrt(self.alphas_cumprod_prev) * self.betas / (1 - self.alphas_cumprod)
        # This is coefficient of x_t in the DDPM section
        self.posterior_mean_coef2 = torch.sqrt(self.alphas) * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)

        ##################################################################
        # Compute posterior variance.
        ##################################################################
        # Calculations for posterior q(x_{t-1} | x_t, x_0) in DDPM
        self.posterior_variance = (1 - self.alphas_cumprod_prev) * self.betas / (1 - self.alphas_cumprod)

        self.posterior_log_variance_clipped = torch.log(
            self.posterior_variance.clamp(min =1e-20))

        # sampling related parameters
        self.sampling_timesteps = default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        # self.ddim_sampling_eta = ddim_sampling_eta


    def beta_scheduler(self,timesteps, start=0.0001, end=0.02, mode="linear"): # Beta for forward pass sampling
        """

        Args:
            timesteps (_type_): total timesteps for diffusion process 
            start (float, optional): scheduler start timestep. Defaults to 0.0001.
            end (float, optional): scheduler end timestep. Defaults to 0.02.
            mode (str, optional): linear or cosine scheduler. Defaults to "linear".

        Returns:
            beta: beta schedule
        """
        mode = mode.lower()
        steps = torch.linspace(start, end, timesteps)
        if mode == "linear":
            beta = steps
            return beta # linear schedule 
        elif mode == "cosine": 
            # Calculate the cosine values, scaled to fit the range between start and end
            values = (torch.cos(steps) + 1) / 2  # Rescale cosine to go from 0 to 1
            beta = values * (end - start) + start  # Scale and shift to fit [start, end]
            
            return beta
        else:
            raise ValueError("Available modes are \"linear\" & \"cosine\" scheduler")
    


    def get_posterior_parameters(self,x_0,x_t,t):
        # Compute the posterior mean and variance for x_{t-1}
        # using the coefficients, x_t, and x_0.
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_0 +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped


    def get_index_from_list(self, vals, t, x_shape): # helper function for getting index
        """
        Returns a specific index t of a passed list of values vals
        while considering the batch dimension.
        """
        batch_size = t.shape[0]
        out = vals.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)     
            
    def sample_forward_diffuse_training(self,x_0, t, device):
            """
            Takes a sample and a timestep as input and
            returns the noisy version of it

            x0 (torch Tensor) : original molecules (n_batch, n_feat)
            total_timesteps (int): total sampled timestep 
            t(torch Tensor, dtype = torch.int64) : sampled timesteps (n_timestep, )
            

            """
            # sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
            sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
            sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
            # posterior_variance = (self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)).clamp(min=1e-20)
            noise = torch.randn_like(x_0) # sampling noise
            sqrt_alphas_cumprod_t = self.get_index_from_list(sqrt_alphas_cumprod, t, x_0.shape) # sqrt_alphas (pre-calculated noises)
            sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(
                sqrt_one_minus_alphas_cumprod, t, x_0.shape
            )
            # mean + variance
            return sqrt_alphas_cumprod_t.to(device) * x_0.to(device) \
            + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device), noise.to(device) # noisy version of the image
            
    def model_prediction(self, x_t, t, edge_index):
        ##################################################################
        # Given a noised data x_t, predict x_0 and the additive
        # noise to predict the additive noise, use the denoising model.
        ##################################################################
        """

        Args:
            x_t (_type_): _description_
            t (_type_):(n_batch, )
        """
        
        pred_noise = self.model(x_t, t, edge_index) # why the t is integer but in computing the time encodeing they are tensor???       
        coef1 = extract(self.x_0_pred_coef_1, t, x_t.shape)
        coef2 = extract(self.x_0_pred_coef_2, t, x_t.shape)
        x_0 = coef1 * (x_t - coef2 * pred_noise)
        ##########
        # x_0 = torch.clamp(x_0, 0, 1) # try not clamping
        return (pred_noise, x_0)

    @torch.no_grad()
    def predict_denoised_at_prev_timestep(self, x, t, edge_index):
        #### why t is integer.... turns out t is not an integer wtf
        pred_noise, x_0 = self.model_prediction(x,t, edge_index)
        posterior_mean, posterior_variance, posterior_log_variance_clipped = self.get_posterior_parameters(x_0, x, t)
        pred_molecule = posterior_mean + torch.sqrt(posterior_variance) * torch.randn_like(x) 
        return pred_molecule, x_0
    
    def sample_times(self, total_timesteps, sampling_timesteps):
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        return list(reversed(times.int().tolist()))
    
    @torch.no_grad()
    def sampling_ddpm(self, shape, z, edge_index):
        """
        diffusion sampling 
        """
        molecule = z
        for t in tqdm(
            range(self.num_timesteps-1,0,-1),
            desc="Denoising Timesteps",
            total=self.num_timesteps-1,
            bar_format="{l_bar}{bar}| timestep {n_fmt}/{total_fmt} [{rate_fmt}]"
            ):
            batched_times = torch.full((shape[0],), t, device=self.device, dtype=torch.long)
            molecule, _ = self.predict_denoised_at_prev_timestep(molecule, batched_times, edge_index)
        # img = unnormalize_to_zero_to_one(img) # maybe we have to do a certain level of scaling
        print(molecule)
        molecule = nn.Softmax()(molecule) # Converting the result to probability distribution
        return molecule
    
    @torch.no_grad()
    def sample(self, x_shape: tuple, edge_index):
        z = torch.randn(x_shape, device=self.betas.device) # random sampling noise
        return self.sampling_ddpm(x_shape, z, edge_index.to(self.betas.device))
    
    @torch.no_grad()
    def sample_given_z(self, z, x_shape):
        # sample_fn = self.sample_ddpm if not self.is_ddim_sampling else self.sample_ddim
        z = z.reshape(x_shape)
        return self.sampling_ddpm(x_shape, z)
        