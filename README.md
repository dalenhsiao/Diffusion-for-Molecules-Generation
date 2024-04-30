# Diffusion for Molecule Generation

This project delves into the application of generative diffusion model graph molecular data study, specifically leveraging the QM9 dataset—a collection of over 13,000 molecules described using graph data structures. Each molecule is represented as a graph with nodes corresponding to atoms and edges describing atomic bonds, facilitating detailed chemical analysis and potential discoveries in fields such as drug discovery and protein science. Utilizing the torch geometric library, this project explores the capabilities of Graph Convolutional Networks (GCNs) for processing molecular graphs. The primary focus is on adapting and refining generative diffusion models, specifically Denoising Diffusion Probabilistic Models (DDPM) and their derivatives, to enhance the generation and sampling of molecular structures. This involves a nuanced approach to data embedding and noise application, ensuring that categorical molecular features are aptly processed for generative tasks. Through extensive experimentation and iterative model refinement, this study aims to establish a robust framework that supports the efficient generation of molecular structures, thereby pushing the boundaries of how graph-based models contribute to scientific advancements in quantum chemistry.


## Model Archietecture

The backbone of our proposed generative diffusion model is Graph Neural Network (GNN). The model is designed to have the capability to learn the molecular attributes and the interconnections between atoms in a molecule. The model is inspired by UNet in image generation and contains "Up Sampling" and "Down Sampling" structure. The information extraction is driven by graph convolution layer

()