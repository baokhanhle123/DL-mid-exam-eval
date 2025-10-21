import torch
import torch.nn as nn

class JitWrapper(torch.nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        # This is your custom function for inference
        # 1. Get the GMM parameters from the base model
        pi, mu, sigma = self.model(x)
        
        # 2. Get the most probable mean (your final prediction)
        #    Implement get_argmax_mu logic directly here for JIT compatibility
        max_idx = torch.argmax(pi, dim=1)  # [N x D]
        argmax_mu = torch.gather(input=mu, dim=1, index=max_idx.unsqueeze(dim=1)).squeeze(dim=1)  # [N x D]
        
        # 3. Return only the final prediction
        return argmax_mu

def get_argmax_mu(pi, mu):

    max_idx = torch.argmax(pi, dim=1) # [N x D]
    argmax_mu = torch.gather(input=mu, dim=1, index=max_idx.unsqueeze(dim=1)).squeeze(dim=1) # [N x D]
    return argmax_mu


def save_jit_model(model, path, example_input=None):

    print(f"Wrapping model for JIT tracing...")
    # Wrap the model with JitWrapper
    jit_model = JitWrapper(model)
    
    # Set to evaluation mode (important before tracing)
    jit_model.eval() 
    
    # Create example input if not provided
    if example_input is None:
        example_input = torch.randn(1, 1)  # Default input shape
    
    # Trace the wrapped model instead of scripting
    traced_mdn = torch.jit.trace(jit_model, example_input)
    
    # Save the traced model
    traced_mdn.save(path)
    print(f"JIT-traced model saved successfully to {path}")