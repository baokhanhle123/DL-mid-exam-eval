# %%
import torch
from torch import nn

StudentNumber = "202501848"  # TODO: Replace with your student number


class MixtureDensityNetwork(nn.Module):
    def __init__(self, in_dim=1, out_dim=1, K=2, hidden=64):
        super().__init__()

        self.net = nn.Identity()  # Simple linear layer as placeholder

    def forward(self, x):

        return self.net(x)


class JitWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        # Write your custom function for inference
        return self.model(x) + 1.0  # Example modification


def save_jit_model(model, path):
    # Wrap the model with JitWrapper
    jit_model = JitWrapper(model)
    scripted_mdn = torch.jit.script(jit_model)
    scripted_mdn.save(path)


if __name__ == "__main__":
    # Example usage
    mdn = MixtureDensityNetwork(in_dim=1, out_dim=1, K=3, hidden=64)
    save_jit_model(mdn, f"model/{StudentNumber}.pth")

    # Load the JIT model
    loaded_model = torch.jit.load(f"model/{StudentNumber}.pth")
    x = torch.arange(5).float().unsqueeze(-1)
    output = loaded_model(x)
    print(output)  # Should print input + 1.0
