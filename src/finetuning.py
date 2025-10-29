import torch
import torch.nn as nn
import math

try:
    from utils import download_and_load_model
except:
    from src.utils import download_and_load_model

class LoRA(nn.Module):
    def __init__(self, original_layer, r=4, alpha=32):
        """
        Low-Rank Adaptation (LoRA) module.
        
        Args:
            original_layer (nn.Module): The original layer to which LoRA is applied.
            r (int): Rank of the low-rank approximation.
            alpha (int): Scaling factor for the LoRA module.
        """
        super().__init__()
        # TODO: Initialize LoRA parameters
        self.r = r
        self.alpha = alpha
        self.original_layer = original_layer

        # TODO: Low-rank matrices A and B for LoRA
        self.A = torch.zeros(original_layer.in_features,r)
        self.B = torch.zeros(r,original_layer.out_features)

        # TODO: Initialize LoRA weights (B is zero-initialized, A is random)
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

        # TODO: Scaling factor alpha
        self.scaling = alpha / r

        # TODO: Freeze the original layer parameters
        for param in self.original_layer.parameters():
            param.requires_grad = False
                
    def forward(self, x):
        # TODO: Perform forward pass with low-rank update
        original_output = self.original_layer(x)
        lora_update = x @(self.A @ self.B) 
        return original_output + lora_update* self.scaling
    
def inject_lora_into_model(model, r=4, alpha=32, device='cpu'):
    """
    Inject LoRA layers into the linear layers of the attention modules of the model.
    
    Args:
        model (PreTrainedModel): The pre-trained model.
        r (int): Rank of the low-rank approximation.
        alpha (int): Scaling factor for LoRA.
        device (torch.device): The device to run the model on ('cuda' or 'cpu').
    
    Returns:
        model (PreTrainedModel): The model with LoRA injected into attention layers.
    """
    # TODO: Iterate through all child modules of the model
    for child_name, child_module in model.named_children():
        # TODO: Check if the child module is a linear layer of the attention module
        if hasattr(child_module, "q") and isinstance(child_module.q, nn.Linear):
            child_module.q = LoRA(child_module.q, r=r, alpha=alpha).to(device)
        if hasattr(child_module, "k") and isinstance(child_module.k, nn.Linear):
            child_module.k = LoRA(child_module.k, r=r, alpha=alpha).to(device)
        if hasattr(child_module, "v") and isinstance(child_module.v, nn.Linear):
            child_module.v = LoRA(child_module.v, r=r, alpha=alpha).to(device)
        if hasattr(child_module, "o") and isinstance(child_module.o, nn.Linear):
            child_module.o = LoRA(child_module.o, r=r, alpha=alpha).to(device)
        else:
            # Recorremos recursivamente
            inject_lora_into_model(child_module, r=r, alpha=alpha, device=device)
    return model.to(device)


class SoftPromptEmbedding(nn.Module):
    def __init__(self, prompt_length, model_hidden_size):
        """
        Creates trainable soft prompts to prepend to input embeddings.

        Args:
            prompt_length (int): Number of virtual tokens in the soft prompt.
            model_hidden_size (int): The hidden size of the pre-trained model.
        """
        super().__init__()
        # TODO: Initialize soft prompt embeddings
        self.soft_prompt = nn.Parameter(torch.randn(prompt_length, model_hidden_size))

    def forward(self, input_embeddings):
        """
        Forward pass to prepend soft prompts to input embeddings.

        Args:
            input_embeddings (torch.Tensor): The original input embeddings from the tokenizer.

        Returns:
            torch.Tensor: The concatenated soft prompts and original embeddings.
        """
        # TODO: Expand soft prompt to match batch size
        batch_size = input_embeddings.size(0)
        soft_prompt_expanded = self.soft_prompt.unsqueeze(0).expand(batch_size, -1, -1)

        # TODO: Concatenate soft prompt and input embeddings
        return torch.cat((soft_prompt_expanded, input_embeddings), dim=1)