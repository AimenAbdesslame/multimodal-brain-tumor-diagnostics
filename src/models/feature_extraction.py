import torch
import torch.nn as nn

class UNetFeatureExtractor:
    """
    Wrapper class to extract intermediate activations (bottleneck & decoder)
    from a U-Net model without modifying the underlying architecture.
    """
    def __init__(self, model: nn.Module, target_layers: dict):
        self.model = model
        self.target_layers = target_layers  # Dict mapping layer_name -> module reference
        self.features = {}
        self.handles = []
        
        # -------------------------------------------------------------
        # Register hooks directly inside __init__ (No helper methods!)
        # -------------------------------------------------------------
        for name, layer in self.target_layers.items():
            
            # Closure function to lock the specific layer's name
            def make_hook(layer_name: str):
                def hook_fn(module, input, output):
                    self.features[layer_name] = output.detach()
                return hook_fn

            # Attach hook directly and store handle
            handle = layer.register_forward_hook(make_hook(name))
            self.handles.append(handle)

    def __call__(self, x: torch.Tensor):
        # Forward pass
        model_output = self.model(x)
        # Return both final prediction AND captured features
        return model_output, self.features

    def remove_hooks(self):
        """Clean up handles to prevent GPU memory leaks."""
        for handle in self.handles:
            handle.remove()
        self.handles.clear()