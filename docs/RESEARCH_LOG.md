# 📓 Research & Experimentation Log

## Experiment ID: EXP-2026-07-28-HOOKS-01
**Objective:** Non-invasive extraction of bottleneck and decoder activation tensors using PyTorch forward hooks.  
**Target Submodules:** `encoder_bottleneck` and `decoder_final_conv`.  

### 1. Theoretical Expectations
By attaching forward hook handles to the intermediate activation layers of the segmentation backbone, we can route spatial tensor representations to secondary model heads (e.g., classification MLP) without mutating `backbone.forward()`.

### 2. Empirical Tensor Verification
* **Input Tensor Shape:** `[Batch_Size=2, Channels=3, Height=256, Width=256]`
* **Target Bottleneck Activation Shape:** Expected `[2, 512, 16, 16]`
* **Target Decoder Activation Shape:** Expected `[2, 64, 256, 256]`

### 3. Memory & Autograd Verification
Hooks must explicitly invoke `.detach()` to drop autograd computational graph pointers from intermediate feature dictionaries, preventing VRAM leaks during iterative training passes.