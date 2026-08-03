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


## experiment : 
step 1 : diagnostic test 
1-train the Model 1 U-net (encoder - bottle neck - decoder ) on the Segmentation Task 1 => accuracy A1
2-train the Modle 2  (U-net encoder - U-net bottle neck - MLP ) on the classification Task 2 => accuracy B1
3-train the model 3 on both task 1 and task 2 => accuracy (A-mtl , B-mtl)

step 2 : verify if we have negative transfer 
if (A-mtl < A1) or (B-mtl < B1) then we have negative sampling => step 3 (fix negative sampling via PCgrad)
if (A-mtl > A1) and(B-mtl > B1) then : synergy achieved 

 