# File: nlm.py
# Description: Non-Local Means Denoising
# Created: 2021/10/22 20:50
# Author: Qiu Jueqin (qiujueqin@gmail.com)


import numpy as np
import torch
import torch.nn.functional as F

from .basic_module import BasicModule, register_dependent_modules
from .helpers import pad, shift_array, mean_filter


@register_dependent_modules('csc')
class NLM(BasicModule):
    def __init__(self, cfg):
        super().__init__(cfg)

        # Original LUT for the original execute method
        self.distance_weights_lut = self.get_distance_weights_lut(h=self.params.h)

        # PyTorch device setup
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            if not torch.backends.mps.is_built():
                print("NLM Warning: MPS not available because the current PyTorch install was not "
                      "built with MPS enabled. Falling back to CPU for PyTorch execution.")
                self.device = torch.device("cpu")
            else:
                self.device = torch.device("mps")
        elif torch.cuda.is_available(): # Fallback for CUDA if user has it
            print("NLM Warning: MPS not available. Falling back to CUDA if available.")
            self.device = torch.device("cuda")
        else:
            print("NLM Warning: MPS and CUDA not available. Falling back to CPU for PyTorch execution.")
            self.device = torch.device("cpu")
        
        print(f"NLM module PyTorch execution will use device: {self.device}")

        # PyTorch LUT, converted from the NumPy LUT and moved to the selected device
        self.distance_weights_lut_torch = torch.from_numpy(self.distance_weights_lut).to(self.device)

        # Store parameters from cfg for PyTorch execution
        self.search_window_size = self.params.search_window_size
        self.patch_size = self.params.patch_size

        # Define AvgPool layer once for efficiency in PyTorch execution
        # Padding P//2 ensures that for an odd kernel size P, the output size matches input size.
        # This assumes self.patch_size is odd, which is typical for such filters.
        if self.patch_size % 2 == 0:
            print(f"NLM Warning: patch_size ({self.patch_size}) is even. "
                  "AvgPool2d padding might not perfectly align with odd-kernel expectations.")
        self.avg_pool = torch.nn.AvgPool2d(kernel_size=self.patch_size,
                                           stride=1,
                                           padding=self.patch_size // 2).to(self.device)

    def execute_np(self, data):
        y_image = data['y_image'].astype(np.int32)

        padded_y_image = pad(y_image, pads=self.params.search_window_size // 2)
        shifted_arrays = shift_array(padded_y_image, window_size=self.params.search_window_size)

        nlm_y_image = np.zeros_like(y_image)
        weights = np.zeros_like(y_image)

        for i, shifted_y_image in enumerate(shifted_arrays):
            distance = mean_filter((y_image - shifted_y_image) ** 2, filter_size=self.params.patch_size)
            weight = self.distance_weights_lut[distance]
            nlm_y_image += shifted_y_image * weight
            weights += weight

        nlm_y_image = nlm_y_image / weights
        data['y_image'] = nlm_y_image.astype(np.uint8)

    def execute(self, data):
        y_image_np = data['y_image']  # Expected (H, W), uint8
        H, W = y_image_np.shape

        # Convert input NumPy array to PyTorch tensor
        # Shape: (1, 1, H, W) for (batch, channels, height, width)
        # Dtype: float32 for calculations, move to selected device
        y_image_torch = torch.from_numpy(y_image_np.astype(np.float32)) \
                             .unsqueeze(0).unsqueeze(0).to(self.device)

        # Pad the image for extracting shifted windows (search window)
        pad_amount = self.search_window_size // 2
        # 'reflect' padding mode is similar to np.pad with mode='reflect'
        padded_y_image_torch = F.pad(y_image_torch,
                                     (pad_amount, pad_amount, pad_amount, pad_amount),
                                     mode='reflect')

        nlm_y_image_accumulator = torch.zeros_like(y_image_torch)
        weights_accumulator = torch.zeros_like(y_image_torch)
        
        # Iterate through all S*S shifts (S = search_window_size)
        for r_offset in range(self.search_window_size):
            for c_offset in range(self.search_window_size):
                # Extract the current shifted window from the padded image
                shifted_y_image_tensor = padded_y_image_torch[:, :,
                                                              r_offset:r_offset + H,
                                                              c_offset:c_offset + W]

                # Calculate squared difference: (y_image_original - shifted_y_image_current_context)^2
                diff_sq = (y_image_torch - shifted_y_image_tensor).pow(2)

                # Calculate patch-wise mean of squared differences using AvgPool2d
                distance_values = self.avg_pool(diff_sq)

                # Weight lookup from LUT
                # Clamp distance values to be valid indices for the LUT (0 to LUT_size-1)
                # The original LUT has size 255**2, so max index is 255**2 - 1.
                max_lut_idx = self.distance_weights_lut_torch.shape[0] - 1
                distance_indices = torch.clamp(distance_values, 0, max_lut_idx).long()
                
                # Retrieve weights. LUT is 1D.
                # Original LUT values are int32 (scaled by 1024). Convert to float for calculations.
                # The scaling factor cancels out in the final division.
                weight = self.distance_weights_lut_torch[distance_indices].float()

                # Accumulate weighted pixel values and weights
                nlm_y_image_accumulator += shifted_y_image_tensor * weight
                weights_accumulator += weight

        # Normalize the accumulated values
        # Add a small epsilon to the denominator to prevent division by zero
        epsilon = 1e-8
        nlm_y_image_final = nlm_y_image_accumulator / (weights_accumulator + epsilon)

        # Convert the final tensor back to a NumPy array with uint8 dtype
        nlm_y_image_output_np = nlm_y_image_final.squeeze().cpu().numpy().clip(0, 255).astype(np.uint8)
        
        data['y_image'] = nlm_y_image_output_np

    @staticmethod
    def get_distance_weights_lut(h):
        distance = np.arange(255 ** 2)
        lut = 1024 * np.exp(-distance / h ** 2)
        return lut.astype(np.int32)  # x1024
