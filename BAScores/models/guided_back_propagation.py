import os
from typing import Any, Callable, Optional

import nibabel as nib
import numpy as np
import torch
from torch import nn

from BAScores.utils import save_2d_attention, save_3d_attention


def GuidedBackPropagation(
    device: str,
    mode: str = "regression",
    output_dir: Optional[str] = None,
    return_attention: bool = False,
) -> Callable:
    def decorator(_class: Any) -> Any:
        class Wrapped(_class):  # type: ignore
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                super().__init__(*args, **kwargs)
                self.image_reconstruction = None
                self.activation_maps: list[torch.Tensor] = []
                self.forward_called_ = 0
                self.model_type_ = None

                def forward_hook(_, __, output) -> None:  # type: ignore
                    self.activation_maps.append(output)

                def backward_hook(_, __, grad_out) -> tuple[torch.Tensor]:  # type: ignore
                    grad = self.activation_maps.pop()
                    grad = grad.clone()
                    grad[grad > 0] = 1

                    positive_grad_out = torch.clamp(grad_out[0], min=0.0)
                    new_grad_in = positive_grad_out * grad

                    return (new_grad_in.clone(),)

                if hasattr(self.net, "features") and hasattr(
                    self.net.features, "named_children"
                ):
                    modules = list(self.net.features.named_children())
                elif hasattr(self.net, "named_children"):
                    modules = list(self.net.named_children())
                else:
                    raise TypeError("Couldn't find any layers")

                for _, module in modules:
                    if self.model_type_ is None:
                        if isinstance(module, nn.Conv3d):
                            self.model_type_ = "3D"
                        elif isinstance(module, nn.Conv2d):
                            self.model_type_ = "2D"
                        elif isinstance(module, nn.Conv1d):
                            raise TypeError(
                                "Expected a 2D or a 3D model but got 1D instead"
                            )

                    if isinstance(module, nn.ReLU):
                        module.register_forward_hook(forward_hook)
                        module.register_backward_hook(backward_hook)

            def forward(
                self,
                x: torch.Tensor,
                out_name: str = "gbp_maps_",
                niftii_header: Optional[nib.nifti1.Nifti1Header] = None,
                target_class: Optional[int] = None,
            ) -> Any:
                model_output = self.net(x)
                self.net.zero_grad()

                grad_target_map = torch.zeros(
                    model_output.shape, dtype=torch.float16
                ).to(device)

                if mode == "multiclass":
                    pred_class = model_output.argmax().item()

                    if target_class is not None:
                        grad_target_map[0][target_class] = 1
                    else:
                        grad_target_map[0][pred_class] = 1
                elif mode == "binary":
                    grad_target_map = torch.ones_like(
                        model_output, dtype=torch.float16
                    ).to(device)
                else:  # regression
                    grad_target_map[0][0] = 1

                model_output.backward(grad_target_map)
                self.image_reconstruction = x.grad.clone()

                assert (
                    self.image_reconstruction is not None
                ), "Failed to update image reconstruction"
                if self.model_type_ == "2D":  # type: ignore
                    result = self.image_reconstruction.data[0].permute(1, 2, 0)
                else:
                    result = np.squeeze(self.image_reconstruction)

                if output_dir is not None:
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)

                    if self.model_type_ == "2D":

                        if out_name != "gbp_maps_":
                            save_2d_attention(result, output_dir, out_name)
                        else:
                            save_2d_attention(
                                result, output_dir, out_name, self.forward_called_
                            )
                    else:
                        if niftii_header is None:
                            raise TypeError(
                                "Expected niftii header. Please make sure you pass niftii_header as a parameter and assign it to the niftii header of the original image"
                            )

                        save_3d_attention(
                            result,
                            niftii_header,
                            output_dir,
                            out_name,
                        )

                self.forward_called_ += 1

                if return_attention:
                    return model_output, result

                return model_output

        return Wrapped

    return decorator
