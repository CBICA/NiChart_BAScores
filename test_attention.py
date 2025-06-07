import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torchio as tio

subject = tio.Subject(
    mri=tio.ScalarImage(
        "../Datasets/ADNI/ad_regression/eval_AD/019_S_5019_2013-02-11_T1_LPS_dlicv_aligned.nii.gz"
    )
)
resize = tio.Resize((128, 128, 128))
resized_img = resize(subject)
resized_img.mri.save("base.nii.gz")

base_img = nib.load("base.nii.gz").get_fdata()
attention = nib.load("../AD_attention_maps/019_S_5019_2013-02-11.nii.gz").get_fdata()

slice_idx = base_img.shape[2] // 2

plt.figure(figsize=(8, 8))
plt.imshow(base_img[:, :, slice_idx], cmap="gray")
plt.imshow(attention[:, :, slice_idx], cmap="jet", alpha=0.5)
plt.axis("off")
plt.show()
plt.savefig("ad_test.png")
