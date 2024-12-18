---
layout: default
---

<figure align="center"  style="margin-top: 2em; margin-bottom: 2em">
  <img src="assets/images/WACV2023.png" width="450" />
  <figcaption>Left: Inpainting result on ultra high-resolution images. Right: Comparison of LPIPS performance and Latency among different state-of-the-art methods..</figcaption>
</figure>

## Abstract

Existing image inpainting methods have shown impressive completion results for low-resolution images. However, most of these algorithms fail at high resolutions and require powerful hardware, limiting their deployment on edge devices. Motivated by this, we propose the first baseline for REal-Time High-resolution image INpainting on Edge Devices (RETHINED) that is able to inpaint at ultra-high-resolution and can run in real-time ($\leq$ 30ms) in a wide variety of mobile devices. A simple, yet effective novel method formed by a lightweight Convolutional Neural Network (CNN) to recover structure, followed by a resolution-agnostic patch replacement mechanism to provide detailed texture. Specially our pipeline leverages the structural capacity of CNN and the high-level detail of patch-based methods, which is a key component for high-resolution image inpainting. To demonstrate the real application of our method, we conduct an extensive analysis on various mobile-friendly devices and demonstrate similar inpainting performance while being $\mathrm{100 \times faster}$ than existing state-of-the-art 
methods. Furthemore,
we realease DF8K-Inpainting, the first free-form mask UHD inpainting dataset.

## Method

<figure align="center"  style="margin-top: 2em; margin-bottom: 2em">
  <img src="assets/images/methodcvpr2023V3.png" width="450" />
  <figcaption>Given a HR image and a binary mask  with corrupted pixels as inputs (left), our model first  downsamples to a lower resolution and  forwards it to the coarse model, obtaining a coarse inpainting representation. It is then refined by the NeuralPatchMatch module. Finally, our Attention Upscaling  module yields the final HR image.</figcaption>
</figure>

Given a high-resolution RGB image $\mathbf{y} \in \mathbb{R}^{H_{\text{HR}}\times W_{\text{HR}}\times 3}$ (where $H_{\text{HR}}$ and $W_{\text{HR}}$ denote, respectively, the height and width of the high-resolution image in pixels) and a binary mask $\mathbf{m} \in \mathbb{R}^{H_{\text{HR}}\times W_{\text{HR}}}$ containing the corrupted pixels, our goal is to fill-in with plausible information the masked image $\mathbf{x} = \mathbf{y} \odot \mathbf{m}$.
To achieve this goal, we first downsample $\mathbf{x}$ to a lower resolution obtaining $\mathbf{x_{\text{LR}}}\in \mathbb{R}^{H\times W\times 3}$ (where $H<H_{\text{HR}}$ and $W<W_{\text{HR}}$) and forward it to the coarse model, obtaining the coarse inpainted image $\hat{\mathbf{x}}_{\text{coarse}}$ of size $H\times W$ . Then, we use the NeuralPatchMatch module to refine $\hat{\mathbf{x}}_{\text{coarse}}$ by propagating known content from the input image $\mathbf{x_{\text{LR}}}$, obtaining $\hat{\mathbf{x}}_{\text{LR}}$ and the corresponding attention map $\mathbf{A}$.
Finally our Attention Upscaling  module uses the learned attention map $\mathbf{A}$ and $\mathbf{x}$ to resemble high texture details found on the base image, finally obtaining a high-resolution image $\hat{\mathbf{x}}_{\text{HR}}$.

<figure align="center"  style="margin-top: 2em; margin-bottom: 2em">
  <img src="assets/images/neuralPatchMatchV2.jpg" width="700" />
  <figcaption>Figure 3. <b>\textbf{Proposed NeuralPatchMatch Inpainting Module.} (Corrupted patches are displayed as red \fcolorbox{red}{white}{\rule{0pt}{5pt}\rule{5pt}{0pt}} while uncorrupted ones as green \fcolorbox{green}{white}{\rule{0pt}{5pt}\rule{5pt}{0pt}}.) First, we project patch embedding to embedding space of dimension $d_{k}$ (Sect.~\ref{sec:neural_patch_match}). Then token similarity is computed in a self-attention manner, obtaining attention map $\mathbf{A}$ (where lighter colors \fcolorbox{black}{yellow}{\rule{0pt}{5pt}\rule{5pt}{0pt}} correspond to a large softmax value while darker colors \fcolorbox{black}{black}{\rule{0pt}{5pt}\rule{5pt}{0pt}} correspond to a low value). The self-attention masking allows to inpaint only on corrupted regions, maintaining high-frequency details from uncorrupted zones. To obtain the final inpainted image, we mix the tokens via a weighted sum based on the attention map $\mathbf{A}$..</figcaption>
</figure>

## Results

The proposed method is compared against parametric model-based methods like [DECA](https://arxiv.org/abs/2012.04012) in single-view face reconstruction, and per-scene-optimization approaches like [H3D-Net](https://openaccess.thecvf.com/content/ICCV2021/html/Ramon_H3D-Net_Few-Shot_High-Fidelity_3D_Head_Reconstruction_ICCV_2021_paper.html) and [SIRA](https://arxiv.org/abs/2209.03027) in single-view and multi-view full-head reconstruction, using the [H3DS](https://openaccess.thecvf.com/content/ICCV2021/html/Ramon_H3D-Net_Few-Shot_High-Fidelity_3D_Head_Reconstruction_ICCV_2021_paper.html) dataset.

<figure align="center"  style="margin-top: 2em; margin-bottom: 2em">
  <img src="assets/images/qualitative_experiments_front.png" width="700" />
  <figcaption>Figure 4. <b>Qualitative results.</b> We compare qualitatively InstantAvatar with other state of the art methods on H3DS dataset for 1 view, 3 views and 6 views. 1 view: SIRA is able to better capture the identity of the subject, however, it takes 10 min of training. DECA on the other hand, can only predict the face region. 3 views: H3D-Net achieves good bias but at a high variance where we can clearly see artifacts on the chin and the hair. 6 views: H3D-Net is able to recover the hair and face regions with similar quality as InstantAvatar.</figcaption>
</figure>

<figure align="center"  style="margin-top: 2em; margin-bottom: 2em">
  <img src="assets/images/qualitative_ablation_ICCV.png" width="700" />
  <figcaption>Figure 5. <b>Ablation: Qualitative comparison.</b> We conduct an ablation study to qualitatively compare variations of our model using a H3Ds dataset scene in the multi-view setting (6 views). The bottom row zooms into the face region to better appreciate the differences among configurations. Both our final approach and the one without normals supervision outperform the rest of alternatives. However, when normals supervision is not considered the resulting shape tends to be excessively sharp (e.g. the outermost part of the eyebrows) or erroneous (hair). The single grid and the 8-layer MLP (without grid) results are comparable, although they are both unable to capture the high-frequency details obtained with our final model.</figcaption>
</figure>

<figure align="center"  style="margin-top: 2em; margin-bottom: 2em">
  <img src="assets/images/celeb2.png" width="700" />
  <figcaption>Figure 6. <b>Qualitative results.</b> InstantAvatar results on CelebHQ dataset for a single input image.</figcaption>
</figure>

<figure align="center"  style="margin-top: 2em; margin-bottom: 2em">
  <img src="assets/images/1 view - 34f084f75deb512f.jpg" width="300" />
  <br>
  <img src="assets/images/1 view - 34f084f75deb512f.gif" width="300" />
  <figcaption>Figure 7. <b>3D reconstruction optimization process</b> on commodity hardware (input: <b>1 view</b>).</figcaption>
</figure>

<figure align="center"  style="margin-top: 2em; margin-bottom: 2em">
  <img src="assets/images/3 views - 5d0e87f3adb80226.jpg" width="300" />
  <br>
  <img src="assets/images/3 views - 5d0e87f3adb80226.gif" width="300" />
  <figcaption>Figure 8. <b>3D reconstruction optimization process</b> on commodity hardware (input: <b>3 views</b>).</figcaption>
</figure>

<figure align="center"  style="margin-top: 2em; margin-bottom: 2em">
  <img src="assets/images/6 views - 797a46725454b9c0.jpg" width="300" />
  <br>
  <img src="assets/images/6 views - 797a46725454b9c0.gif" width="300" />
  <figcaption>Figure 9. <b>3D reconstruction optimization process</b> on commodity hardware (input: <b>6 views</b>).</figcaption>
</figure>

## BibTeX

```latex
@InProceedings{canela2024instantavatar,
  title={InstantAvatar: Efficient 3D Head Reconstruction via Surface Rendering},
  author={Canela, Antonio and Caselles, Pol and Malik, Ibrar and Ramon, Eduard and Garcia, Jaime and Sanchez-Riera, Jordi and Triginer, Gil and Moreno-Noguer, Francesc},
  booktitle = {International Conference on 3D Vision (3DV)},
  year={2024}
}
```