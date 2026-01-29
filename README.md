
# RETHINED: A New Benchmark and Baseline for Real-Time High-Resolution Image Inpainting On Edge Devices (WACV 2025 Oral)

**[Paper](https://arxiv.org/abs/2503.14757) | [Project Page](https://crisalixsa.github.io/rethined/)**


Marcelo Sanchez, Gil Triginer, Ignacio Sarasua, Lara Raad, Coloma Ballester

---

### Abstract
Existing image inpainting methods have shown impressive completion results for low-resolution images. However, most of these algorithms fail at high resolutions and require powerful hardware, limiting their deployment on edge devices. Motivated by this, we propose the first baseline for REal-Time High-resolution image INpainting on Edge Devices (RETHINED) that is able to inpaint at ultra-high-resolution and can run in real-time (≤ 30ms) in a wide variety of mobile devices. A simple, yet effective novel method formed by a lightweight Convolutional Neural Network (CNN) to recover structure, followed by a resolution-agnostic patch replacement mechanism to provide detailed texture. Specially our pipeline leverages the structural capacity of CNN and the high-level detail of patch-based methods, which is a key component for high-resolution image inpainting. To demonstrate the real application of our method, we conduct an extensive analysis on various mobile-friendly devices and demonstrate similar inpainting performance while being 100x faster than existing state-of-the-art methods. Furthemore, we realease DF8K-Inpainting, the first free-form mask UHD inpainting dataset.

---

### Dependencies and Installation

- **PyTorch**
- Other required packages are listed in `pyproject.toml`.

To install the project and its dependencies:

```bash
# 1. Clone this repository
# git clone ...
cd rethined

# 2. Create a virtual environment
uv venv
source .venv/bin/activate

# 3. Install the package in editable mode
pip install -e .
```

---

### Citation

If you find this work useful for your research, please consider citing:

```bibtex
@INPROCEEDINGS{10943510,
  author={Sanchez, Marcelo and Triginer, Gil and Ballester, Coloma and Sarasua, Ignacio and Raad, Lara},
  booktitle={2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)}, 
  title={A New Benchmark and Baseline for Real-Time High-Resolution Image Inpainting on Edge Devices}, 
  year={2025},
  volume={},
  number={},
  pages={1133-1143},
  keywords={Performance evaluation;Computer vision;Image resolution;Limiting;Image edge detection;Pipelines;Real-time systems;Mobile handsets;Hardware;Convolutional neural networks;computer-vision;inpainting;real-time;edge-devices;fast-infernce},
  doi={10.1109/WACV61041.2025.00118}
}
```

---

### Contact
If you have any questions, please feel free to reach me out at `marcelosanchezortega@gmail.com`.

## Acknowledgments

* We thank [advimman/lama](https://github.com/advimman/lama) for the mask generation algorithm.
* Segmentation code and models if form [CSAILVision](https://github.com/CSAILVision/semantic-segmentation-pytorch).
* LPIPS metric is from [richzhang](https://github.com/richzhang/PerceptualSimilarity)
* SSIM is from [Po-Hsun-Su](https://github.com/Po-Hsun-Su/pytorch-ssim)
* FID is from [mseitzer](https://github.com/mseitzer/pytorch-fid)
