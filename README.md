
# RETHINED: A New Benchmark and Baseline for Real-Time High-Resolution Image Inpainting On Edge Devices (WACV 2025 Oral)

**[Paper](https://arxiv.org/abs/2503.14757) | [Project Page](https://crisalixsa.github.io/rethined/)**


Marcelo Sanchez, Gil Triginer, Ignacio Sarasua, Lara Raad, Coloma Ballester

---

### Abstract
Existing image inpainting methods have shown impressive completion results for low-resolution images. However, most of these algorithms fail at high resolutions and require powerful hardware, limiting their deployment on edge devices. Motivated by this, we propose the first baseline for REal-Time High-resolution image INpainting on Edge Devices (RETHINED) that is able to inpaint at ultra-high-resolution and can run in real-time (≤ 30ms) in a wide variety of mobile devices. A simple, yet effective novel method formed by a lightweight Convolutional Neural Network (CNN) to recover structure, followed by a resolution-agnostic patch replacement mechanism to provide detailed texture. Specially our pipeline leverages the structural capacity of CNN and the high-level detail of patch-based methods, which is a key component for high-resolution image inpainting. To demonstrate the real application of our method, we conduct an extensive analysis on various mobile-friendly devices and demonstrate similar inpainting performance while being 100 × faster than existing state-of-the-art methods. Furthemore, we realease DF8K-Inpainting, the first free-form mask UHD inpainting dataset.

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

### Download Test Masks
To download the test masks, you need to install `gdown` first:
```bash
pip install gdown
```
Then, run the provided script:
```bash
./bin/download_dataset.sh
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

## DF8K-Inpainting dataset
We release the DF8K-Inpainting dataset, a mix of DF2K and CAFHQ datasets, providing free-form inpainting masks only for the test set.

The CAFHQ dataset is available here: [repository](https://github.com/owenzlz/SuperCAF)

The DF2K dataset can be downloaded from [kaggle](https://www.kaggle.com/datasets/anvu1204/df2kdata)

The masks generated only for the test set can be found here: [Google Drive](https://drive.google.com/file/d/1BzTVrzZ5Z4rKPPp0K5SO5fRe01Fs4dw5/view?usp=sharing)

## Contact
If you have any questions, please feel free to reach me out at `marcelosanchezortega@gmail.com`.

## Acknowledgments

* We thank [advimman/lama](https://github.com/advimman/lama) for the mask generation algorithm.
* Segmentation code and models are from [CSAILVision](https://github.com/CSAILVision/semantic-segmentation-pytorch).
* LPIPS metric is from [richzhang](https://github.com/richzhang/PerceptualSimilarity)
* SSIM is from [Po-Hsun-Su](https://github.com/Po-Hsun-Su/pytorch-ssim)
* FID is from [mseitzer](https://github.com/mseitzer/pytorch-fid)
* CAFHQ dataset from [SuperCAF](https://github.com/owenzlz/SuperCAF)
