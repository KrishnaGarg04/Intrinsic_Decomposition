# Prompt-Guided Material Editing using Intrinsic Decomposition and LoRA Diffusion

A hybrid computer vision pipeline for transforming the material appearance of objects in images while preserving geometry and scene structure.

The project combines:

- Automatic object segmentation
- Intrinsic image decomposition
- Frequency-domain material manipulation
- LoRA-enhanced Stable Diffusion refinement
- Interactive Streamlit dashboard

The system allows users to upload an image and continuously transform object surfaces between **plastic** and **metallic** appearances using a single material slider.

---

## Demo

### Dashboard Interface

![Dashboard](images/dashboard.png)

### Example Material Transformation

| Original | Metallic Edit |
|----------|---------------|
| ![](images/original.png) | ![](images/metallic.png) |

### Plastic Transformation

| Original | Plastic Edit |
|----------|---------------|
| ![](images/original2.png) | ![](images/plastic.png) |

---

## Project Motivation

Material editing is a challenging computer vision task because appearance changes must occur without modifying object geometry.

Traditional diffusion models often struggle to preserve structure during image editing.

This project explores a hybrid approach where:

1. Computer vision techniques perform the primary material transformation.
2. Diffusion models contribute only lightweight visual refinement.
3. Object geometry and scene context remain unchanged.

---

## System Pipeline

```text
Input Image
      │
      ▼
Automatic Object Segmentation
      │
      ▼
Intrinsic Image Decomposition
(Albedo / Shading / Specularity)
      │
      ▼
Frequency Band Extraction
      │
      ▼
Material Editing
(Metal ↔ Plastic)
      │
      ▼
Image Reconstruction
      │
      ▼
Optional LoRA Diffusion Refinement
      │
      ▼
Final Output
```

---

## Features

### Automatic Segmentation

The uploaded image is automatically segmented to isolate the foreground object.

Material edits are applied only to the detected object region while preserving the background.

### Intrinsic Editing

The image is decomposed into:

- Albedo
- Shading
- Specularity

Material transformations are applied directly to these components.

### Frequency-Based Material Modeling

The system manipulates:

- High-frequency reflections
- Specular highlights
- Surface roughness
- Reflectance characteristics

to simulate metallic and plastic materials.

### LoRA-Guided Refinement

A Stable Diffusion LoRA model trained on synthetic material transformations can optionally provide visual refinement and texture enhancement.

---

## Repository Structure

```text
.
├── app.py
├── utils
│   └── inference.py
├── images
│   ├── dashboard.png
│   ├── original.png
│   ├── metallic.png
│   └── plastic.png
├── requirements.txt
└── README.md
```

---

## Trained LoRA Model

The repository does **not include the trained LoRA weights**.

The LoRA model was trained separately using:

- Stable Diffusion v1.5
- LoRA Rank = 8
- ~60,000 synthetic material transformation samples
- 5,000 optimization steps

The final weights file:

```text
pytorch_lora_weights.safetensors
```

has been omitted from this repository due to storage limitations.

To use diffusion refinement, place the trained LoRA weights inside:

```text
material_lora/
```

and update the model path inside the inference code if required.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/<your-username>/<repo-name>.git

cd <repo-name>
```

Create environment:

```bash
python -m venv venv

source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Dashboard

Launch the Streamlit application:

```bash
streamlit run app.py
```

Open:

```text
http://localhost:8501
```

in your browser.

---

## Training Configuration

The LoRA model was trained using:

| Parameter | Value |
|------------|--------|
| Base Model | Stable Diffusion 1.5 |
| LoRA Rank | 8 |
| Resolution | 512 × 512 |
| Batch Size | 1 |
| Gradient Accumulation | 4 |
| Learning Rate | 1e-4 |
| Training Steps | 5000 |
| Hardware | Apple Silicon (MPS) |

---

## Experimental Observations

The project revealed that diffusion models alone struggle to perform controlled material editing.

Observed limitations included:

- Geometry distortion at high img2img strength
- Weak material changes at low strength
- Difficulty learning high-frequency material cues
- Entanglement of geometry, lighting, and material properties

The final hybrid pipeline achieved significantly better geometry preservation while maintaining realistic material transformations.

---

## Future Work

Potential improvements include:

- Larger real-world material datasets
- Neural intrinsic decomposition networks
- SAM2-based segmentation
- Additional material categories (glass, fabric, wood)
- Quantitative realism metrics
- Multi-view material editing

---

## References

- Ho et al. — Denoising Diffusion Probabilistic Models
- Rombach et al. — Stable Diffusion
- Hu et al. — LoRA: Low-Rank Adaptation
- Kirillov et al. — Segment Anything
- Bell et al. — Intrinsic Images in the Wild
