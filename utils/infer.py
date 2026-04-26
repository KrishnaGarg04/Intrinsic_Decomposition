import torch
from diffusers import StableDiffusionImg2ImgPipeline, AutoencoderKL
from PIL import Image
import argparse

# -----------------------------
# ARGUMENTS
# -----------------------------

parser = argparse.ArgumentParser()

parser.add_argument("--image", type=str, required=True)
parser.add_argument("--prompt", type=str, required=True)

# slider value
# -1 = plastic
# 0 = original
# +1 = metallic
parser.add_argument("--strength", type=float, default=0.5)

args = parser.parse_args()


# -----------------------------
# PATHS
# -----------------------------

MODEL_ID = "runwayml/stable-diffusion-v1-5"
LORA_PATH = "/Volumes/T7 Shield/material_lora"


# -----------------------------
# DEVICE
# -----------------------------

device = "mps" if torch.backends.mps.is_available() else "cpu"

print("Using device:", device)


# -----------------------------
# LOAD PIPELINE
# -----------------------------

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32   # IMPORTANT for Mac MPS
)

pipe = pipe.to(device)


# -----------------------------
# BETTER VAE
# -----------------------------

pipe.vae = AutoencoderKL.from_pretrained(
    "stabilityai/sd-vae-ft-mse",
    torch_dtype=torch.float32
).to(device)


# -----------------------------
# LOAD LORA
# -----------------------------

pipe.load_lora_weights(
    LORA_PATH,
    weight_name="pytorch_lora_weights.safetensors"
)


# -----------------------------
# LOAD INPUT IMAGE
# -----------------------------

image = Image.open(args.image).convert("RGB")
image = image.resize((512, 512))


# -----------------------------
# SLIDER CONTROL
# -----------------------------

scale = max(-1.0, min(1.0, args.strength))

# LoRA influence
lora_scale = abs(scale) * 0.6


# -----------------------------
# PROMPT MODIFICATION
# -----------------------------

prompt = args.prompt

if scale > 0:
    prompt += ", shiny chrome metal surface, reflective metallic material"

elif scale < 0:
    prompt += ", matte plastic material, smooth plastic surface"


print("Prompt:", prompt)
print("LoRA scale:", lora_scale)


# -----------------------------
# GENERATE IMAGE
# -----------------------------

result = pipe(
    prompt=prompt,
    image=image,
    strength=0.25,
    num_inference_steps=20,
    guidance_scale=4.5,
    cross_attention_kwargs={"scale": 0.45}
).images[0]


# -----------------------------
# SAVE RESULT
# -----------------------------

result.save("edited.png")

print("\nSaved -> edited.png")