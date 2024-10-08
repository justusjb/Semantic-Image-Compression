import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader, random_split
from scripts.dataset import Flickr8kDataset,Only_images_Flickr8kDataset
from itertools import islice
from ldm.util import instantiate_from_config
from PIL import Image
import PIL
import torch
import numpy as np
import argparse, os
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from imwatermark import WatermarkEncoder
from ldm.models.diffusion.ddim import DDIMSampler
from tqdm import tqdm
import lpips as lp
from einops import rearrange, repeat
from torch import autocast
from tqdm import tqdm, trange
from transformers import pipeline, logging
from scripts.qam import qam16ModulationTensor, qam16ModulationString
import time

from SSIM_PIL import compare_ssim
from diffusers import FluxPipeline


'''

INIT DATASET AND DATALOADER

'''
capt_file_path  =  "path/to/captions.txt"          #"G:/Giordano/Flickr8kDataset/captions.txt"
images_dir_path =  "path/to/Images"                #"G:/Giordano/Flickr8kDataset/Images/"
batch_size      =  1

dataset = Only_images_Flickr8kDataset(images_dir_path)

test_dataloader=DataLoader(dataset=dataset,batch_size=batch_size, shuffle=True)


'''
MODEL CHECKPOINT

'''


model_ckpt_path = "path/to/model-checkpoint" #"G:/Giordano/stablediffusion/checkpoints/v1-5-pruned.ckpt"             #v2-1_512-ema-pruned.ckpt"        
config_path     = "path/to/model-config"     #"G:/Giordano/stablediffusion/configs/stable-diffusion/v1-inference.yaml"


def load_flux_model(model_path):
    print(f"Loading FLUX model from {model_path}")
    try:
        pipeline = FluxPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16)
        pipeline.enable_model_cpu_offload()
        return pipeline
    except Exception as e:
        print(f"Error loading FLUX model: {e}")
        print("Attempting to load from 'black-forest-labs/FLUX.1-schnell'...")
        pipeline = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
        pipeline.enable_model_cpu_offload()
        return pipeline


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def load_img(path):
    image = Image.open(path).convert("RGB")
    w, h = (512,512)#image.size
    #print(f"loaded input image of size ({w}, {h}) from {path}")
    w, h = map(lambda x: x - x % 64, (w, h))  # resize to integer multiple of 64
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2. * image - 1.


def test(dataloader,
         snr=10,
         num_images=100,
         batch_size=1,
         outpath='',
         model=None,
         device=None,
         strength=0.8,
         scale=9.0):
    blip = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large", device=device)
    i = 0

    sample_path = os.path.join(outpath, f"Test-samples-{snr}")
    os.makedirs(sample_path, exist_ok=True)

    text_path = os.path.join(outpath, f"Test-text-samples-{snr}")
    os.makedirs(text_path, exist_ok=True)

    sample_orig_path = os.path.join(outpath, f"Test-samples-orig-{snr}")
    os.makedirs(sample_orig_path, exist_ok=True)

    lpips = lp.LPIPS(net='alex').to(device)  # Move LPIPS to the specified device
    lpips_values = []
    ssim_values = []
    time_values = []

    tq = tqdm(dataloader, total=num_images)
    for batch in tq:
        img_file_path = batch[0]

        # Open Image
        init_image = Image.open(img_file_path).convert("RGB")
        init_image = init_image.resize((512, 512), resample=PIL.Image.LANCZOS)

        # Automatically extract caption using BLIP model
        prompt = blip(init_image, max_new_tokens=20)[0]["generated_text"]
        prompt_original = prompt

        base_count = len(os.listdir(sample_path))

        # Apply channel simulation to the prompt
        prompt = qam16ModulationString(prompt, snr_db=snr)

        start_time = time.time()

        # Generate image with FLUX
        with torch.no_grad():
            output = model(
                prompt,
                guidance_scale=scale,
                num_inference_steps=int(50 * strength),
                max_sequence_length=256,
                generator=torch.Generator("cpu").manual_seed(0)
            )

        generated_image = output.images[0]

        # Ensure generated image has the same size as the original
        generated_image = generated_image.resize((512, 512), resample=PIL.Image.LANCZOS)

        end_time = time.time()
        execution_time = end_time - start_time
        time_values.append(execution_time)

        # Save generated image
        generated_image.save(os.path.join(sample_path, f"{base_count:05}.png"))

        # Save text
        with open(os.path.join(text_path, f"{base_count:05}.txt"), "w") as f:
            f.write(prompt_original)

        # Save original image
        init_image.save(os.path.join(sample_orig_path, f"{base_count:05}.png"))

        # Compute SSIM
        try:
            ssim_value = compare_ssim(init_image, generated_image)
            ssim_values.append(ssim_value)
        except AttributeError:
            print(f"SSIM comparison failed for image {base_count}. Skipping.")

        # Compute LPIPS
        init_image_tensor = torch.from_numpy(np.array(init_image)).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1
        generated_image_tensor = torch.from_numpy(np.array(generated_image)).permute(2, 0, 1).unsqueeze(
            0).float() / 127.5 - 1

        init_image_tensor = init_image_tensor.to(device)
        generated_image_tensor = generated_image_tensor.to(device)

        with torch.no_grad():  # Add this to prevent unnecessary gradient computation
            lp_score = lpips(init_image_tensor, generated_image_tensor).item()

        #lp_score = lpips(init_image_tensor.to(device), generated_image_tensor.to(device)).item()

        tq.set_postfix(lpips=lp_score)

        if not np.isnan(lp_score):
            lpips_values.append(lp_score)

        i += 1
        if i == num_images:
            break

    print(f'mean lpips score at snr={snr} : {sum(lpips_values) / len(lpips_values)}')
    print(f'mean ssim score at snr={snr} : {sum(ssim_values) / len(ssim_values)}')
    print(f'mean time with sampling iterations {int(50 * strength)} : {sum(time_values) / len(time_values)}')
    return 1


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/img2img-samples"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )


    opt = parser.parse_args()
    seed_everything(opt.seed)

    config = OmegaConf.load(f"{config_path}")
    #model = load_model_from_config(config, f"{model_ckpt_path}")
    #model = load_flux_model(f"{model_ckpt_path}")  # or "stabilityai/flux" if using the pre-trained model

    device = torch.device("cuda") #if torch.cuda.is_available() else torch.device("cpu")
    #model = model.to(device)
    model = load_flux_model("LELELEL")

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    # INIZIO TEST
    for snr in [10, 8.75, 7.50, 6.25, 5]:
        test(test_dataloader, snr=snr, num_images=100, batch_size=1, outpath=outpath,
             model=model, device="cuda", strength=0.6, scale=9)

    #Strength is used to modulate the number of sampling steps. Steps=50*strength 

    
    