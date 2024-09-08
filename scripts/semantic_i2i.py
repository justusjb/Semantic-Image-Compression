import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader, random_split
from scripts.dataset import Only_images_Flickr8kDataset
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
from transformers import pipeline
import time

from SSIM_PIL import compare_ssim

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
         num_images=100,
         batch_size=1,
         num_images_per_sample=2,
         outpath='',
         model=None,
         device=None,
         sampler=None,
         strength=0.8,
         ddim_steps=50,
         scale=9.0):
    """
    Modified testing function for the image-to-image model without noisy channel, using both text and embeddings.
    """
    # Initialize BLIP model for image captioning
    blip = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large")

    # Set up sampling parameters
    sampling_steps = int(strength * 50)
    print(f"Sampling steps: {sampling_steps}")
    sampler.make_schedule(ddim_num_steps=50, ddim_eta=0.0, verbose=False)

    # Create output directories
    sample_path = os.path.join(outpath, f"Test-samples-{sampling_steps}")
    os.makedirs(sample_path, exist_ok=True)
    text_path = os.path.join(outpath, f"Test-text-samples-{sampling_steps}")
    os.makedirs(text_path, exist_ok=True)
    sample_orig_path = os.path.join(outpath, f"Test-samples-orig-{sampling_steps}")
    os.makedirs(sample_orig_path, exist_ok=True)

    # Initialize LPIPS for perceptual similarity measurement
    lpips = lp.LPIPS(net='alex')
    lpips_values = []
    ssim_values = []
    time_values = []

    # Main testing loop
    tq = tqdm(dataloader, total=num_images)
    for batch in tq:
        img_file_path = batch[0]

        # Open and process the input image
        init_image = Image.open(img_file_path)
        prompt = blip(init_image)[0]["generated_text"]  # Generate caption using BLIP

        base_count = len(os.listdir(sample_path))

        # Prepare the input image for the model
        init_image = load_img(img_file_path).to(device)
        init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)
        init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))

        # Prepare data for the model
        data = [batch_size * [prompt]]
        t_enc = int(strength * ddim_steps)

        # Generate the output image
        with torch.no_grad():
            with autocast("cuda"):
                with model.ema_scope():
                    all_samples = list()
                    for n in range(1):
                        for prompts in data:
                            start_time = time.time()
                            uc = None
                            if scale != 1.0:
                                uc = model.get_learned_conditioning(batch_size * [""])
                            if isinstance(prompts, tuple):
                                prompts = list(prompts)
                            c = model.get_learned_conditioning(prompts)

                            # Use both text conditioning and image embedding
                            z_enc = sampler.stochastic_encode(init_latent,
                                                              torch.tensor([t_enc] * batch_size).to(device))
                            samples = sampler.decode(z_enc, c, t_enc, unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=uc, )
                            x_samples = model.decode_first_stage(samples)

                            x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                            end_time = time.time()
                            execution_time = end_time - start_time
                            time_values.append(execution_time)

                            # Save generated images, original images, and captions
                            for x_sample in x_samples:
                                x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                img = Image.fromarray(x_sample.astype(np.uint8))
                                img.save(os.path.join(sample_path, f"{base_count:05}.png"))

                                # Save the caption
                                with open(os.path.join(text_path, f"{base_count:05}.txt"), "a") as f:
                                    f.write(prompt)

                                # Save the original image
                                init_image_copy = Image.open(img_file_path)
                                init_image_copy = init_image_copy.resize((512, 512), resample=PIL.Image.LANCZOS)
                                init_image_copy.save(os.path.join(sample_orig_path, f"{base_count:05}.png"))

                                # Compute SSIM
                                ssim_values.append(compare_ssim(init_image_copy, img))
                                base_count += 1
                            all_samples.append(x_samples)

                    # Compute LPIPS
                    sample_out = (all_samples[0][0] * 2) - 1
                    lp_score = lpips(init_image[0].cpu(), sample_out.cpu()).item()

                    tq.set_postfix(lpips=lp_score)

                    if not np.isnan(lp_score):
                        lpips_values.append(lp_score)

        if len(lpips_values) == num_images:
            break

    # Print average scores
    print(f'Mean LPIPS score: {sum(lpips_values) / len(lpips_values)}')
    print(f'Mean SSIM score: {sum(ssim_values) / len(ssim_values)}')
    print(f'Mean execution time with {sampling_steps} sampling iterations: {sum(time_values) / len(time_values)}')
    return 1


if __name__ == "__main__":
    # Set up command-line arguments
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

    # Load model configuration and checkpoint
    config = OmegaConf.load(f"{config_path}")
    model = load_model_from_config(config, f"{model_ckpt_path}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    # Run test without noisy channel
    test(test_dataloader, num_images=100, batch_size=1, num_images_per_sample=1, outpath=outpath,
         model=model, device=device, sampler=sampler, strength=0.6, scale=9)
