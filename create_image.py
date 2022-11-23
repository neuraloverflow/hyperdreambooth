# copy from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py
import inspect
from typing import List, Optional, Union
from PIL import Image
import torch

from diffusers import LMSDiscreteScheduler


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def create_image(
    text_encoder, vae, unet, tokenizer, scheduler,
    prompt: Union[str, List[str]],
    device=torch.device("cuda"),
    height: Optional[int] = 512,
    width: Optional[int] = 512,
    num_inference_steps: Optional[int] = 50,
    guidance_scale: Optional[float] = 7.5,
    eta: Optional[float] = 0.0,
    generator: Optional[torch.Generator] = None,
    latents: Optional[torch.FloatTensor] = None,
    output_type: Optional[str] = "pil",
    return_dict: bool = True,
    # **kwargs,
):
    if isinstance(prompt, str):
        batch_size = 1
    elif isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

    if height % 8 != 0 or width % 8 != 0:
        raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

    # get prompt text embeddings
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids

    if text_input_ids.shape[-1] > tokenizer.model_max_length:
        removed_text = tokenizer.batch_decode(text_input_ids[:, tokenizer.model_max_length :])
        # logger.warning(
        #     "The following part of your input was truncated because CLIP can only handle sequences up to"
        #     f" {tokenizer.model_max_length} tokens: {removed_text}"
        # )
        text_input_ids = text_input_ids[:, : tokenizer.model_max_length]
    text_embeddings = text_encoder(text_input_ids.to(device))[0]

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    do_classifier_free_guidance = guidance_scale > 1.0
    # get unconditional embeddings for classifier free guidance
    if do_classifier_free_guidance:
        max_length = text_input_ids.shape[-1]
        uncond_input = tokenizer(
            [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]

        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

    # get the initial random noise unless the user supplied it

    # Unlike in other pipelines, latents need to be generated in the target device
    # for 1-to-1 results reproducibility with the CompVis implementation.
    # However this currently doesn't work in `mps`.
    latents_device = "cpu" if device.type == "mps" else device
    latents_shape = (batch_size, unet.in_channels, height // 8, width // 8)
    if latents is None:
        latents = torch.randn(
            latents_shape,
            generator=generator,
            device=latents_device,
            dtype=text_embeddings.dtype,
        )
    else:
        if latents.shape != latents_shape:
            raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {latents_shape}")
        latents = latents.to(latents_device)

    # set timesteps
    scheduler.set_timesteps(num_inference_steps)

    # Some schedulers like PNDM have timesteps as arrays
    # It's more optimzed to move all timesteps to correct device beforehand
    if torch.is_tensor(scheduler.timesteps):
        timesteps_tensor = scheduler.timesteps.to(device)
    else:
        timesteps_tensor = torch.tensor(scheduler.timesteps.copy(), device=device)

    # if we use LMSDiscreteScheduler, let's make sure latents are multiplied by sigmas
    if isinstance(scheduler, LMSDiscreteScheduler):
        latents = latents * scheduler.sigmas[0]

    # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
    # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
    # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
    # and should be between [0, 1]
    accepts_eta = "eta" in set(inspect.signature(scheduler.step).parameters.keys())
    extra_step_kwargs = {}
    if accepts_eta:
        extra_step_kwargs["eta"] = eta

    for i, t in enumerate(timesteps_tensor):
        # expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        if isinstance(scheduler, LMSDiscreteScheduler):
            sigma = scheduler.sigmas[i]
            # the model input needs to be scaled to match the continuous ODE formulation in K-LMS
            latent_model_input = latent_model_input / ((sigma**2 + 1) ** 0.5)

        # predict the noise residual
        noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

        # perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # compute the previous noisy sample x_t -> x_t-1
        if isinstance(scheduler, LMSDiscreteScheduler):
            latents = scheduler.step(noise_pred, i, latents, **extra_step_kwargs).prev_sample
        else:
            latents = scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

    # scale and decode the image latents with vae
    latents = 1 / 0.18215 * latents
    image = vae.decode(latents).sample

    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()

    image = numpy_to_pil(image)[0]
    return image
