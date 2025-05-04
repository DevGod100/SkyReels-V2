import argparse
import gc
import os
import random
import time

import imageio
import torch
from diffusers.utils import load_image

from skyreels_v2_infer.modules import download_model
from skyreels_v2_infer.pipelines import (
    Image2VideoPipeline,
    PromptEnhancer,
    resizecrop,
    Text2VideoPipeline,
)

MODEL_ID_CONFIG = {
    "text2video": [
        "Skywork/SkyReels-V2-T2V-14B-540P",
        "Skywork/SkyReels-V2-T2V-14B-720P",
    ],
    "image2video": [
        "Skywork/SkyReels-V2-I2V-1.3B-540P",
        "Skywork/SkyReels-V2-I2V-14B-540P",
        "Skywork/SkyReels-V2-I2V-14B-720P",
    ],
}


def main(
    image: str,
    prompt: str,
    length: int,
    outdir: str = "video_out",
    model_id: str = "Skywork/SkyReels-V2-I2V-14B-540P",
    resolution: str = "540P",
    num_frames: int = None,
    guidance_scale: float = 6.0,
    shift: float = 8.0,
    inference_steps: int = 30,
    use_usp: bool = False,
    offload: bool = False,
    fps: int = 24,
    seed: int = None,
    prompt_enhancer: bool = False,
    teacache: bool = False,
    teacache_thresh: float = 0.2,
    use_ret_steps: bool = False,
) -> str:
    # default num_frames based on length if not provided
    if num_frames is None:
        num_frames = length * fps // 5  # adjust this mapping as you like

    # Build args namespace
    args = argparse.Namespace(
        outdir=outdir,
        model_id=model_id,
        resolution=resolution,
        num_frames=num_frames,
        image=image,
        guidance_scale=guidance_scale,
        shift=shift,
        inference_steps=inference_steps,
        use_usp=use_usp,
        offload=offload,
        fps=fps,
        seed=seed,
        prompt=prompt,
        prompt_enhancer=prompt_enhancer,
        teacache=teacache,
        teacache_thresh=teacache_thresh,
        use_ret_steps=use_ret_steps,
    )

    # ==== start original logic ====
    args.model_id = download_model(args.model_id)
    if args.seed is None:
        random.seed(time.time())
        args.seed = int(random.randrange(4294967294))

    if args.resolution == "540P":
        height, width = 544, 960
    else:
        height, width = 720, 1280

    image_obj = load_image(args.image).convert("RGB") if args.image else None
    negative_prompt = (
        "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, "
        "images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, "
        "incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, "
        "misshapen limbs, fused fingers, still picture, messy background, three legs, many people in "
        "the background, walking backwards"
    )

    if image_obj is None:
        pipe = Text2VideoPipeline(
            model_path=args.model_id,
            dit_path=args.model_id,
            use_usp=args.use_usp,
            offload=args.offload,
        )
    else:
        pipe = Image2VideoPipeline(
            model_path=args.model_id,
            dit_path=args.model_id,
            use_usp=args.use_usp,
            offload=args.offload,
        )
        image_width, image_height = image_obj.size
        if image_height > image_width:
            height, width = width, height
        image_obj = resizecrop(image_obj, height, width)

    if args.prompt_enhancer and image_obj is None:
        enhancer = PromptEnhancer()
        prompt = enhancer(prompt)
        del enhancer
        gc.collect()
        torch.cuda.empty_cache()

    if args.teacache:
        pipe.transformer.initialize_teacache(
            enable_teacache=True,
            num_steps=args.inference_steps,
            teacache_thresh=args.teacache_thresh,
            use_ret_steps=args.use_ret_steps,
            ckpt_dir=args.model_id,
        )

    kwargs = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "num_frames": args.num_frames,
        "num_inference_steps": args.inference_steps,
        "guidance_scale": args.guidance_scale,
        "shift": args.shift,
        "generator": torch.Generator(device="cuda").manual_seed(args.seed),
        "height": height,
        "width": width,
    }
    if image_obj is not None:
        kwargs["image"] = image_obj

    save_dir = os.path.join("result", args.outdir)
    os.makedirs(save_dir, exist_ok=True)

    with torch.cuda.amp.autocast(dtype=pipe.transformer.dtype), torch.no_grad():
        video_frames = pipe(**kwargs)[0]

    current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    sanitized = prompt[:50].replace("/", "")
    video_filename = f"{sanitized}_{args.seed}_{current_time}.mp4"
    output_path = os.path.join(save_dir, video_filename)
    imageio.mimwrite(
        output_path,
        video_frames,
        fps=args.fps,
        quality=8,
        output_params=["-loglevel", "error"],
    )
    # ==== end original logic ====

    return output_path


if __name__ == "__main__":
    import sys

    img_path = sys.argv[1]
    prm = sys.argv[2]
    ln = int(sys.argv[3])
    result = main(image=img_path, prompt=prm, length=ln)
    print(result)
