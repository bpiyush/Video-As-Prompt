import torch
from diffusers import (
    AutoencoderKLCogVideoX,
    CogVideoXImageToVideoMOTPipeline,
    CogVideoXTransformer3DMOTModel,
)
from diffusers.utils import export_to_video, load_video
from PIL import Image

model_dir = "/work/piyush/pretrained_checkpoints/VideoAsPrompt/"
vae = AutoencoderKLCogVideoX.from_pretrained(f"{model_dir}/Video-As-Prompt-CogVideoX-5B", subfolder="vae", torch_dtype=torch.bfloat16)
transformer = CogVideoXTransformer3DMOTModel.from_pretrained(f"{model_dir}/Video-As-Prompt-CogVideoX-5B", torch_dtype=torch.bfloat16, subfolder="transformer")
pipe = CogVideoXImageToVideoMOTPipeline.from_pretrained(
    f"{model_dir}/Video-As-Prompt-CogVideoX-5B", vae=vae, transformer=transformer, torch_dtype=torch.bfloat16,
)
# Memory optimization: enable sequential CPU offload for layer-by-layer offloading
# This reduces memory from ~40GB to max around 7.5GB
# Alternative: use pipe.enable_model_cpu_offload() for module-level offloading (~30GB)
pipe.enable_sequential_cpu_offload()


n_frames = 49
height = 480
width = 720

ref_video = load_video("assets/videos/demo/object-725.mp4")
image = Image.open("assets/images/demo/animal-2.jpg").convert("RGB")
idx = torch.linspace(0, len(ref_video) - 1, n_frames).long().tolist()
ref_frames = [ref_video[i] for i in idx]

output_frames = pipe(
    image=image,
    ref_videos=[ref_frames],
    prompt="A chestnut-colored horse stands on a grassy hill against a backdrop of distant, snow-dusted mountains. The horse begins to inflate, its defined, muscular body swelling and rounding into a smooth, balloon-like form while retaining its rich, brown hide color. Without changing its orientation, the now-buoyant horse lifts silently from the ground. It begins a steady vertical ascent, rising straight up and eventually floating out of the top of the frame. The camera remains completely static throughout the entire sequence, holding a fixed shot on the landscape as the horse transforms and departs, ensuring the verdant hill and mountain range in the background stay perfectly still.",
    prompt_mot_ref=[
      "A hand holds up a single beige sneaker decorated with gold calligraphy and floral illustrations, with small green plants tucked inside. The sneaker immediately begins to inflate like a balloon, its shape distorting as the decorative details stretch and warp across the expanding surface. It rapidly transforms into a perfectly smooth, matte beige sphere, inheriting the primary color from the original shoe. Once the transformation is complete, the new balloon-like object quickly ascends, moving straight up and exiting the top of the frame. The camera remains completely static and the plain white background is unchanged throughout the entire sequence."
    ],
    height=height,
    width=width,
    num_frames=n_frames,
    frames_selection="evenly",
    use_dynamic_cfg=True,
).frames[0]
export_to_video(output_frames, "cog_demo.mp4", fps=10)