from diffusers_helper.hf_login import login

import os
import argparse
import math
import traceback

import gradio as gr
import torch
import einops
import safetensors.torch as sf
import numpy as np
from PIL import Image

# Establece carpeta de descarga de Hugging Face
os.environ['HF_HOME'] = os.path.abspath(
    os.path.realpath(
        os.path.join(os.path.dirname(__file__), './hf_download')
    )
)

# Importaciones de transformers y diffusers
from diffusers import AutoencoderKLHunyuanVideo
from transformers import (
    LlamaModel,
    CLIPTextModel,
    LlamaTokenizerFast,
    CLIPTokenizer,
    SiglipImageProcessor,
    SiglipVisionModel
)
from diffusers_helper.hunyuan import encode_prompt_conds, vae_decode, vae_encode, vae_decode_fake
from diffusers_helper.utils import (
    save_bcthw_as_mp4,
    crop_or_pad_yield_mask,
    soft_append_bcthw,
    resize_and_center_crop,
    state_dict_weighted_merge,
    state_dict_offset_merge,
    generate_timestamp
)
from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
from diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
from diffusers_helper.memory import (
    cpu,
    gpu,
    move_model_to_device_with_memory_preservation,
    offload_model_from_device_for_memory_preservation,
    fake_diffusers_current_device,
    DynamicSwapInstaller,
    unload_complete_models,
    load_model_as_complete
)
from diffusers_helper.thread_utils import AsyncStream, async_run
from diffusers_helper.gradio.progress_bar import make_progress_bar_css, make_progress_bar_html
from diffusers_helper.clip_vision import hf_clip_vision_encode
from diffusers_helper.bucket_tools import find_nearest_bucket

# Argumentos de la CLI
parser = argparse.ArgumentParser()
parser.add_argument('--share', action='store_true')
parser.add_argument('--server', type=str, default='0.0.0.0')
parser.add_argument('--port', type=int, required=False)
parser.add_argument('--inbrowser', action='store_true')
args = parser.parse_args()

# Ejecutar siempre en CPU
high_vram = False
print(f'High-VRAM Mode: {high_vram} (Forzado por CPU)')

# Carga de modelos en CPU
tokenizer = LlamaTokenizerFast.from_pretrained(
    'hunyuanvideo-community/HunyuanVideo', subfolder='tokenizer'
)
tokenizer_2 = CLIPTokenizer.from_pretrained(
    'hunyuanvideo-community/HunyuanVideo', subfolder='tokenizer_2'
)

text_encoder = LlamaModel.from_pretrained(
    'hunyuanvideo-community/HunyuanVideo',
    subfolder='text_encoder',
    torch_dtype=torch.float16
).to(cpu)

text_encoder_2 = CLIPTextModel.from_pretrained(
    'hunyuanvideo-community/HunyuanVideo',
    subfolder='text_encoder_2',
    torch_dtype=torch.float16
).to(cpu)

vae = AutoencoderKLHunyuanVideo.from_pretrained(
    'hunyuanvideo-community/HunyuanVideo',
    subfolder='vae',
    torch_dtype=torch.float16
).to(cpu)

feature_extractor = SiglipImageProcessor.from_pretrained(
    'lllyasviel/flux_redux_bfl', subfolder='feature_extractor'
)
image_encoder = SiglipVisionModel.from_pretrained(
    'lllyasviel/flux_redux_bfl',
    subfolder='image_encoder',
    torch_dtype=torch.float16
).to(cpu)

transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained(
    'lllyasviel/FramePackI2V_HY',
    torch_dtype=torch.bfloat16
).to(cpu)

# Modo eval y sin gradientes
for model in [text_encoder, text_encoder_2, vae, image_encoder, transformer]:
    model.eval()
    model.requires_grad_(False)

# Optimización para CPU (secuencial y slicing)
if not high_vram:
    vae.enable_slicing()
    vae.enable_tiling()
    DynamicSwapInstaller.install_model(transformer, device=cpu)
    DynamicSwapInstaller.install_model(text_encoder, device=cpu)

# Salida de precisión alta en FP32 para inference
transformer.high_quality_fp32_output_for_inference = True

# Cola de streaming y carpeta de salidas
stream = AsyncStream()
outputs_folder = './outputs'
os.makedirs(outputs_folder, exist_ok=True)

# Función worker adaptada para CPU
@torch.no_grad()
def worker(
    input_image, prompt, n_prompt, seed,
    total_second_length, latent_window_size,
    steps, cfg, gs, rs,
    gpu_memory_preservation, use_teacache, mp4_crf
):
    total_latent_sections = max(round((total_second_length * 30) / (latent_window_size * 4)), 1)
    job_id = generate_timestamp()
    stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Starting ...'))))

    try:
        # Encoding de texto
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Text encoding ...'))))
        llama_vec, clip_l_pooler = encode_prompt_conds(
            prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2
        )
        if cfg == 1:
            llama_vec_n = torch.zeros_like(llama_vec)
            clip_l_pooler_n = torch.zeros_like(clip_l_pooler)
        else:
            llama_vec_n, clip_l_pooler_n = encode_prompt_conds(
                n_prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2
            )
        llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
        llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(llama_vec_n, length=512)

        # Procesamiento de imagen
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Image processing ...'))))
        H, W, C = input_image.shape
        height, width = find_nearest_bucket(H, W, resolution=640)
        input_image_np = resize_and_center_crop(
            input_image, target_width=width, target_height=height
        )
        Image.fromarray(input_image_np).save(
            os.path.join(outputs_folder, f'{job_id}.png')
        )
        input_image_pt = (
            torch.from_numpy(input_image_np).float() / 127.5 - 1
        ).permute(2, 0, 1)[None, :, None].to(cpu)

        # VAE encoding
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'VAE encoding ...'))))
        start_latent = vae_encode(input_image_pt, vae)

        # CLIP vision
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'CLIP Vision encoding ...'))))
        image_encoder_out = hf_clip_vision_encode(
            input_image_np, feature_extractor, image_encoder
        )
        image_encoder_last_hidden_state = image_encoder_out.last_hidden_state

        # Casting a tipos correctos
        llama_vec = llama_vec.to(transformer.dtype)
        llama_vec_n = llama_vec_n.to(transformer.dtype)
        clip_l_pooler = clip_l_pooler.to(transformer.dtype)
        clip_l_pooler_n = clip_l_pooler_n.to(transformer.dtype)
        image_encoder_last_hidden_state = image_encoder_last_hidden_state.to(transformer.dtype)

        # Muestreo
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Start sampling ...'))))
        rnd = torch.Generator("cpu").manual_seed(seed)
        num_frames = latent_window_size * 4 - 3

        history_latents = torch.zeros(
            (1, 16, 1 + 2 + 16, height // 8, width // 8), dtype=torch.float32
        ).cpu()
        history_pixels = None
        total_generated_latent_frames = 0

        for latent_padding in reversed(range(total_latent_sections)):
            latent_pad = latent_padding * latent_window_size
            indices = torch.arange(
                0, 1 + latent_pad + latent_window_size + 1 + 2 + 16
            ).unsqueeze(0)
            clean_pre, blank, latents_i, clean_post, lat2x, lat4x = indices.split([
                1, latent_pad, latent_window_size, 1, 2, 16
            ], dim=1)
            clean_indices = torch.cat([clean_pre, clean_post], dim=1)

            clean_pre_lat = start_latent.to(history_latents)
            post_lats, lats2x, lats4x = history_latents[:, :, :1 + 2 + 16].split([1, 2, 16], dim=2)
            clean_lats = torch.cat([clean_pre_lat, post_lats], dim=2)

            transformer.initialize_teacache(use_teacache, steps)

            def callback(data):
                prev = vae_decode_fake(data['denoised'])
                img = (prev * 255.0).detach().cpu().numpy().clip(0, 255).astype(np.uint8)
                img = einops.rearrange(img, 'b c t h w -> (b h) (t w) c')
                step = data['i'] + 1
                pct = int(100 * step / steps)
                desc = f"Generated frames: {total_generated_latent_frames * 4 - 3}"
                stream.output_queue.push(
                    ('progress', (img, desc, make_progress_bar_html(pct, f'Sampling {step}/{steps}')))
                )

            generated = sample_hunyuan(
                transformer=transformer,
                sampler='unipc',
                width=width,
                height=height,
                frames=num_frames,
                real_guidance_scale=cfg,
                distilled_guidance_scale=gs,
                guidance_rescale=rs,
                num_inference_steps=steps,
                generator=rnd,
                prompt_embeds=llama_vec,
                prompt_embeds_mask=llama_attention_mask,
                prompt_poolers=clip_l_pooler,
                negative_prompt_embeds=llama_vec_n,
                negative_prompt_embeds_mask=llama_attention_mask_n,
                negative_prompt_poolers=clip_l_pooler_n,
                device=cpu,
                dtype=torch.bfloat16,
                image_embeddings=image_encoder_last_hidden_state,
                latent_indices=latents_i,
                clean_latents=clean_lats,
                clean_latent_indices=clean_indices,
                clean_latents_2x=lats2x,
                clean_latent_2x_indices=lat2x,
                clean_latents_4x=lats4x,
                clean_latent_4x_indices=lat4x,
                callback=callback,
            )

            total_generated_latent_frames += generated.shape[2]
            history_latents = torch.cat([generated.to(history_latents), history_latents], dim=2)
            real_hist = history_latents[:, :, :total_generated_latent_frames, :, :]
            decoded = vae_decode(real_hist, vae).cpu()
            history_pixels = decoded if history_pixels is None else soft_append_bcthw(decoded, history_pixels, latent_window_size * 4 - 3)
            out_file = os.path.join(outputs_folder, f'{job_id}_{total_generated_latent_frames}.mp4')
            save_bcthw_as_mp4(history_pixels, out_file, fps=30, crf=mp4_crf)
            stream.output_queue.push(('file', out_file))

    except Exception:
        traceback.print_exc()

    stream.output_queue.push(('end', None))


def process(
    input_image, prompt, n_prompt, seed,
    total_second_length, latent_window_size,
    steps, cfg, gs, rs,
    gpu_memory_preservation, use_teacache, mp4_crf
):
    global stream
    yield None, None, '', '', gr.update(interactive=False), gr.update(interactive=True)
    stream = AsyncStream()
    async_run(
        worker, input_image, prompt, n_prompt, seed,
        total_second_length, latent_window_size,
        steps, cfg, gs, rs,
        gpu_memory_preservation, use_teacache, mp4_crf
    )

    while True:
        flag, data = stream.output_queue.next()
        if flag == 'file':
            yield data, gr.update(), gr.update(), gr.update(), gr.update(interactive=False), gr.update(interactive=True)
        elif flag == 'progress':
            img, desc, html = data
            yield gr.update(), gr.update(visible=True, value=img), desc, html, gr.update(interactive=False), gr.update(interactive=True)
        elif flag == 'end':
            yield None, gr.update(visible=False), gr.update(), '', gr.update(interactive=True), gr.update(interactive=False)
            break


def end_process():
    stream.input_queue.push('end')

# Construcción de la interfaz Gradio
css = make_progress_bar_css()
block = gr.Blocks(css=css).queue()
with block:
    gr.Markdown('# FramePack CPU')
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(sources='upload', type='numpy', label='Image', height=320)
            prompt = gr.Textbox(label='Prompt', value='')
            example_quick_prompts = gr.Dataset(
                samples=[['The girl dances gracefully, with clear movements, full of charm.']],
                label='Quick List', samples_per_page=1000, components=[prompt]
            )
            example_quick_prompts.click(
                lambda x: x[0], inputs=[example_quick_prompts], outputs=prompt,
                show_progress=False, queue=False
            )
            with gr.Row():
                start_button = gr.Button(value='Start Generation')
                end_button = gr.Button(value='End Generation', interactive=False)
            with gr.Group():
                use_teacache = gr.Checkbox(label='Use TeaCache', value=True)
                n_prompt = gr.Textbox(label='Negative Prompt', value='', visible=False)
                seed = gr.Number(label='Seed', value=31337, precision=0)
                total_second_length = gr.Slider(
                    label='Total Video Length (Seconds)', minimum=1, maximum=120,
                    value=5, step=0.1
                )
                latent_window_size = gr.Slider(
                    label='Latent Window Size', minimum=1, maximum=33,
                    value=9, step=1, visible=False
                )
                steps = gr.Slider(label='Steps', minimum=1, maximum=100, value=25, step=1)
                cfg = gr.Slider(label='CFG Scale', minimum=1.0, maximum=32.0, value=1.0, step=0.01, visible=False)
                gs = gr.Slider(label='Distilled CFG Scale', minimum=1.0, maximum=32.0, value=10.0, step=0.01)
                rs = gr.Slider(label='CFG Re-Scale', minimum=0.0, maximum=1.0, value=0.0, step=0.01, visible=False)
                gpu_memory_preservation = gr.Slider(label='Reserved Memory (GB)', minimum=6, maximum=128, value=6, step=0.1)
                mp4_crf = gr.Slider(label='MP4 Compression', minimum=0, maximum=100, value=16, step=1)
        with gr.Column():
            preview_image = gr.Image(label='Next Latents', height=200, visible=False)
            result_video = gr.Video(label='Finished Frames', autoplay=True, show_share_button=False, height=512, loop=True)
            progress_desc = gr.Markdown('', elem_classes='no-generating-animation')
            progress_bar = gr.HTML('', elem_classes='no-generating-animation')
    start_button.click(
        fn=process,
        inputs=[input_image, prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, mp4_crf],
        outputs=[result_video, preview_image, progress_desc, progress_bar, start_button, end_button]
    )
    end_button.click(fn=end_process)

block.launch(
    server_name=args.server,
    server_port=args.port,
    share=args.share,
    inbrowser=args.inbrowser
)
