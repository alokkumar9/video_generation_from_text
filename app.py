import torch
from diffusers import TextToVideoSDPipeline
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video
from PIL import Image
import gradio as gr

#pipe = DiffusionPipeline.from_pretrained("cerspense/zeroscope_v2_576w", torch_dtype=torch.float16)
pipe1 = DiffusionPipeline.from_pretrained("damo-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float16, variant="fp16")
pipe1.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

pipe1.enable_model_cpu_offload()
#pipe1.unet.enable_forward_chunking(chunk_size=1, dim=1)
pipe1.enable_vae_slicing()

def generate(description):

  video_description = description
  video_frames = pipe1(video_description, num_inference_steps=25).frames
  video_path = export_to_video(video_frames)
  return video_path

css='''
#title_head{
text-align: center;
text-weight: bold;
text-size:30px;
}
#name_head{
text-align: center;
}'''

with gr.Blocks(css=css) as front:
    with gr.Row():
      gr.Markdown("<h1>Video Generation from Text</h1>", elem_id='title_head')

    with gr.Row():
      description = gr.Text(
      label='video description',
      show_label=False,
      max_lines=1,
      placeholder='Enter short video description to generate',
      elem_id='text-input').style(container=False)
      btn = gr.Button('Generate video').style(full_width=False)

    with gr.Row():
      result = gr.Video(label='Result', show_label=False, elem_id='video')
    btn.click(generate,inputs=description,outputs=[result])




front.launch()