import torch
import tensorflow as tf
from diffusers import TextToVideoSDPipeline
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video
from PIL import Image
import spacy
import gradio as gr
nlp=spacy.load('en_core_web_sm')
import re

#loading diffusion model
pipe1 = DiffusionPipeline.from_pretrained("damo-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float16, variant="fp16")

pipe1.enable_model_cpu_offload()
pipe1.enable_vae_slicing()

def stopWords(str):
  doc=nlp(str)
  filtered_tokens = [token.text for token in doc if not token.is_stop]
  description = ""
  for word in filtered_tokens:
    description += word + " "
  return description.lstrip().rstrip()

def generate_video(description):
  description=re.sub('[^A-Za-z0-9 ]+','', description)
  pipe1.scheduler = DPMSolverMultistepScheduler.from_config(pipe1.scheduler.config)
  
  if description:
    video_description = stopWords(description)
    # num_frames can be changed to control length of video. If set higher than 60, GPU runtime may fail.
    video_frames = pipe1(video_description, num_inference_steps=25, num_frames=30).frames
    video_path = export_to_video(video_frames)
    
    pipe1.enable_vae_slicing()
        
    return video_path
  else:
    gr.Warning("Please enter video description first")


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
      description = gr.Textbox(
      label='video description',
      show_label=False,
      max_lines=1,
      placeholder='Enter short video description to generate',
      elem_id='text-input').style(container=False)
      btn = gr.Button('Generate video').style(full_width=False)

    with gr.Row():
      result = gr.Video(label='Result', show_label=False, elem_id='video', height=500, width=700)
    btn.click(generate_video,inputs=description,outputs=[result])

front.launch()