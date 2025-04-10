import gradio as gr
import edge_tts
import asyncio
import moviepy as mp
import torch
import math
import numpy as np
import PIL.Image
from typing import Union, List
from diffusers import CogVideoXPipeline, CogVideoXDPMScheduler
from diffusers.image_processor import VaeImageProcessor
from diffusers.utils import export_to_video 
import os, random, gc
import google.generativeai as genai
from dotenv import load_dotenv
import nest_asyncio
from huggingface_hub import login
import uuid
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

# Load environment variables
load_dotenv()

# Login to HuggingFace
huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
if huggingface_token:
    print("Logging in to HuggingFace...")
    login(token=huggingface_token)
else:
    print("HuggingFace token not found. Make sure it is set in the .env file.")

# Configure Gemini API
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY not found in .env file.")

print("Configuring Gemini API...")
genai.configure(api_key=gemini_api_key)
model = genai.GenerativeModel('gemini-pro')

print("Setting device...")
device = "cpu"

print("Loading CogVideoX pipeline...")
pipe = CogVideoXPipeline.from_pretrained(
    "THUDM/CogVideoX-2b",
   torch_dtype=torch.float32
).to(device)
print("Pipeline loaded.")

pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
print("Scheduler configured.")

pipe.enable_model_cpu_offload()
pipe.vae.enable_tiling()
pipe.vae.enable_slicing()
print("Pipeline configuration complete.")

# Create output directory if it doesn't exist
os.makedirs("outputs", exist_ok=True)

# Function to save video
def save_video(output_filename, tensor: Union[List[np.ndarray], List[PIL.Image.Image]], fps: int = 8):
    video_path = output_filename
    export_to_video(tensor, video_path, fps=fps)
    return video_path

# Function to generate text using Gemini
async def generate_text(prompt):
    print(f"Generating text with prompt: {prompt}")
    response = model.generate_content(prompt)
    content = response.text.strip()
    print(f"Response: {content}")
    return content

# Function to generate voice over using Gemini and edge-tts
async def generate_voice_over(user_text):
    print("Generating voice-over script...")
    voice_over_text = await generate_text(f"Please generate a 1-minute voice-over script using the provided text: {user_text}. Only provide the script.")
    voice = edge_tts.Communicate(voice_over_text, voice="en-US-AriaNeural")
    await voice.save("voice_over.mp3")
    print("Voice-over saved.")

# Function to generate descriptions and prompts
async def generate_descriptions_and_prompts(user_text, num_videos):
    descriptions = await generate_text(f"Generate {num_videos} short descriptions based on this text, only respond with the descriptions separated by a newline: {user_text}")
    prompts = await generate_text(f"Generate {num_videos} very detailed prompts based on the description: {descriptions}. Only respond with the prompts separated by a newline: ")
    prompts = prompts.replace('\n\n', '\n').strip()
    desc_list = descriptions.split('\n')
    prompt_list = prompts.split('\n')
    return {i: desc.strip() for i, desc in enumerate(desc_list)}, prompt_list

# Function to generate video using CogVideoX
def generate_video(prompt, output_filename):
    seed = random.randint(0, 2 ** 8 - 1)
    with torch.inference_mode():
        video_pt = pipe(
            prompt=prompt,
            num_videos_per_prompt=1,
            num_inference_steps=50,
            num_frames=49,
            guidance_scale=7,
            use_dynamic_cfg=True,
            output_type="pt",
            generator=torch.Generator(device="cuda").manual_seed(seed),
        ).frames

    batch_size = video_pt.shape[0]
    batch_video_frames = []
    for batch_idx in range(batch_size):
        pt_image = video_pt[batch_idx]
        pt_image = torch.stack([pt_image[i] for i in range(pt_image.shape[0])])
        image_np = VaeImageProcessor.pt_to_numpy(pt_image)
        image_pil = VaeImageProcessor.numpy_to_pil(image_np)
        batch_video_frames.append(image_pil)

    save_video(output_filename, batch_video_frames[0], fps=math.ceil((len(batch_video_frames[0]) - 1) / 6))

# Function to combine audio and videos
def combine_audio_and_videos(audio_file, video_files):
    audio = mp.AudioFileClip(audio_file)
    video_clips = [mp.VideoFileClip(video_file) for video_file in video_files]
    final_video = mp.concatenate_videoclips(video_clips)
    final_video = final_video.set_audio(audio)
    
    # Generate a unique filename
    unique_id = str(uuid.uuid4())
    output_filename = f"outputs/final_output_{unique_id}.mp4"
    final_video.write_videofile(output_filename)
    
    # Clean up
    audio.close()
    for clip in video_clips:
        clip.close()
    final_video.close()
    
    return output_filename

# Main function to process user input
async def process_input(user_text, num_videos):
    descriptions, prompts = await generate_descriptions_and_prompts(user_text, num_videos)
    
    await generate_voice_over(user_text)
    
    video_files = []
    for i, prompt in enumerate(prompts[:num_videos]):  # Limit to selected number of videos
        output_filename = f"video_{i}.mp4"
        generate_video(prompt, output_filename)
        video_files.append(output_filename)
    
    final_output = combine_audio_and_videos("voice_over.mp3", video_files)
    
    # Clean up temporary files
    try:
        os.remove("voice_over.mp3")
        for video_file in video_files:
            os.remove(video_file)
    except Exception as e:
        print(f"Error during cleanup: {e}")
    
    return final_output

# Create FastAPI app for video downloads
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/download/{video_name}")
async def download_video(video_name: str):
    video_path = f"outputs/{video_name}"
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Video not found")
    return FileResponse(video_path, media_type="video/mp4", filename=video_name)

# Create Gradio interface
iface = gr.Interface(
    fn=lambda user_text, num_videos: asyncio.run(process_input(user_text, num_videos)),
    inputs=[
        gr.Textbox(label="Enter your text"),
        gr.Slider(minimum=1, maximum=10, step=1, value=5, label="Number of videos to generate")
    ],
    outputs=gr.Video(label="Generated Video"),
    title="AI Video Generator",
    description="Generate a video based on your input text using AI",
    allow_flagging="never"
)

# Mount Gradio app to FastAPI
app = gr.mount_gradio_app(app, iface, path="/")

# Run the app
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
