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
import os
import random
import gc
import google.generativeai as genai
from dotenv import load_dotenv
import nest_asyncio
from huggingface_hub import login

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
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Setup Gemini model
model = genai.GenerativeModel('gemini-2.0-flash')

def save_video(output_filename, tensor: Union[List[np.ndarray], List[PIL.Image.Image]], fps: int = 8):
    video_path = output_filename
    export_to_video(tensor, video_path, fps=fps)
    return video_path

device = "cuda" if torch.cuda.is_available() else "cpu"

# Function to generate text using Gemini API
async def generate_text(prompt):
    try:
        print(f"Generating text with prompt: {prompt[:50]}...")
        response = await asyncio.to_thread(
            model.generate_content,
            prompt
        )
        return response.text.strip()
    except Exception as e:
        print(f"Error generating text: {e}")
        return f"Error generating text: {e}"

# Function to generate descriptions and prompts
async def generate_descriptions_and_prompts(user_text, num_videos):
    print(f"Generating {num_videos} descriptions and prompts...")
    
    # System prompt for descriptions
    descriptions_prompt = f"""
    Generate {num_videos} short descriptions based on this text:
    "{user_text}"
    
    Only provide the descriptions without any additional text. 
    Each description should be on a new line.
    Don't include numbers, bullet points, or any formatting.
    """
    
    descriptions = await generate_text(descriptions_prompt)
    
    # Clean up descriptions (remove any extra text or numbering)
    descriptions = descriptions.strip()
    desc_list = [line.strip() for line in descriptions.split('\n') if line.strip()]
    
    # Limit to the requested number
    desc_list = desc_list[:num_videos]
    
    # If we didn't get enough descriptions, add generic ones
    while len(desc_list) < num_videos:
        desc_list.append(f"Scene {len(desc_list) + 1} based on the theme")
    
    # System prompt for video generation prompts
    all_descriptions = "\n".join(desc_list)
    prompts_prompt = f"""
    For each of these descriptions, create a very detailed visual prompt that could be used for AI video generation.
    Make each prompt visually rich with details about lighting, style, mood, and scene elements.
    Return only the prompts, one per line:
    
    {all_descriptions}
    """
    
    prompts_text = await generate_text(prompts_prompt)
    
    # Clean up prompts
    prompts_text = prompts_text.strip()
    prompt_list = [line.strip() for line in prompts_text.split('\n') if line.strip()]
    
    # Limit to the requested number and ensure we have enough
    prompt_list = prompt_list[:num_videos]
    while len(prompt_list) < num_videos:
        prompt_list.append(f"Visual scene based on {user_text}")
    
    # Create a dict of descriptions and a list of prompts
    descriptions_dict = {i: desc for i, desc in enumerate(desc_list)}
    
    return descriptions_dict, prompt_list

# Function to generate voice over using Gemini and edge-tts
async def generate_voice_over(user_text):
    print("Generating voice-over script...")
    voice_over_prompt = f"""
    Create a conversational 1-minute voice-over script for a video about:
    "{user_text}"
    
    The script should flow naturally when read aloud and should tell a coherent story.
    Only return the script text, with no additional formatting or notes.
    """
    
    voice_over_text = await generate_text(voice_over_prompt)
    
    print("Creating voice-over audio...")
    voice = edge_tts.Communicate(voice_over_text)
    await voice.save("voice_over.mp3")
    print("Voice-over created successfully!")
    return voice_over_text

print(f"Loading CogVideoX model on {device}...")
pipe = CogVideoXPipeline.from_pretrained(
        "THUDM/CogVideoX-2b",
        torch_dtype=torch.float32
    ).to(device)
pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
pipe.enable_model_cpu_offload()
pipe.vae.enable_tiling()
pipe.vae.enable_slicing()
print("Model loaded successfully!")

# Function to generate video using CogVideoX
def generate_video(prompt, output_filename):
    print(f"Generating video for prompt: {prompt[:50]}...")
    seed = random.randint(0, 2 ** 8 - 1)
    try:
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
        print(f"Video saved to {output_filename}")
        return output_filename
    except Exception as e:
        print(f"Error generating video: {e}")
        return None
    
# Function to combine audio and videos
def combine_audio_and_videos(audio_file, video_files):
    print("Combining audio and videos...")
    # Filter out any None values from video files
    video_files = [f for f in video_files if f and os.path.exists(f)]
    
    if not video_files:
        print("No valid video files to combine!")
        return None
        
    try:
        audio = mp.AudioFileClip(audio_file)
        video_clips = [mp.VideoFileClip(video_file) for video_file in video_files]
        
        # Concatenate video clips
        final_video = mp.concatenate_videoclips(video_clips)
        
        # If audio is longer than video, loop the video
        if audio.duration > final_video.duration:
            final_video = final_video.loop(duration=audio.duration)
        # If video is longer than audio, trim the video
        elif final_video.duration > audio.duration:
            final_video = final_video.subclip(0, audio.duration)
            
        final_video = final_video.set_audio(audio)
        final_output = "final_output.mp4"
        final_video.write_videofile(final_output)
        print(f"Final video saved to {final_output}")
        return final_output
    except Exception as e:
        print(f"Error combining audio and videos: {e}")
        return None

# Main function to process user input
async def process_input(user_text, num_videos):
    print(f"Processing input with {num_videos} videos requested...")
    
    # Create output directories if they don't exist
    os.makedirs("output", exist_ok=True)
    
    descriptions, prompts = await generate_descriptions_and_prompts(user_text, num_videos)
    
    # Display generated descriptions and prompts
    print("\nGenerated Descriptions:")
    for i, desc in descriptions.items():
        if i < num_videos:
            print(f"{i+1}. {desc}")
    
    print("\nGenerated Prompts:")
    for i, prompt in enumerate(prompts[:num_videos]):
        print(f"{i+1}. {prompt}")
    
    # Generate voice-over
    voice_over_text = await generate_voice_over(user_text)
    print(f"Voice-over script: {voice_over_text[:100]}...")
    
    # Generate videos
    video_files = []
    for i, prompt in enumerate(prompts[:num_videos]):
        output_filename = f"video_{i}.mp4"
        video_file = generate_video(prompt, output_filename)
        video_files.append(video_file)
    
    # Combine audio and videos
    final_output = combine_audio_and_videos("voice_over.mp3", video_files)
    
    # If combining failed, return an error message
    if not final_output:
        return "Error: Failed to generate video. Please check the logs."
    
    # Clean up temporary files
    try:
        # Close all video file handles before removing
        for video_file in video_files:
            if video_file and os.path.exists(video_file):
                clip = mp.VideoFileClip(video_file)
                clip.close()
        
        # Close the audio file handle
        audio_clip = mp.AudioFileClip("voice_over.mp3")
        audio_clip.close()
        
        # Close the final output file handle
        if os.path.exists(final_output):
            final_clip = mp.VideoFileClip(final_output)
            final_clip.close()
        
        # Now attempt to remove the files
        print("Cleaning up temporary files...")
        if os.path.exists("voice_over.mp3"):
            os.remove("voice_over.mp3")
        for video_file in video_files:
            if video_file and os.path.exists(video_file):
                os.remove(video_file)
        print("Cleanup complete!")
    except Exception as e:
        print(f"Error during cleanup: {e}")
    
    return final_output

# Create Gradio interface
iface = gr.Interface(
    fn=lambda user_text, num_videos: asyncio.run(process_input(user_text, num_videos)),
    inputs=[
        gr.Textbox(
            label="Enter your text", 
            placeholder="Enter the text you want to convert into a video...",
            lines=5
        ),
        gr.Slider(
            minimum=1, 
            maximum=10, 
            step=1, 
            value=3, 
            label="Number of videos to generate"
        )
    ],
    outputs=gr.Video(label="Generated Video"),
    title="AI Video Generator",
    description="Generate a video based on your input text using AI. This tool uses Google's Gemini API for text generation and CogVideoX for video creation.",
    allow_flagging="never",  # This disables the flagging button
    examples=[
        ["A journey through the history of space exploration, from the first satellites to modern rovers on Mars.", 3],
        ["The importance of sustainable farming practices for the future of our planet.", 2],
        ["A tour of the most beautiful natural landscapes on Earth.", 4]
    ]
)

if __name__ == "__main__":
    print("Starting AI Video Generator...")
    iface.launch()