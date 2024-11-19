import gradio as gr
from huggingface_hub import login
import requests

def validate_hf_token(token):
    """Validate Hugging Face token"""
    try:
        login(token=token, add_to_git_credential=False)
        return True
    except Exception:
        return False

def chat_response(message, history, token):
    """Generate chat response using Hugging Face API"""
    headers = {
        "Authorization": f"{token}",
        "Content-Type": "application/json"
    }
    payload = {
        "inputs": message,
        "parameters": {
            "max_new_tokens": 100,
            "temperature": 0.7
        }
    }
    
    try:
        response = requests.post(
            "https://api-inference.huggingface.co/models/microsoft/Orca-2-13b", 
            headers=headers, 
            json=payload
        )
        result = response.json()[0]['generated_text']
        return result
    except Exception as e:
        return f"Error: {str(e)}"

def generate_image(prompt, token):
    """Generate image from text prompt using Hugging Face API"""
    headers = {
        "Authorization": f"{token}",
        "Content-Type": "application/json"
    }
    payload = {
        "inputs": prompt
    }
    
    try:
        response = requests.post(
            "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0", 
            headers=headers, 
            json=payload
        )
        return response.content
    except Exception as e:
        return f"Error: {str(e)}"

def transcribe_audio(audio, token):
    """Transcribe audio file using Hugging Face API"""
    if audio is None:
        return "No audio file provided"
    
    headers = {
        "Authorization": f"{token}",
        "Content-Type": "audio/wav"
    }
    
    try:
        with open(audio, "rb") as f:
            audio_data = f.read()
        
        response = requests.post(
            "https://api-inference.huggingface.co/models/openai/whisper-large-v2", 
            headers=headers, 
            data=audio_data
        )
        result = response.json()['text']
        return result
    except Exception as e:
        return f"Error: {str(e)}"

def create_gradio_app():
    """Create the main Gradio application"""
    with gr.Blocks(theme=gr.themes.Soft()) as demo:

        # Token Authentication Section
        with gr.Row() as token_section:
            token_input = gr.Textbox(label="Enter Your Hugging Face Token", type="password")
            token_submit = gr.Button("Authenticate")
            token_status = gr.Textbox(label="Status", interactive=False)

        # Tabbed Interface (initially hidden)
        with gr.Tabs(visible=False) as tabs:
            # Chat Bot Tab
            with gr.TabItem("AI Chat-bot"):
                chatbot = gr.Chatbot()
                chat_input = gr.Textbox(label="Your Message")
                chat_submit = gr.Button("Send")
                chat_clear = gr.Button("Clear Chat")

            # Text-to-Image Tab
            with gr.TabItem("Text-to-Image"):
                image_prompt = gr.Textbox(label="Image Prompt")
                image_output = gr.Image(label="Generated Image")
                image_submit = gr.Button("Beep Boop Beep ..... Generate Image")
                image_clear = gr.Button("Clear")

            # Audio-to-Text Tab
            with gr.TabItem("Audio-to-Text"):
                audio_input = gr.Audio(sources=["microphone", "upload"], type="filepath")
                text_output = gr.Textbox(label="Transcribed Text")
                audio_submit = gr.Button("Transcribe")
                audio_clear = gr.Button("Clear")

        # Token Submission Event
        token_submit.click(
            fn=lambda token: "Token Validated Successfully" if token else "Invalid Token", 
            inputs=[token_input], 
            outputs=[token_status]
        ).then(
            fn=lambda: return [gr.update(visible=False), gr.update(visible=False)],
            inputs=None,
            outputs=[token_input, token_submit]
        ).then(
            fn=lambda: gr.update(visible=True),
            inputs=None,
            outputs=[tabs]
        )

        # Chat Bot Events
        chat_submit.click(
            fn=chat_response,
            inputs=[chat_input, chatbot, token_input],
            outputs=[chatbot, chat_input]
        )
        chat_clear.click(fn=lambda: None, inputs=None, outputs=[chatbot])

        # Image Generation Events
        image_submit.click(
            fn=generate_image,
            inputs=[image_prompt, token_input],
            outputs=[image_output]
        )
        image_clear.click(fn=lambda: None, inputs=None, outputs=[image_prompt, image_output])

        # Audio Transcription Events
        audio_submit.click(
            fn=transcribe_audio,
            inputs=[audio_input, token_input],
            outputs=[text_output]
        )
        audio_clear.click(fn=lambda: None, inputs=None, outputs=[audio_input, text_output])

    return demo
if __name__=="__main__":
  # Launch the Gradio app
  demo = create_gradio_app()
  demo.launch()
