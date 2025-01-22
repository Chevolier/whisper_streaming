# import gradio as gr
# from transformers import pipeline
# import numpy as np

# transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-base.en")

# def transcribe(audio):
#     sr, y = audio
    
#     # Convert to mono if stereo
#     if y.ndim > 1:
#         y = y.mean(axis=1)
        
#     y = y.astype(np.float32)
#     y /= np.max(np.abs(y))

#     return transcriber({"sampling_rate": sr, "raw": y})["text"]  

# demo = gr.Interface(
#     transcribe,
#     gr.Audio(sources="microphone"),
#     "text",
# )

# demo.launch(share=True)

import gradio as gr
import numpy as np
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset


device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = '/home/ec2-user/SageMaker/efs/Models/whisper-large-v3' # "openai/whisper-large-v3"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

transcriber = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)


def transcribe(stream, new_chunk):
    sr, y = new_chunk
    
    # Convert to mono if stereo
    if y.ndim > 1:
        y = y.mean(axis=1)
        
    y = y.astype(np.float32)
    y /= np.max(np.abs(y))

    if stream is not None:
        stream = np.concatenate([stream, y])
    else:
        stream = y
    return stream, transcriber({"sampling_rate": sr, "raw": stream})["text"]  

demo = gr.Interface(
    transcribe,
    ["state", gr.Audio(sources=["microphone", "upload"], streaming=True)],
    ["state", "text"],
    live=True,
)

demo.launch(share=True)
