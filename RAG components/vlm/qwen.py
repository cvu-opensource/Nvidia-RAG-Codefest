# API packages
from pydantic import BaseModel
from typing import List, Optional
from fastapi import FastAPI, File, UploadFile
from fastapi import FastAPI, HTTPException, UploadFile, File

# Qwen packages
from qwen_vl_utils import process_vision_info
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    
class QwenVLM:
    def __init__(self):
        # default: Load the model on the available device(s)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-7B-Instruct", torch_dtype="auto", device_map="auto", cache_dir="/qwen_cp"
        )

        self.processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen2-VL-7B-Instruct", cache_dir="/qwen_cp"
        )

    # The default range for the number of visual tokens per image in the model is 4-16384.
    # You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
    # min_pixels = 256*28*28
    # max_pixels = 1280*28*28
    # processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

    def invoke(self, prompt, image):
        """
        This tool is meant to be called to give further context on the user's prompt.
        
        
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {
                        "type": "text",
                        "text": "Describe the contents of this image."
                    },
                ],
            }
        ]

        # Preparation for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Inference: Generation of the output
        generated_ids = self.model.generate(**inputs, max_new_tokens=8196)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        return output_text
    
app = FastAPI()
qwen = QwenVLM()
    
# @app.post("/invoke")
# async def process_data(request):
#     json = request.dict()
#     try:
#         image_bytes = json['image'].read()
#         vlm_output = qwen.invoke(json['query'], image_bytes)
#         return {"status": "success", "vlm_response": vlm_output}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

@app.post("/invoke")
async def process_data(prompt: str, image: UploadFile = File(None)):
    # Process the prompt and the image (if provided)
    if image:
        image_content = await image.read()  # Read image content
        vlm_output = qwen.invoke(json['query'], image_content)
        return {"status": "success", "vlm_response": vlm_output}