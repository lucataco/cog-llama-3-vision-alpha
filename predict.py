# Prediction interface for Cog ⚙️
# https://cog.run/python

from cog import BasePredictor, Input, Path, ConcatenateIterator
import os
import time
import torch
import subprocess
import torch.nn as nn
from PIL import Image
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    LlamaForCausalLM,
    SiglipImageProcessor,
    SiglipVisionModel,
)
from threading import Thread
from transformers.generation import TextIteratorStreamer

MODEL_CACHE = "./model-cache"
EMBED_CACHE = "./embed-cache"
BASE_MODEL = "unsloth/llama-3-8b-Instruct"
EMBED_MODEL = "google/siglip-so400m-patch14-384"
MODEL_URL = "https://weights.replicate.delivery/default/unsloth/llama-3-8b-Instruct/model.tar"
EMBED_URL = "https://weights.replicate.delivery/default/google/siglip-so400m-patch14-384/model.tar"

def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-x", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)

class ProjectionModule(nn.Module):
    def __init__(self, mm_hidden_size, hidden_size):
        super(ProjectionModule, self).__init__()
        # Directly set up the sequential model
        self.model = nn.Sequential(
            nn.Linear(mm_hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, x):
        return self.model(x)
    
class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        # Download model weights
        if not os.path.exists(MODEL_CACHE):
            download_weights(MODEL_URL, MODEL_CACHE)
        if not os.path.exists(EMBED_CACHE):
            download_weights(EMBED_URL, EMBED_CACHE)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL,
            use_fast=True,
            cache_dir=MODEL_CACHE
        )
        self.model = LlamaForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.float16,
            device_map="auto",
            quantization_config=bnb_config,
            cache_dir=MODEL_CACHE
        )
        for param in self.model.base_model.parameters():
            param.requires_grad = False

        vision_model = SiglipVisionModel.from_pretrained(
            EMBED_MODEL, torch_dtype=torch.float16
        )
        self.processor = SiglipImageProcessor.from_pretrained(
            EMBED_MODEL,
            cache_dir=EMBED_CACHE
        )
        self.vision_model = vision_model.to("cuda")

    def tokenizer_image_token(self, prompt, tokenizer, image_token_index=-200):
        prompt_chunks = prompt.split("<image>")
        tokenized_chunks = [tokenizer(chunk).input_ids for chunk in prompt_chunks]
        input_ids = tokenized_chunks[0]
        for chunk in tokenized_chunks[1:]:
            input_ids.append(image_token_index)
            input_ids.extend(chunk[1:])  # Exclude BOS token on nonzero index

        return torch.tensor(input_ids, dtype=torch.long)

    def process_tensors(self, input_ids, image_features, embedding_layer):
        # Find the index of -200 in input_ids
        split_index = (input_ids == -200).nonzero(as_tuple=True)[1][0]
        # Split the input_ids at the index found, excluding -200
        input_ids_1 = input_ids[:, :split_index]
        input_ids_2 = input_ids[:, split_index + 1 :]
        # Convert input_ids to embeddings
        embeddings_1 = embedding_layer(input_ids_1)
        embeddings_2 = embedding_layer(input_ids_2)
        device = image_features.device
        token_embeddings_part1 = embeddings_1.to(device)
        token_embeddings_part2 = embeddings_2.to(device)
        # Concatenate the token embeddings and image features
        concatenated_embeddings = torch.cat(
            [token_embeddings_part1, image_features, token_embeddings_part2], dim=1
        )
        # Create the corrected attention mask
        attention_mask = torch.ones(
            concatenated_embeddings.shape[:2], dtype=torch.long, device=device
        )
        return concatenated_embeddings, attention_mask

    def predict(
        self,
        image: Path = Input(description="Input image"),
        prompt: str = Input(description="Input prompt", default="Describe the image"),
    ) -> ConcatenateIterator[str]:
        """Run a single prediction on the model"""
        mm_hidden_size=1152
        hidden_size=4096
        projection_module = ProjectionModule(mm_hidden_size, hidden_size)
        checkpoint = torch.load("./mm_projector.bin")
        checkpoint = {k.replace("mm_projector.", ""): v for k, v in checkpoint.items()}
        projection_module.load_state_dict(checkpoint)
        projection_module = projection_module.to('cuda').half()

        img = Image.open(image).convert("RGB")
        self.tokenizer.eos_token = "<|eot_id|>"
        question = "<image>" + prompt
        prompt = f"<|start_header_id|>user<|end_header_id|>\n\n{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        input_ids = (
            self.tokenizer_image_token(prompt, self.tokenizer)
            .unsqueeze(0)
            .to('cuda')
        )
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        image_inputs = self.processor(
            images=[img],
            return_tensors="pt",
            do_resize=True,
            size={"height": 384, "width": 384},
        ).to("cuda")
        image_inputs = image_inputs["pixel_values"].squeeze(0)
        image_forward_outs = self.vision_model(
            image_inputs.to(device="cuda", dtype=torch.float16).unsqueeze(0),
            output_hidden_states=True,
        )
        image_features = image_forward_outs.hidden_states[-2]
        projected_embeddings = projection_module(image_features).to("cuda")
        embedding_layer = self.model.get_input_embeddings()
        new_embeds, attn_mask = self.process_tensors(input_ids, projected_embeddings, embedding_layer)
        attn_mask = attn_mask.to('cuda')
        new_embeds = new_embeds.to('cuda')
        model_kwargs = {
            "inputs_embeds": new_embeds,
            "max_new_tokens": 2000,
            "return_dict_in_generate": True,
            "temperature": 0.2,
            "streamer": streamer,
            "pad_token_id": self.tokenizer.eos_token_id,
            "do_sample": True,
        }
        thread = Thread(target=self.model.generate, kwargs=model_kwargs)
        thread.start()
        for _, new_text in enumerate(streamer):
            yield new_text
        thread.join()