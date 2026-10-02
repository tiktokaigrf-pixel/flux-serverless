# server.py
import base64
import io
import logging
import os
import time
import uuid

import torch
import uvicorn
from diffusers import FluxPipeline
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Flux Image Generation API")

# --- Globale Pipeline (wird einmal beim Startup geladen) ---
pipe: FluxPipeline | None = None


class ImageRequest(BaseModel):
    prompt: str
    size: str = "1024x1024"          # Format: "WxH"
    n: int = 1                        # Anzahl Bilder (erstmal nur 1)
    num_inference_steps: int = 4      # Flux-schnell: 1-4 Steps reichen
    guidance_scale: float = 0.0       # Flux-schnell: 0.0 empfohlen
    seed: int | None = None


class ImageData(BaseModel):
    b64_json: str


class ImageResponse(BaseModel):
    created: int
    data: list[ImageData]


@app.on_event("startup")
async def load_model():
    """Flux-Pipeline beim Server-Start laden."""
    global pipe
    model_name = os.environ.get("MODEL_NAME", "black-forest-labs/FLUX.1-schnell")
    logger.info(f"Loading model: {model_name}")

    pipe = FluxPipeline.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
    )
    pipe.to("cuda")

    # Speicher-Optimierungen
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    logger.info("Application startup complete.")  # ← PyWorker on_load Trigger!


@app.post("/v1/images/generations", response_model=ImageResponse)
async def generate_image(request: ImageRequest):
    if pipe is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    # Größe parsen
    try:
        width, height = map(int, request.size.split("x"))
    except ValueError:
        raise HTTPException(status_code=400, detail="size must be 'WxH', e.g. '1024x1024'")

    # Seed setzen
    generator = None
    if request.seed is not None:
        generator = torch.Generator("cuda").manual_seed(request.seed)

    logger.info(f"Generating: prompt='{request.prompt[:80]}...', size={width}x{height}, steps={request.num_inference_steps}")

    # Flux Inference
    result = pipe(
        prompt=request.prompt,
        width=width,
        height=height,
        num_inference_steps=request.num_inference_steps,
        guidance_scale=request.guidance_scale,
        max_sequence_length=256,
        generator=generator,
    )

    image = result.images[0]

    # PIL Image → Base64
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    b64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return ImageResponse(
        created=int(time.time()),
        data=[ImageData(b64_json=b64_image)],
    )


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": pipe is not None}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=18000)
