# worker.py
from vastai import (
    Worker,
    WorkerConfig,
    HandlerConfig,
    BenchmarkConfig,
    LogActionConfig,
)

MODEL_SERVER_URL = "http://127.0.0.1"
MODEL_SERVER_PORT = 18000
MODEL_LOG_FILE = "/var/log/portal/flux.log"


def image_workload_calculator(payload: dict) -> float:
    """Workload basierend auf Bildgröße und Steps berechnen.
    
    Größeres Bild + mehr Steps = mehr GPU-Arbeit = höhere Workload.
    """
    size_str = payload.get("size", "1024x1024")
    try:
        w, h = map(int, size_str.split("x"))
    except (ValueError, AttributeError):
        w, h = 1024, 1024

    steps = payload.get("num_inference_steps", 4)
    # Normalisierte Workload: Pixel * Steps / Referenzwert
    return (w * h * steps) / (1024 * 1024)


worker_config = WorkerConfig(
    model_server_url=MODEL_SERVER_URL,
    model_server_port=MODEL_SERVER_PORT,
    model_log_file=MODEL_LOG_FILE,
    handlers=[
        HandlerConfig(
            route="/v1/images/generations",
            # Flux kann nur 1 Bild gleichzeitig auf der GPU generieren
            allow_parallel_requests=False,
            max_queue_time=120.0,       # Bildgenerierung dauert länger als Text
            workload_calculator=image_workload_calculator,
            benchmark_config=BenchmarkConfig(
                dataset=[
                    {
                        "prompt": "A beautiful sunset over mountains, photorealistic",
                        "size": "1024x1024",
                        "num_inference_steps": 4,
                        "guidance_scale": 0.0,
                    },
                ],
                runs=4,
                concurrency=1,  # Seriell, da GPU-bound
            ),
        ),
    ],
    log_action_config=LogActionConfig(
        on_load=[
            "Application startup complete.",
        ],
        on_error=[
            "Traceback (most recent call last):",
            "CUDA out of memory",
            "RuntimeError:",
            "torch.OutOfMemoryError",
        ],
        on_info=[
            "Loading model:",
            "Generating:",
        ],
    ),
)

Worker(worker_config).run()
