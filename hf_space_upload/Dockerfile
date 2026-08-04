FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /app

# Install git and system build tools
RUN apt-get update && apt-get install -y git build-essential && rm -rf /var/lib/apt/lists/*

# Qwen3.5 needs current Transformers, and PEFT must support Transformers v5.
RUN pip install --no-cache-dir \
    git+https://github.com/huggingface/transformers.git \
    "peft==0.19.1" \
    "accelerate>=1.0.0" \
    "datasets>=3.0.0" \
    matplotlib "pandas==3.0.5" "jinja2==3.1.6" scipy pyyaml jsonlines tqdm jupyter "gradio==6.20.0"

# Copy workspace
COPY . /app

EXPOSE 7860

# Run the live Gradio dashboard while the notebook experiment executes in the
# background. The dashboard exposes the Space health endpoint on port 7860.
CMD ["python3", "app.py"]
