# Antigravity Skill: OpenEnv Development & Training Guide

[cite_start]This skill encompasses the fundamentals, setup, deployment, and training workflows for OpenEnv environments using Hugging Face (HF) Spaces, TRL, and Unsloth[cite: 1, 2, 3, 4, 5, 6].

---

## 1. OpenEnv Fundamentals & Architecture

[cite_start]Every HF Space acts as the infrastructure for OpenEnv environments, providing three core components[cite: 17, 18, 19]:

| Component | What it provides | How to access | Used as |
| :--- | :--- | :--- | :--- |
| **Server** | Running environment endpoint | [cite_start]`https://<username>-<space-name>.hf.space` [cite: 20] | [cite_start]Agent and Public API [cite: 20] |
| **Repository** | Installable Python package | [cite_start]`pip install git+https://huggingface.co/spaces/<username>-<space-name>` [cite: 20] | [cite_start]Code and client [cite: 20] |
| **Registry** | Docker container image | [cite_start]`docker pull registry.hf.space/<username>-<space-name>:latest` [cite: 20] | [cite_start]Deployment [cite: 20] |

### Server Endpoints
[cite_start]The server provides several protocols to interact with the environment[cite: 24, 25]:
* [cite_start]`/ws` (WebSocket): Persistent session (used by client)[cite: 25].
* [cite_start]`/health` (HTTP GET): Health check[cite: 25]. [cite_start]Example: `curl https://openenv-echo-env.hf.space/health` returns `{"status": "healthy"}`[cite: 35, 36, 37].
* [cite_start]`/reset` (HTTP POST): Reset environment (stateless)[cite: 25].
* [cite_start]`/step` (HTTP POST): Execute action (stateless)[cite: 25].
* [cite_start]`/state` (HTTP GET): Get current state[cite: 25].
* [cite_start]`/docs` (HTTP GET): OpenAPI documentation[cite: 25].
* [cite_start]`/web` (HTTP GET): Interactive web UI[cite: 25].

---

## 2. Environment Setup & Initialization

To create a new OpenEnv environment, follow these steps:

1.  **Install the core package:**
    ```bash
    uv pip install openenv-core
    ```
    *(Note: This requires a Python virtual environment managed by `uv` or `conda`)*[cite: 211, 215, 226].

2.  **Initialize the environment:**
    ```bash
    openenv init <your_env_name>
    cd <your_env_name>
    ```
    *This generates essential files including `app.py`, `Dockerfile`, `models.py`, `openenv.yaml`, and your core environment logic Python file.*[cite: 232, 233, 234, 249].

---

## 3. Local Development & Docker

[cite_start]You can develop and run OpenEnv locally using multiple approaches[cite: 140, 160].

### Cloning and Running Locally
* **Clone the space:**
    ```bash
    git clone [https://huggingface.co/spaces/](https://huggingface.co/spaces/)<username>/<space-name>
    cd <space-name>
    ```
    [cite_start][cite: 143, 144, 145].
* **Install in editable mode and run:**
    ```bash
    uv sync
    uv run server
    ```
    [cite_start][cite: 146, 147, 148, 149].

### Running Uvicorn Directly
[cite_start]For full control over the Uvicorn server[cite: 153, 154]:
* **With reload for development:**
    ```bash
    uvicorn benchmark.server.app:app --host 0.0.0.0 --port 8000 --reload
    ```
    [cite: 156, 157].
* **Multi-Worker mode:**
    ```bash
    uvicorn benchmark.server.app:app --host 0.0.0.0 --port 8000 --workers 4
    ```
    [cite: 158, 159].

### Docker Build & Run
* **Build the image using OpenEnv CLI:**
    ```bash
    openenv build -t <image-name>:latest
    ```
    [cite: 167, 168].
* **Run the container:**
    ```bash
    docker run -d -p 8000:8000 <image-name>:latest
    ```
    [cite: 172, 173].

---

## 4. Deploying to Hugging Face Spaces

[cite_start]The OpenEnv CLI simplifies deployment directly to HF Spaces[cite: 181, 185, 186].

* **Deploy to your namespace:**
    ```bash
    openenv push
    ```
    [cite_start][cite: 188].
* **Deploy to a specific repository:**
    ```bash
    openenv push --repo-id username/my-env
    ```
    [cite_start][cite: 189, 190].
* **Deploy as a private space:**
    ```bash
    openenv push --repo-id username/my-env --private
    ```
    [cite_start][cite: 191, 192].

---

## 5. Training Integration

[cite_start]OpenEnv integrates seamlessly with popular training libraries, ensuring the same methods (`reset()`, `step()`, `state()`, `close()`) work everywhere[cite: 560].

### Training with TRL (GRPO)
[cite_start]HuggingFace TRL integrates natively with OpenEnv for GRPO (Group Relative Policy Optimization) training without needing a critic network[cite: 579, 580, 612, 614].

1.  **Load the model:**
    ```python
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    ```
    [cite: 586, 587, 588].
2.  **Define the reward function:** Evaluate each completion in an isolated environment[cite: 589, 616].
    ```python
    def openenv_reward(completions, **kwargs):
        rewards = []
        for completion in completions:
            with HackEnv(base_url="...").sync() as env:
                env.reset()
                result = env.step(completion)
                rewards.append(result.reward)
        return rewards
    ```
    [cite: 590, 591, 592, 593, 594, 595, 596, 597, 598].
3.  **Train using GRPOTrainer:**
    ```python
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=openenv_reward,
        args=config,
        train_dataset=dataset,
    )
    trainer.train()
    ```
    [cite: 622, 624, 625, 626, 627, 628].

### Fast Training with Unsloth
[cite_start]Unsloth provides 2x faster training and 70% less memory usage via 4-bit quantization and LoRA[cite: 632, 653, 655, 657, 658, 661, 662].

1.  **Load model with Unsloth:**
    ```python
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen2.5-7B-Instruct",
        max_seq_length=2048,
        load_in_4bit=True
    )
    ```
    [cite: 638, 639, 640, 641, 642].
2.  **Apply LoRA adapters:**
    ```python
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    ```
    [cite: 644, 645, 646, 647, 648, 649, 650, 651].
3.  Use the exact same `openenv_reward` function and `GRPOTrainer` as standard TRL[cite: 667, 668].

---

## 6. Accessing Infrastructure (Hugging Face Jobs)

[cite_start]HF Jobs allows you to run your compute workloads with pay-as-you-go billing on any hardware[cite: 703, 705, 708]. 

* **View available hardware profiles:**
    ```bash
    hf jobs hardware
    ```
    [cite_start][cite: 761, 762]. [cite_start]*Note: A T4 GPU (`t4-small` or `t4-medium`) is generally recommended for cost-effective execution[cite: 722].*
* **Run a CLI job on specific hardware:**
    ```bash
    hf jobs uv run --flavor t4-small nvidia-smi
    ```
    [cite: 774].