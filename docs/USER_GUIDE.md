# User Guide

This comprehensive guide covers all training methods supported by the Model Training Manager, including how to use trained models with Ollama and LM Studio.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Training Methods](#training-methods)
   - [QLoRA Training](#qlora-training)
   - [Unsloth Training](#unsloth-training)
   - [RAG Training](#rag-training)
   - [Standard Fine-tuning](#standard-fine-tuning)
3. [Model Output & Deployment](#model-output--deployment)
4. [API Integration](#api-integration)

## Quick Start

1. **Upload Dataset**: Upload your training data in CSV or JSON format via the Datasets page
2. **Create Training Job**: Configure training parameters and select your preferred training method
3. **Download & Deploy**: Download your trained model and deploy with Ollama or LM Studio

## Training Methods

### QLoRA Training

#### Overview

QLoRA (Quantized Low-Rank Adaptation) is a memory-efficient fine-tuning technique that enables training of large language models on consumer hardware. It uses 4-bit quantization to reduce memory usage while maintaining model quality through low-rank adaptation.

**Key Benefits:**
- Up to 65% memory reduction compared to full fine-tuning
- Produces small adapter files (typically 10-100MB)
- Compatible with base models in Ollama and LM Studio
- Ideal for limited GPU memory scenarios (8-16GB VRAM)

#### Prerequisites & Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install core dependencies
pip install torch>=2.0.0
pip install transformers>=4.46.0
pip install peft>=0.13.0
pip install bitsandbytes>=0.44.0
pip install accelerate>=1.0.0
pip install datasets>=3.0.0

# Verify CUDA is available (recommended)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

#### Using with Ollama

To use your QLoRA adapter with Ollama, create a Modelfile that references your base model and applies the adapter weights:

```bash
# Navigate to your model directory
cd ./my_model/qlora_model

# Create a Modelfile
cat > Modelfile << 'EOF'
FROM llama3.2:3b
ADAPTER ./adapter_model.safetensors
PARAMETER temperature 0.7
PARAMETER top_p 0.9
SYSTEM "You are a helpful assistant trained with custom data."
EOF

# Create the model in Ollama
ollama create my-qlora-model -f Modelfile

# Verify the model was created
ollama list

# Test the model
ollama run my-qlora-model "Hello, how are you?"
```

**Python 3 Example:**

```python
import requests

# Using the Ollama API
response = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "my-qlora-model",
        "prompt": "Your prompt here",
        "stream": False
    }
)
result = response.json()
print(result["response"])

# Or using the ollama Python library
# pip install ollama
import ollama

response = ollama.generate(
    model="my-qlora-model",
    prompt="Your prompt here"
)
print(response["response"])
```

**PHP 8 Example:**

```php
<?php
// Using cURL
$ch = curl_init();
curl_setopt_array($ch, [
    CURLOPT_URL => "http://localhost:11434/api/generate",
    CURLOPT_POST => true,
    CURLOPT_RETURNTRANSFER => true,
    CURLOPT_HTTPHEADER => ["Content-Type: application/json"],
    CURLOPT_POSTFIELDS => json_encode([
        "model" => "my-qlora-model",
        "prompt" => "Your prompt here",
        "stream" => false
    ])
]);

$response = curl_exec($ch);
curl_close($ch);

$result = json_decode($response, true);
echo $result["response"];
```

**CURL Example:**

```bash
# Generate a response
curl -X POST http://localhost:11434/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "my-qlora-model",
    "prompt": "Your prompt here",
    "stream": false
  }'

# Chat completion
curl -X POST http://localhost:11434/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "my-qlora-model",
    "messages": [
      {"role": "user", "content": "Your message here"}
    ],
    "stream": false
  }'
```

#### Using with LM Studio

LM Studio supports loading LoRA adapters alongside base models:

```bash
# Find LM Studio models directory
# macOS: ~/.cache/lm-studio/models/
# Windows: C:\Users\<user>\.cache\lm-studio\models\
# Linux: ~/.cache/lm-studio/models/

# Copy your adapter to a subdirectory
mkdir -p ~/.cache/lm-studio/models/my-qlora-adapter
cp -r ./my_model/qlora_model/* ~/.cache/lm-studio/models/my-qlora-adapter/

# Restart LM Studio and load the adapter from the UI
# Select your base model, then apply the LoRA adapter
```

**Python 3 Example (LM Studio API):**

```python
import requests

# LM Studio uses OpenAI-compatible API
response = requests.post(
    "http://localhost:1234/v1/chat/completions",
    headers={"Content-Type": "application/json"},
    json={
        "model": "local-model",
        "messages": [
            {"role": "user", "content": "Your prompt here"}
        ],
        "temperature": 0.7,
        "max_tokens": 500
    }
)

result = response.json()
print(result["choices"][0]["message"]["content"])

# Using OpenAI Python library
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="not-needed"
)

response = client.chat.completions.create(
    model="local-model",
    messages=[{"role": "user", "content": "Your prompt here"}],
    temperature=0.7
)
print(response.choices[0].message.content)
```

#### Recommended Parameters

| Parameter | Recommended Value | Description |
|-----------|------------------|-------------|
| LoRA Rank (r) | 16-64 | Higher = more capacity |
| LoRA Alpha | 32-128 | Typically 2x rank |
| Dropout | 0.05-0.1 | Regularization |
| Batch Size | 4-16 | Depends on VRAM |
| Learning Rate | 1e-4 to 3e-4 | Training speed |
| Epochs | 1-5 | Monitor for overfitting |

---

### Unsloth Training

#### Overview

Unsloth is an optimized training framework that provides 2x faster LoRA fine-tuning with reduced memory usage. It uses custom CUDA kernels and optimized attention implementations.

**Key Benefits:**
- 2x faster training compared to standard PEFT
- 30% less memory usage
- Compatible with 4-bit and 8-bit quantization
- Supports most popular LLM architectures (Llama, Mistral, etc.)

#### Prerequisites & Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# For CUDA 12.1+ (recommended)
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps trl peft accelerate bitsandbytes

# For CUDA 11.8
pip install "unsloth[cu118] @ git+https://github.com/unslothai/unsloth.git"

# Alternative: pip install (may be older version)
pip install unsloth>=2024.11
pip install trl>=0.11.0

# Verify installation
python -c "from unsloth import FastLanguageModel; print('Unsloth ready!')"
```

#### Using with Ollama

```bash
# Navigate to your Unsloth model output
cd ./my_model/lora_model

# Create Modelfile
cat > Modelfile << 'EOF'
FROM llama3.2:3b
ADAPTER ./adapter_model.safetensors
PARAMETER temperature 0.7
PARAMETER num_ctx 4096
SYSTEM "You are an AI assistant fine-tuned with Unsloth."
EOF

# Build and run with Ollama
ollama create my-unsloth-model -f Modelfile
ollama run my-unsloth-model "Explain quantum computing"
```

#### Using with LM Studio

```bash
# Copy to LM Studio models directory
cp -r ./my_model/lora_model ~/.cache/lm-studio/models/unsloth-adapter/

# In LM Studio:
# 1. Load your base model (e.g., llama-3.2-3b)
# 2. Go to Model Settings → LoRA Adapters
# 3. Select your unsloth-adapter folder
# 4. Start the server and use the API
```

#### Unsloth Tips

- **GPU Requirement**: Unsloth requires NVIDIA GPU with CUDA support
- **Memory Efficiency**: Use `load_in_4bit=True` for maximum memory savings
- **Gradient Checkpointing**: Enable for large models to reduce VRAM usage
- **Flash Attention**: Automatically used when available for faster training

---

### RAG Training

#### Overview

Retrieval-Augmented Generation (RAG) combines information retrieval with text generation. This training method creates a vector index from your dataset and optionally fine-tunes the generation component.

**Key Benefits:**
- Ideal for knowledge-intensive applications
- Reduces hallucination by grounding responses in data
- Easily updatable - add new documents without retraining
- Works with any embedding model for retrieval

#### Prerequisites & Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Core dependencies
pip install torch>=2.0.0
pip install transformers>=4.46.0
pip install sentence-transformers>=2.2.0

# Vector store options
pip install faiss-cpu  # CPU-only
# OR
pip install faiss-gpu  # GPU accelerated

# Alternative vector stores
pip install chromadb  # ChromaDB
pip install qdrant-client  # Qdrant

# LangChain for RAG orchestration (optional)
pip install langchain langchain-community
```

#### Using Your RAG Model

RAG training produces a vector index and configuration files. Query the index to retrieve relevant context, then pass it to your LLM.

**Python 3 Example:**

```python
import json
import numpy as np
from sentence_transformers import SentenceTransformer

# Load the RAG configuration
with open('./my_model/rag_model/rag_config.json') as f:
    config = json.load(f)

# Load embedding model
embedder = SentenceTransformer(config.get('embedding_model', 'all-MiniLM-L6-v2'))

# Load vector index (FAISS example)
import faiss
index = faiss.read_index('./my_model/rag_model/vector_index.faiss')

# Load document store
with open('./my_model/rag_model/documents.json') as f:
    documents = json.load(f)

def retrieve(query: str, top_k: int = 3):
    """Retrieve relevant documents for a query."""
    query_embedding = embedder.encode([query])
    distances, indices = index.search(query_embedding, top_k)
    return [documents[i] for i in indices[0]]

# Use with Ollama
import requests

def rag_query(question: str):
    # Retrieve context
    context_docs = retrieve(question)
    context = "\n\n".join(context_docs)
    
    # Generate with context
    prompt = f"""Based on the following context, answer the question.

Context:
{context}

Question: {question}

Answer:"""
    
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "llama3.2:3b", "prompt": prompt, "stream": False}
    )
    return response.json()["response"]

# Example usage
answer = rag_query("What is the main topic of the training data?")
print(answer)
```

**PHP 8 Example:**

```php
<?php
class RAGClient {
    private string $ollamaUrl = "http://localhost:11434";
    private string $ragServiceUrl = "http://localhost:8001"; // Python RAG service
    
    public function query(string $question, int $topK = 3): string {
        $context = $this->retrieve($question, $topK);
        
        $prompt = "Based on the following context, answer the question.\n\n";
        $prompt .= "Context:\n" . implode("\n\n", $context) . "\n\n";
        $prompt .= "Question: {$question}\n\nAnswer:";
        
        return $this->generate($prompt);
    }
    
    private function retrieve(string $query, int $topK): array {
        $ch = curl_init();
        curl_setopt_array($ch, [
            CURLOPT_URL => "{$this->ragServiceUrl}/retrieve",
            CURLOPT_POST => true,
            CURLOPT_RETURNTRANSFER => true,
            CURLOPT_HTTPHEADER => ["Content-Type: application/json"],
            CURLOPT_POSTFIELDS => json_encode([
                "query" => $query,
                "top_k" => $topK
            ])
        ]);
        $response = curl_exec($ch);
        curl_close($ch);
        return json_decode($response, true)["documents"];
    }
    
    private function generate(string $prompt): string {
        $ch = curl_init();
        curl_setopt_array($ch, [
            CURLOPT_URL => "{$this->ollamaUrl}/api/generate",
            CURLOPT_POST => true,
            CURLOPT_RETURNTRANSFER => true,
            CURLOPT_HTTPHEADER => ["Content-Type: application/json"],
            CURLOPT_POSTFIELDS => json_encode([
                "model" => "llama3.2:3b",
                "prompt" => $prompt,
                "stream" => false
            ])
        ]);
        $response = curl_exec($ch);
        curl_close($ch);
        return json_decode($response, true)["response"];
    }
}

$rag = new RAGClient();
$answer = $rag->query("What is the main topic?");
echo $answer;
```

#### RAG Output Files

| File | Description |
|------|-------------|
| `rag_config.json` | Configuration and metadata |
| `vector_index.faiss` | FAISS vector index |
| `documents.json` | Document store for retrieval |
| `embeddings_metadata.json` | Embedding model info |

---

### Standard Fine-tuning

#### Overview

Standard fine-tuning performs full model training or supervised fine-tuning (SFT) on your dataset. This approach modifies all model weights and produces a complete fine-tuned model.

**Key Benefits:**
- Maximum model customization potential
- Best for significant behavior changes
- Produces standalone model files

**Trade-offs:**
- Requires more compute and storage
- Larger output files

#### Prerequisites & Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Core dependencies
pip install torch>=2.0.0
pip install transformers>=4.46.0
pip install datasets>=3.0.0
pip install accelerate>=1.0.0
pip install trl>=0.11.0  # For SFTTrainer

# For distributed training (optional)
pip install deepspeed

# Verify GPU access
python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}')"
```

#### Using with Ollama

Standard fine-tuned models need to be converted to GGUF format for use with Ollama:

```bash
# Clone llama.cpp for conversion tools
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# Install conversion dependencies
pip install -r requirements.txt

# Convert your model to GGUF format
python convert_hf_to_gguf.py ../my_model/standard_model \
  --outfile my-model.gguf \
  --outtype q4_k_m  # Quantization type

# Create Ollama Modelfile
cat > Modelfile << 'EOF'
FROM ./my-model.gguf
PARAMETER temperature 0.7
PARAMETER num_ctx 4096
SYSTEM "You are a custom fine-tuned assistant."
EOF

# Create Ollama model
ollama create my-custom-model -f Modelfile

# Test your model
ollama run my-custom-model "Hello!"
```

#### Using with LM Studio

```bash
# After converting to GGUF
mkdir -p ~/.cache/lm-studio/models/my-custom-model
cp my-model.gguf ~/.cache/lm-studio/models/my-custom-model/

# In LM Studio:
# 1. Open LM Studio
# 2. Go to "My Models" or "Local Models"
# 3. Your model should appear in the list
# 4. Click to load and start chatting!
```

#### Standard Training Output Files

| File | Description |
|------|-------------|
| `model.safetensors` | Model weights |
| `config.json` | Model configuration |
| `tokenizer.json` | Tokenizer data |
| `tokenizer_config.json` | Tokenizer settings |
| `special_tokens_map.json` | Special tokens |

#### Resource Requirements

| Model Size | VRAM | RAM | Storage |
|------------|------|-----|---------|
| 3B parameters | 16-24GB | 32GB+ | 20GB+ |
| 7B parameters | 40-48GB | 64GB+ | 50GB+ |

---

## Model Output & Deployment

### Downloading Trained Models

After training completes, download your model from the Training Jobs page. Multi-file models are automatically packaged as `.tar.gz` archives.

```bash
# Extract the downloaded model archive
tar -xzf model_job_1.tar.gz -C ./my_model

# List extracted files
ls -la ./my_model/
```

### Model Storage Location

Models are saved to: `backend/uploads/models/job_{id}/`

| Training Type | Directory |
|--------------|-----------|
| QLoRA | `qlora_model/` |
| Unsloth | `lora_model/` |
| RAG | `rag_model/` |
| Standard | `standard_model/` |

---

## API Integration

### Training Job API

**Create a Training Job (POST):**

```bash
curl -X POST http://localhost:8000/api/v1/jobs/ \
  -H "Content-Type: application/json" \
  -d '{
    "name": "My Training Job",
    "description": "Fine-tuning with custom data",
    "training_type": "qlora",
    "model_name": "llama3.2:3b",
    "dataset_id": 1,
    "epochs": 3,
    "batch_size": 4,
    "learning_rate": 0.0002
  }'
```

**Check Job Status (GET):**

```bash
curl -X GET http://localhost:8000/api/v1/jobs/{job_id}/status
```

**Download Trained Model (GET):**

```bash
curl -X GET -O http://localhost:8000/api/v1/jobs/{job_id}/download
```

### Python Client Example

```python
import requests

class TrainerClient:
    def __init__(self, base_url="http://localhost:8000/api/v1"):
        self.base_url = base_url
    
    def create_job(self, name, dataset_id, training_type="qlora", **kwargs):
        response = requests.post(
            f"{self.base_url}/jobs/",
            json={
                "name": name,
                "dataset_id": dataset_id,
                "training_type": training_type,
                **kwargs
            }
        )
        return response.json()
    
    def get_status(self, job_id):
        response = requests.get(f"{self.base_url}/jobs/{job_id}/status")
        return response.json()
    
    def download_model(self, job_id, output_path):
        response = requests.get(
            f"{self.base_url}/jobs/{job_id}/download",
            stream=True
        )
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

# Usage
client = TrainerClient()
job = client.create_job("My Job", dataset_id=1, epochs=3)
print(f"Job created: {job['id']}")
```

---

## Troubleshooting

### Common Issues

1. **Out of Memory (OOM)**
   - Reduce batch size
   - Use QLoRA instead of standard fine-tuning
   - Enable gradient checkpointing

2. **Slow Training**
   - Use Unsloth for 2x faster training
   - Ensure GPU is being utilized
   - Reduce dataset size for testing

3. **Model Not Loading in Ollama**
   - Verify Modelfile syntax
   - Check adapter file paths
   - Ensure base model is available

4. **Download Failed**
   - Check job status is "completed"
   - Verify model_path exists
   - Check disk space

For more help, see the [main documentation](../README.md) or file an issue.
