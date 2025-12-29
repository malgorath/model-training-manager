import { useState } from 'react';
import { 
  BookOpen, 
  Zap, 
  Database, 
  Cpu, 
  ChevronDown,
  Terminal,
  Download,
  ExternalLink
} from 'lucide-react';
import { clsx } from 'clsx';
import CodeExample, { createOllamaExamples, createLMStudioExamples } from '../components/CodeExample';

/**
 * Training method section configuration.
 */
interface TrainingSection {
  id: string;
  title: string;
  icon: React.ElementType;
  description: string;
}

const trainingSections: TrainingSection[] = [
  {
    id: 'qlora',
    title: 'QLoRA Training',
    icon: Zap,
    description: 'Quantized Low-Rank Adaptation for memory-efficient fine-tuning',
  },
  {
    id: 'unsloth',
    title: 'Unsloth Training',
    icon: Cpu,
    description: 'Optimized LoRA training with 2x faster performance',
  },
  {
    id: 'rag',
    title: 'RAG Training',
    icon: Database,
    description: 'Retrieval-Augmented Generation for knowledge-intensive tasks',
  },
  {
    id: 'standard',
    title: 'Standard Fine-tuning',
    icon: BookOpen,
    description: 'Traditional supervised fine-tuning approach',
  },
];

/**
 * User Guide page component providing comprehensive documentation
 * for all training methods with code examples.
 */
export default function UserGuidePage() {
  const [expandedSection, setExpandedSection] = useState<string | null>('qlora');

  const toggleSection = (id: string) => {
    setExpandedSection(expandedSection === id ? null : id);
  };

  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-white">User Guide</h1>
        <p className="mt-2 text-surface-400">
          Comprehensive documentation for training methods, model deployment, and API integration.
        </p>
      </div>

      {/* Quick Start */}
      <div className="rounded-xl border border-surface-700 bg-surface-800/50 p-6">
        <h2 className="text-lg font-semibold text-white mb-4">Quick Start</h2>
        <div className="grid gap-4 md:grid-cols-3">
          <div className="rounded-lg bg-surface-700/50 p-4">
            <div className="flex items-center gap-2 mb-2">
              <span className="flex h-6 w-6 items-center justify-center rounded-full bg-primary-500/20 text-primary-400 text-sm font-bold">1</span>
              <span className="font-medium text-white">Upload Dataset</span>
            </div>
            <p className="text-sm text-surface-400">
              Upload your training data in CSV or JSON format via the Datasets page.
            </p>
          </div>
          <div className="rounded-lg bg-surface-700/50 p-4">
            <div className="flex items-center gap-2 mb-2">
              <span className="flex h-6 w-6 items-center justify-center rounded-full bg-primary-500/20 text-primary-400 text-sm font-bold">2</span>
              <span className="font-medium text-white">Create Training Job</span>
            </div>
            <p className="text-sm text-surface-400">
              Configure training parameters and select your preferred training method.
            </p>
          </div>
          <div className="rounded-lg bg-surface-700/50 p-4">
            <div className="flex items-center gap-2 mb-2">
              <span className="flex h-6 w-6 items-center justify-center rounded-full bg-primary-500/20 text-primary-400 text-sm font-bold">3</span>
              <span className="font-medium text-white">Download & Deploy</span>
            </div>
            <p className="text-sm text-surface-400">
              Download your trained model and deploy with Ollama or LM Studio.
            </p>
          </div>
        </div>
      </div>

      {/* Training Methods */}
      <div className="space-y-4">
        <h2 className="text-lg font-semibold text-white">Training Methods</h2>
        
        {trainingSections.map((section) => (
          <div
            key={section.id}
            className="rounded-xl border border-surface-700 bg-surface-800/50 overflow-hidden"
          >
            {/* Section Header */}
            <button
              onClick={() => toggleSection(section.id)}
              className="w-full flex items-center justify-between p-4 hover:bg-surface-700/30 transition-colors"
            >
              <div className="flex items-center gap-3">
                <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary-500/10">
                  <section.icon className="h-5 w-5 text-primary-400" />
                </div>
                <div className="text-left">
                  <h3 className="font-medium text-white">{section.title}</h3>
                  <p className="text-sm text-surface-400">{section.description}</p>
                </div>
              </div>
              <ChevronDown
                className={clsx(
                  'h-5 w-5 text-surface-400 transition-transform',
                  expandedSection === section.id && 'rotate-180'
                )}
              />
            </button>

            {/* Section Content */}
            {expandedSection === section.id && (
              <div className="border-t border-surface-700 p-6 space-y-6">
                {section.id === 'qlora' && <QLoRAContent />}
                {section.id === 'unsloth' && <UnslothContent />}
                {section.id === 'rag' && <RAGContent />}
                {section.id === 'standard' && <StandardContent />}
              </div>
            )}
          </div>
        ))}
      </div>

      {/* General Information */}
      <div className="rounded-xl border border-surface-700 bg-surface-800/50 p-6 space-y-6">
        <h2 className="text-lg font-semibold text-white">Model Output & Deployment</h2>
        
        <div className="space-y-4">
          <div>
            <h3 className="font-medium text-white mb-2 flex items-center gap-2">
              <Download className="h-4 w-4 text-primary-400" />
              Downloading Trained Models
            </h3>
            <p className="text-sm text-surface-400 mb-3">
              After training completes, you can download your model from the Training Jobs page.
              Multi-file models are automatically packaged as <code className="text-primary-400">.tar.gz</code> archives.
            </p>
            <CodeExample
              title="Extract Downloaded Model"
              examples={[
                {
                  language: 'bash',
                  label: 'BASH',
                  code: `# Extract the downloaded model archive
tar -xzf model_job_1.tar.gz -C ./my_model

# List extracted files
ls -la ./my_model/

# For single-file downloads, no extraction needed`,
                },
              ]}
            />
          </div>

          <div>
            <h3 className="font-medium text-white mb-2 flex items-center gap-2">
              <ExternalLink className="h-4 w-4 text-primary-400" />
              Model Storage Location
            </h3>
            <p className="text-sm text-surface-400">
              Models are saved to: <code className="text-primary-400">backend/uploads/models/job_{'{'} id {'}'}/</code>
            </p>
            <ul className="mt-2 text-sm text-surface-400 list-disc list-inside space-y-1">
              <li>QLoRA models: <code className="text-primary-400">qlora_model/</code></li>
              <li>Unsloth models: <code className="text-primary-400">lora_model/</code></li>
              <li>RAG models: <code className="text-primary-400">rag_model/</code></li>
              <li>Standard models: <code className="text-primary-400">standard_model/</code></li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}

/**
 * QLoRA training documentation content.
 */
function QLoRAContent() {
  return (
    <div className="space-y-6">
      {/* Overview */}
      <div>
        <h4 className="font-medium text-white mb-2">Overview</h4>
        <p className="text-sm text-surface-400">
          QLoRA (Quantized Low-Rank Adaptation) is a memory-efficient fine-tuning technique that enables 
          training of large language models on consumer hardware. It uses 4-bit quantization to reduce 
          memory usage while maintaining model quality through low-rank adaptation.
        </p>
        <ul className="mt-3 text-sm text-surface-400 list-disc list-inside space-y-1">
          <li>Up to 65% memory reduction compared to full fine-tuning</li>
          <li>Produces small adapter files (typically 10-100MB)</li>
          <li>Compatible with base models in Ollama and LM Studio</li>
          <li>Ideal for limited GPU memory scenarios (8-16GB VRAM)</li>
        </ul>
      </div>

      {/* Prerequisites */}
      <div>
        <h4 className="font-medium text-white mb-2 flex items-center gap-2">
          <Terminal className="h-4 w-4 text-primary-400" />
          Prerequisites & Installation
        </h4>
        <CodeExample
          title="Install Dependencies"
          description="Required Python packages for QLoRA training"
          examples={[
            {
              language: 'bash',
              label: 'BASH',
              code: `# Create virtual environment
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
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"`,
            },
          ]}
        />
      </div>

      {/* Ollama Integration */}
      <div>
        <h4 className="font-medium text-white mb-2">Using with Ollama</h4>
        <p className="text-sm text-surface-400 mb-3">
          To use your QLoRA adapter with Ollama, you need to create a Modelfile that references 
          your base model and applies the adapter weights.
        </p>
        
        <CodeExample
          title="Create Ollama Modelfile"
          examples={[
            {
              language: 'bash',
              label: 'BASH',
              code: `# Navigate to your model directory
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
ollama run my-qlora-model "Hello, how are you?"`,
            },
          ]}
        />

        <div className="mt-4">
          <CodeExample
            title="Query Your Model"
            examples={createOllamaExamples('my-qlora-model')}
          />
        </div>
      </div>

      {/* LM Studio Integration */}
      <div>
        <h4 className="font-medium text-white mb-2">Using with LM Studio</h4>
        <p className="text-sm text-surface-400 mb-3">
          LM Studio supports loading LoRA adapters alongside base models. Import your adapter through the UI
          or place it in the models directory.
        </p>
        
        <CodeExample
          title="Setup for LM Studio"
          examples={[
            {
              language: 'bash',
              label: 'BASH',
              code: `# Find LM Studio models directory
# macOS: ~/.cache/lm-studio/models/
# Windows: C:\\Users\\<user>\\.cache\\lm-studio\\models\\
# Linux: ~/.cache/lm-studio/models/

# Copy your adapter to a subdirectory
mkdir -p ~/.cache/lm-studio/models/my-qlora-adapter
cp -r ./my_model/qlora_model/* ~/.cache/lm-studio/models/my-qlora-adapter/

# Restart LM Studio and load the adapter from the UI
# Select your base model, then apply the LoRA adapter`,
            },
          ]}
        />

        <div className="mt-4">
          <CodeExample
            title="Query via LM Studio API"
            examples={createLMStudioExamples()}
          />
        </div>
      </div>

      {/* Training Parameters */}
      <div>
        <h4 className="font-medium text-white mb-2">Recommended Parameters</h4>
        <div className="grid gap-4 md:grid-cols-2">
          <div className="rounded-lg bg-surface-700/50 p-4">
            <h5 className="text-sm font-medium text-white mb-2">LoRA Configuration</h5>
            <ul className="text-sm text-surface-400 space-y-1">
              <li><strong>Rank (r):</strong> 16-64 (higher = more capacity)</li>
              <li><strong>Alpha:</strong> 32-128 (typically 2x rank)</li>
              <li><strong>Dropout:</strong> 0.05-0.1</li>
              <li><strong>Target modules:</strong> q_proj, v_proj, k_proj, o_proj</li>
            </ul>
          </div>
          <div className="rounded-lg bg-surface-700/50 p-4">
            <h5 className="text-sm font-medium text-white mb-2">Training Settings</h5>
            <ul className="text-sm text-surface-400 space-y-1">
              <li><strong>Batch size:</strong> 4-16 (depends on VRAM)</li>
              <li><strong>Learning rate:</strong> 1e-4 to 3e-4</li>
              <li><strong>Epochs:</strong> 1-5 (monitor for overfitting)</li>
              <li><strong>Warmup ratio:</strong> 0.03-0.1</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}

/**
 * Unsloth training documentation content.
 */
function UnslothContent() {
  return (
    <div className="space-y-6">
      {/* Overview */}
      <div>
        <h4 className="font-medium text-white mb-2">Overview</h4>
        <p className="text-sm text-surface-400">
          Unsloth is an optimized training framework that provides 2x faster LoRA fine-tuning with 
          reduced memory usage. It uses custom CUDA kernels and optimized attention implementations 
          to accelerate training while maintaining full compatibility with Hugging Face ecosystem.
        </p>
        <ul className="mt-3 text-sm text-surface-400 list-disc list-inside space-y-1">
          <li>2x faster training compared to standard PEFT</li>
          <li>30% less memory usage</li>
          <li>Compatible with 4-bit and 8-bit quantization</li>
          <li>Supports most popular LLM architectures (Llama, Mistral, etc.)</li>
        </ul>
      </div>

      {/* Prerequisites */}
      <div>
        <h4 className="font-medium text-white mb-2 flex items-center gap-2">
          <Terminal className="h-4 w-4 text-primary-400" />
          Prerequisites & Installation
        </h4>
        <CodeExample
          title="Install Unsloth"
          description="Installation varies by CUDA version"
          examples={[
            {
              language: 'bash',
              label: 'BASH',
              code: `# Create virtual environment
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
python -c "from unsloth import FastLanguageModel; print('Unsloth ready!')"`,
            },
          ]}
        />
      </div>

      {/* Ollama Integration */}
      <div>
        <h4 className="font-medium text-white mb-2">Using with Ollama</h4>
        <p className="text-sm text-surface-400 mb-3">
          Unsloth produces standard LoRA adapter files that work seamlessly with Ollama.
        </p>
        
        <CodeExample
          title="Create Ollama Model from Unsloth Adapter"
          examples={[
            {
              language: 'bash',
              label: 'BASH',
              code: `# Navigate to your Unsloth model output
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
ollama run my-unsloth-model "Explain quantum computing"`,
            },
          ]}
        />

        <div className="mt-4">
          <CodeExample
            title="API Access"
            examples={createOllamaExamples('my-unsloth-model')}
          />
        </div>
      </div>

      {/* LM Studio Integration */}
      <div>
        <h4 className="font-medium text-white mb-2">Using with LM Studio</h4>
        <CodeExample
          title="Load in LM Studio"
          examples={[
            {
              language: 'bash',
              label: 'BASH',
              code: `# Unsloth adapters are compatible with LM Studio
# Copy to LM Studio models directory

# Linux/macOS
cp -r ./my_model/lora_model ~/.cache/lm-studio/models/unsloth-adapter/

# Then in LM Studio:
# 1. Load your base model (e.g., llama-3.2-3b)
# 2. Go to Model Settings → LoRA Adapters
# 3. Select your unsloth-adapter folder
# 4. Start the server and use the API`,
            },
          ]}
        />

        <div className="mt-4">
          <CodeExample
            title="LM Studio API Usage"
            examples={createLMStudioExamples()}
          />
        </div>
      </div>

      {/* Unsloth-Specific Tips */}
      <div>
        <h4 className="font-medium text-white mb-2">Unsloth Tips</h4>
        <div className="rounded-lg bg-surface-700/50 p-4">
          <ul className="text-sm text-surface-400 space-y-2">
            <li><strong>GPU Requirement:</strong> Unsloth requires NVIDIA GPU with CUDA support</li>
            <li><strong>Memory Efficiency:</strong> Use <code className="text-primary-400">load_in_4bit=True</code> for maximum memory savings</li>
            <li><strong>Gradient Checkpointing:</strong> Enable for large models to reduce VRAM usage</li>
            <li><strong>Flash Attention:</strong> Automatically used when available for faster training</li>
          </ul>
        </div>
      </div>
    </div>
  );
}

/**
 * RAG training documentation content.
 */
function RAGContent() {
  return (
    <div className="space-y-6">
      {/* Overview */}
      <div>
        <h4 className="font-medium text-white mb-2">Overview</h4>
        <p className="text-sm text-surface-400">
          Retrieval-Augmented Generation (RAG) combines information retrieval with text generation.
          This training method creates a vector index from your dataset and optionally fine-tunes 
          the generation component to better utilize retrieved context.
        </p>
        <ul className="mt-3 text-sm text-surface-400 list-disc list-inside space-y-1">
          <li>Ideal for knowledge-intensive applications</li>
          <li>Reduces hallucination by grounding responses in data</li>
          <li>Easily updatable - add new documents without retraining</li>
          <li>Works with any embedding model for retrieval</li>
        </ul>
      </div>

      {/* Prerequisites */}
      <div>
        <h4 className="font-medium text-white mb-2 flex items-center gap-2">
          <Terminal className="h-4 w-4 text-primary-400" />
          Prerequisites & Installation
        </h4>
        <CodeExample
          title="Install RAG Dependencies"
          examples={[
            {
              language: 'bash',
              label: 'BASH',
              code: `# Create virtual environment
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
pip install langchain langchain-community`,
            },
          ]}
        />
      </div>

      {/* Using RAG Model */}
      <div>
        <h4 className="font-medium text-white mb-2">Using Your RAG Model</h4>
        <p className="text-sm text-surface-400 mb-3">
          RAG training produces a vector index and configuration files. You'll query the index 
          to retrieve relevant context, then pass it to your LLM.
        </p>
        
        <CodeExample
          title="Load and Query RAG Model"
          examples={[
            {
              language: 'python',
              label: 'Python 3',
              code: `import json
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
    context = "\\n\\n".join(context_docs)
    
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
print(answer)`,
            },
            {
              language: 'php',
              label: 'PHP 8',
              code: `<?php
// RAG implementation requires a Python backend for embeddings
// PHP handles the orchestration and LLM queries

class RAGClient {
    private string $ollamaUrl = "http://localhost:11434";
    private string $ragServiceUrl = "http://localhost:8001"; // Python RAG service
    
    public function query(string $question, int $topK = 3): string {
        // Get relevant context from RAG service
        $context = $this->retrieve($question, $topK);
        
        // Build prompt with context
        $prompt = "Based on the following context, answer the question.\\n\\n";
        $prompt .= "Context:\\n" . implode("\\n\\n", $context) . "\\n\\n";
        $prompt .= "Question: {$question}\\n\\nAnswer:";
        
        // Generate response with Ollama
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

// Usage
$rag = new RAGClient();
$answer = $rag->query("What is the main topic?");
echo $answer;`,
            },
            {
              language: 'bash',
              label: 'CURL',
              code: `# RAG requires a retrieval step before generation
# First, query your RAG service for context

# Step 1: Retrieve relevant documents (assuming Python RAG service running)
CONTEXT=$(curl -s -X POST http://localhost:8001/retrieve \\
  -H "Content-Type: application/json" \\
  -d '{"query": "What is the main topic?", "top_k": 3}' | jq -r '.documents | join("\\n\\n")')

# Step 2: Generate response with context
curl -X POST http://localhost:11434/api/generate \\
  -H "Content-Type: application/json" \\
  -d "{
    \\"model\\": \\"llama3.2:3b\\",
    \\"prompt\\": \\"Context:\\n\$CONTEXT\\n\\nQuestion: What is the main topic?\\n\\nAnswer:\\",
    \\"stream\\": false
  }"`,
            },
          ]}
        />
      </div>

      {/* RAG Files */}
      <div>
        <h4 className="font-medium text-white mb-2">Output Files</h4>
        <div className="rounded-lg bg-surface-700/50 p-4">
          <p className="text-sm text-surface-400 mb-2">RAG training produces:</p>
          <ul className="text-sm text-surface-400 space-y-1 list-disc list-inside">
            <li><code className="text-primary-400">rag_config.json</code> - Configuration and metadata</li>
            <li><code className="text-primary-400">vector_index.faiss</code> - FAISS vector index</li>
            <li><code className="text-primary-400">documents.json</code> - Document store for retrieval</li>
            <li><code className="text-primary-400">embeddings_metadata.json</code> - Embedding model info</li>
          </ul>
        </div>
      </div>
    </div>
  );
}

/**
 * Standard fine-tuning documentation content.
 */
function StandardContent() {
  return (
    <div className="space-y-6">
      {/* Overview */}
      <div>
        <h4 className="font-medium text-white mb-2">Overview</h4>
        <p className="text-sm text-surface-400">
          Standard fine-tuning performs full model training or supervised fine-tuning (SFT) on your dataset.
          This approach modifies all model weights and produces a complete fine-tuned model rather than 
          small adapter files.
        </p>
        <ul className="mt-3 text-sm text-surface-400 list-disc list-inside space-y-1">
          <li>Maximum model customization potential</li>
          <li>Best for significant behavior changes</li>
          <li>Requires more compute and storage</li>
          <li>Produces standalone model files</li>
        </ul>
      </div>

      {/* Prerequisites */}
      <div>
        <h4 className="font-medium text-white mb-2 flex items-center gap-2">
          <Terminal className="h-4 w-4 text-primary-400" />
          Prerequisites & Installation
        </h4>
        <CodeExample
          title="Install Dependencies"
          examples={[
            {
              language: 'bash',
              label: 'BASH',
              code: `# Create virtual environment
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
python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}')"`,
            },
          ]}
        />
      </div>

      {/* Ollama Integration */}
      <div>
        <h4 className="font-medium text-white mb-2">Using with Ollama</h4>
        <p className="text-sm text-surface-400 mb-3">
          Standard fine-tuned models need to be converted to GGUF format for use with Ollama.
        </p>
        
        <CodeExample
          title="Convert and Deploy to Ollama"
          examples={[
            {
              language: 'bash',
              label: 'BASH',
              code: `# Clone llama.cpp for conversion tools
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# Install conversion dependencies
pip install -r requirements.txt

# Convert your model to GGUF format
python convert_hf_to_gguf.py ../my_model/standard_model \\
  --outfile my-model.gguf \\
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
ollama run my-custom-model "Hello!"`,
            },
          ]}
        />

        <div className="mt-4">
          <CodeExample
            title="API Access"
            examples={createOllamaExamples('my-custom-model')}
          />
        </div>
      </div>

      {/* LM Studio Integration */}
      <div>
        <h4 className="font-medium text-white mb-2">Using with LM Studio</h4>
        <p className="text-sm text-surface-400 mb-3">
          LM Studio can load GGUF models directly. Convert your model and place it in the models directory.
        </p>
        
        <CodeExample
          title="Setup for LM Studio"
          examples={[
            {
              language: 'bash',
              label: 'BASH',
              code: `# After converting to GGUF (see Ollama section)
# Copy to LM Studio models directory

# Linux/macOS
mkdir -p ~/.cache/lm-studio/models/my-custom-model
cp my-model.gguf ~/.cache/lm-studio/models/my-custom-model/

# Windows
# Copy to C:\\Users\\<user>\\.cache\\lm-studio\\models\\my-custom-model\\

# Then:
# 1. Open LM Studio
# 2. Go to "My Models" or "Local Models"
# 3. Your model should appear in the list
# 4. Click to load and start chatting!`,
            },
          ]}
        />

        <div className="mt-4">
          <CodeExample
            title="LM Studio API Usage"
            examples={createLMStudioExamples()}
          />
        </div>
      </div>

      {/* Output Files */}
      <div>
        <h4 className="font-medium text-white mb-2">Output Files</h4>
        <div className="rounded-lg bg-surface-700/50 p-4">
          <p className="text-sm text-surface-400 mb-2">Standard training produces:</p>
          <ul className="text-sm text-surface-400 space-y-1 list-disc list-inside">
            <li><code className="text-primary-400">model.safetensors</code> - Model weights</li>
            <li><code className="text-primary-400">config.json</code> - Model configuration</li>
            <li><code className="text-primary-400">tokenizer.json</code> - Tokenizer data</li>
            <li><code className="text-primary-400">tokenizer_config.json</code> - Tokenizer settings</li>
            <li><code className="text-primary-400">special_tokens_map.json</code> - Special tokens</li>
          </ul>
        </div>
      </div>

      {/* Resource Requirements */}
      <div>
        <h4 className="font-medium text-white mb-2">Resource Requirements</h4>
        <div className="grid gap-4 md:grid-cols-2">
          <div className="rounded-lg bg-surface-700/50 p-4">
            <h5 className="text-sm font-medium text-white mb-2">3B Parameter Model</h5>
            <ul className="text-sm text-surface-400 space-y-1">
              <li><strong>VRAM:</strong> 16-24GB</li>
              <li><strong>RAM:</strong> 32GB+</li>
              <li><strong>Storage:</strong> 20GB+ for checkpoints</li>
            </ul>
          </div>
          <div className="rounded-lg bg-surface-700/50 p-4">
            <h5 className="text-sm font-medium text-white mb-2">7B Parameter Model</h5>
            <ul className="text-sm text-surface-400 space-y-1">
              <li><strong>VRAM:</strong> 40-48GB</li>
              <li><strong>RAM:</strong> 64GB+</li>
              <li><strong>Storage:</strong> 50GB+ for checkpoints</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}
