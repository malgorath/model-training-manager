import { useState } from 'react';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { Copy, Check } from 'lucide-react';
import { clsx } from 'clsx';

/**
 * Language configuration for code examples.
 */
type Language = 'python' | 'php' | 'bash';

/**
 * Code example entry with language and code content.
 */
interface CodeEntry {
  language: Language;
  label: string;
  code: string;
}

/**
 * Props for the CodeExample component.
 */
interface CodeExampleProps {
  /** Title for the code example section */
  title?: string;
  /** Array of code examples in different languages */
  examples: CodeEntry[];
  /** Optional description */
  description?: string;
}

/**
 * Maps our language types to syntax highlighter language identifiers.
 */
const languageMap: Record<Language, string> = {
  python: 'python',
  php: 'php',
  bash: 'bash',
};

/**
 * CodeExample component displays code snippets with syntax highlighting
 * and tabs for switching between different language implementations.
 */
export default function CodeExample({ title, examples, description }: CodeExampleProps) {
  const [activeTab, setActiveTab] = useState(0);
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    const code = examples[activeTab]?.code || '';
    await navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  if (examples.length === 0) {
    return null;
  }

  const activeExample = examples[activeTab];

  return (
    <div className="rounded-lg border border-surface-700 bg-surface-800/50 overflow-hidden">
      {/* Header with title and tabs */}
      <div className="flex items-center justify-between border-b border-surface-700 px-4 py-2">
        <div className="flex items-center gap-4">
          {title && (
            <h4 className="text-sm font-medium text-surface-200">{title}</h4>
          )}
          <div className="flex gap-1">
            {examples.map((example, index) => (
              <button
                key={example.language}
                onClick={() => setActiveTab(index)}
                className={clsx(
                  'px-3 py-1.5 text-xs font-medium rounded-md transition-colors',
                  activeTab === index
                    ? 'bg-primary-500/20 text-primary-400'
                    : 'text-surface-400 hover:text-surface-200 hover:bg-surface-700'
                )}
              >
                {example.label}
              </button>
            ))}
          </div>
        </div>
        <button
          onClick={handleCopy}
          className="flex items-center gap-1.5 px-2 py-1 text-xs text-surface-400 hover:text-surface-200 transition-colors"
          title="Copy to clipboard"
        >
          {copied ? (
            <>
              <Check className="h-3.5 w-3.5 text-green-400" />
              <span className="text-green-400">Copied!</span>
            </>
          ) : (
            <>
              <Copy className="h-3.5 w-3.5" />
              <span>Copy</span>
            </>
          )}
        </button>
      </div>

      {/* Description if provided */}
      {description && (
        <div className="px-4 py-2 border-b border-surface-700 bg-surface-800/30">
          <p className="text-xs text-surface-400">{description}</p>
        </div>
      )}

      {/* Code block */}
      <div className="overflow-x-auto">
        <SyntaxHighlighter
          language={languageMap[activeExample.language]}
          style={oneDark}
          customStyle={{
            margin: 0,
            padding: '1rem',
            background: 'transparent',
            fontSize: '0.8125rem',
            lineHeight: '1.5',
          }}
          showLineNumbers={activeExample.code.split('\n').length > 5}
          wrapLines
        >
          {activeExample.code.trim()}
        </SyntaxHighlighter>
      </div>
    </div>
  );
}

/**
 * Helper function to create code examples for Ollama usage.
 */
export function createOllamaExamples(modelName: string): CodeEntry[] {
  return [
    {
      language: 'python',
      label: 'Python 3',
      code: `import requests

# Using the Ollama API
response = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "${modelName}",
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
    model="${modelName}",
    prompt="Your prompt here"
)
print(response["response"])`,
    },
    {
      language: 'php',
      label: 'PHP 8',
      code: `<?php
// Using cURL
$ch = curl_init();
curl_setopt_array($ch, [
    CURLOPT_URL => "http://localhost:11434/api/generate",
    CURLOPT_POST => true,
    CURLOPT_RETURNTRANSFER => true,
    CURLOPT_HTTPHEADER => ["Content-Type: application/json"],
    CURLOPT_POSTFIELDS => json_encode([
        "model" => "${modelName}",
        "prompt" => "Your prompt here",
        "stream" => false
    ])
]);

$response = curl_exec($ch);
curl_close($ch);

$result = json_decode($response, true);
echo $result["response"];

// Using Guzzle HTTP client
// composer require guzzlehttp/guzzle
use GuzzleHttp\\Client;

$client = new Client();
$response = $client->post("http://localhost:11434/api/generate", [
    "json" => [
        "model" => "${modelName}",
        "prompt" => "Your prompt here",
        "stream" => false
    ]
]);

$result = json_decode($response->getBody(), true);
echo $result["response"];`,
    },
    {
      language: 'bash',
      label: 'CURL',
      code: `# Generate a response
curl -X POST http://localhost:11434/api/generate \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "${modelName}",
    "prompt": "Your prompt here",
    "stream": false
  }'

# Chat completion
curl -X POST http://localhost:11434/api/chat \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "${modelName}",
    "messages": [
      {"role": "user", "content": "Your message here"}
    ],
    "stream": false
  }'`,
    },
  ];
}

/**
 * Helper function to create code examples for LM Studio usage.
 */
export function createLMStudioExamples(): CodeEntry[] {
  return [
    {
      language: 'python',
      label: 'Python 3',
      code: `import requests

# LM Studio uses OpenAI-compatible API
# Default endpoint: http://localhost:1234/v1

response = requests.post(
    "http://localhost:1234/v1/chat/completions",
    headers={"Content-Type": "application/json"},
    json={
        "model": "local-model",  # LM Studio uses generic model name
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
# pip install openai
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="not-needed"  # LM Studio doesn't require API key
)

response = client.chat.completions.create(
    model="local-model",
    messages=[{"role": "user", "content": "Your prompt here"}],
    temperature=0.7
)
print(response.choices[0].message.content)`,
    },
    {
      language: 'php',
      label: 'PHP 8',
      code: `<?php
// Using cURL with OpenAI-compatible API
$ch = curl_init();
curl_setopt_array($ch, [
    CURLOPT_URL => "http://localhost:1234/v1/chat/completions",
    CURLOPT_POST => true,
    CURLOPT_RETURNTRANSFER => true,
    CURLOPT_HTTPHEADER => [
        "Content-Type: application/json"
    ],
    CURLOPT_POSTFIELDS => json_encode([
        "model" => "local-model",
        "messages" => [
            ["role" => "user", "content" => "Your prompt here"]
        ],
        "temperature" => 0.7,
        "max_tokens" => 500
    ])
]);

$response = curl_exec($ch);
curl_close($ch);

$result = json_decode($response, true);
echo $result["choices"][0]["message"]["content"];

// Using Guzzle
use GuzzleHttp\\Client;

$client = new Client(["base_uri" => "http://localhost:1234"]);
$response = $client->post("/v1/chat/completions", [
    "json" => [
        "model" => "local-model",
        "messages" => [
            ["role" => "user", "content" => "Your prompt here"]
        ]
    ]
]);

$result = json_decode($response->getBody(), true);
echo $result["choices"][0]["message"]["content"];`,
    },
    {
      language: 'bash',
      label: 'CURL',
      code: `# Chat completion (OpenAI-compatible)
curl -X POST http://localhost:1234/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "local-model",
    "messages": [
      {"role": "user", "content": "Your prompt here"}
    ],
    "temperature": 0.7,
    "max_tokens": 500
  }'

# Text completion (legacy endpoint)
curl -X POST http://localhost:1234/v1/completions \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "local-model",
    "prompt": "Your prompt here",
    "max_tokens": 500
  }'`,
    },
  ];
}
