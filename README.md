# GlitchMiner: Mining Glitch Tokens in Large Language Models via Gradient-based Discrete Optimization

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg?style=flat-square)](https://www.python.org/)
[![HuggingFace Transformers](https://img.shields.io/badge/HuggingFace-Transformers-orange?style=flat-square)](https://huggingface.co/transformers/)
[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen?style=flat-square)](https://github.com/wooozihui/GlitchMiner/pulls)

## Update

- 2024/11/7 We add reproduction code of the two baseline method in this paper.

**GlitchMiner** is a robust framework designed to detect glitch tokens—tokens that cause unexpected behaviors in large language models (LLMs). These anomalies can severely impact model outputs, particularly in sensitive applications such as healthcare or finance.

[Read our paper](https://arxiv.org/pdf/2410.15052) for detailed insights.

---

## 🧐 What are Glitch Tokens?

**Glitch tokens** are errors within LLMs that can trigger:
- **Incorrect outputs:** Misleading information in critical decision-making.
- **Hallucinated content:** Unreliable or irrelevant responses.
- **Security risks:** Exposure to potential adversarial attacks.

### Causes of Glitch Tokens:
- **Inadequate training data** for certain tokens.
- **Suboptimal token embeddings** affecting model performance.
- **Rare token occurrences** that the model fails to handle properly.

---

## 🔍 How GlitchMiner Works

**GlitchMiner** uses gradient-based discrete optimization to identify glitch tokens effectively, enhancing both the security and reliability of LLMs.

### Key Features:
- **Entropy Maximization:** Targets tokens where the model's output is most uncertain.
- **Gradient-Guided Search:** Employs model gradients to pinpoint potential glitches.
- **Dynamic Token Filtering:** Refines the focus on relevant tokens, improving search efficiency.
- **Support for Multiple Models:** Compatible with popular models like Llama, Qwen, and Gemma.

---

## 🛠️ Getting Started

Install GlitchMiner with pip:
```bash
pip install git+https://github.com/wooozihui/GlitchMiner.git
```

### Usage Example

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from glitchminer import GlitchMiner
import torch

if __name__ == "__main__":
    model_path = "Qwen/Qwen2.5-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="cuda",
            torch_dtype=torch.bfloat16,
        )

    # Run GlitchMiner for glitch token detection
    glitch_tokens, glitch_token_ids = GlitchMiner(
        model,
        tokenizer,
        num_iterations=125,
        batch_size=8,
        k=32,
        if_print=True,
        print_language="CN",
    )
```

### Strictly Glitch Token Verification
To eliminate false positives, we recommend using the `strictly_glitch_verification` function for cross-validation.
```python
    from glitchminer import strictly_glitch_verification
    glitch_count, verified_glitch_ids = strictly_glitch_verification(model, tokenizer, glitch_token_ids)
    print(glitch_count)
```

---

## ⚙️ GlitchMiner Parameters

Here are the configurable parameters for GlitchMiner, with explanations of their purpose and usage:

| Parameter         | Type     | Default Value | Description                                                                                       |
|-------------------|----------|---------------|---------------------------------------------------------------------------------------------------|
| model           | Model    | **Required**  | A Hugging Face AutoModelForCausalLM model used for glitch token detection.                       |
| tokenizer       | Tokenizer| **Required**  | A Hugging Face AutoTokenizer for encoding and decoding tokens.                                   |
| num_iterations  | int    | 125            | The number of iterations to run the glitch token search.                                           |
| batch_size      | int    | 8             | Number of tokens processed per batch during the search process.                                    |
| k               | int    | 32            | Number of top similar tokens to evaluate during each iteration using cosine similarity.            |
| if_print        | bool   | True        | If True, prints detailed progress and results during execution.                                  |
| print_language  | str    | "CN"        | Output language for printed messages. Supports "CN" for Chinese and "ENG" for English.          |
| skip_tokens     | list   | None        | Optional list of token IDs to exclude from the glitch detection process.                           |

---

## 📊 Impact

GlitchMiner has proven effective across various LLM architectures, significantly reducing the risk of error in critical applications.

---

## 🌟 Citing

If you find GlitchMiner helpful in your research, please consider give us a star or cite:
```bibtex
@misc{wu2024glitchminerminingglitchtokens,
      title={GlitchMiner: Mining Glitch Tokens in Large Language Models via Gradient-based Discrete Optimization}, 
      author={Zihui Wu and Haichang Gao and Ping Wang and Shudong Zhang and Zhaoxiang Liu and Shiguo Lian},
      year={2024},
      eprint={2410.15052},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2410.15052}, 
}
```
