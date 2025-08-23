# Dense Abstract Language (DAL) System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

An advanced implementation of the **Dense Abstract Language** pipeline with dynamic processing capabilities. DAL transforms natural language prompts into structured, coherent responses through a multi-stage pipeline that includes segmentation, tagging, vectorization, and intelligent stitching.

## ✨ Features

- **Dynamic Segmentation**: Automatically adjusts the number of segments based on input length
- **Adaptive Generation**: Smartly determines optimal generation parameters for each segment
- **Enhanced Expansion**: Optional LLM-based segment expansion with improved quality
- **Benchmarking Tools**: Built-in tools for performance evaluation and comparison
- **Lightweight**: Minimal external dependencies, with optional components for advanced features

## 🏗️ Project Structure

```
dal/
├── __init__.py           # Package exports and version
├── vectorizer.py        # Text to vector conversion
├── tagging.py           # Keyword-based tag generation
├── segmenter.py         # Prompt segmentation logic
├── markov.py            # Markov chain implementation
├── stitcher.py          # Segment assembly
├── pipeline.py          # Original pipeline implementation
├── pipeline_v2.py       # Enhanced pipeline with dynamic processing
├── expander.py          # Basic segment expansion
└── expander_v2.py       # Enhanced expansion with dynamic processing

benchmark/               # Benchmarking tools
├── __init__.py
├── dal_runner.py
├── gpt2_runner.py
└── requirements.txt

run_dal.py              # Main CLI entry point
benchmark_enhanced.py   # Enhanced benchmark script
README.md               # This file
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip
- (Optional) CUDA-enabled GPU for faster processing

### Installation

1. Clone the repository:
   ```bash
   git clone git@github.com:bimdev1/DAL.git
   cd DAL
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r benchmark/requirements.txt
   ```

### Basic Usage

```bash
# Run with default settings
python run_dal.py --prompt "Why do complex systems fail?" --segments 3

# Enable LLM-based expansion
python run_dal.py --prompt "Explain quantum computing" --expand

# Show vector representations
python run_dal.py --prompt "Describe machine learning" --show-vectors
```

### Advanced Usage with Enhanced Pipeline

The enhanced pipeline offers better handling of various input lengths and more consistent output quality:

```python
from dal import run_enhanced_pipeline

result = run_enhanced_pipeline(
    "Explain the impact of artificial intelligence on modern society",
    n_segments=4,
    expand=True,
    show_vectors=False
)

print(result["answer"])
```

## 📊 Benchmarking

Compare the performance of different pipeline versions:

```bash
python benchmark_enhanced.py
```

This will run a series of tests and generate a `benchmark_results_enhanced.json` file with detailed metrics.
expand each segment using a small local language model (configured in
`dal/expander.py`).  The expansion module relies on the
`transformers` library and downloads a summarisation model on first
use.  When enabled, segments are passed through the expander before
stitching, producing a longer, more detailed answer.  Use
`--show-vectors` to see the vector representations of each original
segment and debug the expansion.

## 🏭 Architecture Overview

### Core Components

1. **Dynamic Segmentation**
   - Splits input into meaningful segments based on content length
   - Adjusts segment count automatically (1-3 segments)
   - Preserves context and coherence

2. **Adaptive Processing**
   - Smartly determines optimal generation parameters
   - Adjusts max_length based on input size
   - Handles both short and long inputs effectively

3. **Enhanced Expansion**
   - Optional LLM-based segment enhancement
   - Dynamic prompt formatting
   - Fallback mechanisms for reliability

4. **Intelligent Stitching**
   - Markov chain-based segment ordering
   - Context-aware assembly
   - Duplicate detection and removal

### Extending the System

1. **Custom Models**
   ```python
   from dal.expander_v2 import ExpanderV2
   
   # Use a different model
   custom_expander = ExpanderV2(model_name="facebook/bart-large-cnn")
   ```

2. **Custom Processing**
   - Override segmentation logic
   - Add custom taggers
   - Implement new stitching strategies

3. **Integration**
   ```python
   from dal import run_enhanced_pipeline
   
   # Integrate with your application
   response = run_enhanced_pipeline(
       user_prompt,
       n_segments=3,
       expand=True
   )
   ```

## 📈 Performance

The enhanced pipeline shows significant improvements in response quality:

- **2.4x faster** processing for some inputs
- **2-3x longer** and more detailed responses
- Better handling of technical content
- More coherent output structure

## 🤝 Contributing

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Resources

- [DAL Documentation](docs/)
- [API Reference](docs/API.md)
- [Benchmark Results](benchmark_results_enhanced.json)
  stitching.
* Add a reflexive revision pass that recomputes tags after stitching
  and loops back to refine the answer.

Contributions and experiments are welcome; please open issues or pull
requests to discuss improvements.