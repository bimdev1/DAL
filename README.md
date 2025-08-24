# Dense Abstract Language (DAL) System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

An advanced implementation of the **Dense Abstract Language** pipeline with dynamic processing capabilities. # DAL System (v1.3.0)

Document Augmentation and Language processing system for structured text generation and manipulation.

## What's New in v1.3.0 (Phase 2C)

### Reflexive Revision Engine
- **Consistency Checking**: Automatically detects and reports issues like redundancy, contradictions, and style mismatches
- **Smart Reordering**: Optimizes content flow by reordering blocks based on semantic importance and structure
- **Seamless Stitching**: Creates smooth transitions between segments with optional AI-assisted mini-generations
- **Configurable**: Control the level of intervention with `use_reflexive` and `reflexive_allow_mini_gen` flags

### Enhanced SDT Injection
- **Normalized Attributes**: Consistent handling of tone, depth, and format attributes
- **Prompt Styles**: Multiple prompt formats supported (`llama_inst`, `chatml`, `plain`)
- **Constraints System**: Configurable constraints to guide model behavior
- **Detailed Metadata**: Comprehensive tracking of applied SDT attributes

### Improved Pipeline Integration
- **Metrics Collection**: Detailed performance and quality metrics for the reflexive pass
- **Backward Compatibility**: All existing functionality remains unchanged
- **Error Handling**: Graceful degradation when reflexive processing encounters issues

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
├── pipeline_v3.py       # Enhanced pipeline with reflexive revision
├── expander.py          # Basic segment expansion
├── expander_v2.py       # Enhanced expansion with dynamic processing
├── reflexive_revision.py # Reflexive revision engine
└── sdt_injector.py      # SDT injection module

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
   git clone https://github.com/yourusername/dal_system.git
   cd dal_system
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -e ".[local]"  # For local model support
   # or
   pip install -e ".[all]"    # For all optional dependencies
   ```

### Using the Enhanced Pipeline with Reflexive Revision

```python
from dal.pipeline_v3 import run_enhanced_pipeline_v3, PipelineConfig
from dal.sdt_injector import SDTOpts

# Configure the pipeline with reflexive revision enabled
config = PipelineConfig(
    use_local_generation=True,
    use_reflexive=True,  # Enable reflexive revision
    reflexive_allow_mini_gen=True,  # Allow small generations for transitions
    sdt_control=True,
    sdt_opts=SDTOpts(
        prompt_style='llama_inst',  # or 'chatml', 'plain'
        include_constraints=True,
        extra_constraints=["Be concise and factual"]
    )
)

# Run the pipeline
result = run_enhanced_pipeline_v3(
    "Your input text here",
    config=config
)

print(result["text"])
print("Reflexive metrics:", result.get("metrics", {}).get("reflexive", {}))
```

### Command Line Usage

```bash
# Run with default settings
python -m dal.run_dal_v3 input.txt output.txt

# Enable reflexive revision with mini-generations
python -m dal.run_dal_v3 --use-reflexive --allow-mini-gen input.txt output.txt

# Customize SDT options
python -m dal.run_dal_v3 \
    --prompt-style chatml \
    --constraint "Be concise" \
    input.txt output.txt
```

## 🤝 Contributing

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) for details.

## Metrics and Monitoring

The reflexive revision engine collects detailed metrics that are included in the pipeline output:

```python
{
  "reflexive": {
    "issues_found": 2,
    "issues": [
      {
        "block_index": 1,
        "issue_type": "redundancy",
        "severity": "medium",
        "description": "Content overlaps with block 3",
        "related_blocks": [3]
      }
    ],
    "reordered": true,
    "new_order": [1, 0, 2],
    "blocks_modified": [0, 2],
    "transitions_added": 1,
    "mini_generations": 1,
    "duration": 0.87
  }
}
```

## Version History

### v1.3.0 (Current)
- Added Reflexive Revision Engine
- Enhanced SDT Injection with normalization and constraints
- Improved prompt engineering and formatting
- Comprehensive metrics collection

### v1.2.0
- Added local model support
- SDT-aware generation
- Pipeline v3 with metrics

### v1.1.0
- Initial release with core functionality
- Basic pipeline implementation
- SDT tag support

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Resources

- [DAL Documentation](docs/)
- [API Reference](docs/API.md)
  stitching.
* Add a reflexive revision pass that recomputes tags after stitching
  and loops back to refine the answer.

Contributions and experiments are welcome; please open issues or pull
requests to discuss improvements.