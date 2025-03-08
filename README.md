# Mathematical LaTeX Document Generator

A hierarchical multi-agent system for automatically generating high-quality mathematics papers in LaTeX. This system uses a structured approach with multiple layers of specialized agents working together to create coherent, accurate mathematical content.

## 🔍 Overview

This project implements a collaborative bot system where each bot has a specialized role in the document creation process:

- **L1 Bot**: Handles high-level document construction and structure
- **L2 Bot**: Manages section-level writing 
- **L3 Bot**: Manages subsection-level writing
- **L4 Bot**: Creates mathematical content (theorems, lemmas, examples, etc.)
- **Review Bot**: Reviews mathematical content for accuracy

The system also includes a tournament mechanism to generate multiple candidate outputs and select the best ones through a round-robin competition.

## 🏗️ Architecture

The system follows a hierarchical structure:

```
L1 Bot (Document)
  └── L2 Bot (Section)
       └── L3 Bot (Subsection)
            └── L4 Bot (Mathematical Content)
                 └── Review Bot (Content Verification)
```

Each level operates with its own context and responsibilities:

1. **L1 Bot**: Structures the overall document, delegates section writing to L2 bots
2. **L2 Bot**: Creates sections, delegates subsection writing to L3 bots
3. **L3 Bot**: Creates subsections, delegates mathematical content to L4 bots
4. **L4 Bot**: Constructs mathematical environments (theorems, proofs, examples)
5. **Review Bot**: Performs multi-step review of mathematical content

## 🔧 Installation

### Prerequisites

- Python 3.10 or higher
- Google Cloud API key with access to Gemini models

### Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/justincasher/math_recursion
   cd math_recursion
   ```

2. Install dependencies:
   ```bash
   pip install google google-generativeai
   ```

3. Configure your API key:
   - Set the `GEMINI_API_KEY` environment variable:
     ```bash
     export GEMINI_API_KEY="your-api-key"
     ```
   - Or directly edit the `config.py` file (not recommended for security)

## 📝 Usage

1. Place your initial LaTeX document or problem statement in a text file
2. Update the `config.py` file to point to your input and output files:
   ```python
   DOC_LOAD = "your_input_file.txt"
   DOC_SAVE = "your_output_file.txt"
   ```
3. Run the system:
   ```bash
   python main.py
   ```
4. If `VISUALIZER` is enabled (default), a GUI will appear showing the document creation progress

## 🖥️ Visualizer

The system includes a visualization tool to monitor the document creation process:

- Left panel: Shows the current document/section/subsection content
- Right panel: Displays metadata, instructions, and navigation buttons
- Pause/Resume button: Controls the update process

## ⚙️ Configuration

Edit `config.py` to customize the system behavior:

```python
# Basic configuration
DEFAULT_MODEL_NAME = "gemini-2.0-flash"  # LLM model to use
PARALLEL = True                          # Enable parallel processing
VISUALIZER = True                        # Enable visualization GUI

# Iteration limits
L1_REASONING_STEPS = 5                   # Maximum L1 bot iterations
L2_REASONING_STEPS = 2                   # Maximum L2 bot iterations
L3_REASONING_STEPS = 2                   # Maximum L3 bot iterations
L4_REASONING_STEPS = 2                   # Maximum L4 bot iterations

# Number of bots at each level for tournament selection
NUM_L2_BOTS = 1                          # Number of L2 bots
NUM_L3_BOTS = 1                          # Number of L3 bots
NUM_L4_BOTS = 3                          # Number of L4 bots
NUM_REVIEWERS = 3                        # Number of review bots

# IO configuration
DOC_LOAD = "test_problems/Putnam_2024_A6.txt"  # Input file
DOC_SAVE = "math_new.txt"                      # Output file
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

For questions or suggestions, please contact me at justinchadwickasher@gmail.com

## 📄 License

This project is licensed under the MIT License - see below for details:

```
MIT License

Copyright (c) 2025 [Your Name or Organization]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

The MIT License allows you to freely use, modify, and distribute this code in both private and commercial projects. You just need to include the original copyright notice and license text.

## 🙏 Acknowledgements

This project uses Google's Gemini API for language model capabilities.
