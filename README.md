# Adam: A Modular LLM Application

Adam is a compact yet powerful language model (LLM) application designed for flexibility and efficiency (~500 MB). Built on the GPT-2 architecture, Adam allows you to fine-tune and adapt the model to meet your specific needs. Whether you need a basic solution or an advanced setup, Adam can scale up to GPT-3.5, provided you have a capable machine and the necessary dataset.

### Key Features

- **End-to-End Customization:** Modify the architecture and fine-tune the model to suit your goals. Adam is highly adaptable for a variety of LLM applications.
  
- **Fine-Tuning for Specific Tasks:** Tailor the model's behavior by training it on your own dataset for a more accurate and targeted performance.

- **Modular Design:** Adam’s architecture is modular, enabling you to swap out components easily. If you prefer a web interface over the terminal, you can replace the CLI layer with a web API.

- **Runs Locally:** The entire application resides on your machine, ensuring full control over your environment and data.

### Requirements

The app requires Nvidia GPU to run. It is recommended to have VRAM of 16 GB or higher. For storage you'll need at least 1.5 GB for torch.

### Installation

It is highly recommended to install and run the app inside virtual environment. For more information, please see [virtual environment](https://docs.python.org/3/library/venv.html) from python documentation. Download the repo and then run

```bash
pip3 install -e .
```

to install the dependencies first.

### Terminal Interface Commands

| Command    | Description                                                                 | Options/Arguments                                                                                                      |
|------------|-----------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------|
| `version`  | Prints the current version of Adam.                                          | None                                                                                                                   |
| `pretrain` | Pretrains Adam's base model. Requires a file path, or optionally a directory. | - `--file` / `-f`: Path to a specific file (optional).<br> - `--dir` / `-d`: Directory for multiple text files (default: current directory).<br> - `--epochs` / `-e`: Number of training epochs (default: 10). |
| `train`    | Trains the model with a given file.                                          | - `--file` / `-f`: Path to the training file.<br> - `--epochs` / `-e`: Number of training epochs (default: 10).        |
| `generate` | Generates text based on the provided input.                                 | - `<text>`: The input text for text generation.                                                                         |
| `ask`      | Asks a question to the model and gets a response.                           | - `<question>`: The question to ask the model.                                                                          |

### Training Datasets

Adam uses two distinct training datasets: one for the base model and another for fine-tuning based on your specific needs.

- **Base Model**: This dataset is responsible for building the core vocabulary and prediction capabilities for your application. Currently, the base model is being trained using open-source texts from the Gutenberg Project.

- **Instruction Dataset**: This dataset is used to fine-tune the model for better response generation. It’s based on Stanford's Alpaca-style templates and stored under the `instructions` directory.

You can easily upload your own datasets for both the base model and instructions. Just place your files in the appropriate folders, and then use the terminal commands with the directory flags to train on them all at once.
