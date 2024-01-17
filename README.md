# Active Learning in Image Data Research Project

## Project Overview
This project explores the application of active learning (AL) in image data scenarios characterized by limited sample sizes and constrained data querying opportunities. It focuses on the practical challenges of limited data availability in real-world settings and investigates uncertainty-based methods within deep learning frameworks.

### Research Questions
- **RQ1**: Efficiency of Querying Strategies: Examines the effectiveness of Random, Maximum Entropy, and BALD AL strategies in handling limited sample sizes and noise-influenced image datasets.
- **RQ2**: Adaptability and Performance of Deep Learning Models: Assesses whether LeNet or ResNet18 demonstrates superior adaptability and effectiveness in AL environments with noisy image data.

## Getting Started

### Dependencies
- Python 3.9
- PyTorch
- NumPy
- scikit-learn
- tqdm
- IPython Debugger (ipdb)
- Weights & Biases (wandb)

### Installation
Clone the repository and install the required packages:
```bash
git clone https://github.com/lucapantea/active-learning.git 
cd active-learning
```

### Environment Setup
Create the Conda environment using the provided `environment.yml` file:
```bash
conda env create -f environment.yml
```

## Configuration Details
The `config.py` file contains various configuration settings and defaults for running the experiments in this project. Here's a breakdown of the key configurations:

1. **Dataset Settings**
   - `dataset`: The dataset to use for the experiments (default: 'mnist').
   - `data_dir`: Directory where the dataset is stored (default: 'data').
   - `num_valid`: Number of validation samples (default: 1000).

2. **Training Settings**
   - `batch_size`: Batch size for training (default: 64).
   - `epochs`: Number of training epochs (default: 10).
   - `num_workers`: Number of workers for data loading (default: 0).
   - `seed`: Seed for random number generators (default: 42).

3. **Model and Learning Settings**
   - `model`: The deep learning model to use (default: 'lenet').
   - `lr`: Learning rate for the optimizer (default: 0.001).

4. **Active Learning Strategy Settings**
   - `strategy`: The active learning strategy to use (default: 'random').
   - `n_init_labeled`: Number of initially labelled samples (default: 10000).
   - `n_query`: Number of samples to query in each round (default: 1000).
   - `n_round`: Number of active learning rounds (default: 10).

5. **Noise Settings**
   - `noise`: The type of noise to add to the dataset, can be 'gaussian', 'salt_and_pepper', or 'none' (default: 'none').
   - `noise_rate`: Rate of noise to apply to the dataset (default: 0.0).

6. **Experiment and Debug Settings**
   - `wandb`: Flag to enable Weights & Biases logging (default: True).
   - `experiment`: Flag to run the project in experimental mode (default: False).
   - `debug`: Flag to enable debug mode (default: False).

7. **Logging Configuration**
   - `LOG_LEVEL`: Default logging level.
   - `LOG_FORMAT`: Format for logging messages.

The `get_logger` function in `config.py` is used to set up logging with the specified configuration. The logger named 'ProjectLogger' is initialized as a singleton for use across the project.

Please refer to the `config.py` file for any additional details and to modify these settings as per your experiment requirements.

## Usage
Run the project using:
```bash
python main.py [arguments]
```
Specify arguments according to the `config.py` file for custom configurations.

## Experiment Details
The project conducts experiments to test different active learning strategies and deep learning models under various conditions, focusing on noise effects and malicious labelling.

## Contributing
Contributions are welcome. Please follow standard GitHub pull request processes for proposing changes.

## Contact
If you have any questions or contributions, please contact Luca Pantea at luca.p.pantea@gmail.com.

## Acknowledgements
This project is part of the research project for the Human in the Loop Machine Learning course at the University of Amsterdam. We acknowledge the use of public datasets and open-source software in this project.

## Full Report
For detailed information and results, please refer to the attached project report: `HITL_ML_Project.pdf`.
