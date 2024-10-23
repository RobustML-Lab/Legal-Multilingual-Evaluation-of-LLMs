#!/bin/bash

#SBATCH --job-name="MultiEurlex"
#SBATCH --time=10:00:00            # Set an estimated time for your job (1 hour)
#SBATCH --ntasks=1                 # Using 1 task since this is not an MPI-parallel job
#SBATCH --cpus-per-task=24          # Number of CPUs per task (adjust based on your job requirements)
#SBATCH --partition=gpu        # Default partition to use
#SBATCH --mem=32GB                 # Adjust memory based on the dataset and Hugging Face model requirements
#SBATCH --gpus-per-task=1               # Request one GPU
#SBATCH --account=Education-EEMCS-BSc-TI       # Set your account here

# Load necessary modules
module load 2023r1
module load python/3.8.12          # Loads a compatible Python version
module load py-numpy               # Load numpy for array operations
module load cuda/11.6

# Optionally, create a virtual environment (if not done yet)
# You can skip this if your environment is already set up
python3 -m venv myenv
source myenv/bin/activate

# Upgrade pip and install necessary Python libraries
pip install --upgrade pip

# Run the Python script
srun python evaluation_classifiers_multilingual.py | tee output.log

