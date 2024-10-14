#!/bin/bash

#SBATCH --account=def-accountname
#SBATCH --mem=200G
#SBATCH --cpus-per-task=10
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=5
#SBATCH --time=40:50:0
#SBATCH --mail-user=<xxxxxxxxx@gmail.com>
#SBATCH --mail-type=ALL

python Train_icecream.py

#trap finish EXIT
