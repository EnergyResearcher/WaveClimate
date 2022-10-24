#!/bin/bash
#SBATCH --partition=par-single
#SBATCH --time=45:00:00
#SBATCH --mem=60G
#SBATCH --ntasks=16
#SBATCH -e outputs/%j.err
#SBATCH -o outputs/%j.out


source $HOME/.conda/envs/ncdf/bin/activate

python no_hpo.py