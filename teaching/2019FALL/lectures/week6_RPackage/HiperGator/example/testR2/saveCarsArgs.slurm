#!/bin/sh
#SBATCH --job-name=serial_job_test    # Job name
#SBATCH --account=phc6068             # your own sponser or account from this class
#SBATCH --qos=phc6068                 # your own sponser or account from this class
#SBATCH --mail-type=ALL               # Mail events
#SBATCH --mail-user=xx@xx.xx          # Where to send email	
#SBATCH --ntasks=1                    # Run on a single machine (node)
#SBATCH --cpus-per-task 1			  # Run on a single CPU
#SBATCH --mem=8gb                    # Memory limit
#SBATCH --time=04:00:00               # Time: hrs:min:sec
#SBATCH --output=serial_test_%j.out   # Output and error log 

pwd; hostname; date 

module load R 

echo "Running save cars script on a single CPU core" 

R --no-save --quiet --slave --args 1 < saveCarsArgs.R 

date
