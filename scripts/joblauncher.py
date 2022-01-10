import os
import sys
import glob
import subprocess

from production_variables import queue_limit
from production_variables import jobsdir
from production_variables import get_cut_and_num
from production_variables import queue_state_command
from production_variables import joblaunch_command

from time import sleep

jobs = glob.glob(jobsdir + "/*")
#no me hace falta ordenarlos, además que tendría que cambiar la key porque
#estos jobs no tienen la misma estructura... (la _ no está, ni hay num solo cutnum)
#jobs = sorted(jobs, key = get_cut_and_num)

def check_jobs(cmd, nmin=10, wait=1):
    j = nmin
    while j>nmin-1:
        j = subprocess.run(cmd, shell=True, capture_output=True)
        j = int(j.stdout)
        sleep(wait)

############
# LAUNCHER
############


for job in jobs:

	check_jobs(queue_state_command, nmin = queue_limit)

	#launch job
	cmd = joblaunch_command.format(job_filename = job)
	print("Launching job", job)
	subprocess.run(cmd, shell = True)
