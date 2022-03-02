import os
import glob
import re


#Directory that will contain the created jobs/configs and the output files
basedir = os.path.expandvars("/mnt/lustre/scratch/home/usc/ie/mpm/beersh_labelling_4mm_fid_corrected")

#Part of the in/out filename
tag = "Tl208_NEW_v1_03_01_nexus_v5_03_04"

#number of jobs to launch (max is 200 in cesga)
queue_limit = 198

#directory of the job and config templates, write / at the end
jobTemp_dir    = os.path.expandvars("/home/usc/ie/mpm/data_labelling/templates/")
configTemp_dir = os.path.expandvars("/home/usc/ie/mpm/data_labelling/templates/")

#name for the in/out files and names of the job/config templates
in_filename         = "{tag}_cut{cutnum}.beersheba_*.root.h5"
out_filename        = "{tag}_cut{cutnum}.beersheba_label_4mm_fid_corrected.h5"
jobTemp_filename    = "jobTemplate.sh"
configTemp_filename = "configTemplate.conf"

#directory of the input files to be processed, remember to write / at the end
indir = "/mnt/lustre/scratch/home/usc/ie/mpm/data_production_chain/prod/beersheba/"

#path of the script to run
scriptdir = "/home/usc/ie/mpm/data_labelling/scripts/create_labelled_dataset.py"

#function to get the cutnum and the num of each file
def get_cut_and_num(filename):
	cut = filename.split("/")[-1].split("_")[-2].split(".")[0]
	num = filename.split("/")[-1].split("_")[-1].split(".")[0]
	match = re.match(r"([a-z]+)([0-9]+)", cut, re.I)
	if match:
	        items = match.groups()
	cutnum = items[-1]
	return cutnum, num

#checks if a directory exists, if not, it creates it
def checkmakedir(path):
	if os.path.isdir(path):
		print('hey, directory already exists!:\n' + path)
	else:
		os.makedirs(path)
		print('creating directory...\n' + path)

#this function creates the output tree of directories (all the jobs and configs, the data
#production and the logs
def create_out_dirs():
        proddir = basedir + "/prod/"
        jobsdir = basedir + "/jobs/"
        confdir = basedir + "/config/"
        logsdir = basedir + "/logs/"
        checkmakedir(jobsdir)
        checkmakedir(confdir)
        checkmakedir(logsdir)
        checkmakedir(proddir)
        
        return proddir, jobsdir, confdir, logsdir

proddir, jobsdir, confdir, logsdir = create_out_dirs()


##############
# INPUT FILES
##############

#takes all the .h5 files in the specified indir. we will make a loop on them, but as I
#want to grab all the files with the same cut, in the job creator script I will assure that
#once one cut job/config is created, no other configs are created
files_in = glob.glob(indir + "/" + tag + "*.h5") #para que me haga todo poner *.h5 al final

#for f in files_in: 
#	check_filename_structure(f)

#sorts all the files, first in the cut number and then in the number
files_in = sorted(files_in, key = get_cut_and_num)


##############
# JOB LAUNCH
##############

#commands for CESGA
queue_state_command = "squeue |grep usciempm |wc -l"
joblaunch_command   = "sbatch {job_filename}"
jobtime             = "4:00:00"
