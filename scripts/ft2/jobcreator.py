from production_variables import *

in_filename_structure  = "{dir}" + in_filename
out_filename_structure = "{dir}" + out_filename

if __name__ == "__main__":
    
    #full path of the job template
    job_temp = jobTemp_dir + jobTemp_filename
    
    #opens the template to copy its content to the new files
    job_file = open(job_temp).read() 

    for f in files_in:
        cutnum, num = get_cut_and_num(f)

        #creates the new job file full path
        job = jobsdir + "cut{cutnum}.job".format(cutnum = cutnum) 

        #checks if the job exists so no repeated jobs are created
        if os.path.exists(job):
            continue

        #completes the input/output files with the directory, tag and cutnum (as we only require this)
        files_in = in_filename_structure.format(dir    = indir,
                                                tag    = tag,
                                                cutnum = cutnum)
        file_out = out_filename_structure.format(dir    = proddir,
                                                 tag    = tag,
                                                 cutnum = cutnum)
        #full path of the config template
        config_temp = configTemp_dir + configTemp_filename

        #opens the template to copy its content to the new created files
        config_file = open(config_temp).read()

        #creates the new config file full path
        config = confdir + "cut{cutnum}.conf".format(cutnum = cutnum)

        #we open this new config file to write the information
        with open(config, "w") as config_write:
            #we write using the config template and formatting the specific information in each case
            config_write.write(config_file.format(files_in = files_in,
                                                  file_out = file_out))
            
        #we create the commands to be written in the job file, such as the scripth path
        commands = "python {script_directory} {config_directory}".format(script_directory = scriptdir,
                                                                         config_directory = config)
        #we open the new job file to write the information
        with open(job, "w") as job_write:
            #we write using the job template and formatting the specific information in each case
            job_write.write(job_file.format(jobtime = jobtime,
                                            jobname = "cut{cutnum}".format(cutnum = cutnum),
                                            logfilename = logsdir + "cut{cutnum}.log".format(cutnum = cutnum),
                                            errfilename = logsdir + "cut{cutnum}.err".format(cutnum = cutnum),
                                            commands = commands))
