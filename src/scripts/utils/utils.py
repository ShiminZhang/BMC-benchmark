import os

DEBUG = False

def run_slurm_job_wrap(cmd, output, job_name,wait_id=None,mem="16g", time="20:00:00"):
    if DEBUG:
        print(f"Running command: {cmd}")
        os.system(cmd)
        return
    activate_python = "source ../../general/bin/activate"
    wrap = f"{activate_python} && {cmd}"
    # os.system(f"sbatch --job-name={job_name} --output={output} --mem={mem} --time={time} --wrap=\"{wrap}\"")
    if wait_id is None: 
        full_cmd = f"sbatch --job-name={job_name} --output={output} --mem={mem} --time={time} --wrap=\"{wrap}\""
    else:
        full_cmd = f"sbatch --dependency=afterok:{wait_id} --job-name={job_name} --output={output} --mem={mem} --time={time} --wrap=\"{wrap}\""
    # print(full_cmd)
    job_id = os.popen(full_cmd).read().split()[-1]
    return job_id


