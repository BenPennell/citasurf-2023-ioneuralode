# How do?

I have a particular setup that vibes with how my brain works. Here is an insight, for those brave enough, into my brain so you can run my sunnyvale jobs. 

Here's the scheme:
- Have a script to run with command line arguments
- `.pbs` file will send the script as a job to the cluster with particular arguments
- Use a julia file to mass produce `.pbs` files and run each of them with different arguments

## 1. Test script
```bash
    julia -t 12 --project=/fs/lustre/scratch/bpennel/github /fs/lustre/scratch/bpennel/jobs/test_job.jl 5 cats
```
Which outputs
```
    Running on 12 threads!
    ["5", "cats"]
```
It is important to note that the julia environment is in `./github` and the files jobs are stored in `./jobs`. Though, all commands will be executed from a designated output folder of your choice, for example `/fs/lustre/scratch/bpennel/run_jobs_here`. Do not run it in `/bpennel`, as you will clutter that directory. To run your command, simply cd into the designated output folder and run your command from there. You want this to be your working directory for the output to go there. This should also be the directory to save files to in your jobs.

## 2. Test `.pbs` file
note that we're using `workq` queue, you can see this declared on line 7 of `test_pbs.pbs`. Perchance you will, perchance you won't want to use this! `starq` has the fanciest stuff if you want to be fancy like that, up to you. Check this for more info:

`https://wiki.cita.utoronto.ca/index.php/Sunnyvale_Upgrade_June_2022`

Notice that `test_pbs.pbs` is also in the `run_jobs` folder and it contains two simple commands (after the PBS stuff)
```bash
    cd /fs/lustre/scratch/bpennel
    julia -t 12 --project=./github ./jobs/test_job.jl 5 cats
```
So you can see that it's just a script that does precicely what we did in step 1. You can run it with
```bash
    qsub /fs/lustre/scratch/bpennel/jobs/test_pbs.pbs
```
Which, if you run it correctly, should output its id:
```bash
    247605.bob
``` 
And you can use `qstat` (and scroll all the way to the bottom) to see your job! Alternatively, make use of these:
```bash
    qstat $247605
```
```bash
    qstat -u bpennel
```
Once it's done you should see an output in your working directory `test_pbs.pbs.o247605`. Congratulations, you ran a job on Sunnyvale! Your parents must be so proud.

## 3. Expand, expand, expand
Here's what you want to do. Create three folders in `/bpennel`, each will have a different use:
- `pbs_dump` for the julia script to throw all the generated pbs files into
- `run_jobs_here` file to cd into and call the generator file from. This will store the (blank) returns from the job
- `Recfast_Batches` the ACTUAL output folder for the files that we save. 

There is a **generator** file in `/fs/lustre/scratch/bpennel/jobs` called `batch_submission.jl`. This file will create and dump pbs files into `pbs_dump` and run them. The run jobs will then output an empty file (unless there is an error) into the active directory `run_jobs_here` and save the results of the script into `Recfast_Batches`. All the names can be changed, of course, to suit your individual needs. This is all handled within `batch_submission.jl` To run the file, it is extremely straight forward, simply cd into the chosen active directory (such as `run_jobs_here`) and call the file:

```bash
cd /fs/lustre/scratch/bpennel/run_jobs_here
julia /fs/lustre/scratch/bpennel/jobs/batch_submission.jl
```

and that's it!