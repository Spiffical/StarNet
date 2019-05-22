The following will create a Compute Canada job script in `$SCRATCH/jobs/todo` named `job.sh` (as defined by `--output_path`).
It is also necessary to define the path to the h5 training dataset given by `--data_path`, the number of training examples 
given by `--num_train`, etc.

```
python /path/to/starnet/compute-canada/create_trainStarNet_jobscript.py \
--output_path $SCRATCH/jobs/todo/job.sh \
--data_path /path/to/training_dataset.h5 \ # test
--num_train 200000 \
--targets teff logg M_H a_M \
--spec_key spectra_starnetnorm \
--save_folder trained_model/ \
--batch_size 32 \
--epochs 35 \
--zeros 0.10 \
--telluric_file /path/to/data/telluric_lines.txt \
--finetune_model weights.best.h5 \
--model_to_train StarNet2017 
```

Once this job script is created, you can then run `queue_cc.py` from [compute-canada-goodies](https://github.com/vcg-uvic/compute-canada-goodies)
which will queue any jobs it finds in your `jobs/todo` directory. An example of how to use this script to queue up to 10 jobs
with 16 CPUs, 1 GPU, and 32GB per job is as follows:

```
python compute-canada-goodies/python/queue_cc.py \
--account def-sfabbro \
--num_jobs 10 \
--num_runs 4 \
--num_cpu 16 \
--num_gpu 1 \
--mem 32G
```
where `--account` is the Compute Canada account you are associated with, and `--num_runs` is the number of times a particular 
job will run: for example, if the value is 4 then one such job will be scheduled, by default, for increments of *3 hours each*
(for minimum turnaround time) with each subsequent iteration beginning after the previous one. This is used in our case to
train a StarNet model and save the current best model weights to `weights.best.h5`, which each iteration of a job will load 
and fine-tune.
