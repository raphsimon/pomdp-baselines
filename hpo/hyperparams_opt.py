# -*- coding: future_fstrings -*-
import sys, os, time

t0 = time.time()
import numpy as np
import torch
import psutil
from ruamel.yaml import YAML
from absl import flags
from utils import system, logger
from pathlib import Path

from torchkit.pytorch_utils import set_gpu_mode
from policies.learner import Learner

import optuna
from optuna.storages import RetryFailedTrialCallback
from optuna.trial import TrialState
from optuna.study import MaxTrialsCallback
from sqlalchemy.pool import NullPool

"""
This script performs a hyperparameter search for the SACD algorithm.
What's expected as input is first and foremost a complete config, just
so that all the values can be set to perform some of the decisions.
Then, we sample the hyperparameters and create and instance of the
Learner class and train the agent. The reported score is the return
after the last performed evaluation during the training run. This 
value is then stored alongside other trial details in the database,
and also used to prune other trials.
"""

DATABASE_URL = os.environ.get("OPTUNA_DB_URL", "sqlite:////home/rsimon/optuna_hpo.db")

def optimize_hyperparameters(study_name, optimize_trial, n_trials=20, max_total_trials=None, n_jobs=1):
    # Add stream handler of stdout to show the messages
    #optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))

    logger.log(f"Provided database: {DATABASE_URL}")

    sqlite_timeout = 300
    engine_kwargs = None
    if "sqlite" in DATABASE_URL:
        engine_kwargs={
            'connect_args': {'timeout': sqlite_timeout},
        }
    elif "postgresql" in DATABASE_URL:
        engine_kwargs = {"poolclass": NullPool}

    storage = optuna.storages.RDBStorage(
        DATABASE_URL,
        engine_kwargs=engine_kwargs,
        heartbeat_interval=60,
        grace_period=120,
        failed_trial_callback=RetryFailedTrialCallback(max_retry=2),
    )
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        direction='maximize',
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=400)
    ) # No sampler is specified, so a default sampler (TPE) is used.
    
    if max_total_trials is not None:
        # Note: we count already running trials here otherwise we get
        #  (max_total_trials + number of workers) trials in total.
        counted_states = [
            TrialState.COMPLETE,
            TrialState.RUNNING,
            TrialState.PRUNED,
        ]
        completed_trials = len(study.get_trials(states=counted_states))
        if completed_trials < max_total_trials:
            return study.optimize(
                    optimize_trial,
                    n_trials=n_trials,
                    callbacks=[
                        MaxTrialsCallback(
                            max_total_trials,
                            states=counted_states,
                        )   
                    ],
                    n_jobs=n_jobs,  
                    gc_after_trial=True
                )
    else:
        return study.optimize(optimize_trial, n_trials=n_trials, n_jobs=n_jobs, gc_after_trial=True)


def suggest_sacd_params(trial: optuna.Trial):
    """
    Sampler for SAC hyperparams.

    :param trial:
    :return: dictionary with all the sampled hyperparameters.
    """
    gamma = trial.suggest_categorical("gamma", [0.9, 0.95, 0.975, 0.99])                                    # |V| = 4
    learning_rate = trial.suggest_categorical("learning_rate", [3e-05, 0.0001, 0.0003, 0.001, 0.003])       # |V| = 5
    buffer_size = trial.suggest_categorical("buffer_size", [int(1e4), int(1e5), int(1e6)])                  # |V| = 3
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])                                # |V| = 4
    sampled_seq_len = trial.suggest_categorical("sampled_seq_len", ["all", "batch_size"])                   # |V| = 2
    # This is how I understood things relating to the sampled sequence length
    # But I'm not sure I understand it fully. What does "all" mean? How long
    # is the sampled sequence length really then?
    if sampled_seq_len == "all":
        sampled_seq_len = -1
    else:
        sampled_seq_len = batch_size
    # The smaller num_updates_per_iter, the less often the policy is updated
    # Basically, for num_updates_per_iter = 0.004, the policy is updated every
    # 250 iterations.
    tau = trial.suggest_categorical("tau", [0.001, 0.005, 0.01, 0.02, 0.05])                                # |V| = 5
    observ_embedding_size = trial.suggest_categorical("observ_embedding_size", [8, 16, 32, 64])             # |V| = 4
    action_embedding_size = trial.suggest_categorical("action_embedding_size", [8, 16, 32, 64])             # |V| = 4
    entropy_target = trial.suggest_categorical("entropy_target", [0.9, 0.95, 0.975, 0.99])                  # |V| = 4

    hyperparams = {
        "gamma": gamma,
        "learning_rate": learning_rate,
        "buffer_size": buffer_size,
        "batch_size": batch_size,
        "tau": tau,
        "sampled_seq_len": sampled_seq_len,
        "observ_embedding_size": observ_embedding_size,
        "action_embedding_size": action_embedding_size,
        "seq_model": seq_model,
        "entropy_target": entropy_target
    }
    return hyperparams


def setup_logging(exp_id, pid, logger_formats, yaml_conf):
    os.makedirs(exp_id, exist_ok=True)
    log_folder = os.path.join(exp_id, system.now_str())
    if yaml_conf["eval"]["log_tensorboard"]:
        logger_formats.append("tensorboard")
    logger.configure(dir=log_folder, format_strs=logger_formats, precision=4)
    logger.log(f"preload cost {time.time() - t0:.2f}s")

    # os.system(f"cp -r policies/ {log_folder}") Don't need this for hpo
    # For the naming with the pid, we assume that every trial is run in a 
    # different process. Otherwise it doesn't help us.
    yaml.dump(yaml_conf, Path(f"{log_folder}/parameters_{pid}.yml")) # This line causes problems.
    key_flags = FLAGS.get_key_flags_for_module(sys.argv[0])
    logger.log("\n".join(f.serialize() for f in key_flags) + "\n")
    os.makedirs(os.path.join(logger.get_dir(), "save"), exist_ok=True)


if __name__ == '__main__':
    # Setting up the arguments to parse
    FLAGS = flags.FLAGS
    flags.DEFINE_string("cfg", None, "path to configuration file")
    flags.DEFINE_string("env", None, "env_name")
    flags.DEFINE_string("algo", None, '["td3", "sac", "sacd"]')
    flags.DEFINE_string("output", None, "base path for the output")

    flags.DEFINE_boolean("automatic_entropy_tuning", None, "for [sac, sacd]")
    flags.DEFINE_float("target_entropy", None, "for [sac, sacd]")
    flags.DEFINE_float("entropy_alpha", None, "for [sac, sacd]")

    flags.DEFINE_integer("cuda", None, "cuda device id")
    flags.DEFINE_boolean(
        "oracle",
        False,
        "whether observe the privileged information of POMDP, reduced to MDP",
    )
    flags.DEFINE_boolean("debug", False, "debug mode")
    flags.DEFINE_integer("trials", 20, "Number of trials to run")
    flags.DEFINE_integer("n_jobs", 1, "Number of jobs to run in parallel")
    flags.DEFINE_integer("max_total_trials", None, "Maxumim number of trials for the study")
    flags.DEFINE_integer("n_evals", 10, "Number of evaluations to perform during trial")
    flags.DEFINE_integer("num_iters", None, "Number of iterations to perform in the environment during one trial")
    flags.DEFINE_integer("num_init_rollouts", None, "Number of initial rollouts before training")


    flags.FLAGS(sys.argv)

    yaml = YAML()
    v = yaml.load(open(FLAGS.cfg))

    # overwrite config params
    if FLAGS.env is not None:
        v["env"]["env_name"] = FLAGS.env
    if FLAGS.algo is not None:
        v["policy"]["algo_name"] = FLAGS.algo
    if FLAGS.n_evals is not None:
        v["env"]["num_eval_tasks"] = FLAGS.n_evals
    if FLAGS.num_iters is not None:
        v["train"]["num_iters"] = FLAGS.num_iters
    if FLAGS.num_init_rollouts is not None:
        v["train"]["num_init_rollouts_pool"] = FLAGS.num_init_rollouts
    
    if FLAGS.output is not None:
        # Use it as prefix for the output path
        output_base = FLAGS.output
    else:
        # We just leave it blank and let the code do what it always did.
        output_base = ''

    seq_model, algo = v["policy"]["seq_model"], v["policy"]["algo_name"]
    assert seq_model in ["mlp", "lstm", "gru", "lstm-mlp", "gru-mlp"]
    assert algo in ["td3", "sac", "sacd"]

    if FLAGS.automatic_entropy_tuning is not None:
        v["policy"][algo]["automatic_entropy_tuning"] = FLAGS.automatic_entropy_tuning
    if FLAGS.entropy_alpha is not None:
        v["policy"][algo]["entropy_alpha"] = FLAGS.entropy_alpha
    if FLAGS.target_entropy is not None:
        v["policy"][algo]["target_entropy"] = FLAGS.target_entropy
    if FLAGS.cuda is not None:
        v["cuda"] = FLAGS.cuda
    if FLAGS.oracle:
        v["env"]["oracle"] = True

    env_name = v["env"]["env_name"]
    study_name = v["study_name"]

    def hyperparams_search(n_trials=50, max_total_trials=None, n_jobs=1):

        # system: device, threads, seed, pid
        seed = np.random.randint(0, 1e9)
        system.reproduce(seed)
        v["seed"] = seed

        np.set_printoptions(precision=3, suppress=True)
        torch.set_printoptions(precision=3, sci_mode=False)

        set_gpu_mode(torch.cuda.is_available() and v["cuda"] >= 0, v["cuda"])

        print(f"{torch.get_num_threads()=}")
        print(f"{torch.get_num_interop_threads()=}")

        def optimize_trial(trial):
            # Here we need to sample hyperparams and run the training
            sampled_hyperparams = suggest_sacd_params(trial)

            # Place the parameters in v
            v["train"]["buffer_size"] = sampled_hyperparams["buffer_size"]
            v["train"]["batch_size"] = sampled_hyperparams["batch_size"]
            v["train"]["sampled_seq_len"] = sampled_hyperparams["sampled_seq_len"]
            v["policy"]["seq_model"] = sampled_hyperparams["seq_model"]
            v["policy"]["observ_embedding_size"] = sampled_hyperparams["observ_embedding_size"]
            v["policy"]["action_embedding_size"] = sampled_hyperparams["action_embedding_size"]
            v["policy"]["learning_rate"] = sampled_hyperparams["learning_rate"]
            v["policy"]["gamma"] = sampled_hyperparams["gamma"]
            v["policy"]["tau"] = sampled_hyperparams["tau"]
            v["policy"]["sacd"]["target_entropy"] = sampled_hyperparams["entropy_target"]

            # logs
            if FLAGS.debug:
                exp_id = output_base + "debug/hpo/"
                logger_formats = ["stdout", "log", "csv"]
            else:
                exp_id = output_base + "logs/hpo/"
                logger_formats = ["stdout", "log", "csv"]

            # Setup the experiment name and logging according to the sampled
            # hyperparameters
            env_type = v["env"]["env_type"]
            env_name = v["env"]["env_name"]
            exp_id += f"{env_type}/{env_name}/"
            
            if "rnn_num_layers" in v["policy"]:
                rnn_num_layers = v["policy"]["rnn_num_layers"]
                if rnn_num_layers == 1:
                    rnn_num_layers = ""
                else:
                    rnn_num_layers = str(rnn_num_layers)
            else:
                rnn_num_layers = ""
            exp_id += f"{algo}_{rnn_num_layers}{v['policy']['seq_model']}"
            if "separate" in v["policy"] and v["policy"]["separate"] == False:
                exp_id += "_shared"
            exp_id += "/"

            if algo in ["sac", "sacd"]:
                if not v["policy"][algo]["automatic_entropy_tuning"]:
                    exp_id += f"alpha-{v['policy'][algo]['entropy_alpha']}/"
                elif "target_entropy" in v["policy"]:
                    exp_id += f"ent-{v['policy'][algo]['target_entropy']}/"

            exp_id += f"gamma-{v['policy']['gamma']}/"
            pid = str(os.getpid())
            setup_logging(exp_id, pid, logger_formats, v)

            # start training
            learner = Learner(
                env_args=v["env"],
                train_args=v["train"],
                eval_args=v["eval"],
                policy_args=v["policy"],
                seed=seed,
            )
            logger.log(
                f"total RAM usage: {psutil.Process().memory_info().rss / 1024 ** 3 :.2f} GB\n"
            )
            score = learner.train(trial)

            return score

        optimize_hyperparameters(study_name, optimize_trial, n_trials, max_total_trials, n_jobs)

    hyperparams_search(FLAGS.trials, FLAGS.max_total_trials, FLAGS.n_jobs)
