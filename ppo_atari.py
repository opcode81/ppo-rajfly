import argparse
import json
import os
import pathlib
import time
import uuid

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.distributions import Categorical, Distribution
from torch.optim.lr_scheduler import LambdaLR

from atari_network import DQN, layer_init, scale_obs
from atari_wrapper import make_atari_env
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.policy import PPOPolicy
from tianshou.trainer import OnpolicyTrainer
from tianshou.utils.net.common import ActorCritic
from tianshou.utils.net.discrete import Actor, Critic


def actor_init(layer):
    if isinstance(layer, nn.Linear):
        torch.nn.init.orthogonal_(layer.weight, 0.01)
        torch.nn.init.constant_(layer.bias, 0.0)


def critic_init(layer):
    if isinstance(layer, nn.Linear):
        torch.nn.init.orthogonal_(layer.weight, 1)
        torch.nn.init.constant_(layer.bias, 0.0)


def train_atari(args: argparse.Namespace):
    from tianshou.utils import logging
    logging.configure()

    # make env
    env, train_envs, test_envs = make_atari_env(
        task=f"{args.env}NoFrameskip-v4", seed=args.seed, training_num=8, test_num=8
    )

    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n

    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = DQN(
        *state_shape,
        action_shape,
        device=device,
        features_only=True,
        output_dim=512,
        layer_init=layer_init,
    )
    net = scale_obs(net)
    actor = Actor(net, action_shape, softmax_output=False, device=device)
    critic = Critic(net, device=device)
    actor.last.apply(actor_init)
    critic.last.apply(critic_init)

    optim = torch.optim.Adam(
        ActorCritic(actor, critic).parameters(), lr=2.5e-4, eps=1e-5
    )

    # decay learning rate to 0 linearly
    step_per_collect = 128 * 8
    step_per_epoch = round(100000 // step_per_collect) * step_per_collect
    epoch = int(10000000 // step_per_epoch)
    max_update_num = np.ceil(step_per_epoch / step_per_collect) * epoch
    lr_scheduler = LambdaLR(optim, lr_lambda=lambda e: 1 - e / max_update_num)

    def dist(logits: torch.Tensor) -> Distribution:
        return Categorical(logits=logits)

    # policy
    policy: PPOPolicy = PPOPolicy(
        actor=actor,
        critic=critic,
        optim=optim,
        dist_fn=dist,
        action_space=env.action_space,
        eps_clip=0.1,
        dual_clip=None,
        value_clip=True,
        advantage_normalization=True,
        recompute_advantage=False,
        vf_coef=0.5,
        ent_coef=0.01,
        max_grad_norm=0.5,
        gae_lambda=0.95,
        discount_factor=0.99,
        reward_normalization=False,
        deterministic_eval=False,
        observation_space=env.observation_space,
        action_scaling=False,
        lr_scheduler=lr_scheduler,
    ).to(device)

    train_buffer = VectorReplayBuffer(
        128 * 8,
        buffer_num=len(train_envs),
        ignore_obs_next=True,
        save_only_last_obs=True,
        stack_num=4,
    )

    train_collector = Collector(
        policy, train_envs, train_buffer, exploration_noise=True
    )
    test_collector = Collector(policy, test_envs, exploration_noise=True)

    from tianshou.highlevel.logger import LoggerFactoryDefault
    logger_factory = LoggerFactoryDefault()
    logger = logger_factory.create_logger(
        log_dir="log",
        experiment_name=f"rajfly-{args.env}",
        run_id=None,
    )

    start_time = time.time()

    # train
    result = OnpolicyTrainer(
        policy=policy,
        max_epoch=epoch,
        batch_size=256,
        train_collector=train_collector,
        test_collector=test_collector,
        buffer=None,
        step_per_epoch=step_per_epoch,
        repeat_per_collect=4,
        episode_per_test=8,
        update_per_step=1.0,
        step_per_collect=step_per_collect,
        episode_per_collect=None,
        train_fn=None,
        test_fn=None,
        stop_fn=None,
        save_best_fn=None,
        save_checkpoint_fn=None,
        resume_from_log=False,
        reward_metric=None,
        logger=logger,
        verbose=True,
        show_progress=True,
        test_in_train=False,
    ).run()

    train_end_time = time.time()

    progress_df = pd.DataFrame(logger.progress_data)
    progress_df.to_csv(os.path.join(args.path, "progress.csv"), index=False)

    # eval
    policy.eval()
    test_collector = Collector(policy, test_envs, exploration_noise=False)
    result = test_collector.collect(n_episode=100)
    eval_end_time = time.time()
    args.eval_mean_reward = result.returns_stat.mean
    args.training_time_h = ((train_end_time - start_time) / 60) / 60
    args.total_time_h = ((eval_end_time - start_time) / 60) / 60


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-g",
        "--gpu",
        type=int,
        help="Specify GPU index",
        default=0,
    )
    parser.add_argument(
        "-e",
        "--env",
        type=str,
        help="Specify Atari or MuJoCo environment w/o version",
        default="Pong",
    )
    parser.add_argument(
        "-t",
        "--trials",
        type=int,
        help="Specify number of trials",
        default=5,
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        help="Random seed",
        default=23,
    )
    args = parser.parse_args()
    seed = args.seed
    for _ in range(args.trials):
        args.id = uuid.uuid4().hex
        args.path = os.path.join("trials", "ppo", args.env, args.id)
        args.seed = seed

        # create dir
        pathlib.Path(args.path).mkdir(parents=True, exist_ok=True)

        # set gpu
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu}"

        train_atari(args)

        # save trial info
        with open(os.path.join(args.path, "info.json"), "w") as f:
            json.dump(vars(args), f, indent=4)

        seed += 2