# main function that sets up environments
# perform training loop

import json
import os
from collections import deque

import fire
import numpy as np
import torch
from unityagents import UnityEnvironment

from maddpg import MADDPG, ReplayBuffer


def save(save_path, maddpg, episode, scores):
    save_dict_list = []

    for i in range(2):
        save_dict = {
            "actor_params": maddpg.maddpg_agent[i].actor.state_dict(),
            "actor_optim_params": maddpg.maddpg_agent[i].actor_optimizer.state_dict(),
            "critic_params": maddpg.maddpg_agent[i].critic.state_dict(),
            "critic_optim_params": maddpg.maddpg_agent[i].critic_optimizer.state_dict(),
        }
        save_dict_list.append(save_dict)

    torch.save(
        save_dict_list,
        os.path.join(save_path + "/model_dir", "episode-{}.pt".format(episode)),
    )
    with open(os.path.join(save_path, "scores.json"), "w") as fp:
        fp.write(json.dumps(scores))


def check_solved(rolling_score, reward_this_episode, episode, model_dir):
    if np.mean(rolling_score) > 0.5:
        print(f"SOLVED in {episode} episodes")
        with open(f"{model_dir}/rewards.json", "w") as fp:
            fp.write(
                json.dumps(
                    {"agent0": reward_this_episode[0], "agent1": reward_this_episode[1]}
                )
            )
        return True
    return False


def print_status_update(
    print_every, episode, number_of_episodes, rolling_score, reward_this_episode, noise
):
    if episode % print_every == 0 or episode == number_of_episodes - 1:
        print(
            f"Episode: {episode}\tAverage Score: {np.mean(rolling_score)}\tScore: {rolling_score[-1]}\tAgent1 {reward_this_episode[0]}\tAgent2 {reward_this_episode[1]}\tNoise:{noise}"
        )


def main(
    env_path="/root/code/deep-reinforcement-learning/p3_collab-compet/Tennis_Linux_NoVis/Tennis.x86_64",
    gamma=0.99,
    tau=1e-3,
    # number of training episodes.
    # change this to higher number to experiment. say 30000.
    number_of_episodes=10000,
    batchsize=256,
    save_info=True,
    num_episodes_win_condition=100,
    noise_scale=1,
    # amplitude of OU noise
    # this slowly decreases to 0
    noise_reduction=0.999,
    # how many episodes before update
    episode_per_update=1,
    updates_per_round=2,
    print_every=1,
    nn_size=256,
    save_path=".",
    lr_actor=1.0e-4,
    lr_critic=1.0e-3,
    weight_decay=0,
    mu=0,
    theta=0.15,
    sigma=1.0,
    buffer_size=int(5e5),
    seed=2,
):

    config = {
        "gamma": gamma,
        "tau": tau,
        # number of training episodes.
        # change this to higher number to experiment. say 30000.
        "number_of_episodes": number_of_episodes,
        "batchsize": batchsize,
        "save_info": save_info,
        "num_episodes_win_condition": num_episodes_win_condition,
        "noise_scale": noise_scale,
        # amplitude of OU noise
        # this slowly decreases to 0
        "noise_reduction": noise_reduction,
        # how many episodes before update
        "episode_per_update": episode_per_update,
        "updates_per_round": updates_per_round,
        "print_every": print_every,
        "nn_size": nn_size,
        "save_path": save_path,
        "lr_actor": lr_actor,
        "lr_critic": lr_critic,
        "weight_decay": weight_decay,
        "mu": mu,
        "theta": theta,
        "sigma": sigma,
        "buffer_size": buffer_size,
        "seed": seed,
    }

    max_score = -10

    os.makedirs(save_path + "/model_dir", exist_ok=True)

    env = UnityEnvironment(file_name=env_path)

    brain_name = env.brain_names[0]

    env_info = env.reset(train_mode=True)[brain_name]
    num_agents = len(env_info.agents)

    buffer = ReplayBuffer(buffer_size, num_agents)

    # initialize policy and critic
    maddpg = MADDPG(
        env_info.vector_observations.shape[1],
        env.brains[brain_name].vector_action_space_size,
        config,
    )
    rolling_score = deque(maxlen=num_episodes_win_condition)
    scores = []

    # training loop
    # show progressbar

    # use keep_awake to keep workspace from disconnecting
    for episode in range(0, number_of_episodes):

        reward_this_episode = np.zeros(2)
        env_info = env.reset(train_mode=True)[brain_name]
        noise_scale *= noise_reduction
        maddpg.noise_reset()

        states = env_info.vector_observations

        while True:

            # explore = only explore for a certain number of episodes
            # action input needs to be transposed
            actions = maddpg.act(states, noise_scale=noise_scale)

            # step forward one frame
            env_info = env.step(actions)[brain_name]

            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done

            # add data to buffer
            buffer.add(states, actions, rewards, next_states, dones)
            states = next_states

            reward_this_episode += rewards

            if np.any(dones):
                break

        # update once after every episode_per_update
        if len(buffer) > batchsize and episode % episode_per_update == 0:
            for _ in range(updates_per_round):
                samples = buffer.sample(batchsize)
                maddpg.learn(samples, noise_scale=0.0)

        rolling_score.append(max(reward_this_episode[0], reward_this_episode[1]))
        scores.append(reward_this_episode.tolist())
        max_score = max(max_score, np.mean(rolling_score))

        print_status_update(
            print_every,
            episode,
            number_of_episodes,
            rolling_score,
            reward_this_episode,
            noise_scale,
        )

        if episode == number_of_episodes - 1:
            print(f"Max score ever found: {max_score}")

        # saving model
        if save_info:
            save(save_path, maddpg, episode, scores)

        if check_solved(
            rolling_score, reward_this_episode, episode, save_path + "/model_dir"
        ):
            break

    env.close()


if __name__ == "__main__":
    fire.Fire(main)
