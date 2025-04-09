import json
import numpy as np
from os.path import join
import pdb

from diffuser.guides.policies import Policy
import diffuser.datasets as datasets
import diffuser.utils as utils


class Parser(utils.Parser):
    dataset: str = 'pointmaze-umaze-v2'
    config: str = 'config.maze2d'
num_success = 0

args = Parser().parse_args('plan')
diffusion_experiment = utils.load_diffusion(args.logbase, args.dataset, args.diffusion_loadpath, epoch=args.diffusion_epoch)

diffusion = diffusion_experiment.ema
dataset = diffusion_experiment.dataset
renderer = diffusion_experiment.renderer

policy = Policy(diffusion, dataset.normalizer)
for i in range(50):

    #---------------------------------- setup ----------------------------------#


    # logger = utils.Logger(args)

    env = datasets.load_environment(args.dataset, reset_target=False)

    #---------------------------------- main loop ----------------------------------#
    observation, _ = env.reset()

    if args.conditional:
        print('Resetting target')
        env.set_target()

    ## set conditioning xy position to be the goal
    target = np.array(observation['desired_goal'])
    print("Target")
    print(target)
    print(i)
    cond = {
        diffusion.horizon - 1: np.array([*target, 0, 0]),
    }
    # cond = { 
    #     diffusion.horizon - 1: np.array([1, 1, 0, 0]),
    # }
    ## observations for rendering
    rollout = [observation['observation']]

    total_reward = 0
    for t in range(env.max_episode_steps):
        state = observation['observation']

        ## can replan if desired, but the open-loop plans are good enough for maze2d
        ## that we really only need to plan once
        if t % 20 == 0:
            cond[0] = observation['observation']
            # print("Condition")
            # print(cond)
            action, samples = policy(cond, batch_size=args.batch_size)
            actions = samples.actions[0]
            sequence = samples.observations[0]

        # pdb.set_trace()

        # ####
        if t < len(sequence) - 1:
            next_waypoint = sequence[t+1]
        else:
            next_waypoint = sequence[-1].copy()
            next_waypoint[2:] = 0
            # pdb.set_trace()

        ## can use actions or define a simple controller based on state predictions
        action = next_waypoint[:2] - state[:2] + (next_waypoint[2:] - state[2:])

        action = action - 0.3

        # pdb.set_trace()
        ####

        # else:
        #     actions = actions[1:]
        #     if len(actions) > 1:
        #         action = actions[0]
        #     else:
        #         # action = np.zeros(2)
        #         action = -state[2:]
        #         pdb.set_trace()


        next_observation, reward, terminal, _, _ = env.step(action)
        total_reward += reward
        if total_reward > 0: 
            num_success += 1
            break
        # score = env.get_normalized_score(total_reward)
        score = total_reward
        # print(
        #     f't: {t} | r: {reward:.2f} |  R: {total_reward:.2f} | score: {score:.4f} | '
        #     f'{action}'
        # )

        if 'pointmaze' in args.dataset:
            xy = next_observation['observation'][:2]
            goal = next_observation['desired_goal']
            # print(
            #     f'maze | pos: {xy} | goal: {goal}'
            # )

        ## update rollout observations
        rollout.append(next_observation['observation'])

        # logger.log(score=score, step=t)

        # if t % args.vis_freq == 0 or terminal:
            # fullpath = join(args.savepath, f'{t}.png')

            # if t == 0: 
                # renderer.composite(fullpath, samples.observations, ncol=1)



            # renderer.render_plan(join(args.savepath, f'{t}_plan.mp4'), samples.actions, samples.observations, state)

            ## save rollout thus far
            # renderer.composite(join(args.savepath, 'rollout.png'), np.array(rollout)[None], ncol=1)

            # renderer.render_rollout(join(args.savepath, f'rollout.mp4'), rollout, fps=80)

            # logger.video(rollout=join(args.savepath, f'rollout.mp4'), plan=join(args.savepath, f'{t}_plan.mp4'), step=t)

        if terminal:
            break

        observation = next_observation

print("Num Success")
print(num_success)
    # logger.finish(t, env.max_episode_steps, score=score, value=0)

