import nengo
from nengo_gym import GymEnv

model = nengo.Network(seed=13)

# Some example environments for reference
env_names = [
    'MountainCarContinuous-v0',
    'CartPole-v1',
    'LunarLanderContinuous-v2',
    'Breakout-v0',
    'Acrobot-v1',
    #'Pendulum-v0',
    #'Ant-v1',
    #'Humanoid-v1',
    'Skiing-v0',

]

with model:

    # dt of CartPole is 0.02
    # dt of Nengo is 0.001
    env = GymEnv(
        env_name=env_names[0],
        reset_signal=False,
        reset_when_done=True,
        return_reward=True,
        return_done=False,
        render=True,
        nengo_steps_per_update=20,
    )

    env_node = nengo.Node(
        env,
        size_in=env.size_in,
        size_out=env.size_out
    )

    action = nengo.Node([0]*env.act_dim)

    reward = nengo.Ensemble(n_neurons=50, dimensions=1)

    nengo.Connection(action, env_node)
    nengo.Connection(env_node[env.obs_dim], reward)

    # if env.obs_dim < 50:
    #     observation = nengo.Ensemble(n_neurons=env.obs_dim*50, dimensions=env.obs_dim)
    #     nengo.Connection(env_node[:env.obs_dim], observation)


def on_close(sim):
    env.close()


if __name__ == '__main__':

    sim = nengo.Simulator(model)
    sim.run(10)
