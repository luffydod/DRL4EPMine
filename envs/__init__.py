from gymnasium.envs.registration import register

register(
    id='EpMineEnv-v0',
    entry_point='envs.SingleAgent:EpMineEnv',
    max_episode_steps=1800
)

register(
    id='EpMineEnv-v1',
    entry_point='envs.SingleAgent:NewEpMineEnv',
    max_episode_steps=1800
)