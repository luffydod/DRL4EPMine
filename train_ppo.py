import argparse
import os

from mindspore import context
from mindspore import dtype as mstype
from mindspore.communication import get_rank, init

import mindspore_rl.distribution.distribution_policies as DP
from mindspore_rl.algorithm.ppo import config
from mindspore_rl.algorithm.ppo.ppo_session import PPOSession
from mindspore_rl.algorithm.ppo.ppo_trainer import PPOTrainer
from envs.SingleAgent.mine_toy import EpMineEnv


parser = argparse.ArgumentParser(description="MindSpore Reinforcement PPO for EpMineEnv")
parser.add_argument("--episode", type=int, default=1000, help="total episode numbers.")
parser.add_argument(
    "--device_target",
    type=str,
    default="GPU",
    choices=["Ascend", "CPU", "GPU", "Auto"],
    help="Choose a device to run the ppo example(Default: Auto).",
)
parser.add_argument(
    "--precision_mode",
    type=str,
    default="fp32",
    choices=["fp32", "fp16"],
    help="Precision mode",
)
parser.add_argument(
    "--env_yaml",
    type=str,
    default="conf/EpMineEnv.yaml",
    help="Choose an environment yaml to update the ppo example(Default: EpMineEnv.yaml).",
)
parser.add_argument(
    "--algo_yaml",
    type=str,
    default="conf/PPO.yaml",
    help="Choose an algo yaml to update the ppo example(Default: PPO.yaml).",
)
parser.add_argument(
    "--enable_distribute",
    type=bool,
    default=False,
    help="Train in distribute mode (Default: False).",
)
parser.add_argument(
    "--worker_num", type=int, default=1, help="Worker num (Default: 1)."
)
parser.add_argument(
    "--graph_op_run", type=int, default=1, help="Run kernel by kernel (Default: 1)."
)
options, _ = parser.parse_known_args()


def train(episode=options.episode):
    try:
        """PPO train entry."""
        if options.device_target != "Auto":
            context.set_context(device_target=options.device_target)
        if context.get_context("device_target") in ["CPU"]:
            context.set_context(enable_graph_kernel=True)
        if context.get_context("device_target") in ["Ascend"] and options.graph_op_run:
            os.environ["GRAPH_OP_RUN"] = "1"

        compute_type = (
            mstype.float32 if options.precision_mode == "fp32" else mstype.float16
        )
        config.algorithm_config["policy_and_network"]["params"][
            "compute_type"
        ] = compute_type
        if compute_type == mstype.float16 and options.device_target != "Ascend":
            raise ValueError("Fp16 mode is supported by Ascend backend.")
        duration = config.trainer_params.get("duration")
        context.set_context(mode=context.GRAPH_MODE, max_call_depth=100000)
        is_distribte = options.enable_distribute
        if is_distribte:
            context.set_context(enable_graph_kernel=False)
            dp = config.deploy_config.get("distribution_policy")
            if dp == DP.SingleActorLearnerMultiEnvHeterDP:
                init("mccl")
                rank_id = get_rank()
                if rank_id == 0:
                    context.set_context(device_target="GPU")
            config.deploy_config["worker_num"] = options.worker_num
            config.deploy_config["auto_distribution"] = is_distribte
        print(options.env_yaml)
        ppo_session = PPOSession(options.env_yaml, options.algo_yaml, is_distribte)
        print(options.env_yaml)
        ppo_session.run(class_type=PPOTrainer, episode=episode, duration=duration)
    except KeyboardInterrupt:
        print("运行中断，正在关闭环境")
    finally:
        ppo_session.env.close()
        print("环境已关闭")


if __name__ == "__main__":
    train()