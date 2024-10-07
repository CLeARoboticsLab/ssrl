from brax.envs.go1_deltaxy_pd_slow import Go1DeltaXyPdSlow
from ssrl_ros_go1.bag import read_bag
from ssrl_ros_go1.plot_rollout import plot_rollout

from omegaconf import DictConfig
import hydra
from pathlib import Path
import os


@hydra.main(config_path="configs", config_name="go1")
def main(cfg: DictConfig):
    data_path = (Path(os.path.abspath(__file__)).parent.parent
                 / 'data'
                 / 'data.bag')
    env = Go1DeltaXyPdSlow()
    ts, obses, qs, qds, q_deses, qd_deses, Kps, Kds, actions = read_bag(
        data_path,
        q_idxs=env._q_idxs,
        qd_idxs=env._qd_idxs
    )
    plot_rollout(cfg, env, ts, obses, qs, qds, q_deses, qd_deses, Kps, Kds, actions)


if __name__ == '__main__':
    main()