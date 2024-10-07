#!/usr/bin/env python
from ssrl_ros_go1.controller import Controller
from pathlib import Path
import rospy
from omegaconf import DictConfig
from hydra.core.global_hydra import GlobalHydra
from hydra import compose, initialize


def main(cfg: DictConfig):
    try:
        data_path = Path(__file__).parent.parent / 'data'
        node = Controller(cfg, data_path)
        node.run()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    GlobalHydra.instance().clear()
    initialize(config_path="configs")
    cfg = compose("go1.yaml")
    main(cfg)
