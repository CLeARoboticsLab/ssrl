import rosbag
from jax import numpy as jp


def read_bag(data_path: str, q_idxs=jp.s_[4:16], qd_idxs=jp.s_[22:34]):
    bag = rosbag.Bag(data_path, 'r')
    topics = ['observation', 'pd_target', 'action']
    ts = []
    obses = []
    qs = []
    qds = []
    q_deses = []
    qd_deses = []
    Kps = []
    Kds = []
    actions = []
    for topic, msg, t in bag.read_messages(topics=topics):
        if topic == 'observation':
            obs = jp.array(msg.observation)
            obses.append(obs)
            qs.append(obs[q_idxs])
            qds.append(obs[qd_idxs])
            ts.append(t.to_sec())
        elif topic == 'pd_target':
            q_deses.append(msg.q_des[:12])
            qd_deses.append(msg.qd_des[:12])
            Kps.append(msg.Kp[:12])
            Kds.append(msg.Kd[:12])
        elif topic == 'action':
            actions.append(jp.array(msg.action))
    bag.close()
    assert len(obses) == len(q_deses) == len(actions)
    ts = [t - ts[0] for t in ts]

    return (ts, obses, qs, qds, q_deses, qd_deses, Kps, Kds, actions)
