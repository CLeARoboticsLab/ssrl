from absl.testing import absltest
from absl.testing import parameterized
from brax import envs
from brax.training.agents.ssrl import train as ssrl
from jax import numpy as jp


def low_level_control(action: jp.ndarray,
                      obs: jp.ndarray) -> jp.ndarray:
    return action


def approx_dynamics(obs: jp.ndarray, u: jp.ndarray,
                    pred: jp.ndarray = None) -> jp.ndarray:
    return obs + pred


def compute_reward(obs: jp.ndarray, prev_obs: jp.ndarray,
                   unused_u: jp.ndarray,
                   action: jp.ndarray) -> jp.ndarray:
    return obs[0]


class ssrlTest(parameterized.TestCase):
    """Tests for ssrl module."""

    def testTrain(self):
        """Test ssrl with a simple env."""
        fast = envs.get_environment('fast')
        _, _, metrics = ssrl.train(
            fast,
            # num_timesteps=2**15,
            episode_length=128,
            # num_envs=64,
            # learning_rate=3e-4,
            # discounting=0.99,
            # batch_size=64,
            # normalize_observations=True,
            # reward_scaling=10,
            # grad_updates_per_step=64,
            # num_evals=3,
            # seed=0
        )
        # self.assertGreater(metrics['eval/episode_reward'], 140 * 0.995)
        # self.assertEqual(fast.reset_count, 2)
        # once for prefill, once for train, once for eval
        # self.assertEqual(fast.step_count, 3)  # type: ignore


if __name__ == '__main__':
    fast = envs.get_environment('fast')

    def progress_fn(steps, metrics):
        print(steps, metrics)

    _, _, metrics = ssrl.train(
        fast,
        low_level_control_fn=low_level_control,
        dynamics_fn=approx_dynamics,
        reward_fn=compute_reward,
        model_output_dim=2,
        num_epochs=15,
        init_exploration_steps=1000,
        episode_length=128,
        num_envs=1,
        env_steps_per_epoch=1000,
        num_evals=15,
        obs_history_length=2,
        progress_fn=progress_fn,
        deterministic_eval=True,
        model_check_done_condition=False,
        model_loss_horizon=10
    )
    # absltest.main()
