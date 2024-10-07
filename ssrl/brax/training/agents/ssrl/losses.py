from brax.training import types
from brax.training.agents.ssrl import networks as ssrl_networks
from brax.training.agents.ssrl import base
from jax import numpy as jp
import jax


def make_losses(
    ensemble_model: ssrl_networks.EnsembleModel,
    preprocess_fn: lambda x, y: (x, y),
    c: base.Constants,
    mean_loss_over_horizon: bool = False  # used only for model evaluation
):

    def model_loss(model_params: types.Params,
                   vars: types.Params,
                   scaler_params: base.ScalerParams,
                   obs_stack_r: types.Observation,
                   obs_next_stack_r: types.Observation,
                   actions_r: types.Action,
                   discount_r: jp.ndarray):

        # _r indicates actual data from the rollout; XX_r shapes:
        # (batch_size, horizon, ensemble_size, dim)

        obs_next, logvars = propagate_obs_batch(
            model_params, vars, scaler_params, obs_stack_r,
            actions_r, ensemble_model, preprocess_fn, c)
        obs_next_r = obs_next_stack_r[:, :, :, :c.obs_size]

        error = obs_next - obs_next_r
        # TODO: discount error against horizon with a discount factor by
        # multiplying by an array of shape:
        # (batch_size, horizon, ensemble_size, dim)

        # do not propagate the loss when a done is hit (discount = 0)
        discount_mask = jp.where(
            jp.expand_dims(jp.cumprod(discount_r, axis=1) == 0, axis=-1),
            jp.zeros_like(error), jp.ones_like(error))
        error = discount_mask * error

        # compute the loss only for the mean, for each ensemble model, as an
        # auxiliary output
        if mean_loss_over_horizon:
            mean_loss = jp.mean(error**2, axis=(0, 2, 3))
        else:
            mean_loss = jp.mean(error**2, axis=(0, 1, 3))

        # loss for all models (average over batch, horizon, and dim; sum over
        # ensembles)
        if c.model_probabilistic:
            inv_vars = jp.exp(-logvars)
            mse_loss = jp.mean(inv_vars*error**2, axis=(0, 1, 3))
            var_loss = jp.mean(discount_mask*logvars, axis=(0, 1, 3))
            total_loss = jp.sum(mse_loss) + jp.sum(var_loss)
        else:
            # if not probabilistic, just use the mean loss
            total_loss = jp.sum(mean_loss)

        # add weight decay (L2 regularization) to the loss
        if c.model_training_weight_decay:
            for layer, decay in ensemble_model.weight_decays.items():
                weights = model_params[layer]['kernel']
                total_loss += 0.5 * decay * jp.sum(weights**2)

        return total_loss, mean_loss

    return model_loss


def propagate_obs_batch(
        model_params: types.Params,
        vars: types.Params,  # model variables, not variances
        scaler_params: base.ScalerParams,
        obs_stack_r: types.Observation,
        actions_r: types.Action,
        ensemble_model: ssrl_networks.EnsembleModel,
        preprocess_fn: lambda x, y: (x, y),
        c: base.Constants):

    # obs_stack_r, obs_next_stack_r, actions_r shapes:
    # (batch_size, horizon, ensemble_size, dim)

    def propagate_obs(obs_stack_r, actions_r):

        # input shapes: (horizon, ensemble_size, dim)

        def outer(carry_outer, in_element_outer):

            # shape of carry items and in_element: (ensemble_size, dim)
            obs_stack = carry_outer
            actions_r = in_element_outer

            obs = obs_stack[:, :c.obs_size]

            if c.model_training_stop_gradient:
                obs = jax.lax.stop_gradient(obs)

            proc_obs_stack, proc_act_r = preprocess_fn(obs_stack, actions_r,
                                                       scaler_params)
            x = jp.concatenate([proc_obs_stack, proc_act_r], axis=-1)
            means, logvars = ensemble_model.apply(
                {'params': model_params, **vars}, x)

            # propagate the mean through the dynamics
            def inner(carry, in_element_unused):
                obs = carry
                u = jax.vmap(c.low_level_control_fn)(actions_r, obs)
                obs_next = jax.vmap(c.dynamics_fn)(obs, u, means)
                return obs_next, None
            obs_next, _ = jax.lax.scan(inner, obs, (), length=c.policy_repeat)

            # update the obs stack
            obs_stack_next = jp.concatenate(
                [obs_next, obs_stack[:, :c.obs_size*(c.obs_hist_len-1)]],
                axis=-1
            )

            return obs_stack_next, (obs_next, logvars)

        _, (obs_next, logvars) = jax.lax.scan(
            outer, obs_stack_r[0, :, :], actions_r)

        return obs_next, logvars

    # output shape: (batch_size, horizon, ensemble_size, dim)
    return jax.vmap(propagate_obs)(obs_stack_r, actions_r)
