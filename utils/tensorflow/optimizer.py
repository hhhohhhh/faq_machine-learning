#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file:
@version:
@desc:
@time: 2021/6/21 14:34

https://github.com/OverLordGoldDragon/keras-adamw/blob/master/keras_adamw/optimizers_v2.py
<DECOUPLED WEIGHT DECAY REGULARIZATION> 论文中提到 将  decoupled 梯度计算(含有 L2正则的) 与 weight decay

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/6/21 14:34   wangfc      1.0         None


tf的自动求导就是靠各式各样的 OptimizerV2 类进行的，我们只需要在程序中构建前向图，然后加上Optimizer，再调用minimize()方法就可以完成梯度的反向传播。
    minimize()函数完成：
    1. 计算每一个部分的梯度，self._compute_gradients()
    2. 根据需要对梯度进行处理
    3. 把梯度更新到参数上，self.apply_gradients()

    self._create_slots函数表示创建一些优化器自带的一些参数，比如AdamOptimizer的m和v， beta1 的 t 次方（beta1_power）和 beta2的 t 次方（beta2_power）
    self._prepare()函数的作用是在apply梯度前创建好所有必须的tensors。
    self._prepare_local():


    初始化优化器时
    创建一个变量 optimier._iterations 用于记录迭代的次数
    所有优化器都可用的参数:参数 clipnorm 和 clipvalue 是所有优化器都可以使用的参数,用于对梯度进行裁剪.



### Creating a custom optimizer

If you intend to create your own optimization algorithm, simply inherit from
this class and override the following methods:

    - `_resource_apply_dense` (update variable given gradient tensor is dense)
    - `_resource_apply_sparse` (update variable given gradient tensor is sparse)
    - `_create_slots`


      (if your optimizer algorithm requires additional variables)
    - `get_config`
      (serialization of the optimizer, include all hyper parameters)


    sparse layers: - e.g. Embedding -
    dense layers :  with everything else

"""
from typing import Optional, List
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.keras import backend_config
from tensorflow.python.keras.optimizer_v2.optimizer_v2 import OptimizerV2
from tensorflow.python.keras.optimizer_v2 import learning_rate_schedule
from tensorflow.python.ops import array_ops, control_flow_ops, math_ops, state_ops
from tensorflow.python.util.tf_export import keras_export
from tensorflow.python.keras import backend as K

from .utils import _init_weight_decays, _apply_weight_decays, _check_args
from .utils import _update_t_cur_eta_t_v2, _apply_lr_multiplier, _set_autorestart
from .utils import K_eval as KE

from bert4keras.optimizers import extend_with_piecewise_linear_lr, Adam
from transformers.optimization_tf import WarmUp, AdamWeightDecay


def K_eval(x):
    return KE(x, K)


def create_optimizer(
        optimizer_name: str,
        learning_rate: float,
        num_train_steps: int = None,
        num_warmup_steps: int = None,
        model=None,
        train_step_per_epoch: int = None,
        min_lr_ratio: float = 0.0,
        power: float = 1.0,
        use_cosine_annealing=True,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        adam_epsilon: float = 1e-8,
        weight_decay_rate: float = 0.01,
        exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"],
        include_in_weight_decay: Optional[List[str]] = None,
        clipnorm: float = 1.0,

        **kwargs
):
    """
    Creates an optimizer with a learning rate schedule using a warmup phase followed by a linear decay.

    Args:
        init_lr (:obj:`float`):
            The desired learning rate at the end of the warmup phase.
        num_train_steps (:obj:`int`):
            The total number of training steps.
        num_warmup_steps (:obj:`int`):
            The number of warmup steps.
        min_lr_ratio (:obj:`float`, `optional`, defaults to 0):
            The final learning rate at the end of the linear decay will be :obj:`init_lr * min_lr_ratio`.
        adam_beta1 (:obj:`float`, `optional`, defaults to 0.9):
            The beta1 to use in Adam.
        adam_beta2 (:obj:`float`, `optional`, defaults to 0.999):
            The beta2 to use in Adam.
        adam_epsilon (:obj:`float`, `optional`, defaults to 1e-8):
            The epsilon to use in Adam.
        weight_decay_rate (:obj:`float`, `optional`, defaults to 0):
            The weight decay to use.
        power (:obj:`float`, `optional`, defaults to 1.0):
            The power to use for PolynomialDecay.
        include_in_weight_decay (:obj:`List[str]`, `optional`):
            List of the parameter names (or re patterns) to apply weight decay to. If none is passed, weight decay is
            applied to all parameters except bias and layer norm parameters.
    """
    if num_warmup_steps:
        decay_steps = num_train_steps - num_warmup_steps
        # Implements linear decay of the learning rate.
        lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=learning_rate,
            decay_steps=decay_steps,
            end_learning_rate=learning_rate * min_lr_ratio,
            power=power,
        )

        lr_schedule = WarmUp(
            initial_learning_rate=learning_rate,
            decay_schedule_fn=lr_schedule,
            warmup_steps=num_warmup_steps,
        )
    else:
        lr_schedule = learning_rate

    """
    tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.0, nesterov=False, name='SGD', **kwargs)
        1.当momentum为0时，更新权重w与梯度g的公式如下:
        w = w - learning_rate * g
        2.当momentum不为0时，变成SGDM, 考虑了一阶动量. 更新公式如下
        velocity = momentum * velocity - learning_rate * g
        w = w + velocity
        3.当 nesterov 为True时,更新公式如下: 变成NAG，即 Nesterov Accelerated Gradient，在计算梯度时计算的是向前走一步所在位置的梯度
        velocity = momentum * velocity - learning_rate * g
        w = w + momentum * velocity - learning_rate * g
    
    Adagrad, 考虑了二阶动量，对于不同的参数有不同的学习率，即自适应学习率。缺点是学习率单调下降，可能后期学习速率过慢乃至提前停止学习。
    
    RMSprop, 考虑了二阶动量，对于不同的参数有不同的学习率，即自适应学习率，对Adagrad进行了优化，通过指数平滑只考虑一定窗口内的二阶动量。
    
    Adadelta, 考虑了二阶动量，与RMSprop类似，但是更加复杂一些，自适应性更强。
    
    Adam, 同时考虑了一阶动量和二阶动量，可以看成RMSprop上进一步考虑了一阶动量。
    
    Nadam, 在Adam基础上进一步考虑了 Nesterov Acceleration。
    """
    if optimizer_name.lower() == 'sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0, decay=0)
    elif optimizer_name.lower() == 'adagrad':
        optimizer = tf.keras.optimizers.Adagrad(learning_rate=lr_schedule)
    elif optimizer_name.lower() == 'rmsprop':
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr_schedule)
    elif optimizer_name.lower() == "adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule,
                                             beta_1=adam_beta1,
                                             beta_2=adam_beta2,
                                             epsilon=adam_epsilon)
    elif optimizer_name.lower() == "hf_adamw" and weight_decay_rate > 0.0:
        # 使用 hugging_face transformers 的 adamw
        allowed_kwargs = {"clipnorm", "clipvalue", "lr", "decay"}
        optimizer = AdamWeightDecay(
            learning_rate=lr_schedule,
            weight_decay_rate=weight_decay_rate,
            beta_1=adam_beta1,
            beta_2=adam_beta2,
            epsilon=adam_epsilon,
            exclude_from_weight_decay=exclude_from_weight_decay,
            include_in_weight_decay=include_in_weight_decay,
            clipnorm=clipnorm,
        )
    elif optimizer_name == 'adamlr':
        # 使用 bert4keras 的 adamlr
        AdamLR = extend_with_piecewise_linear_lr(Adam, name='AdamLR', clipnorm=clipnorm)
        optimizer = AdamLR(learning_rate=learning_rate, lr_schedule={num_warmup_steps: 1, decay_steps: 0.1})
    elif optimizer_name == 'adamw':
        # 使用 github上的 adamw
        optimizer = AdamW(learning_rate=lr_schedule,
                          model=model,
                          total_iterations=train_step_per_epoch,
                          autorestart=True,
                          use_cosine_annealing=use_cosine_annealing,
                          clipnorm=clipnorm)

    # We return the optimizer and the LR scheduler in order to better track the
    # evolution of the LR independently of the optimizer.
    return optimizer, lr_schedule


@keras_export("keras.optimizers.SGDW")
class SGDW(OptimizerV2):
    """Stochastic gradient descent optimizer.
    Includes support for momentum,
    learning rate decay, and Nesterov momentum.
    # Arguments
        learning_rate: float hyperparameter >= 0. Learning rate.
        momentum: float hyperparameter >= 0 that accelerates SGDW in the relevant
          direction and dampens oscillations.
        nesterov: boolean. Whether to apply Nesterov momentum.
        name: Optional name prefix for the operations created when applying
            gradients.  Defaults to 'SGDW'.
        **kwargs: keyword arguments. Allowed to be {`clipnorm`, `clipvalue`, `lr`,
            `decay`}. `clipnorm` is clip gradients by norm; `clipvalue` is clip
            gradients by value, `decay` is included for backward compatibility to
            allow time inverse decay of learning rate. `lr` is included for
            backward compatibility, recommended to use `learning_rate` instead.
        model: keras.Model/tf.keras.Model/None. If not None, automatically
            extracts weight penalties from layers, and overrides `weight_decays`.
        zero_penalties: bool. If True and `model` is passed, will zero weight
            penalties (loss-based). (RECOMMENDED; see README "Use guidelines").
        total_iterations: int >= 0. Total expected iterations / weight updates
                          throughout training, used for normalization; <1>
        lr_multipliers:   dict / None. Name-value pairs specifying per-layer lr
                          multipliers, as {<layer name>:<multiplier value>}; <2>
        weight_decays:    dict / None. Name-value pairs specifying weight decays,
                          as {<weight matrix name>:<weight decay value>}; <2>
        use_cosine_annealing: bool. If True, multiplies lr each train iteration
                              as a function of eta_min, eta_max, total_iterations,
                              and t_cur (current); [2]-Appendix, 2
        autorestart: bool / None. If True, will automatically do Warm Restarts
                     by resetting `t_cur=0` after `total_iterations`. If None,
                     will default to same as `use_cosine_annealing`. If True
                     but `use_cosine_annealing` is False, will raise ValueError.
                     Note: once optimizer is built (happens on first model fit),
                     changing `autorestart` has no effect; optimizer needs to be
                     re-built.
        eta_min, eta_max: int, int. Min & max values of cosine annealing
                          lr multiplier; [2]-Appendix, 2
        t_cur: int. Value to initialize t_cur to - used for 'warm restarts'.
               To be used together with use_cosine_annealing==True
        total_iterations_wd: int / None. If not None, weight_decays will be
                     applied according to total_iterations_wd instead of
                     total_iterations, contrary to authors' scheme. Set to
                     sum(total_iterations) over all restarts to normalize over
                     all epochs. May yield improvement over `None`.
        init_verbose: bool. If True, print weight-name--weight-decay, and
                      lr-multiplier--layer-name value pairs set during
                      optimizer initialization (recommended)
    # <1> - if using 'warm restarts', then refers to total expected iterations
        for a given restart; can be an estimate, and training won't stop
        at iterations == total_iterations. [2]-Appendix, pg 1
    # <2> - [AdamW Keras Implementation - Github repository]
        (https://github.com/OverLordGoldDragon/keras_adamw)
    # References
    - [1][Adam - A Method for Stochastic Optimization]
         (http://arxiv.org/abs/1412.6980v8)
    - [2][Fixing Weight Decay Regularization in Adam]
         (https://arxiv.org/abs/1711.05101)
    """

    def __init__(self, learning_rate=0.01, momentum=0.0, nesterov=False,
                 model=None, zero_penalties=True,
                 total_iterations=0, total_iterations_wd=None,
                 use_cosine_annealing=False, lr_multipliers=None,
                 weight_decays=None, autorestart=None, init_verbose=True,
                 eta_min=0, eta_max=1, t_cur=0, name="SGDW", **kwargs):
        if total_iterations > 1:
            weight_decays = _init_weight_decays(model, zero_penalties,
                                                weight_decays)

        eta_t = kwargs.pop('eta_t', 1.)
        super(SGDW, self).__init__(name, **kwargs)
        self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))
        self._set_hyper("decay", self._initial_decay)

        self._momentum = False
        if isinstance(momentum, ops.Tensor) or callable(momentum) or momentum > 0:
            self._momentum = True
        if isinstance(momentum, (int, float)) and (momentum < 0 or momentum > 1):
            raise ValueError("`momentum` must be between [0, 1].")
        self._set_hyper("momentum", momentum)

        self.nesterov = nesterov
        self.eta_min = K.constant(eta_min, name='eta_min')
        self.eta_max = K.constant(eta_max, name='eta_max')
        self.eta_t = K.variable(eta_t, dtype='float32', name='eta_t')
        self.t_cur = K.variable(t_cur, dtype='int64', name='t_cur')
        self.total_iterations = total_iterations
        self.total_iterations_wd = total_iterations_wd or total_iterations
        self.lr_multipliers = lr_multipliers
        self.weight_decays = weight_decays or {}
        self.init_verbose = init_verbose
        self.use_cosine_annealing = use_cosine_annealing

        _set_autorestart(self, autorestart, use_cosine_annealing)
        _check_args(self, total_iterations, use_cosine_annealing, weight_decays)
        self._init_lr = kwargs.get('lr', learning_rate)  # to print lr_mult setup
        self._updates_processed = 0  # to track num calls to '_resource_apply_...'
        self._init_notified = False

    def _create_slots(self, var_list):
        """
        For each model variable, create the optimizer variable associated with it.
        TensorFlow calls these optimizer variables "slots".
        For momentum optimization, we need one momentum slot per model variable.
        """
        if self._momentum:
            for var in var_list:
                self.add_slot(var, "momentum")
        self._updates_per_iter = len(var_list)

    def _resource_apply_dense(self, grad, var):
        var_dtype = var.dtype.base_dtype
        # handle learning rate decay
        lr_t = self._decayed_lr(var_dtype)

        # Learning rate multipliers
        if self.lr_multipliers is not None:
            lr_t = _apply_lr_multiplier(self, lr_t, var)

        if self._momentum:
            momentum = array_ops.identity(self._get_hyper('momentum', var_dtype))
            # Extract the previous values of Variables and Gradients
            m = self.get_slot(var, 'momentum')
            v = momentum * m - self.eta_t * lr_t * grad  # velocity
            m = state_ops.assign(m, v, use_locking=self._use_locking)

            if self.nesterov:
                var_t = math_ops.sub(
                    var, -momentum * v + self.eta_t * lr_t * grad)
            else:
                var_t = var + v
        else:
            v = - self.eta_t * lr_t * grad
            var_t = var + v

        # Weight decays
        if var.name in self.weight_decays.keys():
            var_t = _apply_weight_decays(self, var, var_t)

        var_update = state_ops.assign(var, var_t, use_locking=self._use_locking)

        # Cosine annealing
        (iteration_done, t_cur_update, eta_t_update
         ) = _update_t_cur_eta_t_v2(self, lr_t, var)
        if iteration_done and not self._init_notified:
            self._init_notified = True

        updates = [var_update]
        if self._momentum:
            updates += [m]
        if iteration_done:
            updates += [t_cur_update]
        if self.use_cosine_annealing and iteration_done:
            updates += [eta_t_update]
        return control_flow_ops.group(*updates)

    def _resource_apply_sparse(self, grad, var, indices):
        var_dtype = var.dtype.base_dtype
        lr_t = self._decayed_lr(var_dtype)

        # Learning rate multipliers
        if self.lr_multipliers is not None:
            lr_t = _apply_lr_multiplier(self, lr_t, var)

        if self._momentum:
            momentum = array_ops.identity(self._get_hyper('momentum', var_dtype))
            m = self.get_slot(var, 'momentum')
            v = momentum * m - self.eta_t * lr_t * grad
            m = state_ops.assign(m, v, use_locking=self._use_locking)

            if self.nesterov:
                var_t = self._resource_scatter_add(
                    var, indices, momentum * v - (self.eta_t * lr_t * grad))
            else:
                var_t = self._resource_scatter_add(var, indices, v)
        else:
            v = - self.eta_t * lr_t * grad
            var_t = var + v

        # Weight decays
        if var.name in self.weight_decays.keys():
            var_t = _apply_weight_decays(self, var, var_t)

        var_update = state_ops.assign(var, var_t, use_locking=self._use_locking)

        # Cosine annealing
        (iteration_done, t_cur_update, eta_t_update
         ) = _update_t_cur_eta_t_v2(self, lr_t, var)
        if iteration_done and not self._init_notified:
            self._init_notified = True

        updates = [var_update]
        if self._momentum:
            updates += [m]
        if iteration_done:
            updates += [t_cur_update]
        if self.use_cosine_annealing and iteration_done:
            updates += [eta_t_update]
        return control_flow_ops.group(*updates)

    def get_config(self):
        config = super(SGDW, self).get_config()
        config.update({
            "learning_rate": self._serialize_hyperparameter("learning_rate"),
            "decay": self._serialize_hyperparameter("decay"),
            "momentum": self._serialize_hyperparameter("momentum"),
            "nesterov": self.nesterov,
            'total_iterations': int(self.total_iterations),
            'weight_decays': self.weight_decays,
            'use_cosine_annealing': self.use_cosine_annealing,
            'autorestart': self.autorestart,
            't_cur': int(K_eval(self.t_cur)),
            'eta_t': float(K_eval(self.eta_t)),
            'eta_min': float(K_eval(self.eta_min)),
            'eta_max': float(K_eval(self.eta_max)),
            'init_verbose': self.init_verbose
        })
        return config


@keras_export('keras.optimizers.AdamW')
class AdamW(OptimizerV2):
    """AdamW optimizer.
    Default parameters follow those provided in the original paper.
    For extended documentation, see optimizer_v2.Adam.__doc__.
    # Arguments
        model: keras.Model/tf.keras.Model. Pass as first positional argument
            to constructor (AdamW(model, ...)). If passed, automatically extracts
            weight penalties from layers and overrides `weight_decays`.
        zero_penalties: bool. If True and `model` is passed, will zero weight
            penalties (loss-based). (RECOMMENDED; see README "Use guidelines").
        learning_rate: A Tensor or a floating point value.  The learning rate.
        beta_1: A float value or a constant float tensor. The exponential decay
            rate for the 1st moment estimates.
        beta_2: A float value or a constant float tensor. The exponential decay
            rate for the 2nd moment estimates.
        epsilon: A small constant for numerical stability. This epsilon is
            "epsilon hat" in the Kingma and Ba paper (in the formula just before
            Section 2.1), not the epsilon in Algorithm 1 of the paper.
        amsgrad: boolean. Whether to apply AMSGrad variant of this algorithm from
            the paper "On the Convergence of Adam and beyond".
        name: Optional name for the operations created when applying gradients.
            Defaults to "Adam".  @compatibility(eager) When eager execution is
            enabled, `learning_rate`, `beta_1`, `beta_2`, and `epsilon` can each
            be a callable that takes no arguments and returns the actual value
            to use. This can be useful for changing these values across different
            invocations of optimizer functions. @end_compatibility
        **kwargs: keyword arguments. Allowed to be {`clipnorm`, `clipvalue`, `lr`,
            `decay`}. `clipnorm` is clip gradients by norm; `clipvalue` is clip
            gradients by value, `decay` is included for backward compatibility to
            allow time inverse decay of learning rate. `lr` is included for
            backward compatibility, recommended to use `learning_rate` instead.
        model: keras.Model/tf.keras.Model/None. If not None, automatically
            extracts weight penalties from layers, and overrides `weight_decays`.
        zero_penalties: bool. If True and `model` is passed, will zero weight
            penalties (loss-based). (RECOMMENDED; see README "Use guidelines").
        total_iterations: int >= 0. Total expected iterations / weight updates
                          throughout training, used for normalization; <1>
        lr_multipliers:   dict / None. Name-value pairs specifying per-layer lr
                          multipliers, as {<layer name>:<multiplier value>}; <2>
        weight_decays:    dict / None. Name-value pairs specifying weight decays,
                          as {<weight matrix name>:<weight decay value>}; <2>
        use_cosine_annealing: bool. If True, multiplies lr each train iteration
                              as a function of eta_min, eta_max, total_iterations,
                              and t_cur (current); [2]-Appendix, 2
        autorestart: bool / None. If True, will automatically do Warm Restarts
                     by resetting `t_cur=0` after `total_iterations`. If None,
                     will default to same as `use_cosine_annealing`. If True
                     but `use_cosine_annealing` is False, will raise ValueError.
                     Note: once optimizer is built (happens on first model fit),
                     changing `autorestart` has no effect; optimizer needs to be
                     re-built.
        eta_min, eta_max: int, int. Min & max values of cosine annealing
                          lr multiplier; [2]-Appendix, 2
        t_cur: int. Value to initialize t_cur to - used for 'warm restarts'.
               To be used together with use_cosine_annealing==True
        total_iterations_wd: int / None. If not None, weight_decays will be
                     applied according to total_iterations_wd instead of
                     total_iterations, contrary to authors' scheme. Set to
                     sum(total_iterations) over all restarts to normalize over
                     all epochs. May yield improvement over `None`.
        init_verbose: bool. If True, print weight-name--weight-decay, and
                      lr-multiplier--layer-name value pairs set during
                      optimizer initialization (recommended)
    # <1> - if using 'warm restarts', then refers to total expected iterations
            for a given restart; can be an estimate, and training won't stop
            at iterations == total_iterations. [2]-Appendix, pg 1
    # <2> - [AdamW Keras Implementation - Github repository]
            (https://github.com/OverLordGoldDragon/keras_adamw)
    # References
        - [1][Adam - A Method for Stochastic Optimization]
             (http://arxiv.org/abs/1412.6980v8)
        - [2][Fixing Weight Decay Regularization in Adam]
             (https://arxiv.org/abs/1711.05101)
    """

    def __init__(self, learning_rate=0.001, beta_1=0.9, beta_2=0.999,
                 epsilon=None, decay=0., amsgrad=False,
                 model=None, zero_penalties=True,
                 total_iterations=0, total_iterations_wd=None,
                 use_cosine_annealing=False, lr_multipliers=None,
                 weight_decays=None, autorestart=None,
                 init_verbose=True, eta_min=0, eta_max=1, t_cur=0,
                 name="AdamW", **kwargs):
        if total_iterations > 1:
            weight_decays = _init_weight_decays(model, zero_penalties,
                                                weight_decays)
        eta_t = kwargs.pop('eta_t', 1.)

        super(AdamW, self).__init__(name, **kwargs)
        self._set_hyper('learning_rate', kwargs.get('lr', learning_rate))
        self._set_hyper('decay', self._initial_decay)
        self._set_hyper('beta_1', beta_1)
        self._set_hyper('beta_2', beta_2)

        self.eta_min = K.constant(eta_min, name='eta_min')
        self.eta_max = K.constant(eta_max, name='eta_max')
        self.eta_t = K.variable(eta_t, dtype='float32', name='eta_t')
        self.t_cur = K.variable(t_cur, dtype='int64', name='t_cur')
        self.total_iterations = total_iterations
        self.total_iterations_wd = total_iterations_wd or total_iterations
        self.lr_multipliers = lr_multipliers
        self.weight_decays = weight_decays or {}
        self.init_verbose = init_verbose
        self.use_cosine_annealing = use_cosine_annealing
        self.epsilon = epsilon or backend_config.epsilon()
        self.amsgrad = amsgrad

        _set_autorestart(self, autorestart, use_cosine_annealing)
        _check_args(self, total_iterations, use_cosine_annealing, weight_decays)
        self._init_lr = kwargs.get('lr', learning_rate)  # to print lr_mult setup
        self._updates_processed = 0  # to track num calls to '_resource_apply_...'
        self._init_notified = False
        # 增加 该参数
        # self.lr_t = kwargs.get('lr', learning_rate)

    def _create_slots(self, var_list):
        # Create slots for the first and second moments.
        # Separate for-loops to respect the ordering of slot variables from v1.
        for var in var_list:
            self.add_slot(var, 'm')
        for var in var_list:
            self.add_slot(var, 'v')
        if self.amsgrad:
            for var in var_list:
                self.add_slot(var, 'vhat')
        self._updates_per_iter = len(var_list)

    def _resource_apply_dense(self, grad, var):
        var_dtype = var.dtype.base_dtype
        lr_t = self._decayed_lr(var_dtype)
        local_step = math_ops.cast(self.iterations + 1, var_dtype)
        m = self.get_slot(var, 'm')
        v = self.get_slot(var, 'v')
        beta_1_t = array_ops.identity(self._get_hyper('beta_1', var_dtype))
        beta_2_t = array_ops.identity(self._get_hyper('beta_2', var_dtype))
        beta_1_power = math_ops.pow(beta_1_t, local_step)
        beta_2_power = math_ops.pow(beta_2_t, local_step)
        epsilon_t = ops.convert_to_tensor(self.epsilon, var_dtype)

        lr_t = lr_t * math_ops.sqrt(1 - beta_2_power) / (1 - beta_1_power)
        # Learning rate multipliers
        if self.lr_multipliers is not None:
            lr_t = _apply_lr_multiplier(self, lr_t, var)

        # 更新该参数
        # self.lr_t = K_eval(lr_t)

        m_t = state_ops.assign(m,
                               beta_1_t * m + (1.0 - beta_1_t) * grad,
                               use_locking=self._use_locking)
        v_t = state_ops.assign(v,
                               beta_2_t * v + (1.0 - beta_2_t
                                               ) * math_ops.square(grad),
                               use_locking=self._use_locking)
        if self.amsgrad:
            vhat = self.get_slot(var, 'vhat')
            vhat_t = state_ops.assign(vhat,
                                      math_ops.maximum(vhat, v_t),
                                      use_locking=self._use_locking)
            var_delta = m_t / (math_ops.sqrt(vhat_t) + epsilon_t)
        else:
            var_delta = m_t / (math_ops.sqrt(v_t) + epsilon_t)
        var_t = math_ops.sub(var, self.eta_t * lr_t * var_delta)

        # Weight decays
        if var.name in self.weight_decays.keys():
            var_t = _apply_weight_decays(self, var, var_t)

        var_update = state_ops.assign(var, var_t, use_locking=self._use_locking)

        # Cosine annealing
        (iteration_done, t_cur_update, eta_t_update
         ) = _update_t_cur_eta_t_v2(self, lr_t, var)
        if iteration_done and not self._init_notified:
            self._init_notified = True

        updates = [var_update, m_t, v_t]
        if iteration_done:
            updates += [t_cur_update]
        if self.use_cosine_annealing and iteration_done:
            updates += [eta_t_update]
        if self.amsgrad:
            updates.append(vhat_t)
        return control_flow_ops.group(*updates)

    def _resource_apply_sparse(self, grad, var, indices):
        var_dtype = var.dtype.base_dtype
        lr_t = self._decayed_lr(var_dtype)
        local_step = math_ops.cast(self.iterations + 1, var_dtype)
        m = self.get_slot(var, 'm')
        v = self.get_slot(var, 'v')
        beta_1_t = array_ops.identity(self._get_hyper('beta_1', var_dtype))
        beta_2_t = array_ops.identity(self._get_hyper('beta_2', var_dtype))
        beta_1_power = math_ops.pow(beta_1_t, local_step)
        beta_2_power = math_ops.pow(beta_2_t, local_step)
        epsilon_t = ops.convert_to_tensor(self.epsilon, var_dtype)

        lr_t = lr_t * math_ops.sqrt(1 - beta_2_power) / (1 - beta_1_power)
        # Learning rate multipliers
        if self.lr_multipliers is not None:
            lr_t = _apply_lr_multiplier(self, lr_t, var)

        m_scaled_g_values = grad * (1 - beta_1_t)
        m_t = state_ops.assign(m, m * beta_1_t, use_locking=self._use_locking)
        with ops.control_dependencies([m_t]):
            m_t = self._resource_scatter_add(m, indices, m_scaled_g_values)

        v_scaled_g_values = (grad * grad) * (1 - beta_2_t)
        v_t = state_ops.assign(v, v * beta_2_t, use_locking=self._use_locking)
        with ops.control_dependencies([v_t]):
            v_t = self._resource_scatter_add(v, indices, v_scaled_g_values)

        if self.amsgrad:
            vhat = self.get_slot(var, 'vhat')
            vhat_t = state_ops.assign(vhat,
                                      math_ops.maximum(vhat, v_t),
                                      use_locking=self._use_locking)
            var_delta = m_t / (math_ops.sqrt(vhat_t) + epsilon_t)
        else:
            var_delta = m_t / (math_ops.sqrt(v_t) + epsilon_t)
        var_t = math_ops.sub(var, self.eta_t * lr_t * var_delta)

        # Weight decays
        if var.name in self.weight_decays.keys():
            var_t = _apply_weight_decays(self, var, var_t)

        var_update = state_ops.assign(var, var_t, use_locking=self._use_locking)

        # Cosine annealing
        (iteration_done, t_cur_update, eta_t_update
         ) = _update_t_cur_eta_t_v2(self, lr_t, var)
        if iteration_done and not self._init_notified:
            self._init_notified = True

        updates = [var_update, m_t, v_t]
        if iteration_done:
            updates += [t_cur_update]
        if self.use_cosine_annealing and iteration_done:
            updates += [eta_t_update]
        if self.amsgrad:
            updates.append(vhat_t)
        return control_flow_ops.group(*updates)

    def set_weights(self, weights):
        params = self.weights
        # If the weights are generated by Keras V1 optimizer, it includes vhats
        # even without amsgrad, i.e, V1 optimizer has 3x + 1 variables, while V2
        # optimizer has 2x + 1 variables. Filter vhats out for compatibility.
        num_vars = int((len(params) - 1) / 2)
        if len(weights) == 3 * num_vars + 1:
            weights = weights[:len(params)]
        super(AdamW, self).set_weights(weights)

    def get_config(self):
        config = super(AdamW, self).get_config()
        config.update({
            'learning_rate': self._serialize_hyperparameter('learning_rate'),
            'decay': self._serialize_hyperparameter('decay'),
            'beta_1': self._serialize_hyperparameter('beta_1'),
            'beta_2': self._serialize_hyperparameter('beta_2'),
            'epsilon': self.epsilon,
            'amsgrad': self.amsgrad,
            'total_iterations': int(self.total_iterations),
            'weight_decays': self.weight_decays,
            'use_cosine_annealing': self.use_cosine_annealing,
            'autorestart': self.autorestart,
            't_cur': int(K_eval(self.t_cur)),
            'eta_t': float(K_eval(self.eta_t)),
            'eta_min': float(K_eval(self.eta_min)),
            'eta_max': float(K_eval(self.eta_max)),
            'init_verbose': self.init_verbose
        })
        return config


@keras_export('keras.optimizers.NadamW')
class NadamW(OptimizerV2):
    """Nesterov Adam optimizer.
    Much like Adam is essentially RMSprop with momentum,
    Nadam is Adam RMSprop with Nesterov momentum.
    Default parameters follow those provided in the paper.
    It is recommended to leave the parameters of this optimizer
    at their default values.
    # Arguments
      learning_rate: A Tensor or a floating point value.  The learning rate.
      beta_1: A float value or a constant float tensor. The exponential decay
        rate for the 1st moment estimates.
      beta_2: A float value or a constant float tensor. The exponential decay
        rate for the exponentially weighted infinity norm.
      epsilon: A small constant for numerical stability.
      name: Optional name for the operations created when applying gradients.
        Defaults to "Adamax".
      **kwargs: keyword arguments. Allowed to be {`clipnorm`, `clipvalue`, `lr`,
        `decay`}. `clipnorm` is clip gradients by norm; `clipvalue` is clip
        gradients by value, `decay` is included for backward compatibility to
        allow time inverse decay of learning rate. `lr` is included for backward
        compatibility, recommended to use `learning_rate` instead.
        model: keras.Model/tf.keras.Model/None. If not None, automatically
            extracts weight penalties from layers, and overrides `weight_decays`.
        zero_penalties: bool. If True and `model` is passed, will zero weight
            penalties (loss-based). (RECOMMENDED; see README "Use guidelines").
        total_iterations: int >= 0. Total expected iterations / weight updates
                          throughout training, used for normalization; <1>
        lr_multipliers:   dict / None. Name-value pairs specifying per-layer lr
                          multipliers, as {<layer name>:<multiplier value>}; <2>
        weight_decays:    dict / None. Name-value pairs specifying weight decays,
                          as {<weight matrix name>:<weight decay value>}; <2>
        use_cosine_annealing: bool. If True, multiplies lr each train iteration
                              as a function of eta_min, eta_max, total_iterations,
                              and t_cur (current); [3]-Appendix, 2
        autorestart: bool / None. If True, will automatically do Warm Restarts
                     by resetting `t_cur=0` after `total_iterations`. If None,
                     will default to same as `use_cosine_annealing`. If True
                     but `use_cosine_annealing` is False, will raise ValueError.
                     Note: once optimizer is built (happens on first model fit),
                     changing `autorestart` has no effect; optimizer needs to be
                     re-built.
        eta_min, eta_max: int, int. Min & max values of cosine annealing
                          lr multiplier; [3]-Appendix, 2
        t_cur: int. Value to initialize t_cur to - used for 'warm restarts'.
               To be used together with use_cosine_annealing==True
        total_iterations_wd: int / None. If not None, weight_decays will be
                     applied according to total_iterations_wd instead of
                     total_iterations, contrary to authors' scheme. Set to
                     sum(total_iterations) over all restarts to normalize over
                     all epochs. May yield improvement over `None`.
        init_verbose: bool. If True, print weight-name--weight-decay, and
                      lr-multiplier--layer-name value pairs set during
                      optimizer initialization (recommended)
    # <1> - if using 'warm restarts', then refers to total expected iterations
        for a given restart; can be an estimate, and training won't stop
        at iterations == total_iterations. [3]-Appendix, pg 1
    # <2> - [AdamW Keras Implementation - Github repository]
            (https://github.com/OverLordGoldDragon/keras_adamw)
    # References
        - [1][Nadam report](http://cs229.stanford.edu/proj2015/054_report.pdf)
        - [2][On the importance of initialization and momentum in deep learning]
             (http://www.cs.toronto.edu/~fritz/absps/momentum.pdf)
        - [3][Fixing Weight Decay Regularization in Adam]
             (https://arxiv.org/abs/1711.05101)
    """

    def __init__(self, learning_rate=0.001, beta_1=0.9, beta_2=0.999,
                 epsilon=1e-7, model=None, zero_penalties=True,
                 total_iterations=0, total_iterations_wd=None,
                 use_cosine_annealing=False, lr_multipliers=None,
                 weight_decays=None, autorestart=None, init_verbose=True,
                 eta_min=0, eta_max=1, t_cur=0, name="NadamW", **kwargs):
        if total_iterations > 1:
            weight_decays = _init_weight_decays(model, zero_penalties,
                                                weight_decays)

        # Backwards compatibility with keras NAdam optimizer.
        kwargs['decay'] = kwargs.pop('schedule_decay', 0.004)
        eta_t = kwargs.pop('eta_t', 1.)
        learning_rate = kwargs.get('lr', learning_rate)
        if isinstance(learning_rate, learning_rate_schedule.LearningRateSchedule):
            raise ValueError('The Nadam optimizer does not support '
                             'tf.keras.optimizers.LearningRateSchedules as the '
                             'learning rate.')

        super(NadamW, self).__init__(name, **kwargs)
        self._set_hyper('learning_rate', kwargs.get('lr', learning_rate))
        self._set_hyper('decay', self._initial_decay)
        self._set_hyper('beta_1', beta_1)
        self._set_hyper('beta_2', beta_2)
        self.epsilon = epsilon or backend_config.epsilon()
        self._m_cache = None

        self.eta_min = K.constant(eta_min, name='eta_min')
        self.eta_max = K.constant(eta_max, name='eta_max')
        self.eta_t = K.variable(eta_t, dtype='float32', name='eta_t')
        self.t_cur = K.variable(t_cur, dtype='int64', name='t_cur')
        self.total_iterations = total_iterations
        self.total_iterations_wd = total_iterations_wd or total_iterations
        self.lr_multipliers = lr_multipliers
        self.weight_decays = weight_decays or {}
        self.init_verbose = init_verbose
        self.use_cosine_annealing = use_cosine_annealing
        self.epsilon = epsilon or backend_config.epsilon()

        _set_autorestart(self, autorestart, use_cosine_annealing)
        _check_args(self, total_iterations, use_cosine_annealing, weight_decays)
        self._init_lr = kwargs.get('lr', learning_rate)  # to print lr_mult setup
        self._updates_processed = 0  # to track num calls to '_resource_apply_...'
        self._init_notified = False

    def _create_slots(self, var_list):
        var_dtype = var_list[0].dtype.base_dtype
        if self._m_cache is None:
            self._m_cache = self.add_weight('momentum_cache', shape=[],
                                            dtype=var_dtype, initializer='ones',
                                            trainable=False)
            self._weights.append(self._m_cache)
        # Separate for-loops to respect the ordering of slot variables from v1.
        for var in var_list:
            self.add_slot(var, 'm')
        for var in var_list:
            self.add_slot(var, 'v')
        self._updates_per_iter = len(var_list)

    def _prepare(self, var_list):
        # Get the value of the momentum cache before starting to apply gradients.
        self._m_cache_read = array_ops.identity(self._m_cache)
        return super(NadamW, self)._prepare(var_list)

    def _resource_apply_dense(self, grad, var):
        var_dtype = var.dtype.base_dtype
        lr_t = array_ops.identity(self._get_hyper('learning_rate', var_dtype))
        beta_1_t = array_ops.identity(self._get_hyper('beta_1', var_dtype))
        beta_2_t = array_ops.identity(self._get_hyper('beta_2', var_dtype))
        epsilon_t = ops.convert_to_tensor(self.epsilon, var_dtype)
        m = self.get_slot(var, 'm')
        v = self.get_slot(var, 'v')
        local_step = math_ops.cast(self.iterations + 1, var_dtype)
        next_step = math_ops.cast(self.iterations + 2, var_dtype)
        decay_base = math_ops.cast(0.96, var_dtype)

        # Learning rate multipliers
        if self.lr_multipliers is not None:
            lr_t = _apply_lr_multiplier(self, lr_t, var)

        # Due to the recommendations in [2], i.e. warming momentum schedule
        momentum_cache_t = beta_1_t * (1. - 0.5 * (
            math_ops.pow(decay_base, self._initial_decay * local_step)))
        momentum_cache_t_1 = beta_1_t * (1. - 0.5 * (
            math_ops.pow(decay_base, self._initial_decay * next_step)))
        m_schedule_new = math_ops.cast(self._m_cache_read,
                                       var_dtype) * momentum_cache_t
        if var_dtype is self._m_cache.dtype:
            m_schedule_new = array_ops.identity(state_ops.assign(
                self._m_cache, m_schedule_new, use_locking=self._use_locking))
        m_schedule_next = m_schedule_new * momentum_cache_t_1

        # the following equations given in [1]
        g_prime = grad / (1. - m_schedule_new)
        m_t = beta_1_t * m + (1. - beta_1_t) * grad
        m_t_prime = m_t / (1. - m_schedule_next)
        v_t = beta_2_t * v + (1. - beta_2_t) * math_ops.square(grad)
        v_t_prime = v_t / (1. - math_ops.pow(beta_2_t, local_step))
        m_t_bar = (1. - momentum_cache_t) * g_prime + (
                momentum_cache_t * m_t_prime)

        m_t = state_ops.assign(m, m_t, use_locking=self._use_locking)
        v_t = state_ops.assign(v, v_t, use_locking=self._use_locking)

        var_t = math_ops.sub(var, self.eta_t * lr_t * m_t_bar / (
            math_ops.sqrt(v_t_prime + epsilon_t)))

        # Weight decays
        if var.name in self.weight_decays.keys():
            var_t = _apply_weight_decays(self, var, var_t)

        var_update = state_ops.assign(var, var_t, use_locking=self._use_locking)

        # Cosine annealing
        (iteration_done, t_cur_update, eta_t_update
         ) = _update_t_cur_eta_t_v2(self, lr_t, var)
        if iteration_done and not self._init_notified:
            self._init_notified = True

        updates = [var_update, m_t, v_t]
        if iteration_done:
            updates += [t_cur_update]
        if self.use_cosine_annealing and iteration_done:
            updates += [eta_t_update]
        return control_flow_ops.group(*updates)

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        var_dtype = var.dtype.base_dtype
        lr_t = array_ops.identity(self._get_hyper('learning_rate', var_dtype))
        beta_1_t = array_ops.identity(self._get_hyper('beta_1', var_dtype))
        beta_2_t = array_ops.identity(self._get_hyper('beta_2', var_dtype))
        epsilon_t = ops.convert_to_tensor(self.epsilon, var_dtype)
        m = self.get_slot(var, 'm')
        v = self.get_slot(var, 'v')
        local_step = math_ops.cast(self.iterations + 1, var_dtype)
        next_step = math_ops.cast(self.iterations + 2, var_dtype)
        decay_base = math_ops.cast(0.96, var_dtype)

        # Learning rate multipliers
        if self.lr_multipliers is not None:
            lr_t = _apply_lr_multiplier(self, lr_t, var)

        momentum_cache_t = beta_1_t * (1. - 0.5 * (
            math_ops.pow(decay_base, self._initial_decay * local_step)))
        momentum_cache_t_1 = beta_1_t * (1. - 0.5 * (
            math_ops.pow(decay_base, self._initial_decay * next_step)))
        m_schedule_new = math_ops.cast(self._m_cache_read,
                                       var_dtype) * momentum_cache_t
        if var_dtype is self._m_cache.dtype:
            m_schedule_new = array_ops.identity(state_ops.assign(
                self._m_cache, m_schedule_new, use_locking=self._use_locking))
        m_schedule_next = m_schedule_new * momentum_cache_t_1

        m_scaled_g_values = grad * (1. - beta_1_t)
        m_t = state_ops.assign(m, m * beta_1_t, use_locking=self._use_locking)
        with ops.control_dependencies([m_t]):
            m_t = self._resource_scatter_add(m, indices, m_scaled_g_values)
            m_t_slice = array_ops.gather(m_t, indices)

        m_t_prime = m_t_slice / (1. - m_schedule_next)
        g_prime = grad / (1. - m_schedule_new)
        m_t_bar = (1. - momentum_cache_t) * g_prime + (
                momentum_cache_t_1 * m_t_prime)

        v_scaled_g_values = (grad * grad) * (1. - beta_2_t)
        v_t = state_ops.assign(v, v * beta_2_t, use_locking=self._use_locking)

        with ops.control_dependencies([v_t]):
            v_t = self._resource_scatter_add(v, indices, v_scaled_g_values)
            v_t_slice = array_ops.gather(v_t, indices)

        v_t_prime_denominator = 1. - math_ops.pow(beta_2_t, local_step)
        v_t_prime = v_t_slice / v_t_prime_denominator
        v_prime_sqrt_plus_eps = math_ops.sqrt(v_t_prime) + epsilon_t

        var_t = self._resource_scatter_add(
            var, indices,
            -self.eta_t * lr_t * m_t_bar / v_prime_sqrt_plus_eps)

        # Weight decays
        if var.name in self.weight_decays.keys():
            var_t = _apply_weight_decays(self, var, var_t)

        var_update = state_ops.assign(var, var_t, use_locking=self._use_locking)

        # Cosine annealing
        (iteration_done, t_cur_update, eta_t_update
         ) = _update_t_cur_eta_t_v2(self, lr_t, var)
        if iteration_done and not self._init_notified:
            self._init_notified = True

        updates = [var_update, m_t_bar, v_t]
        if iteration_done:
            updates += [t_cur_update]
        if self.use_cosine_annealing and iteration_done:
            updates += [eta_t_update]
        return control_flow_ops.group(*updates)

    def get_config(self):
        config = super(NadamW, self).get_config()
        config.update({
            'learning_rate': self._serialize_hyperparameter('learning_rate'),
            'decay': self._serialize_hyperparameter('decay'),
            'beta_1': self._serialize_hyperparameter('beta_1'),
            'beta_2': self._serialize_hyperparameter('beta_2'),
            'epsilon': self.epsilon,
            'total_iterations': int(self.total_iterations),
            'weight_decays': self.weight_decays,
            'use_cosine_annealing': self.use_cosine_annealing,
            'autorestart': self.autorestart,
            't_cur': int(K_eval(self.t_cur)),
            'eta_t': float(K_eval(self.eta_t)),
            'eta_min': float(K_eval(self.eta_min)),
            'eta_max': float(K_eval(self.eta_max)),
            'init_verbose': self.init_verbose
        })
        return config


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'

    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, enable=True)
    print(gpus)
    cifar10 = tf.keras.datasets.cifar10

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)),
        tf.keras.layers.AveragePooling2D(),
        tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.AveragePooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    optimizer = AdamW(learning_rate=lr_schedule(0), weight_decay=wd_schedule(0))
    # optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    tb_callback = tf.keras.callbacks.TensorBoard(os.path.join('logs', 'adamw'), profile_batch=0)

    lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_schedule)
    wd_callback = WeightDecayScheduler(wd_schedule)

    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=40, validation_split=0.1,
              callbacks=[tb_callback, lr_callback, wd_callback])

    model.evaluate(x_test, y_test, verbose=2)
