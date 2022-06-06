#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:
@time: 2021/5/19 21:57

模型定义方法
1. Sequential
    model = Sequential([tf.keras.layers.Flatten(),
                        tf.keras.layers.Dense(128, activation=tf.nn.relu),
                        tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

2. Functional API：
    1）先创建一个输入节点 ： instantiate the input Tensor
       inputs = tf.keras.layers.Input(shape=(28, 28))
    2）可以通过在此 inputs 对象上调用层，在层计算图中创建新的节点：
        # stack the layers using the syntax: new_layer()(previous_layer)
        flatten_layer = tf.keras.layers.Flatten()(input_layer)
        first_dense = tf.keras.layers.Dense(128, activation=tf.nn.relu)(flatten_layer)
        output_layer = tf.keras.layers.Dense(10, activation=tf.nn.softmax)(first_dense)

    3）通过 创建模型的 explict inputs and outputs来创建 Model：
       # 可以定义多个输入输出
       model = Model(inputs=inputs, outputs=outputs, name="mnist_model")


3. 自定义 layer 和 Model
    通过对 tf.keras.Model 进行子类化并定义自己的前向传播来构建完全可自定义的模型。

    在 __init__ 方法中创建层并将它们设置为类实例的属性
    在 call 方法中定义前向传播


@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/5/19 21:57   wangfc      1.0         None
"""
import os
import re
import copy
from collections import defaultdict
import random
from typing import Text, Optional, Any, Union, Tuple, Dict, List,Generator,Iterator

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras.callbacks import Callback, History
from tensorflow.python.keras.engine import training
from tensorflow.python.keras.engine import training_utils
from tensorflow.python.keras.engine import data_adapter
from tensorflow.python.keras.utils import version_utils
from tensorflow.python.eager import context
from tensorflow.python.keras.engine.data_adapter import DataHandler


# from rasa.utils.tensorflow.data_generator import RasaBatchDataGenerator, RasaDataGenerator
# from rasa.utils.train_utils import create_data_generators
# from rasa.utils.tensorflow.model_data import RasaModelData, FeatureSignature
from data_process.training_data.rasa_model_data import RasaBatchDataGenerator, RasaDataGenerator,\
    RasaModelData, FeatureSignature,create_data_generators

from utils.constants import DIAGNOSTIC_DATA
from utils.tensorflow import layers, rasa_layers
from utils.tensorflow.constants import *
import logging
logger = logging.getLogger()


# noinspection PyMethodOverriding
class TmpKerasModel(tf.keras.models.Model):
    """

    """

    @property
    def name(self) -> Text:
        """The name property is a function of the class - its __name__."""
        return self.__class__.__name__

    def _create_metrics(self) -> None:
        raise NotImplementedError

    def _prepare_layers(self) -> None:
        raise NotImplementedError

    def _prepare_loss(self) -> None:
        raise NotImplementedError


    def _get_savedmodel_subdirs(self, output_dir, pattern=r'tf_model_epoch_(\d{0,3})'):
        savedmodel_subdirs = []
        for name in os.listdir(output_dir):
            matched_subdir = re.match(pattern=pattern, string=name)
            if matched_subdir:
                savedmodel_subdirs.append({'subdir_name': name, 'epoch': int(matched_subdir.groups()[0])})
        sorted_savedmodel_subdirs = sorted(savedmodel_subdirs, key=lambda x: x['epoch'])
        return sorted_savedmodel_subdirs

    # def _export_model(self,filepath):
    #     """
    #     Model  cannot be saved because the input shapes have not been set.
    #     Usually, input shapes are automatically determined from calling `.fit()` or `.predict()`.
    #     To manually set the shapes, call `model.build(input_shape)`.
    #     """
    #     # tf.saved_model.save(self.model, export_dir=self._export_model_dir)
    #     tf.keras.models.save_model(model = self, filepath= filepath)

    def _load_savedmodel(self, savedmodel_subdir=None, compile=False) -> tf.keras.Model:
        """
        @time:  2021/6/22 11:19
        @author:wangfc
        @version:
        @description:
        主要有三种方式。
        1) saved_model下的 load API，但是这个要注意一点，就是它只能加载.pb文件，并且提供的路径参数是.pb文件所在的文件夹路径；
        2) model.load_weights API，它只适合加载只保存了权重的文件；
        3) tf.keras.models.load_model


        SavedModel 如何处理自定义对象
        https://www.tensorflow.org/guide/keras/save_and_serialize
            保存模型及其层时， SavedModel 格式存储类名、调用函数、损失和权重（以及配置，如果实现）。call 函数定义了模型/层的计算图。
            在没有模型/层配置的情况下，调用函数用于创建一个与原始模型一样存在的模型，该模型可以被训练、评估和用于推理。
            尽管如此，在编写自定义模型或层类时定义get_config 和from_config方法始终是一个好习惯。这允许您稍后在需要时轻松更新计算。有关 更多信息，请参阅有关自定义对象的部分。


        @params:
        @return:
        """
        # Option 1: Load with the custom_object argument.
        # 加载的模型是使用配置和CustomModel类加载的。
        # model = tf.keras.models.load_model(export_dir, custom_objects={"CustomModel": CustomModel}, compile=True)

        # Option 2: Load without the CustomModel class: custom_objects=None
        # 通过动态创建与原始模型类似的模型类来加载
        if savedmodel_subdir:
            savedmodel_dir = os.path.join(self.output_dir, savedmodel_subdir)
            model = tf.keras.models.load_model(savedmodel_dir, custom_objects=None, compile=compile)
            logger.info(f"Restoring from {savedmodel_dir}")
            return model


    """Temporary solution. Keras model that uses a custom data adapter inside fit."""

    # TODO
    #  we don't need this anymore once
    #  https://github.com/tensorflow/tensorflow/pull/45338
    #  is merged and released

    # This code is adapted from
    # https://github.com/tensorflow/tensorflow/blob/v2.3.1/tensorflow/python/keras/engine/training.py#L824-L1146

    @training.enable_multi_worker
    def fit(
        self,
        x: Optional[
            Union[np.ndarray, tf.Tensor, tf.data.Dataset, tf.keras.utils.Sequence]
        ] = None,
        y: Optional[
            Union[np.ndarray, tf.Tensor, tf.data.Dataset, tf.keras.utils.Sequence]
        ] = None,
        batch_size: Optional[int] = None,
        epochs: int = 1,
        verbose: int = 1,
        callbacks: Optional[List[Callback]] = None,
        validation_split: float = 0.0,
        validation_data: Optional[Any] = None,
        shuffle: bool = True,
        class_weight: Optional[Dict[int, float]] = None,
        sample_weight: Optional[np.ndarray] = None,
        initial_epoch: int = 0,
        steps_per_epoch: Optional[int] = None,
        validation_steps: Optional[int] = None,
        validation_batch_size: Optional[int] = None,
        validation_freq: int = 1,
        max_queue_size: int = 10,
        workers: int = 1,
        use_multiprocessing: bool = False,
    ) -> History:
        """Trains the model for a fixed number of epochs (iterations on a dataset).

        Args:
            x: Input data.
            y: Target data.
            batch_size: Number of samples per gradient update.
            epochs: Number of epochs to train the model.
            verbose: Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per
                     epoch.
            callbacks: List of `keras.callbacks.Callback` instances.
            validation_split: Fraction of the training data to be used as validation
                              data.
            validation_data: Data on which to evaluate the loss and any model metrics
                             at the end of each epoch.
            shuffle: whether to shuffle the training data before each epoch
            class_weight: Optional dictionary mapping class indices (integers)
                          to a weight (float) value, used for weighting the loss
                          function (during training only).
            sample_weight: Optional Numpy array of weights for
                           the training samples, used for weighting the loss function
                           (during training only).
            initial_epoch: Epoch at which to start training
            steps_per_epoch: Total number of steps (batches of samples)
                             before declaring one epoch finished and starting the
                             next epoch.
            validation_steps: Total number of steps (batches of
                              samples) to draw before stopping when performing
                              validation at the end of every epoch.
            validation_batch_size: Number of samples per validation batch.
            validation_freq: specifies how many training epochs to run before a
                             new validation run is performed
            max_queue_size: Maximum size for the generator queue.
            workers: Maximum number of processes to spin up
                     when using process-based threading.
            use_multiprocessing: If `True`, use process-based threading.

        Returns:
            A `History` object. Its `History.history` attribute is
            a record of training loss values and metrics values
            at successive epochs, as well as validation loss values
            and validation metrics values (if applicable).

        Raises:
            RuntimeError: 1. If the model was never compiled or,
            2. If `model.fit` is  wrapped in `tf.function`.

            ValueError: In case of mismatch between the provided input data
                and what the model expects.
        """
        training._keras_api_gauge.get_cell("fit").set(True)
        # Legacy graph support is contained in `training_v1.Model`.
        version_utils.disallow_legacy_graph("Model", "fit")
        self._assert_compile_was_called()
        self._check_call_args("fit")
        training._disallow_inside_tf_function("fit")

        if validation_split:
            # Create the validation data using the training data. Only supported for
            # `Tensor` and `NumPy` input.
            (
                (x, y, sample_weight),
                validation_data,
            ) = data_adapter.train_validation_split(
                (x, y, sample_weight), validation_split=validation_split
            )

        if validation_data:
            val_x, val_y, val_sample_weight = data_adapter.unpack_x_y_sample_weight(
                validation_data
            )

        with self.distribute_strategy.scope(), training_utils.RespectCompiledTrainableState(
            self
        ):
            # Creates a `tf.data.Dataset` and handles batch and epoch iteration.
            # Use our own custom data handler to handle increasing batch size
            data_handler = CustomDataHandler(
                x=x,
                y=y,
                sample_weight=sample_weight,
                batch_size=batch_size,
                steps_per_epoch=steps_per_epoch,
                initial_epoch=initial_epoch,
                epochs=epochs,
                shuffle=shuffle,
                class_weight=class_weight,
                max_queue_size=max_queue_size,
                workers=workers,
                use_multiprocessing=use_multiprocessing,
                model=self,
                steps_per_execution=self._steps_per_execution,
            )

            # Container that configures and calls `tf.keras.Callback`s.
            if not isinstance(callbacks, training.callbacks_module.CallbackList):
                callbacks = training.callbacks_module.CallbackList(
                    callbacks,
                    add_history=True,
                    add_progbar=verbose != 0,
                    model=self,
                    verbose=verbose,
                    epochs=epochs,
                    steps=data_handler.inferred_steps,
                )

            self.stop_training = False
            train_function = self.make_train_function()
            self._train_counter.assign(0)
            callbacks.on_train_begin()
            training_logs = None
            # Handle fault-tolerance for multi-worker.
            data_handler._initial_epoch = self._maybe_load_initial_epoch_from_ckpt(  # pylint: disable=protected-access
                initial_epoch
            )
            for epoch, iterator in data_handler.enumerate_epochs():
                self.reset_metrics()
                callbacks.on_epoch_begin(epoch)
                with data_handler.catch_stop_iteration():
                    for step in data_handler.steps():
                        with training.trace.Trace(
                            "TraceContext",
                            graph_type="train",
                            epoch_num=epoch,
                            step_num=step,
                            batch_size=batch_size,
                        ):
                            callbacks.on_train_batch_begin(step)
                            tmp_logs = train_function(iterator)
                            if data_handler.should_sync:
                                context.async_wait()
                            logs = tmp_logs  # No error, now safe to assign to logs.
                            end_step = step + data_handler.step_increment
                            callbacks.on_train_batch_end(end_step, logs)
                epoch_logs = copy.copy(logs)

                # Run validation.
                if validation_data and self._should_eval(epoch, validation_freq):
                    # Create data_handler for evaluation and cache it.
                    if getattr(self, "_eval_data_handler", None) is None:
                        self._eval_data_handler = CustomDataHandler(
                            x=val_x,
                            y=val_y,
                            sample_weight=val_sample_weight,
                            batch_size=validation_batch_size or batch_size,
                            steps_per_epoch=validation_steps,
                            initial_epoch=0,
                            epochs=1,
                            max_queue_size=max_queue_size,
                            workers=workers,
                            use_multiprocessing=use_multiprocessing,
                            model=self,
                            steps_per_execution=self._steps_per_execution,
                        )
                    val_logs = self.evaluate(
                        x=val_x,
                        y=val_y,
                        sample_weight=val_sample_weight,
                        batch_size=validation_batch_size or batch_size,
                        steps=validation_steps,
                        callbacks=callbacks,
                        max_queue_size=max_queue_size,
                        workers=workers,
                        use_multiprocessing=use_multiprocessing,
                        return_dict=True,
                    )
                    val_logs = {"val_" + name: val for name, val in val_logs.items()}
                    epoch_logs.update(val_logs)

                callbacks.on_epoch_end(epoch, epoch_logs)
                training_logs = epoch_logs
                if self.stop_training:
                    break

            # If _eval_data_handler exists, delete it after all epochs are done.
            if getattr(self, "_eval_data_handler", None) is not None:
                del self._eval_data_handler
            callbacks.on_train_end(logs=training_logs)
            return self.history


class CustomDataHandler(DataHandler):
    """Handles iterating over epoch-level `tf.data.Iterator` objects."""

    def enumerate_epochs(self) -> Generator[Tuple[int, Iterator], None, None]:
        """Yields `(epoch, tf.data.Iterator)`."""
        # TODO
        #  we don't need this anymore once
        #  https://github.com/tensorflow/tensorflow/pull/45338
        #  is merged and released

        # This code is adapted from
        # https://github.com/tensorflow/tensorflow/blob/v2.3.1/tensorflow/python/keras/engine/data_adapter.py#L1135-L1145

        with self._truncate_execution_to_epoch():
            data_iterator = iter(self._dataset)
            for epoch in range(self._initial_epoch, self._epochs):
                if self._insufficient_data:  # Set by `catch_stop_iteration`.
                    break
                if self._adapter.should_recreate_iterator():
                    data_iterator = iter(self._dataset)
                    # update number of steps for epoch as we might have an increasing
                    # batch size
                    self._inferred_steps = len(self._adapter._keras_sequence)
                yield epoch, data_iterator
                self._adapter.on_epoch_end()



# noinspection PyMethodOverriding
class RasaModel(TmpKerasModel):
    """Abstract custom Keras model.

     This model overwrites the following methods:
    - train_step
    - test_step
    - predict_step
    - save
    - load
    Cannot be used as tf.keras.Model.
    """

    def __init__(self, random_seed: Optional[int] = None, **kwargs: Any) -> None:
        """Initialize the RasaModel.

        Args:
            random_seed: set the random seed to get reproducible results
        """
        # make sure that keras releases resources from previously trained model
        tf.keras.backend.clear_session()
        super().__init__(**kwargs)

        self.total_loss = tf.keras.metrics.Mean(name="t_loss")
        self.metrics_to_log = ["t_loss"]

        self._training = None  # training phase should be defined when building a graph

        self.random_seed = random_seed
        self._set_random_seed()

        self._tf_predict_step = None
        self.prepared_for_prediction = False

    def _set_random_seed(self) -> None:
        random.seed(self.random_seed)
        tf.random.set_seed(self.random_seed)
        np.random.seed(self.random_seed)

    def batch_loss(
        self, batch_in: Union[Tuple[tf.Tensor], Tuple[np.ndarray]]
    ) -> tf.Tensor:
        """Calculates the loss for the given batch.

        Args:
            batch_in: The batch.

        Returns:
            The loss of the given batch.
        """
        raise NotImplementedError

    def prepare_for_predict(self) -> None:
        """Prepares tf graph fpr prediction.

        This method should contain necessary tf calculations
        and set self variables that are used in `batch_predict`.
        For example, pre calculation of `self.all_labels_embed`.
        """
        pass

    def batch_predict(
        self, batch_in: Union[Tuple[tf.Tensor], Tuple[np.ndarray]]
    ) -> Dict[Text, Union[tf.Tensor, Dict[Text, tf.Tensor]]]:
        """Predicts the output of the given batch.

        Args:
            batch_in: The batch.

        Returns:
            The output to predict.
        """
        raise NotImplementedError

    def train_step(
        self, batch_in: Union[Tuple[tf.Tensor], Tuple[np.ndarray]]
    ) -> Dict[Text, float]:
        """Performs a train step using the given batch.
        This method is called by `Model.make_train_function`.
        This typically includes the forward pass, loss calculation, backpropagation,and metric updates.

        https://keras.io/guides/customizing_what_happens_in_fit/#supporting-sampleweight-amp-classweight
        自定义 fit() 过程:
        customize what fit(): override the method train_step(self, data)

        相对于源码在计算 loss 的时候： 减少了 sample_weight, loss scale , gradient clipping

        Args:
            batch_in: The batch input.

        Returns:
            Training metrics: a dictionary mapping metric names (including the loss) to their current value.
            A `dict` containing values that will be passed to `tf.keras.callbacks.CallbackList.on_train_batch_end`.
            Example:`{'loss': 0.2, 'accuracy': 0.7}`.


        """
        self._training = True
        # TODO: 为什么要分开计算 loss 呢？
        # 源码中 使用 self.compiled_loss(y, y_pred, sample_weight, regularization_losses=self.losses) 直接计算损失
        # calculate supervision and regularization losses separately
        with tf.GradientTape(persistent=True) as tape:
            #  Forward pass + 计算 loss
            #  源码中可以自定义  computer_loss() 方法 来计算 loss，rasa 这里使用了 batch_loss() 方法
            #  单独获取 supervision losses
            prediction_loss = self.batch_loss(batch_in)
            # 相加所有的  regularization losses
            if self.losses:
                regularization_loss = tf.math.add_n(self.losses)
                total_loss = prediction_loss + regularization_loss
            else:
                total_loss = prediction_loss

        # TODO: 只更新 total_loss，不更新 metric：self.compiled_metrics.update_state(y, y_pred, sample_weight)
        self.total_loss.update_state(total_loss)

        # TODO: 为什么分开计算 prediction_gradients + regularization_gradients ，而后由合并 gradients
        # calculate the gradients that come from supervision signal
        prediction_gradients = tape.gradient(prediction_loss, self.trainable_variables)

        if self.losses:
            # calculate the gradients that come from regularization
            regularization_gradients = tape.gradient(
                regularization_loss, self.trainable_variables
            )
        # delete gradient tape manually
        # since it was created with `persistent=True` option
        del tape

        gradients = []
        if self.losses:
            for pred_grad, reg_grad in zip(prediction_gradients, regularization_gradients):
                if pred_grad is not None and reg_grad is not None:
                    # remove regularization gradient for variables
                    # that don't have prediction gradient
                    gradients.append(
                        pred_grad
                        + tf.where(pred_grad > 0, reg_grad, tf.zeros_like(reg_grad))
                    )
                else:
                    gradients.append(pred_grad)
        else:
            gradients = prediction_gradients
        # 更新参数  Update weights
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self._training = None

        return self._get_metric_results()

    def test_step(
        self, batch_in: Union[Tuple[tf.Tensor], Tuple[np.ndarray]]
    ) -> Dict[Text, float]:
        """Tests the model using the given batch.

        This method is used during validation.

        Args:
            batch_in: The batch input.

        Returns:
            Testing metrics.
        """
        self._training = False

        prediction_loss = self.batch_loss(batch_in)
        regularization_loss = tf.math.add_n(self.losses)
        total_loss = prediction_loss + regularization_loss
        # TODO: 只更新 total_loss
        self.total_loss.update_state(total_loss)

        self._training = None

        return self._get_metric_results()

    def predict_step(
        self, batch_in: Union[Tuple[tf.Tensor], Tuple[np.ndarray]]
    ) -> Dict[Text, tf.Tensor]:
        """Predicts the output for the given batch.

        Args:
            batch_in: The batch to predict.

        Returns:
            Prediction output.
        """
        self._training = False

        if not self.prepared_for_prediction:
            # in case the model is used for prediction without loading, e.g. directly
            # after training, we need to prepare the model for prediction once
            self.prepare_for_predict()
            self.prepared_for_prediction = True

        return self.batch_predict(batch_in)

    @staticmethod
    def _dynamic_signature(
        batch_in: Union[Tuple[tf.Tensor], Tuple[np.ndarray]]
    ) -> List[List[tf.TensorSpec]]:
        element_spec = []
        for tensor in batch_in:
            if len(tensor.shape) > 1:
                shape = [None] * (len(tensor.shape) - 1) + [tensor.shape[-1]]
            else:
                shape = [None]
            element_spec.append(tf.TensorSpec(shape, tensor.dtype))
        # batch_in is a list of tensors, therefore we need to wrap element_spec into
        # the list
        return [element_spec]

    def _rasa_predict(
        self, batch_in: Tuple[np.ndarray]
    ) -> Dict[Text, Union[np.ndarray, Dict[Text, Any]]]:
        """Custom prediction method that builds tf graph on the first call.

        Args:
            batch_in: Prepared batch ready for input to `predict_step` method of model.

        Return:
            Prediction output, including diagnostic data.
        """
        self._training = False
        if not self.prepared_for_prediction:
            # in case the model is used for prediction without loading, e.g. directly
            # after training, we need to prepare the model for prediction once
            self.prepare_for_predict()
            self.prepared_for_prediction = True

        if self._run_eagerly:
            outputs = tf_utils.to_numpy_or_python_type(self.predict_step(batch_in))
            if DIAGNOSTIC_DATA in outputs:
                outputs[DIAGNOSTIC_DATA] = self._empty_lists_to_none_in_dict(
                    outputs[DIAGNOSTIC_DATA]
                )
            return outputs

        if self._tf_predict_step is None:
            self._tf_predict_step = tf.function(
                self.predict_step, input_signature=self._dynamic_signature(batch_in)
            )

        outputs = tf_utils.to_numpy_or_python_type(self._tf_predict_step(batch_in))
        if DIAGNOSTIC_DATA in outputs:
            outputs[DIAGNOSTIC_DATA] = self._empty_lists_to_none_in_dict(
                outputs[DIAGNOSTIC_DATA]
            )
        return outputs

    def run_inference(
        self, model_data: RasaModelData, batch_size: Union[int, List[int]] = 1
    ) -> Dict[Text, Union[np.ndarray, Dict[Text, Any]]]:
        """Implements bulk inferencing through the model.

        Args:
            model_data: Input data to be fed to the model.
            batch_size: Size of batches that the generator should create.

        Returns:
            Model outputs corresponding to the inputs fed.
        """
        outputs = {}
        (data_generator, _,) = create_data_generators(
            model_data=model_data, batch_sizes=batch_size, epochs=1, shuffle=False,
        )
        data_iterator = iter(data_generator)
        while True:
            try:
                # data_generator is a tuple of 2 elements - input and output.
                # We only need input, since output is always None and not
                # consumed by our TF graphs.
                batch_in = next(data_iterator)[0]
                batch_out = self._rasa_predict(batch_in)
                outputs = self._merge_batch_outputs(outputs, batch_out)
            except StopIteration:
                # Generator ran out of batches, time to finish inferencing
                break
        return outputs

    @staticmethod
    def _merge_batch_outputs(
        all_outputs: Dict[Text, Union[np.ndarray, Dict[Text, np.ndarray]]],
        batch_output: Dict[Text, Union[np.ndarray, Dict[Text, np.ndarray]]],
    ) -> Dict[Text, Union[np.ndarray, Dict[Text, np.ndarray]]]:
        """Merges a batch's output into the output for all batches.

        Function assumes that the schema of batch output remains the same,
        i.e. keys and their value types do not change from one batch's
        output to another.

        Args:
            all_outputs: Existing output for all previous batches.
            batch_output: Output for a batch.

        Returns:
            Merged output with the output for current batch stacked
            below the output for all previous batches.
        """
        if not all_outputs:
            return batch_output
        for key, val in batch_output.items():
            if isinstance(val, np.ndarray):
                all_outputs[key] = np.concatenate(
                    [all_outputs[key], batch_output[key]], axis=0
                )

            elif isinstance(val, dict):
                # recurse and merge the inner dict first
                all_outputs[key] = RasaModel._merge_batch_outputs(all_outputs[key], val)

        return all_outputs

    @staticmethod
    def _empty_lists_to_none_in_dict(input_dict: Dict[Text, Any]) -> Dict[Text, Any]:
        """Recursively replaces empty list or np array with None in a dictionary."""

        def _recurse(x: Union[Dict[Text, Any], List[Any], np.ndarray]) -> Optional[Any]:
            if isinstance(x, dict):
                return {k: _recurse(v) for k, v in x.items()}
            elif (isinstance(x, list) or isinstance(x, np.ndarray)) and np.size(x) == 0:
                return None
            return x

        return _recurse(input_dict)

    def _get_metric_results(self, prefix: Optional[Text] = "") -> Dict[Text, float]:
        return {
            f"{prefix}{metric.name}": metric.result()
            for metric in self.metrics
            if metric.name in self.metrics_to_log
        }

    def save(self, model_file_name: Text, overwrite: bool = True) -> None:
        """Save the model to the given file.

        Args:
            model_file_name: The file name to save the model to.
            overwrite: If 'True' an already existing model with the same file name will
                       be overwritten.
        """
        self.save_weights(model_file_name, overwrite=overwrite, save_format="tf")

    @classmethod
    def load(
        cls,
        model_file_name: Text,
        model_data_example: RasaModelData,
        predict_data_example: Optional[RasaModelData] = None,
        finetune_mode: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> "RasaModel":
        """Loads a model from the given weights.

        Args:
            model_file_name: Path to file containing model weights.
            model_data_example: Example data point to construct the model architecture.
            predict_data_example: Example data point to speed up prediction during
              inference.
            finetune_mode: Indicates whether to load the model for further finetuning.
            *args: Any other non key-worded arguments.
            **kwargs: Any other key-worded arguments.

        Returns:
            Loaded model with weights appropriately set.
        """
        logger.debug(
            f"Loading the model from {model_file_name} "
            f"with finetune_mode={finetune_mode}..."
        )
        # create empty model
        model = cls(*args, **kwargs)
        learning_rate = kwargs.get("config", {}).get(LEARNING_RATE, 0.001)
        # need to train on 1 example to build weights of the correct size
        # load model 的时候 增加 run_eagerly 参数用于调试
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                      run_eagerly= kwargs.get("run_eagerly",True) )
        data_generator = RasaBatchDataGenerator(model_data_example, batch_size=1)
        model.fit(data_generator, verbose=False)
        # load trained weights
        model.load_weights(model_file_name)

        # predict on one data example to speed up prediction during inference
        # the first prediction always takes a bit longer to trace tf function
        if not finetune_mode and predict_data_example:
            model.run_inference(predict_data_example)

        logger.debug("Finished loading the model.")
        return model

    @staticmethod
    def batch_to_model_data_format(
        batch: Union[Tuple[tf.Tensor], Tuple[np.ndarray]],
        data_signature: Dict[Text, Dict[Text, List[FeatureSignature]]],
    ) -> Dict[Text, Dict[Text, List[tf.Tensor]]]:
        """Convert input batch tensors into batch data format.

        Batch contains any number of batch data. The order is equal to the
        key-value pairs in session data. As sparse data were converted into (indices,
        data, shape) before, this method converts them into sparse tensors. Dense
        data is kept.
        """
        # during training batch is a tuple of input and target data
        # as our target data is inside the input data, we are just interested in the
        # input data
        if isinstance(batch[0], Tuple):
            batch = batch[0]

        batch_data = defaultdict(lambda: defaultdict(list))
        # 根据 data_signature 转换为  batch_data
        idx = 0
        for key, values in data_signature.items():
            for sub_key, signature in values.items():
                for is_sparse, feature_dimension, number_of_dimensions in signature:
                    # we converted all 4D features to 3D features before
                    number_of_dimensions = (
                        number_of_dimensions if number_of_dimensions != 4 else 3
                    )
                    if is_sparse:
                        tensor, idx = RasaModel._convert_sparse_features(
                            batch, feature_dimension, idx, number_of_dimensions
                        )
                    else:
                        tensor, idx = RasaModel._convert_dense_features(
                            batch, feature_dimension, idx, number_of_dimensions
                        )
                    batch_data[key][sub_key].append(tensor)

        return batch_data

    @staticmethod
    def _convert_dense_features(
        batch: Union[Tuple[tf.Tensor], Tuple[np.ndarray]],
        feature_dimension: int,
        idx: int,
        number_of_dimensions: int,
    ) -> Tuple[tf.Tensor, int]:
        if isinstance(batch[idx], tf.Tensor):
            # explicitly substitute last dimension in shape with known
            # static value
            if number_of_dimensions > 1 and (
                batch[idx].shape is None or batch[idx].shape[-1] is None
            ):
                shape: List[Optional[int]] = [None] * (number_of_dimensions - 1)
                shape.append(feature_dimension)
                batch[idx].set_shape(shape)

            return batch[idx], idx + 1

        # convert to Tensor
        return (
            tf.constant(batch[idx], dtype=tf.float32, shape=batch[idx].shape),
            idx + 1,
        )

    @staticmethod
    def _convert_sparse_features(
        batch: Union[Tuple[tf.Tensor], Tuple[np.ndarray]],
        feature_dimension: int,
        idx: int,
        number_of_dimensions: int,
    ) -> Tuple[tf.SparseTensor, int]:
        # explicitly substitute last dimension in shape with known
        # static value
        shape = [batch[idx + 2][i] for i in range(number_of_dimensions - 1)] + [
            feature_dimension
        ]
        return tf.SparseTensor(indices=batch[idx], values=batch[idx + 1], dense_shape=shape), idx + 3

    def call(
        self,
        inputs: Union[tf.Tensor, List[tf.Tensor]],
        training: Optional[tf.Tensor] = None,
        mask: Optional[tf.Tensor] = None,
    ) -> Union[tf.Tensor, List[tf.Tensor]]:
        """Calls the model on new inputs.

        Arguments:
            inputs: A tensor or list of tensors.
            training: Boolean or boolean scalar tensor, indicating whether to run
              the `Network` in training mode or inference mode.
            mask: A mask or list of masks. A mask can be
                either a tensor or None (no mask).

        Returns:
            A tensor if there is a single output, or
            a list of tensors if there are more than one outputs.
        """
        # This method needs to be implemented, otherwise the super class is raising a
        # NotImplementedError('When subclassing the `Model` class, you should
        #   implement a `call` method.')
        pass


# noinspection PyMethodOverriding
class TransformerRasaModel(RasaModel):
    def __init__(
        self,
        name: Text,
        config: Dict[Text, Any],
        data_signature: Dict[Text, Dict[Text, List[FeatureSignature]]],
        label_data: RasaModelData,
    ) -> None:
        super().__init__(
            name=name, random_seed=config[RANDOM_SEED],
        )

        self.config = config
        # 输入数据的格式：
        self.data_signature = data_signature
        # 输出数据的格式
        self.label_signature = label_data.get_signature()

        self._check_data()
        # 获取 tf_label_data : key，sub_key 对应的 features
        label_batch = RasaDataGenerator.prepare_batch(label_data.data)
        self.tf_label_data = self.batch_to_model_data_format(
            label_batch, self.label_signature
        )

        # set up tf layers
        self._tf_layers: Dict[Text, tf.keras.layers.Layer] = {}

    def _check_data(self) -> None:
        raise NotImplementedError

    def _prepare_layers(self) -> None:
        raise NotImplementedError

    def _prepare_label_classification_layers(self, predictor_attribute: Text) -> None:
        """Prepares layers & loss for the final label prediction step.
        建立 2个 编码层  ： 用于 train + infernce 的时候计算 编码
        1. embedding sentence 等 predictor_attribute
        2. embedding label

        建立 1 个 loss层 ： 只用于 train ： 计算 similarity 和 loss
                          inference 的时候计算  confidence
        """
        self._prepare_embed_layers(predictor_attribute)
        self._prepare_embed_layers(LABEL)

        self._prepare_dot_product_loss(LABEL, self.config[SCALE_LOSS])

    def _prepare_embed_layers(self, name: Text, prefix: Text = "embed") -> None:
        self._tf_layers[f"{prefix}.{name}"] = layers.Embed(
            self.config[EMBEDDING_DIMENSION],
            self.config[REGULARIZATION_CONSTANT],
            name,
        )

    def _prepare_ffnn_layer(
        self,
        name: Text,
        layer_sizes: List[int],
        drop_rate: float,
        prefix: Text = "ffnn",
    ) -> None:
        self._tf_layers[f"{prefix}.{name}"] = layers.Ffnn(
            layer_sizes,
            drop_rate,
            self.config[REGULARIZATION_CONSTANT],
            self.config[CONNECTION_DENSITY],
            layer_name_suffix=name,
        )

    def _prepare_dot_product_loss(
        self, name: Text, scale_loss: bool, prefix: Text = "loss"
    ) -> None:
        self._tf_layers[f"{prefix}.{name}"] = layers.DotProductLoss(
            self.config[NUM_NEG],
            self.config[LOSS_TYPE],
            self.config[MAX_POS_SIM],
            self.config[MAX_NEG_SIM],
            self.config[USE_MAX_NEG_SIM],
            self.config[NEGATIVE_MARGIN_SCALE],
            scale_loss,
            similarity_type=self.config[SIMILARITY_TYPE],
            constrain_similarities=self.config[CONSTRAIN_SIMILARITIES],
            model_confidence=self.config[MODEL_CONFIDENCE],
        )

    def _prepare_entity_recognition_layers(self) -> None:
        for tag_spec in self._entity_tag_specs:
            name = tag_spec.tag_name
            num_tags = tag_spec.num_tags
            self._tf_layers[f"embed.{name}.logits"] = layers.Embed(
                num_tags, self.config[REGULARIZATION_CONSTANT], f"logits.{name}"
            )
            self._tf_layers[f"crf.{name}"] = layers.CRF(
                num_tags, self.config[REGULARIZATION_CONSTANT], self.config[SCALE_LOSS]
            )
            self._tf_layers[f"embed.{name}.tags"] = layers.Embed(
                self.config[EMBEDDING_DIMENSION],
                self.config[REGULARIZATION_CONSTANT],
                f"tags.{name}",
            )

    @staticmethod
    def _last_token(x: tf.Tensor, sequence_lengths: tf.Tensor) -> tf.Tensor:
        last_sequence_index = tf.maximum(0, sequence_lengths - 1)
        batch_index = tf.range(tf.shape(last_sequence_index)[0])

        indices = tf.stack([batch_index, last_sequence_index], axis=1)
        return tf.gather_nd(x, indices)

    def _get_mask_for(
        self,
        tf_batch_data: Dict[Text, Dict[Text, List[tf.Tensor]]],
        key: Text,
        sub_key: Text,
    ) -> Optional[tf.Tensor]:
        if key not in tf_batch_data or sub_key not in tf_batch_data[key]:
            return None

        sequence_lengths = tf.cast(tf_batch_data[key][sub_key][0], dtype=tf.int32)
        return rasa_layers.compute_mask(sequence_lengths)

    def _get_sequence_feature_lengths(
        self, tf_batch_data: Dict[Text, Dict[Text, List[tf.Tensor]]], key: Text
    ) -> tf.Tensor:
        """Fetches the sequence lengths of real tokens per input example.

        The number of real tokens for an example is the same as the length of the
        sequence of the sequence-level (token-level) features for that input example.
        """
        if key in tf_batch_data and SEQUENCE_LENGTH in tf_batch_data[key]:
            return tf.cast(tf_batch_data[key][SEQUENCE_LENGTH][0], dtype=tf.int32)

        batch_dim = self._get_batch_dim(tf_batch_data[key])
        return tf.zeros([batch_dim], dtype=tf.int32)

    def _get_sentence_feature_lengths(
        self, tf_batch_data: Dict[Text, Dict[Text, List[tf.Tensor]]], key: Text,
    ) -> tf.Tensor:
        """Fetches the sequence lengths of sentence-level features per input example.

        This is needed because we treat sentence-level features as token-level features
        with 1 token per input example. Hence, the sequence lengths returned by this
        function are all 1s if sentence-level features are present, and 0s otherwise.
        """
        batch_dim = self._get_batch_dim(tf_batch_data[key])

        if key in tf_batch_data and SENTENCE in tf_batch_data[key]:
            return tf.ones([batch_dim], dtype=tf.int32)

        return tf.zeros([batch_dim], dtype=tf.int32)

    @staticmethod
    def _get_batch_dim(attribute_data: Dict[Text, List[tf.Tensor]]) -> int:
        # All the values in the attribute_data dict should be lists of tensors, each
        # tensor of the shape (batch_dim, ...). So we take the first non-empty list we
        # encounter and infer the batch size from its first tensor.
        for key, data in attribute_data.items():
            if data:
                return tf.shape(data[0])[0]
        return None

    def _calculate_entity_loss(
        self,
        inputs: tf.Tensor,
        tag_ids: tf.Tensor,
        mask: tf.Tensor,
        sequence_lengths: tf.Tensor,
        tag_name: Text,
        entity_tags: Optional[tf.Tensor] = None,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:

        tag_ids = tf.cast(tag_ids[:, :, 0], tf.int32)

        if entity_tags is not None:
            _tags = self._tf_layers[f"embed.{tag_name}.tags"](entity_tags)
            inputs = tf.concat([inputs, _tags], axis=-1)
        # logits = (batch_size, max_seq_length, tag_num)
        logits = self._tf_layers[f"embed.{tag_name}.logits"](inputs)

        # should call first to build weights
        pred_ids, _ = self._tf_layers[f"crf.{tag_name}"](logits, sequence_lengths)
        # 计算 loss
        loss = self._tf_layers[f"crf.{tag_name}"].loss(
            logits, tag_ids, sequence_lengths
        )
        f1 = self._tf_layers[f"crf.{tag_name}"].f1_score(tag_ids, pred_ids, mask)

        return loss, f1, logits

    def batch_loss(
        self, batch_in: Union[Tuple[tf.Tensor], Tuple[np.ndarray]]
    ) -> tf.Tensor:
        """Calculates the loss for the given batch.

        Args:
            batch_in: The batch.

        Returns:
            The loss of the given batch.
        """
        raise NotImplementedError

    def batch_predict(
        self, batch_in: Union[Tuple[tf.Tensor], Tuple[np.ndarray]]
    ) -> Dict[Text, Union[tf.Tensor, Dict[Text, tf.Tensor]]]:
        """Predicts the output of the given batch.

        Args:
            batch_in: The batch.

        Returns:
            The output to predict.
        """
        raise NotImplementedError

