#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file:  from rasa.untils.tensorflow.environment
@version: 
@desc:  
@time: 2021/5/19 22:26 


@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/5/19 22:26   wangfc      1.0         None
"""

from typing import Text, Dict
from utils.tensorflow.constants import ENV_GPU_CONFIG, ENV_CPU_INTER_OP_CONFIG, ENV_CPU_INTRA_OP_CONFIG,CUDA_VISIBLE_DEVICES

import os
import sys
import logging
logger = logging.getLogger(__name__)


def get_gpu_memory_config(gpu_no=0,gpu_memory_fraction=0.5,gpu_memory_total = 1024 * 16):
    gpu_memory =  int(gpu_memory_total*gpu_memory_fraction)
    gpu_memory_config = f"{gpu_no}:{gpu_memory}"
    return gpu_memory_config


def _setup_gpu_environment(gpu_memory_config='0:1024') -> None:
    """Sets configuration for TensorFlow GPU environment based on env variable.
    gpu_memory_config='0:1024' 0 表示 第0块 gpu，1024 表示内存使用 1024M
    """
    import tensorflow as tf
    if gpu_memory_config is None:
        gpu_memory_config = os.getenv(ENV_GPU_CONFIG)

    if not gpu_memory_config:
        return

    parsed_gpu_config = _parse_gpu_config(gpu_memory_config)

    if tf.__version__ >= "2.0":
        from tensorflow import config as tf_config
        # 默认情况下 TensorFlow 会使用其所能够使用的所有 GPU。
        # tf.config.experimental.set_visible_devices(devices=gpus[2:4], device_type='GPU')
        # 设置之后，当前程序只会使用自己可见的设备，不可见的设备不会被当前程序使用。
        # 另一种方式是使用环境变量 CUDA_VISIBLE_DEVICES 也可以控制程序所使用的 GPU:
        # export CUDA_VISIBLE_DEVICES= 2,3
        # import os
        # os.environ[CUDA_VISIBLE_DEVICES] = ",".join(map(str,list(parsed_gpu_config.keys())))

        # Import from tensorflow only if necessary (environment variable was set)

        # 获得当前主机上特定运算设备的列表
        physical_gpus = tf_config.list_physical_devices("GPU")

        # Logic taken from https://www.tensorflow.org/guide/gpu
        if physical_gpus:
            for gpu_id, gpu_id_memory in parsed_gpu_config.items():
                tf.config.experimental.set_visible_devices(devices=physical_gpus[gpu_id], device_type='GPU')
                _allocate_gpu_memory(physical_gpus[gpu_id], gpu_id_memory)
        elif ENV_GPU_CONFIG is not None:
            logger.warning(
                f"You have an environment variable '{ENV_GPU_CONFIG}' set but no GPUs were "
                f"detected to configure."
            )
        else:
            logger.warning(
                f"gpu_memory_config = '{gpu_memory_config}'but no GPUs were "
                f"detected to configure."
            )
        logger.info(f"我们使用 GPU device={gpu_memory_config}")
    else:
        # 使用第一张与第三张GPU卡
        gpu_no = str(list(parsed_gpu_config.keys())[0])
        if gpu_no != '-1':
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_no
            gpu_memory= parsed_gpu_config[int(gpu_no)]
            total_memory = 16*1024
            gpu_memory_fraction = gpu_memory / total_memory
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
            tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
            logger.info(f"我们使用 device = GPU:{gpu_memory_config}")
        else:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def set_ld_library_environment(ld_library_environment_name ="LD_LIBRARY_PATH"):
    """
    LD_LIBRARY_PATH属于进程中的环境变量，Python代码开始运行后是无法修改的，即使通过os.environ['LD_LIBRARY_PATH']='path'来设置也不起作用，
    有个变通的方式是设置os.environ['LD_LIBRARY_PATH']后，调用os.execv重新启动程序。
    前提是python脚本必须是可执行的，如果不可执行，请运行chmod a+x a.py将其变为可执行。

    export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/cuda-10.1/lib64:/usr/local/cuda-10.0/lib64:$HOME/home/anaconda3/lib:"

    """
    original_path = os.getenv(ld_library_environment_name)
    LD_LIBRARY_PATH= "/usr/local/cuda/lib64:/usr/local/cuda-10.1/lib64:/usr/local/cuda-10.0/lib64:$HOME/home/anaconda3/lib:"

    if original_path== LD_LIBRARY_PATH:
        pass
    else:
        if original_path is None:
            os.environ[ld_library_environment_name] = LD_LIBRARY_PATH
        else:
            os.environ[ld_library_environment_name] = original_path+ ":" + LD_LIBRARY_PATH

        # try:
        #     print(script_path)
        #     os.execv(script_path,sys.argv)
        # except Exception as e:
        #     logger.error(e,exc_info=True)
        #     sys.exit(2)
        print(os.environ[ld_library_environment_name])






def _parse_gpu_config(gpu_memory_config: Text) -> Dict[int, int]:
    """Parse GPU configuration variable from a string to a dict.

    Args:
        gpu_memory_config: String containing the configuration for GPU usage.

    Returns:
        Parsed configuration as a dictionary with GPU IDs as keys and requested memory
        as the value.
    """

    # gpu_config is of format "gpu_id_1:gpu_id_1_memory, gpu_id_2: gpu_id_2_memory"
    # Parse it and store in a dictionary
    parsed_gpu_config = {}

    try:
        for instance in gpu_memory_config.split(","):
            instance_gpu_id, instance_gpu_mem = instance.split(":")
            instance_gpu_id = int(instance_gpu_id)
            instance_gpu_mem = int(instance_gpu_mem)

            parsed_gpu_config[instance_gpu_id] = instance_gpu_mem
    except ValueError:
        # Helper explanation of where the error comes from
        raise ValueError(
            f"Error parsing GPU configuration. Please cross-check the format of "
            f"'{ENV_GPU_CONFIG}' at https://rasa.com/docs/rasa/tuning-your-model"
            f"#restricting-absolute-gpu-memory-available ."
        )

    return parsed_gpu_config



def _allocate_gpu_memory(
    gpu_instance: "tf_config.PhysicalDevice", logical_memory: int
) -> None:
    """Create a new logical device for the requested amount of memory.

    ensorFlow 提供两种显存使用策略，让我们能够更灵活地控制程序的显存使用方式：

    1. 仅在需要时申请显存空间（程序初始运行时消耗很少的显存，随着程序的运行而动态申请显存）；
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    2. 限制消耗固定大小的显存（程序不会超出限定的显存大小，若超出的报错）。

    # 只有单GPU的环境模拟多GPU进行调试
    tf.config.experimental.set_virtual_device_configuration(
    gpus[0],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048),
     tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])


    Args:
        gpu_instance: PhysicalDevice instance of a GPU device.
        logical_memory: Absolute amount of memory to be allocated to the new logical
            device.
    """
    import tensorflow as tf
    try:
        if tf.__version__ >= "2.0":
            from tensorflow import config as tf_config
        # 设置Tensorflow固定消耗GPU: gpu_instance  的 logical_memory MB显存
        tf_config.experimental.set_virtual_device_configuration(
            gpu_instance,
            [
                tf_config.experimental.VirtualDeviceConfiguration(
                    memory_limit=logical_memory
                )
            ],
        )

    except RuntimeError:
        # Helper explanation of where the error comes from
        raise RuntimeError(
            "Error while setting up tensorflow environment. "
            "Virtual devices must be set before GPUs have been initialized."
        )


def _setup_cpu_environment(inter_op_parallel_threads='1',intra_op_parallel_threads='1') -> None:
    """Set configuration for the CPU environment based on environment variable."""
    import tensorflow as tf
    # 首先若不加任何配置情况下，是默认使用gpu的
    os.environ[CUDA_VISIBLE_DEVICES] = "-1"

    if inter_op_parallel_threads is None or intra_op_parallel_threads is None:
        inter_op_parallel_threads = os.getenv(ENV_CPU_INTER_OP_CONFIG)
        intra_op_parallel_threads = os.getenv(ENV_CPU_INTRA_OP_CONFIG)

    if not inter_op_parallel_threads and not intra_op_parallel_threads:
        return

    if tf.__version__ >= "2.0":
        from tensorflow import config as tf_config
        if inter_op_parallel_threads:
            try:
                inter_op_parallel_threads = int(inter_op_parallel_threads.strip())
            except ValueError:
                raise ValueError(
                    f"Error parsing the environment variable '{ENV_CPU_INTER_OP_CONFIG}'. "
                    f"Please cross-check the value."
                )

            tf_config.threading.set_inter_op_parallelism_threads(inter_op_parallel_threads)

        if intra_op_parallel_threads:
            try:
                intra_op_parallel_threads = int(intra_op_parallel_threads.strip())
            except ValueError:
                raise ValueError(
                    f"Error parsing the environment variable '{ENV_CPU_INTRA_OP_CONFIG}'. "
                    f"Please cross-check the value."
                )
            tf_config.threading.set_intra_op_parallelism_threads(intra_op_parallel_threads)
    else:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    logger.info("我们使用 device = CPU")



def setup_tf_environment(gpu_memory_config='0:1024') -> None:
    """Setup CPU and GPU related environment settings for TensorFlow."""
    if gpu_memory_config and gpu_memory_config.split(':')[0]!= "-1":
        # use_cpu = False
        _setup_gpu_environment(gpu_memory_config=gpu_memory_config)
    else:
        _setup_cpu_environment()


