#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2021/7/23 9:09 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/7/23 9:09   wangfc      1.0         None
"""
from .text_classifier_task import ClassifierTask


class ImageClassifier(ClassifierTask):
    """
    分类任务的父类
    """

    def __init__(self, dataset=None, data_dir=None, class_num=None, train_epochs=5, train_batch_size=8,
                 loss_name='sparse_categorical_crossentropy', optimizer_name='Adam', learning_rate=0.001,
                 metrics=['accuracy'],
                 model_type=None, input_image_size=(300, 300), output_size=1, output_activation='softmax',
                 run_eagerly=True):
        self.dataset = dataset
        self.data_dir = data_dir
        self.class_num = class_num

        self.train_epochs = train_epochs
        self.train_batch_size = train_batch_size
        self.loss_name = loss_name
        self.optimizer_name = optimizer_name
        self.learning_rate = learning_rate
        self.metrics = metrics

        self.model_type = model_type

        if input_image_size is not None:
            self.input_image_size = input_image_size
            self.input_size = (None,) + input_image_size
        # 在初始化 data 的时候 计算 input_size
        self.build_data()

        self.output_size = output_size
        self.output_activation = output_activation

        # model.compile() 的时候 debug 模式，这样　model.fit() 的时候可以进行断点调试
        self.run_eagerly = run_eagerly

        # self.data = self.build_data()
        self.model = self.build_model()

        logger.info(f"input_image_size={self.input_image_size},output_size={self.output_size},"
                    f"\nmodel_type={self.model_type},loss_name={self.loss_name},optimizer_name={self.optimizer_name}, learning_rate={self.learning_rate}")

    def build_data(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def build_model(self) -> Model:
        # if self.model_type == 'BaiscConvolutionModel':
        #     model = BaiscConvolutionModel(input_size=self.input_size, output_size=self.output_size)
        # elif self.model_type == 'ThreeConvolutionLayerModelV1':
        #     model = ThreeConvolutionLayerModelV1(input_size=self.input_size, output_size=self.output_size)
        # elif self.model_type == 'ThreeConvolutionLayerModelV2':
        #     model = ThreeConvolutionLayerModelV2(input_size=self.input_size, output_size=self.output_size)
        # elif self.model_type == 'TransferInceptionModel':
        #     model = TransferInceptionModel(input_size=self.input_size, output_size=self.output_size)
        #
        # elif self.model_type =='EasyResidualModel':
        #     model = EasyResidualModel(input_size=self.input_size,output_size=self.output_size)

        model_class = self.get_model_class(model_name=self.model_type)
        model = model_class(input_size=self.input_size, output_size=self.output_size)

        self.optimizer = self.build_optimizer()
        model.compile(optimizer=self.optimizer, loss=self.loss_name, metrics=self.metrics,
                      run_eagerly=self.run_eagerly)

        # build 的时候可能因为维度不正确而报错，在compile的时候 run_eagerly 进行调试
        model.build(input_shape=self.input_size)
        inputs = Input(shape=self.input_size[1:])
        outputs = model.call(inputs)

        logger.info(
            f"model_name={model.name},model.count_params={model.count_params()}\nmodel_summary={model.summary()}")  #
        return model
