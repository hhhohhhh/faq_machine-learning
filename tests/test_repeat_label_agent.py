#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2021/4/12 16:39 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/4/12 16:39   wangfc      1.0         None

"""
from parlai.core.agents import Agent
from parlai.core.params import ParlaiParser
from parlai.core.worlds import create_task
import random


class RepeatLabelAgent(Agent):
    # initialize by setting id
    def __init__(self, opt):
        self.id = 'RepeatLabel'
    # store observation for later, return it unmodified
    def observe(self, observation):
        self.observation = observation
        return observation
    # return label from before if available
    def act(self):
        reply = {'id': self.id}
        # 只是简单的返回 labels
        if 'labels' in self.observation:
            reply['text'] = ', '.join(self.observation['labels'])
        # 返回 label_candidates
        elif 'label_candidates' in self.observation:
            cands = self.observation['label_candidates']
            reply['text'] = random.choice(list(cands))
        else:
            reply['text'] = "I don't know."
        return reply

if __name__ == '__main__':
    task = 'convai2'
    teacher = 'both'
    num = 1
    total_task = f'{task}:{teacher}:{num}'
    kwargs = {'task': 'convai2:both:1'}

    parser = ParlaiParser()
    opt = parser.parse_kwargs(**kwargs)
    # 创建 agent = RepeatLabelAgent
    agent = RepeatLabelAgent(opt)
    # 创建 world = parlai.core.worlds.DialogPartnerWorld
    world = create_task(opt, agent)

    for _ in range(10):
        world.parley()
        print(world.display())
        if world.epoch_done():
            print('EPOCH DONE')
            break
