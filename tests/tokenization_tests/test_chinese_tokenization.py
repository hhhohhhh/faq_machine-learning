#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@author: wangfc
@site: http://www.***
@time: 2020/7/14 22:20

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2020/7/14 22:20   wangfc      1.0         None

 * 密级：秘密
 * 版权所有：*** 2019
 * 注意：本内容仅限于***内部传阅，禁止外泄以及用于其他的商业目的

"""
import unittest
from tokenizations.tokenization import chinese_tokenize


class TokenizationTest(unittest.TestCase):
    def test_chinese_tokenizer(self):
        content = '英文里表示char，所在的,第10abc个 词'
        doc_tokens, char_to_word_offset = chinese_tokenize(content)
        self.assertEqual(doc_tokens, ['英', '文', '里', '表', '示', 'char', '，', '所', '在', '的', ',', '第', '10', 'abc','个',' ', '词'])
        self.assertEqual(char_to_word_offset, [0, 1, 2, 3, 4, 5, 5, 5, 5, 6, 7, 8, 9, 10, 11, 12,12, 13,13,13, 14,15,16])


if __name__ == '__main__':
    unittest.main()
