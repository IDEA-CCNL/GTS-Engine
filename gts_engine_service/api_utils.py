# -*- encoding: utf-8 -*-
'''
Copyright 2022 The International Digital Economy Academy (IDEA). CCNL team. All rights reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
@File    :   api_utils.py
@Time    :   2022/10/31 18:30
@Author  :   Kunhao Pan
@Version :   1.0
@Contact :   pankunhao@idea.edu.cn
@License :   (C)Copyright 2022-2023, CCNL-IDEA
'''

import os
import json

def list_task(task_dir):
    if not os.path.exists(task_dir):
        return []
    tasks = os.listdir(task_dir)
    return tasks

def is_task_valid(task_dir, task_id):
    tasks = list_task(task_dir)
    tasks = set(tasks)
    return (task_id in tasks)

def is_data_format_valid(data_path, data_type):
    print(data_path)
    if not os.path.exists(data_path):
        return False
    valid = True
    with open(data_path, 'r', encoding='utf8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
            except:
                valid = False
                break
            if data_type == 'train' or data_type == 'dev':
                if "content" not in data or "label" not in data:
                    valid = False
                    break
            if data_type == 'test':
                if "content" not in data:
                    valid = False
                    break
            if data_type == 'label':
                if "labels" not in data:
                    valid = False 
    return valid

    