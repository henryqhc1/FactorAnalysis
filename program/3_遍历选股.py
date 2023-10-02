"""
2023分享会
author: 邢不行
微信: xbx9585
选股策略框架
"""
import sys
import os
from program.Functions import get_file_in_folder, calc_period_offset
from program.Config import root_path

python_exe = sys.executable
stg_file = get_file_in_folder(root_path + '/program/选股策略/', '.py', filters=['_init_'], drop_type=True)

# 遍历每个策略的周期和offset
period_dict = calc_period_offset(stg_file)

for period in period_dict:
    for file in stg_file:
        cls = __import__('program.选股策略.%s' % file, fromlist=('',))
        if cls.period == period:
            os.system('%s %s/program/2_选股.py %s' % (python_exe, root_path, file))
