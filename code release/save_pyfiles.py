import os

#  Copyright (c) 2016, NVIDIA Corporation
#  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of NVIDIA Corporation nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import glob, os, shutil, sys, time, string, warnings, datetime
from collections import OrderedDict
import numpy as np


# if __name__ != '__main__':
#    import lasagne

# ----------------------------------------------------------------------------

def shape_to_str(shape):
    str = ['%d' % v if v else '?' for v in shape]
    return ', '.join(str) if len(str) else ''


# ----------------------------------------------------------------------------

# def generate_network_topology_info(layers):
#    yield "%-30s%-20s%-10s%-20s%s" % ('LayerName', 'LayerType', 'Params', 'OutputShape', 'WeightShape')
#    yield "%-30s%-20s%-10s%-20s%s" % (('---',) * 5)
#
#    total_params = 0
#    for layer in lasagne.layers.get_all_layers(layers):
#        type_str = type(layer).__name__
#        outshape = lasagne.layers.get_output_shape(layer)
#        try:
#            weights = layer.W.get_value()
#        except:
#            try:
#                weights = layer.W_param.get_value()
#            except:
#                weights = np.zeros(())
#        nparams = lasagne.layers.count_params(layer, trainable = True) - total_params
#
#        weight_str = shape_to_str(weights.shape) if type_str != 'DropoutLayer' else 'p = %g' % layer.p
#        yield "%-30s%-20s%-10d%-20s%s" % (layer.name, type_str, nparams, shape_to_str(outshape), weight_str)
#        total_params += nparams
#
#    yield "%-30s%-20s%-10s%-20s%s" % (('---',) * 5)
#    yield "%-30s%-20s%-10d%-20s%s" % ('Total', '', total_params, '', '')

# ----------------------------------------------------------------------------

def create_result_subdir(result_dir, run_desc):
    ordinal = 0
    for fname in glob.glob(os.path.join(result_dir, '*')):
        try:
            fbase = os.path.basename(fname)
            ford = int(fbase[:fbase.find('-')])
            ordinal = max(ordinal, ford + 1)
        except ValueError:
            pass

    result_subdir = os.path.join(result_dir, '%03d-%s' % (ordinal, run_desc))
    if os.path.isdir(result_subdir):
        return create_result_subdir(result_dir, run_desc)  # Retry.
    if not os.path.isdir(result_subdir):
        os.makedirs(result_subdir)
    return result_subdir


# ----------------------------------------------------------------------------

def export_pyfile(target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    for ext in ('py', 'pyproj', 'sln'):
        for fn in glob.glob('*.' + ext):
            shutil.copy2(fn, target_dir)
        if os.path.isdir('src'):
            for fn in glob.glob(os.path.join('src', '*.' + ext)):
                shutil.copy2(fn, target_dir)


import os
import shutil


def get_MD5(file_path):
    files_md5 = os.popen('md5 %s' % file_path).read().strip()
    file_md5 = files_md5.replace('MD5 (%s) = ' % file_path, '')
    return file_md5


def find_subfolder(path):
    p = path.split("/")

    if p[-1] == "":
        subfolder = p[-2]
    else:
        subfolder = p[-1]
    return subfolder


def export_folders(required_folders, saved_to_path):
    for folder in required_folders:

        subfolder = find_subfolder(folder)
        # print("subfolder", subfolder)
        # print("saved_to_path", saved_to_path)
        if subfolder in saved_to_path:
            return

        back_folder = os.path.join(saved_to_path, folder.split("./")[-1])

        if not os.path.exists(back_folder):
            os.makedirs(back_folder)
        for files in os.listdir(folder):
            name = os.path.join(folder, files)
            back_name = os.path.join(back_folder, files)

            if "__pycache__" in back_name:
                continue

            if os.path.isfile(name):

                if os.path.isfile(back_name):
                    if get_MD5(name) != get_MD5(back_name):
                        shutil.copy(name, back_name)
                else:
                    shutil.copy(name, back_name)
            else:
                if not os.path.isdir(back_name):
                    os.makedirs(back_name)

                export_folders(name, back_name)


def export_folders_v2(required_folders, saved_to_path):
    for folder in required_folders:
        target_path = saved_to_path
        if not os.path.exists(target_path):
            os.makedirs(target_path)
        source_path = folder
        shutil.copytree(source_path, target_path)


# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------

class GenericCSV(object):
    def __init__(self, fname, *fields):
        self.fields = fields
        self.fout = open(fname, 'wt')
        # self.fout.write(string.join(fields, ',') + '\n')
        self.fout.write(",".join(fields) + '\n')
        self.fout.flush()

    def add_data(self, *values):
        assert len(values) == len(self.fields)
        strings = [v if isinstance(v, str) else '%g' % v for v in values]
        # self.fout.write(string.join(strings, ',') + '\n')
        self.fout.write(",".join(strings) + '\n')
        self.fout.flush()

    def close(self):
        self.fout.close()

    def __enter__(self):  # for 'with' statement
        return self

    def __exit__(self, *excinfo):
        self.close()


# ----------------------------------------------------------------------------

def merge_csv_reports(result_dir):
    print('Merging CSV reports in', result_dir)
    print()

    # List runs.

    subdirs = os.listdir(result_dir)
    max_digits = max([3] + [subdir.find('-') for subdir in subdirs if subdir[0] in '0123456789'])

    runs = []
    for subdir in subdirs:
        if subdir[0] in '0123456789':
            run_path = os.path.join(result_dir, subdir)
            if os.path.isdir(run_path):
                run_id = '0' * (max_digits - max(subdir.find('-'), 0)) + subdir
                runs.append((run_id, run_path))
    runs.sort()

    # Collect rows.

    all_rows = []
    for run_id, run_path in runs:
        print(run_id)
        run_rows = []
        for csv in glob.glob(os.path.join(run_path, '*.csv')):
            with open(csv, 'rt') as file:
                lines = [line.strip().split(',') for line in file.readlines()]
            run_rows += [OrderedDict([('RunID', run_id)] + zip(lines[0], line)) for line in lines[1:]]
            if len(lines) >= 2 and 'Epoch' in run_rows[-1] and run_rows[-1]['Epoch']:
                run_rows.append(OrderedDict(run_rows[-1]))
                run_rows[-1]['Epoch'] = ''
        all_rows += run_rows

    # Format output.

    fields = ('Stat', 'Value', 'RunID', 'Epoch')
    lines = []
    for row in all_rows:
        stats = [stat for stat in row.iterkeys() if stat not in fields]
        rest = [row.get(field, '') for field in fields[2:]]
        lines += [[stat, row[stat]] + rest for stat in stats]

    # Write CSV.

    fname = os.path.join(result_dir, 'merged.csv')
    print()
    print("Writing", fname)

    with open(fname, 'wt') as file:
        file.write(string.join(fields, ',') + '\n')
        for line in lines:
            file.write(string.join(line, ',') + '\n')

    print('Done.')
    print()


def save_runfile(save_folder, args):
    #save_folder = os.path.join("./", "output", nowtime+"-"+'%s_%.2f_%s_%s_%s' % (args.dataset, args.r, args.noise_mode, args.partition, str(args.num_users)))
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    # checkpath saving
    folder = os.path.join(save_folder, "folder_for_pyfiles")
    if not os.path.exists(folder):
        os.makedirs(folder)

    # current python file saving
    export_pyfile(folder)
    export_folders(
        required_folders=["./model", "./util"], saved_to_path=folder)
    
    argsDict = args.__dict__
    with open(os.path.join(save_folder, 'args-setting.txt'), 'w') as f:
        f.writelines('------------------ start ------------------' + '\n')
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
        f.writelines('------------------- end -------------------')

    return save_folder





import smtplib
from email.mime.text import MIMEText

def program_start(meg, args):
    text = 'Your program is started! Please check in time.\n\n' \
        + '{}\n\n'.format(meg) \
            + 'This is an automatic Email!\n\nNot reply.\nThanks.'
    msg = MIMEText(text, 'plain', 'utf-8')
    msg['From'] = 'JC <1107609760@qq.com>'
    msg['To'] = 'JC <1107609760@qq.com>'
    msg['Subject'] = 'Your program (Server: {}, Exp ID: {}) is started!'.format(args.server, args.expid)
    from_addr = '1107609760@qq.com'
    password = 'mhqnjaykfytybaei'
    to_addr = '1107609760@qq.com'
    smtp_server = 'smtp.qq.com'
    server = smtplib.SMTP_SSL(smtp_server)
    server.set_debuglevel(1)
    server.login(from_addr, password)
    server.sendmail(from_addr, [to_addr], msg.as_string())
    server.quit()

def program_end(meg, args):
    msg = MIMEText('Your program is finished! Please check in time.\n\n' \
        + '{}\n\n'.format(meg) \
            + 'This is an automatic Email!\n\nNot reply.\nThanks.', 'plain', 'utf-8')
    msg['From'] = 'JC <1107609760@qq.com>'
    msg['To'] = 'JC <1107609760@qq.com>'
    msg['Subject'] = 'Your program (Server: {}, Exp ID: {}) is ended!'.format(args.server, args.expid)
    from_addr = '1107609760@qq.com'
    password = 'mhqnjaykfytybaei'
    to_addr = '1107609760@qq.com'
    smtp_server = 'smtp.qq.com'
    server = smtplib.SMTP_SSL(smtp_server)
    server.set_debuglevel(1)
    server.login(from_addr, password)
    server.sendmail(from_addr, [to_addr], msg.as_string())
    server.quit()


import os
import torch
from tqdm import tqdm
import time

# declare which gpu device to use
cuda_device = '0'

def check_mem(cuda_device):
    devices_info = os.popen('"/usr/bin/nvidia-smi" --query-gpu=memory.total,memory.used --format=csv,nounits,noheader').read().strip().split("\n")
    total, used = devices_info[int(cuda_device)].split(',')
    return total,used

def occumpy_mem(cuda_device, rate=0.90):
    total, used = check_mem(cuda_device)
    total = int(total)
    used = int(used)
    max_mem = int(total * rate)
    block_mem = max_mem - used
    x = torch.cuda.FloatTensor(256,1024,block_mem)
    del x
    for _ in tqdm(range(100000000000000)):
        time.sleep(1)
    
# if __name__ == '__main__':
#     os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device
#     occumpy_mem(cuda_device, rate=0.90)
#     for _ in tqdm(range(60)):
#         time.sleep(1)
#     print('Done!')
#     print(torch.cuda.empty_cache())
#     occumpy_mem(cuda_device, rate=0.50)
#     for _ in tqdm(range(100000000000000)):
#         time.sleep(1)
#     print('Done!!')