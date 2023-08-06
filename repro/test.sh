#!/bin/bash
#此参数用于指定运行作业的名称
#DSUB -n ddq_legacy_test_2023-07-26
#此处需要把“用户名”修改为用户的用户名，例如用户名为 gpuuser001 则此行写为“#DSUB -A root.bingxing2.gpuuser001”
#DSUB -A root.bingxing2.gpuuser194
#默认参数，一般不需要修改
#DSUB -q root.default
#DSUB -l wuhanG5500
#跨节点任务不同类型程序 job_type 会有差异，请参考下文对应跨节点任务模板编写
#DSUB --job_type cosched
#此参数用于指定资源。如申请 6 核 CPU，1 卡 GPU，48GB 内存。
#DSUB -R 'cpu=48;gpu=8;mem=300000'
#此参数用于指定运行作业的机器数量。单节点作业则为 1 。
#DSUB -N 1
# 此参数用于指定日志的输出，%J 表示 JOB_ID。
#DSUB -e %J.out
#DSUB -o %J.out

#加载环境
module load python/3.8.12-gcc-4.8.5-jbm
module load cuda/11.1.0-gcc-4.8.5-67q
module load gcc/9.3.0-gcc-4.8.5-bxl

source ~/ddq/legacy/py38/bin/activate

cd ~/ddq/legacy/proj/DDQ/mmcv-1.4.7
export PYTHONPATH=`pwd`:$PYTHONPATH

#python 运行程序
cd ~/ddq/legacy/proj/DDQ
PORT=50136 sh tools/test.sh  projects/configs/ddq_fcn/ddq_fcn_r50_1x.py ~/data/pretrain_models/ddq_fcn_r50_1x.pth 8 --eval bbox
