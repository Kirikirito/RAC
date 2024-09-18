import json 
import os
import glob
env_dict = {'Ant-v3': 'gym_ant', 'Humanoid-v3': 'gym_humanoid'}
def convert_single_json(data, algorithm, env, run_id):
    # 设定基本参数
    step_increment = 15000  # step与index的关系
    index = data.get("training_iteration", 0)  # index为training_iteration
    step = index * step_increment  # step是index的15000倍
    value = data.get("episode_reward_mean", 0)  # value是episode_reward_mean
    
    # 创建转换后的字典
    converted_record = {
        "index": index,
        "algorithm": algorithm,
        "env": env_dict[env],
        "run": step/2500,  # 可以根据实际需求调整
        "step": step,
        "value": value,
        "metric": "Evaluation/1. TAR-RL iter",
        "run_id": run_id,
        "default": ""
    }
    return converted_record
def convert_json(data_list, algorithm, env, run_id):
    output_data = []
    for data in data_list:
        converted_record = convert_single_json(data, algorithm, env, run_id)
        output_data.append(converted_record)
    return output_data

def load_json(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    data_list = []
    for line in lines:
        data_list.append(json.loads(line.strip()))
    return data_list

def save_json(data, file_path):
    with open(file_path, 'w') as f:
        f.write('[')
        for record in data:
            f.write(json.dumps(record) + '\n')
            f.write(',')
            f.write('\n')
        f.write(']')


def gather_file_by_env(env, root_path,seed=None):
    # gather all the dirs in the root_path that contains the env name
    if seed is not None:
        env_dirs = glob.glob(f'{root_path}/**{env}*,seed={seed}*', recursive=False)
    else:
        env_dirs = glob.glob(f'{root_path}/**{env}*', recursive=False)
    json_files = []
    for env_dir in env_dirs:
        json_files.append(env_dir + '/result.json')
    return json_files




data_list = load_json('/root/dsact/RAC/RAC-SAC/ray_results/RAC-SAC/RAC_SAC_Ant-v3_4bc7e_00000_0_env=Ant-v3,seed=12345_2024-09-14_23-59-17/result.json')
output_data = convert_json(data_list, "RAC-SAC", "Ant-v3", 0)


base_file_path = './output/'
if not os.path.exists(base_file_path):
    os.makedirs(base_file_path)


algorithm = "RAC-SAC"
env_list = ["Ant-v3","Humanoid-v3"]
seed = 12345

for env in env_list:
    output_data = []
    json_files = gather_file_by_env(env, '/root/dsact/RAC/RAC-SAC/ray_results/RAC-SAC/', seed)
    for i, json_file in enumerate(json_files):
        data_list = load_json(json_file)
        output_data.append(convert_json(data_list, algorithm, env, i))
    output_data = [item for sublist in output_data for item in sublist]
    save_json(output_data, base_file_path + f"{algorithm}_{env}.json")