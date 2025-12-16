import os
import numpy as np
current_dir = os.path.dirname(os.path.abspath(__file__))
import matplotlib.pyplot as plt
import itertools

class Config:
    def __init__(self, **kwargs):

        self.dict = kwargs
        # 实验基础配置
        self.suffix = kwargs['suffix']
        self.env_name = kwargs['env_name']
        self.method = kwargs['method']   
        # 功能开关
        self.use_KEM = kwargs['use_KEM']
        self.use_nosie = kwargs['use_nosie']
        # method 参数
        self.traj_name = kwargs['traj_name']
        self.save_path = os.path.join(current_dir , "ControlResults", self.suffix)

    # 派生参数（自动生成）
    @property
    def KEM_FLAG(self):
        return "_UKF" if self.use_KEM else ""
    
    @property
    def NOISE_FLAG(self):
        return "_noise" if self.use_nosie else ""
    
    
    def save_result(self, u, actual_traj, traj, JointAngles):

        save_path = os.path.join(self.save_path, self.traj_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        sub_path = f'{self.method}{self.KEM_FLAG}{self.NOISE_FLAG}'
        error_dis = np.linalg.norm(actual_traj[:,:3] - traj)
        error_angle = np.linalg.norm(actual_traj[:,3:] - JointAngles)
        data_dict = {
            'u': u,
            'traj': actual_traj,
            'ref_traj' : traj,
            'error_dis' :  error_dis,
            'error_angle' :  error_angle,
            'config_dict': self.dict
            }
        print(error_dis, error_angle)
        np.savez(
            os.path.join(save_path, f"{sub_path}.npz"),
            **data_dict
        )

    def load_result(self):

        save_path = os.path.join(self.save_path, self.traj_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        sub_path = f'{self.method}{self.KEM_FLAG}{self.NOISE_FLAG}'
        data_dict = np.load(os.path.join(save_path, f"{sub_path}.npz"), allow_pickle=True)

        return data_dict
    

class ConfigGenerator:
    def __init__(self, base_config):
        self.base = base_config
        # 需要变化的参数空间
        self.parameter_space = {
            'method': ['IBKN'],
            'traj_name': ["Fig8",'FigStar'],
            'use_nosie': [True, False],
            'use_KEM': [True, False],
        }
        # 轨迹配置信息
        self.traj_config = {
            'Fig8': { 'title': 'Fig8'},
            'FigStar': { 'title': 'FigStar'}
        }
        # 方法配置信息
        self.method_config = {
            'IBKN': {
                'name': 'IBKN',
                'color': 'red',
                'line_style': '--'
            },
        }    
        self.robustness_config = {
                'none': {'name': 'IBKN-δMPC', 'alpha': 1.0, 'color': 'red', 'line_style': '-'},
                'KEM': {'name': 'IBKN-δMPC-UKF', 'alpha': 1.0 , 'color': 'blue', 'line_style': '-'},
            }
    # 生成所有有效配置组合
    def generate_configs(self, space):
        keys = list(space.keys())
        value_combinations = itertools.product(*(space[key] for key in keys))
        
        configs = []
        seen = set()
        for combo in value_combinations:
            config = self.base.copy()
            for i, key in enumerate(keys):
                config[key] = combo[i]
            if not config["use_nosie"]:
                config["use_KEM"] = False
            # 创建唯一标识符
            config_id = tuple(sorted(config.items()))
            # 检查是否重复
            if config_id not in seen:
                seen.add(config_id)
                configs.append(config)
        
        
        return configs
    
    def generate_for_comparison(self, scenario):
        """生成特定场景的配置"""
        scenarios = {
            # InvertKoopmanBlinear噪声鲁棒性测试
            'noise_robustness': {
                'method': ['IBKN'],
                'traj_name': ["Fig8", "FigStar"],
                'use_nosie': [True],
                'use_KEM': [True, False],
            },
                
            'base_performance': {
                'method': ['IBKN'],
                'traj_name': ["Fig8", "FigStar"],
                'use_nosie': [False],
                'use_KEM': [False],
            },
        }
        
        if scenario not in scenarios:
            raise ValueError(f"未知场景: {scenario}. 可用场景: {list(scenarios.keys())}")
        
        return self.generate_configs(scenarios[scenario])

    def get_result(self, scenario):

        if scenario == 'base_performance':
            base_performance_list = self.generate_for_comparison(scenario)
            results = {}
            for config_dict in base_performance_list:
                config = Config(**config_dict)
                data_dict = config.load_result()
                traj_name = config_dict['traj_name']
                method = config_dict['method']
                # 初始化数据结构
                if traj_name not in results:
                    results[traj_name] = {
                        'ref_traj': data_dict['ref_traj'],
                        'u': data_dict['u']
                    }
                # 存储方法特定的预测轨迹
                results[traj_name][method] = {
                    'pre_traj': data_dict['traj'],
                    'u': data_dict['u'],
                    'error_dis': data_dict['error_dis'],
                    'error_angle': data_dict['error_angle']
                } 

        if scenario == 'noise_robustness':
            robustness_performance_list = self.generate_for_comparison(scenario)
            results = {}
            for config_dict in robustness_performance_list:
                config = Config(**config_dict)
                data_dict = config.load_result()
                traj_name = config_dict['traj_name']
                method = config_dict['method']
                # 确定鲁棒性方法类型
                robustness_type = 'none'
                if config_dict['use_KEM']:
                    robustness_type = 'KEM'     
                # 初始化数据结构
                if traj_name not in results:
                    results[traj_name] = {
                        'ref_traj': data_dict['ref_traj'],
                        'u': data_dict['u']
                    }
                
                if method not in results[traj_name]:
                    results[traj_name][method] = {}
                
                # 存储结果
                results[traj_name][method][robustness_type] = {
                    'pre_traj': data_dict['traj'],
                    'u': data_dict['u'],
                    'error_dis': data_dict['error_dis'],
                    'error_angle': data_dict['error_angle']
                }

        return results

    def save_table(self, results_dict, save_Fig_path):
        # 添加误差表格
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.axis('off')
        error_table_data = []

        results = results_dict['noise_robustness']
        for traj_name in ['Fig8', 'FigStar']:
            for method in self.method_config:
                if method not in results[traj_name]:
                    continue        
                row_data = [f"{self.method_config[method]['name']} - {traj_name}"]

                error_dis = results_dict['base_performance'][traj_name][method]['error_dis']
                error_angle = results_dict['base_performance'][traj_name][method]['error_angle']
                error_all = np.sqrt(error_dis**2 + error_angle ** 2)
                row_data.append(f"{error_dis:.4f},{error_angle:.4f},{error_all:.4f}")
                for rob_type in ['none', 'KEM']:
                    if rob_type in results[traj_name][method]:
                        error_dis = results[traj_name][method][rob_type]['error_dis']
                        error_angle = results[traj_name][method][rob_type]['error_angle']
                        error_all = np.sqrt(error_dis**2 + error_angle ** 2)
                        row_data.append(f"{error_dis:.4f},{error_angle:.4f},{error_all:.4f}")
                    else:
                        row_data.append("N/A")
                error_table_data.append(row_data)
        
        # 创建表格
        table = ax.table(
            cellText=error_table_data,
            colLabels=['Scenario', 'base','No Filter', 'KF Only'],
            loc='center',
            # bbox=[0.0, -0.3, 1.0, 0.2],
            cellLoc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        # --- 给表格留空间 ---
        plt.subplots_adjust(bottom=0.3)
        plt.savefig(
            os.path.join(save_Fig_path, f"table.png"), format="png",\
                  dpi=500, bbox_inches='tight'
        )
        plt.show()