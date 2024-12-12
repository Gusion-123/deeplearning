import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AlzheimerDataset(Dataset):
    def __init__(self, h5_path, transform=None):
        """
        参数:
            h5_path (str): h5文件的路径
            transform (callable, optional): 数据转换
        """
        self.h5_path = h5_path
        self.transform = transform
        self.samples = []
        self.labels = []
        
        # 加载数据
        self._load_data()
        
    def _load_data(self):
        logger.info(f"正在加载数据从: {self.h5_path}")
        
        try:
            with h5py.File(self.h5_path, 'r') as f:
                # 打印h5文件的结构以便调试
                logger.info("H5文件结构:")
                self._print_h5_structure(f)
                
                # 加载数据
                data = f['data'][:]
                logger.info(f"原始数据形状: {data.shape}")
                
                # 数据预处理
                # 1. 标准化
                data = self._normalize_data(data)
                
                self.samples = data  # 保持5D形状
                
                # 生成标签
                num_samples = len(self.samples)
                
                # 根据数据集大小调整每个类别的样本数
                if num_samples == 300:  # 训练集
                    samples_per_class = 100
                elif num_samples == 116:  # 测试集a
                    samples_per_class = 38
                else:  # 测试集b或其他
                    samples_per_class = num_samples // 3
                
                remainder = num_samples - (samples_per_class * 3)
                if remainder != 0:
                    logger.warning(f"样本总数 {num_samples} 不能被3整除，最后 {remainder} 个样本可能标签不准确")
                
                # 为每个类别生成标签
                self.labels = np.concatenate([
                    np.zeros(samples_per_class),  # 健康样本
                    np.ones(samples_per_class),   # 轻度认知障碍
                    np.ones(samples_per_class) * 2  # 阿尔茨海默症
                ]).astype(np.int64)
                
                if remainder > 0:
                    # 处理剩余样本
                    remaining_labels = np.zeros(remainder)  # 将剩余样本标记为健康类别
                    self.labels = np.concatenate([self.labels, remaining_labels])
                
                logger.info(f"成功加载 {len(self.samples)} 个样本")
                logger.info(f"最终数据形状: {self.samples.shape}")
                logger.info(f"标签分布: 健康: {samples_per_class}, "
                          f"轻度认知障碍: {samples_per_class}, "
                          f"阿尔茨海默症: {samples_per_class}"
                          + (f", 剩余: {remainder}" if remainder > 0 else ""))
                
        except Exception as e:
            logger.error(f"加载数据时出错: {str(e)}")
            raise
    
    def _normalize_data(self, data):
        """标准化数据"""
        # 对每个样本分别进行标准化
        for i in range(len(data)):
            sample = data[i]
            mean = np.mean(sample)
            std = np.std(sample)
            data[i] = (sample - mean) / (std + 1e-8)
        return data
    
    def _print_h5_structure(self, h5file, level=0):
        """递归打印h5文件结构"""
        for key in h5file.keys():
            logger.info("  " * level + f"- {key}: {h5file[key].shape if hasattr(h5file[key], 'shape') else '组'}")
            if isinstance(h5file[key], h5py.Group):
                self._print_h5_structure(h5file[key], level + 1)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]  # 形状: (1, 79, 95, 79)
        label = self.labels[idx]
        
        # 转换为张量
        sample = torch.FloatTensor(sample)
        label = torch.tensor(label, dtype=torch.long)
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample, label

def get_data_loaders(train_path, test_paths, batch_size=8):
    """
    参数:
        train_path (str): 训练数据h5文件路径
        test_paths (list): 测试数据h5文件路径列表
        batch_size (int): 批次大小
    """
    logger.info("正在创建数据加载器...")
    
    try:
        # 加载训练数据
        train_dataset = AlzheimerDataset(train_path)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # 加载多个测试数据集
        test_loaders = []
        for test_path in test_paths:
            test_dataset = AlzheimerDataset(test_path)
            test_loader = DataLoader(test_dataset, batch_size=batch_size)
            test_loaders.append(test_loader)
        
        logger.info("数据加载器创建成功")
        return train_loader, test_loaders
        
    except Exception as e:
        logger.error(f"创建数据加载器时出错: {str(e)}")
        raise