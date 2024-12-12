import torch
import h5py
import numpy as np
import pandas as pd
from models.model3d import AlzheimerNet3D
from torch.cuda.amp import autocast
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def normalize_data(data):
    """标准化数据"""
    for i in range(len(data)):
        sample = data[i]
        mean = np.mean(sample)
        std = np.std(sample)
        data[i] = (sample - mean) / (std + 1e-8)
    return data

def predict_and_save(model_path, test_paths, output_path='submit.csv'):
    """
    使用训练好的模型进行预测并生成提交文件
    
    参数:
        model_path: 模型权重文件路径
        test_paths: 测试数据文件路径列表
        output_path: 输出文件路径
    """
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    
    # 加载模型
    model = AlzheimerNet3D()
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()
    
    # 存储所有预测结果
    all_predictions = []
    all_ids = []
    
    # 对每个测试文件进行预测
    for test_path in test_paths:
        logger.info(f"处理测试文件: {test_path}")
        
        try:
            with h5py.File(test_path, 'r') as f:
                # 加载数据
                data = f['data'][:]
                data = normalize_data(data)
                
                # 批量处理数据
                batch_size = 4  # 可以根据GPU内存调整
                num_samples = len(data)
                
                # 生成当前文件的ID
                if 'testa.h5' in test_path:
                    file_ids = [f'testa_{i}' for i in range(num_samples)]
                else:
                    file_ids = [f'testb_{i}' for i in range(num_samples)]
                
                for i in range(0, num_samples, batch_size):
                    batch_data = data[i:min(i+batch_size, num_samples)]
                    batch_tensor = torch.FloatTensor(batch_data).to(device)
                    
                    # 使用混合精度进行预测
                    with torch.no_grad(), autocast():
                        outputs = model(batch_tensor)
                        _, predicted = torch.max(outputs.data, 1)
                        
                        # 将预测结果添加到列表中
                        predictions = predicted.cpu().numpy()
                        all_predictions.extend(predictions)
                        
                        # 添加对应的ID
                        batch_ids = file_ids[i:min(i+batch_size, num_samples)]
                        all_ids.extend(batch_ids)
                    
                    # 清理GPU缓存
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
        except Exception as e:
            logger.error(f"处理文件 {test_path} 时出错: {str(e)}")
            raise
    
    # 创建提交文件
    try:
        # 将数值标签转换为类别名称
        label_map = {
            0: 'healthy',
            1: 'mci',
            2: 'alzheimer'
        }
        
        predictions_labels = [label_map[pred] for pred in all_predictions]
        
        # 创建DataFrame
        df = pd.DataFrame({
            'testa_id': all_ids,  # 使用testa_id作为列名
            'label': predictions_labels
        })
        
        # 保存为CSV文件
        df.to_csv(output_path, index=False)
        logger.info(f"提交文件已保存到: {output_path}")
        
        # 显示预测结果统计
        label_counts = df['label'].value_counts()
        logger.info("预测结果统计:")
        for label, count in label_counts.items():
            logger.info(f"{label}: {count}")
            
        # 显示ID格式示例
        logger.info("ID格式示例:")
        logger.info(df['testa_id'].head())
            
    except Exception as e:
        logger.error(f"创建提交文件时出错: {str(e)}")
        raise

if __name__ == "__main__":
    # 设置路径
    model_path = "best_model.pth"  # 训练好的模型权重文件
    test_paths = [
        "F:/Program/test/testa.h5",
        "F:/Program/test/testb.h5"
    ]
    
    # 生成预测结果
    predict_and_save(model_path, test_paths) 