import torch
import torch.nn as nn
import torch.optim as optim
from data.data_loader import get_data_loaders
from models.model3d import AlzheimerNet3D
from utils.preprocessing import normalize_scan, resize_volume
import numpy as np
import logging
from torch.cuda.amp import autocast, GradScaler

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_model(model, train_loader, test_loaders, num_epochs=50):
    # 设置设备
    if torch.cuda.is_available():
        device = torch.device("cuda")
        # 设置较小的工作内存限制
        torch.cuda.set_per_process_memory_fraction(0.6)  # 降低内存使用限制
    else:
        device = torch.device("cpu")
    
    logger.info(f"使用设备: {device}")
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    
    # 创建梯度缩放器用于混合精度训练
    scaler = GradScaler()
    
    best_acc = 0.0
    
    try:
        for epoch in range(num_epochs):
            # 训练阶段
            model.train()
            train_loss = 0.0
            batch_count = 0
            
            for inputs, labels in train_loader:
                try:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    
                    # 使用混合精度训练
                    with autocast():
                        optimizer.zero_grad()
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                    
                    # 使用梯度缩放器
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    
                    train_loss += loss.item()
                    batch_count += 1
                    
                    # 定期清理缓存
                    if batch_count % 5 == 0 and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        logger.error("GPU内存不足，跳过当前批次")
                        continue
                    else:
                        raise e
            
            # 验证阶段
            model.eval()
            total_val_loss = 0.0
            total_acc = 0.0
            
            for test_idx, test_loader in enumerate(test_loaders):
                correct = 0
                total = 0
                val_loss = 0.0
                
                with torch.no_grad(), autocast():
                    for inputs, labels in test_loader:
                        inputs = inputs.to(device)
                        labels = labels.to(device)
                        
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        val_loss += loss.item()
                        
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                        
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                
                acc = 100 * correct / total
                logger.info(f'测试集 {test_idx+1} - 准确率: {acc:.2f}%')
                total_val_loss += val_loss
                total_acc += acc
            
            avg_val_loss = total_val_loss / len(test_loaders)
            avg_acc = total_acc / len(test_loaders)
            
            logger.info(f'Epoch [{epoch+1}/{num_epochs}]')
            logger.info(f'训练损失: {train_loss/batch_count:.4f}')
            logger.info(f'平均验证损失: {avg_val_loss/len(test_loader):.4f}')
            logger.info(f'平均准确率: {avg_acc:.2f}%')
            
            scheduler.step(avg_val_loss)
            
            if avg_acc > best_acc:
                best_acc = avg_acc
                torch.save(model.state_dict(), 'best_model.pth')
                logger.info(f'保存新的最佳模型，准确率: {best_acc:.2f}%')
                
    except Exception as e:
        logger.error(f"训练过程中出错: {str(e)}")
        raise

def main():
    # 设置数据路径
    train_path = "F:/Program/train/train_pre_data.h5"
    test_paths = [
        "F:/Program/test/testa.h5",
        "F:/Program/test/testb.h5"
    ]
    
    # 设置更小的batch_size
    batch_size = 2  # 进一步减小批次大小
    
    logger.info("开始加载数据...")
    logger.info(f"训练数据路径: {train_path}")
    logger.info(f"测试数据路径: {test_paths}")
    
    # 获取数据加载器
    train_loader, test_loaders = get_data_loaders(train_path, test_paths, batch_size=batch_size)
    
    # 创建模型
    model = AlzheimerNet3D()
    logger.info("模型创建成功")
    
    # 清理GPU缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 训练模型
    logger.info("开始训练...")
    train_model(model, train_loader, test_loaders)
    logger.info("训练完成!")

if __name__ == "__main__":
    main() 