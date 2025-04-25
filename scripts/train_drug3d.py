import sys
import os
import shutil
import argparse
import numpy as np
sys.path.append('.')

from tqdm import tqdm
import torch
from easydict import EasyDict

# 安全加载设置
import torch.serialization
from numpy.core.multiarray import _reconstruct
import numpy as np
torch.serialization.add_safe_globals([EasyDict, _reconstruct, np.ndarray])
torch.multiprocessing.set_sharing_strategy('file_system')

from torch.nn.utils import clip_grad_norm_
from torch_geometric.loader import DataLoader

from models.model import MolDiff
from utils.dataset import get_dataset
from utils.transforms import FeaturizeMol, Compose
from utils.misc import *
from utils.train import *

def safe_load_config(config_path):
    """安全加载配置文件"""
    try:
        return load_config(config_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load config: {str(e)}")

def initialize_logging(args, config, config_name):
    """初始化日志系统"""
    log_dir = get_new_log_dir(args.logdir, prefix=config_name)
    ckpt_dir = os.path.join(log_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    logger = get_logger('train', log_dir)
    
    logger.info("Program arguments:")
    logger.info(args)
    logger.info("Configuration:")
    logger.info(config)
    
    shutil.copyfile(args.config, os.path.join(log_dir, os.path.basename(args.config)))
    return log_dir, ckpt_dir, logger

def safe_load_dataset(config, transform):
    """安全加载数据集"""
    try:
        return get_dataset(config=config.dataset, transform=transform)
    except Exception as e:
        logger.warning(f"Standard dataset loading failed: {str(e)}")
        logger.info("Attempting with safe unpickling...")
        
        try:
            original_load = torch.load
            torch.load = lambda *args, **kwargs: original_load(*args, **kwargs, weights_only=False)
            
            dataset, subsets = get_dataset(config=config.dataset, transform=transform)
            torch.load = original_load
            return dataset, subsets
        except Exception as e:
            torch.load = original_load
            raise RuntimeError(f"Failed to load dataset: {str(e)}")

def build_data_loaders(config, featurizer):
    """构建数据加载器"""
    logger.info('Loading dataset...')
    # 保留 Compose，用于以后扩展
    transform = Compose([featurizer])
    # 把 featurizer 的属性贴给 Compose 后的 transform
    transform.follow_batch = featurizer.follow_batch
    transform.exclude_keys = featurizer.exclude_keys

    dataset, subsets = safe_load_dataset(config, transform)
    train_set, val_set = subsets['train'], subsets['val']
    
    train_loader = DataLoader(
        train_set,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.train.num_workers,
        pin_memory=config.train.pin_memory,
        follow_batch=transform.follow_batch,
        exclude_keys=transform.exclude_keys,
    )
    train_iterator = inf_iterator(train_loader)
    
    val_loader = DataLoader(
        val_set,
        batch_size=config.train.batch_size,
        shuffle=False,
        follow_batch=transform.follow_batch,
        exclude_keys=transform.exclude_keys
    )
    
    return train_iterator, val_loader

if __name__ == '__main__':
    # 解析参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/train_MolDiff_simple.yml')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--logdir', type=str, default='logs')
    args, _ = parser.parse_known_args()

    # 加载配置
    config = safe_load_config(args.config)
    config_name = os.path.basename(args.config)[:os.path.basename(args.config).rfind('.')]
    seed_all(config.train.seed)

    # 初始化日志系统
    log_dir, ckpt_dir, logger = initialize_logging(args, config, config_name)

    # 数据转换
    featurizer = FeaturizeMol(
        config.chem.atomic_numbers,
        config.chem.mol_bond_types,
        use_mask_node=config.transform.use_mask_node,
        use_mask_edge=config.transform.use_mask_edge
    )

    # 数据加载器
    train_iterator, val_loader = build_data_loaders(config, featurizer)

    # 模型构建
    logger.info('Building model...')
    if config.model.name == 'diffusion':
        model = MolDiff(
            config=config.model,
            num_node_types=featurizer.num_node_types,
            num_edge_types=featurizer.num_edge_types
        ).to(args.device)
    else:
        raise NotImplementedError(f'Model {config.model.name} not implemented')
    
    logger.info(f'Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    # 优化器和调度器
    optimizer = get_optimizer(config.train.optimizer, model)
    scheduler = get_scheduler(config.train.scheduler, optimizer)

    def train(it):
        """训练步骤"""
        try:
            optimizer.zero_grad(set_to_none=True)
            batch = next(train_iterator).to(args.device)
            
            pos_noise = torch.randn_like(batch.node_pos) * config.train.pos_noise_std
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=config.train.use_amp):
                loss_dict = model.get_loss(
                    node_type=batch.node_type,
                    node_pos=batch.node_pos + pos_noise,
                    batch_node=batch.node_type_batch,
                    halfedge_type=batch.halfedge_type,
                    halfedge_index=batch.halfedge_index,
                    batch_halfedge=batch.halfedge_type_batch,
                    num_mol=batch.num_graphs,
                )
            
            loss = loss_dict['loss']
            loss.backward()
            grad_norm = clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
            optimizer.step()

            # 记录训练信息
            log_info = '[Train] Iter %d | ' % it + ' | '.join([
                '%s: %.6f' % (k, v.item()) for k, v in loss_dict.items()
            ])
            logger.info(log_info)
            
            return loss_dict, grad_norm
            
        except Exception as e:
            logger.error(f'Training error at iteration {it}: {str(e)}')
            raise

    def validate(it):
        """验证步骤"""
        sum_loss_dict = {}
        sum_n = 0
        
        try:
            model.eval()
            with torch.no_grad():
                for batch in tqdm(val_loader, desc='Validate'):
                    batch = batch.to(args.device)
                    with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=config.train.use_amp):
                        loss_dict = model.get_loss(
                            node_type=batch.node_type,
                            node_pos=batch.node_pos,
                            batch_node=batch.node_type_batch,
                            halfedge_type=batch.halfedge_type,
                            halfedge_index=batch.halfedge_index,
                            batch_halfedge=batch.halfedge_type_batch,
                            num_mol=batch.num_graphs,
                        )
                    
                    if not sum_loss_dict:
                        sum_loss_dict = {k: v.item() for k, v in loss_dict.items()}
                    else:
                        for key in sum_loss_dict:
                            sum_loss_dict[key] += loss_dict[key].item()
                    sum_n += 1

            avg_loss_dict = {k: v / sum_n for k, v in sum_loss_dict.items()}
            avg_loss = avg_loss_dict['loss']
            
            # 更新学习率
            if hasattr(scheduler, 'step_ReduceLROnPlateau'):
                scheduler.step_ReduceLROnPlateau(avg_loss)
            else:
                scheduler.step(avg_loss if config.train.scheduler.type == 'plateau' else None)

            # 记录验证信息
            log_info = '[Validate] Iter %d | ' % it + ' | '.join([
                '%s: %.6f' % (k, v) for k, v in avg_loss_dict.items()
            ])
            logger.info(log_info)
            
            return avg_loss_dict
            
        except Exception as e:
            logger.error(f'Validation error at iteration {it}: {str(e)}')
            raise

    # 主训练循环
    try:
        logger.info(f'Starting training for {config.train.max_iters} iterations')
        model.train()
        
        for it in range(1, config.train.max_iters + 1):
            try:
                train(it)
                
                if it % config.train.val_freq == 0 or it == config.train.max_iters:
                    validate(it)
                    
                    # 保存检查点
                    ckpt_path = os.path.join(ckpt_dir, f'{it}.pt')
                    torch.save({
                        'config': config,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'iteration': it,
                    }, ckpt_path)
                    logger.info(f'Saved checkpoint at iteration {it}')
                    
                    model.train()
                    
            except RuntimeError as e:
                logger.error(f'Runtime error at iteration {it}: {str(e)}')
                logger.error(f'Skipping iteration {it}')
                continue
                
    except KeyboardInterrupt:
        logger.info('Training interrupted by user')
    finally:
        logger.info('Training completed')
