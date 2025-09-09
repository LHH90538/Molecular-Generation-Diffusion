import os
import sys
import shutil
import argparse
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)


import torch
import numpy as np
from easydict import EasyDict
from rdkit import Chem

# 在导入其他模块前添加安全全局变量
import torch.serialization
torch.serialization.add_safe_globals([EasyDict])

from models.model import MolDiff
from utils.sample import seperate_outputs
from utils.transforms import *
from utils.misc import *
from utils.reconstruct import *

def print_pool_status(pool, logger):
    logger.info('[Pool] Finished %d | Failed %d' % (
        len(pool.finished), len(pool.failed)
    ))

def data_exists(data, prevs):
    for other in prevs:
        if len(data.logp_history) == len(other.logp_history):
            if (data.ligand_context_element == other.ligand_context_element).all().item() and \
                (data.ligand_context_feature_full == other.ligand_context_feature_full).all().item() and \
                torch.allclose(data.ligand_context_pos, other.ligand_context_pos):
                return True
    return False

def safe_load_checkpoint(checkpoint_path, device, logger):
    """安全加载检查点的封装函数"""
    try:
        # 先尝试默认weights_only模式
        logger.info(f"Attempting standard checkpoint loading...")
        return torch.load(checkpoint_path, map_location=device)
    except Exception as e:
        logger.warning(f"Standard loading failed: {str(e)}")
        logger.info("Attempting with explicit safe globals...")
        try:
            # 显式添加安全全局变量后重试
            torch.serialization.add_safe_globals([EasyDict])
            return torch.load(checkpoint_path, map_location=device)
        except Exception as e:
            logger.warning(f"Safe loading failed: {str(e)}")
            logger.warning("Falling back to weights_only=False (USE WITH CAUTION!)")
            # 最后回退到非安全模式
            return torch.load(checkpoint_path, map_location=device, weights_only=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/sample_MolDiff_simple.yml')
    parser.add_argument('--outdir', type=str, default='./outputs')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batch_size', type=int, default=12)
    
    # 处理Jupyter可能传入的额外参数
    args, _ = parser.parse_known_args()
    
    # 设置配置文件绝对路径
    if not os.path.isabs(args.config):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(script_dir)
        args.config = os.path.join(parent_dir, args.config)

    # 设置输出目录到上一级目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    args.outdir = os.path.join(parent_dir, 'outputs')
    
    # 首先初始化日志系统
    log_root = args.outdir.replace('outputs', 'outputs_vscode') if sys.argv[0].startswith('/data') else args.outdir
    log_dir = get_new_log_dir(log_root, prefix='sample')
    logger = get_logger('sample', log_dir)
    
    # Load configs
    logger.info("Loading configuration...")
    config = load_config(args.config)
    config_name = os.path.basename(args.config)[:os.path.basename(args.config).rfind('.')]
    
    # 修复种子类型问题
    seed = int(config.sample.seed) + int(np.sum([ord(s) for s in args.outdir]))
    seed_all(seed)
    logger.info(f"Using random seed: {seed}")
    
    # 设置检查点文件绝对路径
    checkpoint_path = config.model.checkpoint
    if not os.path.isabs(checkpoint_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(script_dir)
        checkpoint_path = os.path.join(parent_dir, checkpoint_path)
    
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    ckpt = safe_load_checkpoint(checkpoint_path, args.device, logger)
    train_config = ckpt['config']

    # 记录配置信息
    logger.info("Program arguments:")
    logger.info(args)
    logger.info("Configuration:")
    logger.info(config)
    shutil.copyfile(args.config, os.path.join(log_dir, os.path.basename(args.config)))

    # Transform
    logger.info('Loading data placeholder...')
    featurizer = FeaturizeMol(
        train_config.chem.atomic_numbers, 
        train_config.chem.mol_bond_types,
        use_mask_node=train_config.transform.use_mask_node,
        use_mask_edge=train_config.transform.use_mask_edge,
    )
    max_size = None
    add_edge = getattr(config.sample, 'add_edge', None)
    
    # Model
    logger.info('Loading diffusion model...')
    if train_config.model.name == 'diffusion':
        model = MolDiff(
            config=train_config.model,
            num_node_types=featurizer.num_node_types,
            num_edge_types=featurizer.num_edge_types
        ).to(args.device)
    else:
        raise NotImplementedError
    
    model.load_state_dict(ckpt['model'])
    model.eval()
    logger.info("Model loaded successfully")
    
    bond_predictor = None
    guidance = None

    pool = EasyDict({
        'failed': [],
        'finished': [],
    })

    # Generating molecules
    logger.info(f"Starting molecule generation (target: {config.sample.num_mols} molecules)")
    while len(pool.finished) < config.sample.num_mols:
        if len(pool.failed) > 3 * config.sample.num_mols:
            logger.warning('Too many failed molecules. Stop sampling.')
            break
        
        batch_size = args.batch_size if args.batch_size > 0 else config.sample.batch_size
        n_graphs = min(batch_size, (config.sample.num_mols - len(pool.finished))*2)
        logger.debug(f"Generating batch of {n_graphs} molecules")
        
        batch_holder = make_data_placeholder(n_graphs=n_graphs, device=args.device, max_size=max_size)
        batch_node, halfedge_index, batch_halfedge = (
            batch_holder['batch_node'], 
            batch_holder['halfedge_index'], 
            batch_holder['batch_halfedge']
        )
        
        # Inference
        try:
            outputs = model.sample(
                n_graphs=n_graphs,
                batch_node=batch_node,
                halfedge_index=halfedge_index,
                batch_halfedge=batch_halfedge,
                bond_predictor=bond_predictor,
                guidance=guidance,
            )
            outputs = {key: [v.cpu().numpy() for v in value] for key, value in outputs.items()}
        except Exception as e:
            logger.error(f"Model sampling failed: {str(e)}")
            continue
        
        # Decode outputs to molecules
        batch_node = batch_node.cpu().numpy()
        halfedge_index = halfedge_index.cpu().numpy()
        batch_halfedge = batch_halfedge.cpu().numpy()
        
        try:
            output_list = seperate_outputs(outputs, n_graphs, batch_node, halfedge_index, batch_halfedge)
        except Exception as e:
            logger.warning(f"Output separation failed: {str(e)}")
            continue
            
        gen_list = []
        for i_mol, output_mol in enumerate(output_list):
            try:
                mol_info = featurizer.decode_output(
                    pred_node=output_mol['pred'][0],
                    pred_pos=output_mol['pred'][1],
                    pred_halfedge=output_mol['pred'][2],
                    halfedge_index=output_mol['halfedge_index'],
                )
                rdmol = reconstruct_from_generated_with_edges(mol_info, add_edge=add_edge)
                mol_info['rdmol'] = rdmol
                smiles = Chem.MolToSmiles(rdmol)
                mol_info['smiles'] = smiles
                
                if '.' in smiles:
                    logger.warning(f'Incomplete molecule: {smiles}')
                    pool.failed.append(mol_info)
                else:
                    logger.info(f'Success: {smiles}')
                    if np.random.rand() < config.sample.save_traj_prob:
                        mol_info['traj'] = [
                            reconstruct_from_generated_with_edges(
                                featurizer.decode_output(
                                    output_mol['traj'][0][t],
                                    output_mol['traj'][1][t],
                                    output_mol['traj'][2][t],
                                    output_mol['halfedge_index']
                                ),
                                False,
                                add_edge=add_edge
                            ) if t < len(output_mol['traj'][0]) else Chem.MolFromSmiles('O')
                            for t in range(len(output_mol['traj'][0]))
                        ]
                    gen_list.append(mol_info)
                    
            except MolReconsError as e:
                pool.failed.append(mol_info if 'mol_info' in locals() else {'error': str(e)})
                logger.warning(f'Reconstruction error: {str(e)}')
                continue
            except Exception as e:
                pool.failed.append({'error': str(e)})
                logger.error(f'Unexpected error: {str(e)}')
                continue

        # Save results
        sdf_dir = os.path.join(log_dir, 'SDF')
        os.makedirs(sdf_dir, exist_ok=True)
        
        with open(os.path.join(log_dir, 'SMILES.txt'), 'a') as smiles_f:
            for i, data in enumerate(gen_list):
                smiles_f.write(data['smiles'] + '\n')
                Chem.MolToMolFile(data['rdmol'], os.path.join(sdf_dir, f'{i+len(pool.finished)}.sdf'))
                
                if 'traj' in data:
                    with Chem.SDWriter(os.path.join(sdf_dir, f'traj_{i+len(pool.finished)}.sdf')) as w:
                        for m in data['traj']:
                            try:
                                w.write(m)
                            except:
                                w.write(Chem.MolFromSmiles('O'))
        
        pool.finished.extend(gen_list)
        print_pool_status(pool, logger)

    # Save final results
    torch.save(pool, os.path.join(log_dir, 'samples_all.pt'))
    logger.info(f"Sampling completed. Results saved to {log_dir}")
    logger.info(f"Successfully generated {len(pool.finished)} molecules")
    logger.info(f"Failed attempts: {len(pool.failed)}")