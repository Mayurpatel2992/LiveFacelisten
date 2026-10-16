import pytorch_lightning as pl
import logging
import time
from pathlib import Path
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from rich.progress import Progress
import torch
import numpy as np

import framework.launch.prepare  # noqa
from framework.model.utils.tools import load_checkpoint, detach_to_numpy
from framework.data.tools.collate import collate_motion_and_audio
from deps.flame.flame_pytorch import FLAME
from generation import cfg_mean_nsamples_resolution, get_path_vae, get_path_vqvae
from framework.data.motion_normalization import denormalize_motion


logger = logging.getLogger(__name__)

emo_dict = {
       "0": "neutral",  # only have one intensity level
       "1": "happy",
       "2": "sad",
       "3": "surprised",
       "4": "fear",
       "5": "disgusted",
       "6": "angry",
       "7": "contempt"
   }

int_dict = {
       "0": "low",
       "1": "medium",
       "2": "high",
   }

@hydra.main(version_base=None, config_path="configs", config_name="evaluation")
def _sample(cfg: DictConfig):
    sample(cfg)


def sample(newcfg: DictConfig):
    # Load previous configs
    prevcfg = OmegaConf.load(Path(newcfg.folder) / ".hydra/config.yaml")
    # Merge configs to overload them
    cfg = OmegaConf.merge(prevcfg, newcfg)

    onesample = cfg_mean_nsamples_resolution(cfg)

    logger.info("Sample script. The outputs will be stored in:")
    folder_name = cfg.folder.split("/")[-1]
    output_dir = Path(cfg.path.code_dir) / f"results/evaluation/{cfg.experiment}/{folder_name}"
    path = None
    if hasattr(cfg.model, 'vae_pred') and cfg.model.vae_pred:
        path = get_path_vae(output_dir, onesample, cfg.mean, cfg.fact)
    if hasattr(cfg.model, 'vqvae_diffusion_pred') and cfg.model.vqvae_diffusion_pred:
        if not cfg.sample:
            path = get_path_vqvae(output_dir, onesample, "none", cfg.k)
        else:
            path = get_path_vqvae(output_dir, onesample, cfg.temperature, cfg.k)
    if hasattr(cfg.model, 'vqvae_pred') and cfg.model.vqvae_pred:
        if not cfg.sample:
            path = get_path_vqvae(output_dir, onesample, "none", cfg.k)
        else:
            path = get_path_vqvae(output_dir, onesample, cfg.temperature, cfg.k)
    is_arflow = hasattr(cfg.model, 'vqvae_arflow_pred') and cfg.model.vqvae_arflow_pred
    if is_arflow:
        # arflow handles its own output paths per mask ratio below
        path = output_dir / "arflow"
        path.mkdir(exist_ok=True, parents=True)
    if path is None:
        raise ValueError("No model specified in the config file.")
    else:
        path.mkdir(exist_ok=True, parents=True)
        logger.info(f"{path}")

    # save config to check
    OmegaConf.save(cfg, output_dir / "merged_config.yaml")
    pl.seed_everything(cfg.seed)

    logger.info("Loading data module")
    # Disable progress bar for faster loading when limiting samples
    if hasattr(cfg, 'max_samples') and cfg.max_samples is not None:
        cfg.data.progress_bar = False
    data_module = instantiate(cfg.data)
    logger.info(f"Data module '{cfg.data.data_name}' loaded")

    logger.info("Loading model")
    last_ckpt_path = cfg.last_ckpt_path
    logger.info(last_ckpt_path)
    model = instantiate(cfg.model,
                        nfeats=data_module.nfeats,
                        split_path=data_module.split_path,
                        one_hot_dim=data_module.one_hot_dim,
                        resumed_training=False,
                        logger_name="none",
                        _recursive_=False)
    logger.info(f"Model '{cfg.model.modelname}' loaded")
    # move model to cuda
    if cfg.device is None:
        device_index = cfg.trainer.devices[0]
    else:
        device_index = cfg.device
    if torch.cuda.is_available():
        num_devices = torch.cuda.device_count()
        if device_index < num_devices:
            model.to(f"cuda:{device_index}")
        else:
            model.to(f"cuda:0")
    # print("device checking:", model.device)

    # load ckpt
    load_checkpoint(model, last_ckpt_path, eval_mode=True, device=model.device)

    # log model parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    logger.info(f"Model parameters: total={total_params:,}  trainable={trainable_params:,}  frozen={frozen_params:,}")
    logger.info(f"Model size: {total_params * 4 / 1024**2:.1f} MB (fp32)")
    if hasattr(cfg.model, 'vae_pred') and cfg.model.vae_pred:
        model.motion_prior.sample_mean = cfg.mean
        model.motion_prior.fact = cfg.fact
    if hasattr(cfg.model, 'vqvae_diffusion_pred') and cfg.model.vqvae_diffusion_pred:
        model.temperature = cfg.temperature
        model.k = cfg.k
        if hasattr(cfg, 'sampling_steps') and cfg.sampling_steps is not None:
            model.sampling_steps = cfg.sampling_steps
    if hasattr(cfg.model, 'vqvae_pred') and cfg.model.vqvae_pred:
        model.temperature = cfg.temperature
        model.k = cfg.k
    if hasattr(cfg.model, 'vqvae_arflow_pred') and cfg.model.vqvae_arflow_pred:
        model.temperature = cfg.temperature
        model.k = cfg.k
        logger.info(f"[arflow] sampling_steps from checkpoint/training config: {model.sampling_steps}")
        if hasattr(cfg, 'sampling_steps') and cfg.sampling_steps is not None:
            model.sampling_steps = cfg.sampling_steps
            logger.info(f"[arflow] sampling_steps overridden via CLI to: {model.sampling_steps}")

    # Determine mask ratio sweep (arflow) or single run (other models)
    if is_arflow:
        mask_ratios = list(cfg.get('infer_mask_ratios', [0.0]))
        extra_str = "" if onesample else "_multi"
    else:
        mask_ratios = [None]   # single pass, no masking concept

    # load test data
    dataset = getattr(data_module, f"{cfg.split}_dataset")
    
    # Limit to N samples if specified
    if hasattr(cfg, 'max_samples') and cfg.max_samples is not None:
        import random
        random.seed(cfg.seed)
        random.shuffle(dataset.keyids)
        dataset.keyids = dataset.keyids[:cfg.max_samples]
        logger.info(f"Limiting evaluation to {len(dataset.keyids)} random samples")

    # remove printing for changing the seed
    logging.getLogger('pytorch_lightning.utilities.seed').setLevel(logging.WARNING)

    # Performance: enable cudnn benchmark for consistent input sizes
    torch.backends.cudnn.benchmark = True

    # Warmup: run one dummy forward pass to eliminate CUDA context / kernel compilation overhead
    if torch.cuda.is_available() and len(dataset.keyids) > 0:
        logger.info("Running warmup inference...")
        warmup_data = dataset.load_keyid(dataset.keyids[0])
        warmup_batch = collate_motion_and_audio([warmup_data])
        with torch.no_grad():
            if hasattr(cfg.model, 'vqvae_arflow_pred') and cfg.model.vqvae_arflow_pred:
                _ = model.forward_multi_sample(warmup_batch.copy(), num_samples=cfg.number_of_samples, generation=False)
            elif hasattr(cfg.model, 'vqvae_diffusion_pred') and cfg.model.vqvae_diffusion_pred:
                _ = model.forward_multi_sample(warmup_batch.copy(), num_samples=cfg.number_of_samples, generation=False)
            else:
                _ = model(warmup_batch.copy(), sample=cfg.sample, generation=False)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        logger.info("Warmup done.")

    # load templates for evaluation
    with open(Path(cfg.region_path)/"lve.txt")as f:
        maps = f.read().split(",")
        mouth_map = [int(i) for i in maps]
    with open(Path(cfg.region_path)/"fdd.txt") as f:
        maps = f.read().split(",")
        upper_map = [int(i) for i in maps]
    templates_ref = np.load(cfg.reference_path)     # (5023, 3) zero pose

    # Loop over mask ratios for arflow, single pass for others
    for imr in mask_ratios:
        # Set mask ratio & output path for this sweep iteration
        if is_arflow:
            model.infer_mask_ratio = imr
            mask_str = f"_um{imr}"
            sweep_path = output_dir / f"arflow{mask_str}{extra_str}"
            sweep_path.mkdir(exist_ok=True, parents=True)
            logger.info(f"[arflow] infer_mask_ratio={imr} -> {sweep_path}")
        else:
            sweep_path = path

        seq_count = 0
        frame_count = 0
        mve_sum = 0.0               # incremental MVE accumulator
        mve_count = 0
        lve_all = []                # lve per-frame values (scalar per frame, small)
        mee_all = []                # mean value
        ce_all = []                 # closest value
        motion_std_difference = []  # fdd: first prediction
        # jitter / smoothness: first prediction vs GT, {name: [sum, count]}
        smooth = {f"{space}_{kind}_{who}": [0.0, 0]
                  for space in ("vert", "param") for kind in ("acc", "jerk") for who in ("pred", "gt")}
        diversity = 0               # 2 subsets
        total_inference_time = 0.0  # accumulated model inference wall-time (seconds)
        total_inference_frames = 0  # total output frames across all sequences
        first_token_times = []      # TTFT per sequence (seconds)
        with torch.no_grad():
            with Progress(transient=True) as progress:
                task = progress.add_task("Sampling", total=len(dataset.keyids))
                for idx, keyid in enumerate(dataset.keyids):
                    progress.update(task, description=f"Sampling {keyid}..")

                    # load gt data
                    motion_gt = np.load(Path(cfg.data.motion_path) / f"{keyid}.npy")    # (1, T, 403)
                    # save gt to check (a small part)
                    keyid_split = keyid.split("_")
                    emo = emo_dict[str(keyid_split[2])]     # retrieve emotion
                    ints = int_dict[str(keyid_split[3])]    # retrieve intensity
                    gt_path = sweep_path / "param" / f"{keyid}_{emo}_{ints}_gt.npy"
                    gt_path_vert = sweep_path / "vert" / f"{keyid}_{emo}_{ints}_gt.npy"

                    if (idx + 1) % 100 == 0:
                        logger.info(f"Saving param: {gt_path.stem}")
                        gt_path.parent.mkdir(exist_ok=True, parents=True)
                        np.save(gt_path, motion_gt)

                    # sample test data
                    test_data = dataset.load_keyid(keyid)
                    batch = collate_motion_and_audio([test_data])
                
                    ce_lve_set = []     # save 10 lve values
                    motion_set = []     # save 10 samples
                    vertices_npy_gt = None
                    flamelayer = None
                    # Use cached multi-sample for diffusion model (HuBERT + condition computed once)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    seq_infer_start = time.perf_counter()
                    if hasattr(cfg.model, 'vqvae_diffusion_pred') and cfg.model.vqvae_diffusion_pred:
                        all_motion_preds = model.forward_multi_sample(batch.copy(), num_samples=cfg.number_of_samples, generation=False)
                    elif hasattr(cfg.model, 'vqvae_arflow_pred') and cfg.model.vqvae_arflow_pred:
                        all_motion_preds = model.forward_multi_sample(batch.copy(), num_samples=cfg.number_of_samples, generation=False)
                    else:
                        all_motion_preds = None
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    seq_infer_end = time.perf_counter()
                    # For multi-sample models the entire batch is generated above;
                    # for single-sample models we time each call inside the loop.
                    if all_motion_preds is not None:
                        seq_infer_elapsed = seq_infer_end - seq_infer_start
                        total_inference_time += seq_infer_elapsed
                        # TTFT: approximate as time for one sample (shared audio encoding)
                        first_token_times.append(seq_infer_elapsed / cfg.number_of_samples)
                    single_sample_time_accum = 0.0
                    for i in range(cfg.number_of_samples):
                        if all_motion_preds is not None:
                            motion_pred = all_motion_preds[i]
                        else:
                            if torch.cuda.is_available():
                                torch.cuda.synchronize()
                            t_single_start = time.perf_counter()
                            motion_pred = model(batch.copy(), sample=cfg.sample, generation=False)
                            if torch.cuda.is_available():
                                torch.cuda.synchronize()
                            t_single_elapsed = time.perf_counter() - t_single_start
                            single_sample_time_accum += t_single_elapsed
                            if i == 0:
                                first_token_times.append(t_single_elapsed)
                        if motion_pred.shape[1] < motion_gt.shape[1] and i == 0:
                            motion_gt = motion_gt[:, :motion_pred.shape[1], :]      # (1, T, 403)
                        assert motion_gt.shape[1] == motion_pred.shape[1], "Length mismatch"

                        # denormalization
                        shape = np.zeros((motion_gt.shape[1], 300))     # (T, 300) use zero shape
                        if cfg.number_of_samples > 1:
                            pred_path = sweep_path/"param"/f"{keyid}_{emo}_{ints}_{i}.npy"
                            pred_path_vert = sweep_path/"vert"/f"{keyid}_{emo}_{ints}_{i}.npy"
                        else:
                            pred_path = sweep_path/"param"/f"{keyid}_{emo}_{ints}_one.npy"
                            pred_path_vert = sweep_path/"vert"/f"{keyid}_{emo}_{ints}_one.npy"

                        inverse_exp_pred, inverse_jaw_pred = denormalization(shape=shape,
                                                                             exp_jaw=detach_to_numpy(motion_pred.squeeze(0)),
                                                                             save_path=pred_path,
                                                                             idx=idx,
                                                                             i=i)
                        if i == 0:  # set the flame layer and compute gt vertices for once
                            # initialize flame layer
                            cfg.flame.batch_size = motion_gt.shape[1]
                            flamelayer = FLAME(cfg.flame).to(model.device)
                            # adjust input shape
                            exp_gt = np.squeeze(motion_gt[:, :, 300:350])           # (T, 50) use first 50 exp
                            exp_suffix = np.zeros((motion_gt.shape[1], 50))         # (T, 50)
                            exp_gt = np.concatenate((exp_gt, exp_suffix), axis=1)   # (T, 100)
                            jaw_gt = np.squeeze(motion_gt[:, :, 400:])              # (T, 3)
                            # transfer gt params to vertex (T, 5023, 3)
                            vertices_npy_gt = transfer_to_vert(flamelayer=flamelayer,
                                                               shape=shape,
                                                               exp=exp_gt,
                                                               jaw=jaw_gt,
                                                               save_path=gt_path_vert,
                                                               device=model.device,
                                                               idx=idx,
                                                               i=i)
                            params_gt = np.concatenate((exp_gt[:, :50], jaw_gt), axis=1)              # (T, 53)
                            params_pred0 = np.concatenate((inverse_exp_pred[:, :50], inverse_jaw_pred), axis=1)
                        assert flamelayer is not None and vertices_npy_gt is not None, "No GT information"

                        # transfer pred params to vertex (T, 5023, 3)
                        vertices_npy_pred = transfer_to_vert(flamelayer=flamelayer,
                                                             shape=shape,
                                                             exp=inverse_exp_pred,
                                                             jaw=inverse_jaw_pred,
                                                             save_path=pred_path_vert,
                                                             device=model.device,
                                                             idx=idx,
                                                             i=i)
                        """CE: compute lve for each samples"""
                        ce_lve = lve_compute(vertices_gt=list(vertices_npy_gt),
                                             vertices_pred=list(vertices_npy_pred),
                                             mouth_map=mouth_map)
                        ce_lve_set.append(ce_lve)                   # (T,)
                    
                        # save 10 samples
                        motion_set.append(vertices_npy_pred)        # (T, 5023, 3)

                    """MVE, LVE: compute per-sequence and accumulate (avoids OOM)"""
                    verts_gt = np.array(vertices_npy_gt)    # (T, 5023, 3)
                    verts_pred = motion_set[0]               # (T, 5023, 3)
                    vertices_dis = np.linalg.norm(verts_gt - verts_pred, axis=2)  # (T, 5023)
                    mve_sum += vertices_dis.sum()
                    mve_count += vertices_dis.size  # T * 5023 (total elements, not just frames)
                    lve_seq = lve_compute(vertices_gt=list(verts_gt),
                                          vertices_pred=list(verts_pred),
                                          mouth_map=mouth_map)
                    lve_all.extend(list(lve_seq))

                    """MEE: mean over 10 samples"""
                    motion_set_stack = np.stack(motion_set, axis=0)
                    vertices_npy_pred_mean = np.mean(motion_set_stack, axis=0)      # (T, 5023, 3)
                    mee_lve = lve_compute(vertices_gt=list(vertices_npy_gt), 
                                          vertices_pred=list(vertices_npy_pred_mean), 
                                          mouth_map=mouth_map)
                    mee_all.extend(list(mee_lve))

                    """CE: closest lve in 10 samples"""
                    smallest_lve = None
                    smallest_lve_value = float('inf')               # start with an infinitely large value
                    for lve_of_one_seq in ce_lve_set:
                        lve_value = np.sum(lve_of_one_seq)
                        if lve_value < smallest_lve_value:
                            smallest_lve_value = lve_value
                            smallest_lve = lve_of_one_seq           # (T,)
                    assert smallest_lve is not None, "No smallest distance found"
                    ce_all.extend(list(smallest_lve))

                    # count sequence, frame numbers
                    frame_count += motion_gt.shape[1]
                    total_inference_frames += motion_gt.shape[1]
                    if all_motion_preds is None:
                        total_inference_time += single_sample_time_accum
                    seq_count += 1

                    """FDD computation: use the first sample"""
                    subject_template = templates_ref.reshape(1, 5023, 3)
                    upper_std_gt = seq_std_compute(motion=vertices_npy_gt - subject_template,
                                                   map=upper_map)
                    upper_std_pred = seq_std_compute(motion=motion_set[0] - subject_template,
                                                     map=upper_map)
                    motion_std_difference.append(upper_std_gt - upper_std_pred)

                    """Jitter: mean |2nd / 3rd temporal difference|, first sample vs GT"""
                    for space, gt_seq, pred_seq in (("vert", vertices_npy_gt, motion_set[0]),
                                                    ("param", params_gt, params_pred0)):
                        for kind, order in (("acc", 2), ("jerk", 3)):
                            for who, seq in (("gt", gt_seq), ("pred", pred_seq)):
                                s, n = temporal_diff_norm(np.asarray(seq), order)
                                smooth[f"{space}_{kind}_{who}"][0] += s
                                smooth[f"{space}_{kind}_{who}"][1] += n

                    """Diversity computation"""
                    np.random.shuffle(motion_set)           # list of (T, 5023, 3)
                    n_samples = len(motion_set)
                    half = n_samples // 2
                    if half >= 1:
                        subset1 = motion_set[:half]
                        subset2 = motion_set[half:half * 2]
                        motion_diversity = 0
                        for sample1, sample2 in zip(subset1, subset2):
                            motion_diversity += np.linalg.norm(sample1 - sample2, axis=2).mean(axis=1).mean()
                        motion_diversity /= len(subset1)
                        diversity += motion_diversity
                    else:
                        diversity += 0.0  # single sample, no diversity

                    # per-sequence live FPS & TTFT
                    if all_motion_preds is not None:
                        seq_time = seq_infer_elapsed
                    else:
                        seq_time = single_sample_time_accum
                    seq_frames = motion_gt.shape[1]
                    seq_fps = seq_frames / seq_time if seq_time > 0 else float('inf')
                    seq_ttft_ms = first_token_times[-1] * 1000.0 if first_token_times else 0.0
                    running_fps = total_inference_frames / total_inference_time if total_inference_time > 0 else float('inf')
                    # print(f"Done sampling: {keyid}  |  seq FPS: {seq_fps:.1f}  |  TTFT: {seq_ttft_ms:.1f} ms  |  running FPS: {running_fps:.1f}")
                    torch.cuda.empty_cache()
                    progress.update(task, advance=1)

        # logging.disable(logging.NOTSET)
        logger.info('Total sequence number: {}'.format(seq_count))
        logger.info('Total frame number: {}'.format(frame_count))
        if seq_count == 0:
            logger.info("No sequences were processed. Unable to compute metrics.")
        else:
            """MVE computation"""
            logger.info('MVE: {:.4e}'.format(mve_sum / mve_count))

            """LVE computation"""
            logger.info('LVE: {:.4e}'.format(np.mean(lve_all)))

            """MEE computation"""
            logger.info('MEE: {:.4e}'.format(np.mean(mee_all)))

            """CE computation"""
            logger.info('CE: {:.4e}'.format(np.mean(ce_all)))

            """FDD computation"""
            logger.info('FDD: {:.4e}'.format(sum(motion_std_difference) / len(motion_std_difference)))

            """Jitter computation"""
            for line in smoothness_lines(smooth):
                logger.info(line)

            """Divertiy computation"""
            logger.info('Diversity: {:.4e}'.format(diversity / seq_count))

            """FPS computation"""
            if total_inference_time > 0:
                fps = total_inference_frames / total_inference_time
                logger.info('FPS (frames/sec): {:.2f}'.format(fps))
                logger.info('Total inference time: {:.2f} sec for {} frames'.format(
                    total_inference_time, total_inference_frames))
            else:
                logger.info('FPS: N/A (no inference time recorded)')

            """TTFT (First Token Output Time) computation"""
            if len(first_token_times) > 0:
                mean_ttft = np.mean(first_token_times) * 1000.0  # convert to ms
                std_ttft = np.std(first_token_times) * 1000.0
                min_ttft = np.min(first_token_times) * 1000.0
                max_ttft = np.max(first_token_times) * 1000.0
                logger.info('TTFT (first token output time): mean={:.2f} ms, std={:.2f} ms, '
                            'min={:.2f} ms, max={:.2f} ms'.format(mean_ttft, std_ttft, min_ttft, max_ttft))
            else:
                logger.info('TTFT: N/A (no first-token times recorded)')

            logger.info(f"All the sampling are done. You can find them here:\n{sweep_path}")

        # Save the same metrics to a text file alongside the samples for this sweep
        metrics_path = sweep_path / "metrics.txt"
        with open(metrics_path, 'w') as f:
            f.write(f"sampling_steps: {getattr(model, 'sampling_steps', None)}\n")
            f.write(f"infer_mask_ratio: {imr}\n")
            f.write(f"number_of_samples: {cfg.number_of_samples}\n")
            f.write('Total sequence number: {}\n'.format(seq_count))
            f.write('Total frame number: {}\n'.format(frame_count))
            if seq_count == 0:
                f.write("No sequences were processed. Unable to compute metrics.\n")
            else:
                f.write('MVE: {:.4e}\n'.format(mve_sum / mve_count))
                f.write('LVE: {:.4e}\n'.format(np.mean(lve_all)))
                f.write('MEE: {:.4e}\n'.format(np.mean(mee_all)))
                f.write('CE: {:.4e}\n'.format(np.mean(ce_all)))
                f.write('FDD: {:.4e}\n'.format(sum(motion_std_difference) / len(motion_std_difference)))
                for line in smoothness_lines(smooth):
                    f.write(line + '\n')
                f.write('Diversity: {:.4e}\n'.format(diversity / seq_count))
                if total_inference_time > 0:
                    f.write('FPS (frames/sec): {:.2f}\n'.format(total_inference_frames / total_inference_time))
                    f.write('Total inference time: {:.2f} sec for {} frames\n'.format(
                        total_inference_time, total_inference_frames))
                else:
                    f.write('FPS: N/A (no inference time recorded)\n')
                if len(first_token_times) > 0:
                    f.write('TTFT (first token output time): mean={:.2f} ms, std={:.2f} ms, '
                            'min={:.2f} ms, max={:.2f} ms\n'.format(
                                np.mean(first_token_times) * 1000.0,
                                np.std(first_token_times) * 1000.0,
                                np.min(first_token_times) * 1000.0,
                                np.max(first_token_times) * 1000.0))
                else:
                    f.write('TTFT: N/A (no first-token times recorded)\n')
        logger.info(f"Metrics saved to: {metrics_path}")


def denormalization(shape, exp_jaw, save_path, idx, i):
    exp_suffix = np.zeros((exp_jaw.shape[0], 50))   # (T, 50)

    inverse_exp, inverse_jaw = denormalize_motion(
        exp_jaw[:, :50], exp_jaw[:, 50:]
    )
    inverse_exp = np.concatenate((inverse_exp, exp_suffix), axis=1)         # (T, 100)
    inverse_exp_jaw = np.concatenate((inverse_exp, inverse_jaw), axis=1)    # (T, 103)
    seq = np.concatenate((shape, inverse_exp_jaw), axis=1)                  # (T, 403)
    seq = np.expand_dims(seq, axis=0)               # (1, T, 403)
    # save params (a small part)
    if (idx+1) % 100 == 0:  # save every 100 keyids for checking
        if (i+1) % 3 == 1:  # save 3 samples in 10
            logger.info(f"Saving param: {save_path.stem}")
            save_path.parent.mkdir(exist_ok=True, parents=True)
            np.save(save_path, seq)
    return inverse_exp, inverse_jaw


def transfer_to_vert(flamelayer, shape, exp, jaw, save_path, device, idx, i):
    input_shape = torch.tensor(shape).to(device)    # [T, 300]
    input_exp = torch.tensor(exp).to(device)        # [T, 100]
    input_global_pose = torch.zeros(jaw.shape[0], 3).to(device)    # [T, 3]
    input_jaw_pose = torch.tensor(jaw).to(device)   # [T, 3]
    input_pose = torch.cat((input_global_pose, input_jaw_pose), dim=1)
    # transfer params to vertices [T, 5023, 3]
    vertices, _ = flamelayer(input_shape.float(), input_exp.float(), input_pose.float())
    vertices_npy = detach_to_numpy(vertices)
    # save vertices (a small part)
    if (idx+1) % 100 == 0:  # save every 100 keyids for checking
        if (i+1) % 3 == 1:  # save 3 samples in 10
            logger.info(f"Saving vert: {save_path.stem}")
            save_path.parent.mkdir(exist_ok=True, parents=True)
            np.save(save_path, vertices_npy)
    return vertices_npy


def lve_compute(vertices_gt, vertices_pred, mouth_map):
    vertices_gt = np.array(vertices_gt)
    vertices_pred = np.array(vertices_pred)
    mouth_idx = np.array(mouth_map)
    L2_dis_mouth_max = np.square(vertices_gt[:, mouth_idx, :] - vertices_pred[:, mouth_idx, :])  # (T, 428, 3)
    L2_dis_mouth_max = np.sum(L2_dis_mouth_max, axis=2)   # (T, 428)
    L2_dis_mouth_max = np.max(L2_dis_mouth_max, axis=1)   # (T,)
    return L2_dis_mouth_max


def temporal_diff_norm(seq, order):
    """Sum and count of the order-th temporal finite-difference magnitude.
    seq: (T, V, 3) vertices -> per-vertex L2; (T, D) params -> per-dim abs.
    Returning (sum, count) lets sequences be pooled frame-weighted."""
    d = np.diff(seq, n=order, axis=0)
    if d.shape[0] == 0:
        return 0.0, 0
    mag = np.linalg.norm(d, axis=-1) if seq.ndim == 3 else np.abs(d)
    return float(mag.sum()), mag.size


def smoothness_lines(smooth):
    """Format pooled jitter metrics (acc = 2nd diff, jerk = 3rd diff, per frame)."""
    lines = []
    for space in ("vert", "param"):
        for kind in ("acc", "jerk"):
            vals = {}
            for who in ("pred", "gt"):
                s, n = smooth[f"{space}_{kind}_{who}"]
                vals[who] = s / n if n else float('nan')
            ratio = vals["pred"] / vals["gt"] if vals["gt"] else float('nan')
            lines.append('{}_{}: pred={:.4e} gt={:.4e} ratio={:.3f}'.format(
                space, kind, vals["pred"], vals["gt"], ratio))
    return lines


def seq_std_compute(motion, map):
    map_idx = np.array(map)
    L2_dis = np.square(motion[:, map_idx, :])    # (T, 1501, 3)
    L2_dis = np.sum(L2_dis, axis=2)               # (T, 1501)
    L2_dis = np.std(L2_dis, axis=0)               # (1501,)
    std = np.mean(L2_dis)
    return std


if __name__ == '__main__':
    _sample()
