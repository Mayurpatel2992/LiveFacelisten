import os
import random
import torch
import torch.nn.functional as F
from torch import Tensor
from typing import List
from hydra.utils import instantiate
import logging

from framework.model.utils.tools import load_checkpoint, create_one_hot, resample_input
from framework.model.utils.diffusion_mlp import DiffusionMLPHead, FlowMatchingScheduler
from framework.model.metrics.compute import ComputeMetrics
from framework.model.base import BaseModel
from framework.data.utils import get_split_keyids

logger = logging.getLogger(__name__)


class VqvaeDiffusionPredict(BaseModel):
    def __init__(self,
                 # passed trough datamodule
                 nfeats: int,
                 split_path: str,
                 one_hot_dim: List,
                 resumed_training: bool,
                 **kwargs):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.nfeats = nfeats
        self.resumed_training = resumed_training
        self.working_dir = self.hparams.working_dir

        self.feature_extractor = instantiate(self.hparams.feature_extractor)
        logger.info(f"1. Audio feature extractor '{self.feature_extractor.hparams.name}' loaded")
        audio_encoded_dim = self.feature_extractor.audio_encoded_dim
        self.audio_dim = audio_encoded_dim * 2  # after frame-pair reshape

        # Style one-hot embedding
        self.all_identity_list = get_split_keyids(path=split_path, split="train")
        self.all_identity_onehot = torch.eye(len(self.all_identity_list))

        # Load motion prior (Stage 1) — frozen
        self.motion_prior = instantiate(self.hparams.motion_prior,
                                        nfeats=self.hparams.nfeats,
                                        logger_name="none",
                                        resumed_training=False,
                                        _recursive_=False)
        logger.info(f"2. '{self.motion_prior.hparams.modelname}' loaded")
        if os.path.exists(self.hparams.ckpt_path_prior):
            load_checkpoint(model=self.motion_prior,
                            ckpt_path=self.hparams.ckpt_path_prior,
                            eval_mode=True,
                            device=self.device)
        else:
            raise ValueError(f"Motion Autoencoder path not found: {self.hparams.ckpt_path_prior}")
        for param in self.motion_prior.parameters():
            param.requires_grad = False

        # Condition encoder (transformer predictor)
        self.feature_predictor = instantiate(self.hparams.feature_predictor,
                                             audio_dim=audio_encoded_dim * 2,   # 768*2
                                             one_hot_dim=sum(one_hot_dim[:]),   # 32+8+3
                                             prosody_dim=103)
        logger.info(f"3. 'Audio Condition Encoder' loaded")

        # Diffusion MLP Head
        diff_cfg = self.hparams.diffusion
        self.diffusion_mlp_head = DiffusionMLPHead(
            latent_dim=diff_cfg.latent_dim,
            cond_dim=diff_cfg.cond_dim,
            time_dim=diff_cfg.time_dim,
            hidden_dim=diff_cfg.hidden_dim,
            audio_dim=diff_cfg.latent_dim,  # processed audio_tokens are 256-dim (after prosody+proj+style)
            audio_proj_dim=diff_cfg.get('audio_proj_dim', 256),
        )
        logger.info(f"4. 'Diffusion MLP Head' loaded")

        # Flow matching scheduler
        self.scheduler = FlowMatchingScheduler()

        # Loss weights
        loss_cfg = self.hparams.loss
        self.lambda_flow = loss_cfg.lambda_diff
        self.lambda_lat = loss_cfg.lambda_lat
        self.lambda_exp = loss_cfg.lambda_exp
        self.lambda_jaw = loss_cfg.lambda_jaw

        # Sampling config
        self.sampling_steps = diff_cfg.sampling_steps

        # Gumbel-Softmax temperature for condition quantization
        self.gumbel_temperature = diff_cfg.get('gumbel_temperature', 1.0)

        # Crop length for training (no padding needed)
        self.crop_length = self.hparams.crop_length

        self.optimizer = instantiate(self.hparams.optim, params=filter(
            lambda p: p.requires_grad, self.parameters()))
        self.metrics = ComputeMetrics()

        # If we want to override it at testing time
        self.temperature = 0.2
        self.k = 1

        self.__post_init__()

    # Forward: audio => motion, called during inference/generation
    def forward(self, batch, sample=False, generation=True) -> Tensor:
        self.feature_predictor.to(self.device)
        self.feature_extractor.to(self.device)
        self.motion_prior.to(self.device)
        self.diffusion_mlp_head.to(self.device)

        # style embedding
        style_one_hot = create_one_hot(keyids=batch["keyid"],
                                       IDs_list=self.all_identity_list,
                                       IDs_labels=self.all_identity_onehot,
                                       one_hot_dim=self.hparams.one_hot_dim)

        # audio feature extraction
        audio_feature = self.feature_extractor(batch['audio'], False)  # list of [B, Ts, 768]
        resample_audio_feature = []
        prosody_feature = []
        for idx in range(len(audio_feature)):
            prosody_feature.append(batch['prosody'][idx])
            if audio_feature[idx].shape[1] % 2 != 0:
                audio_feature_one = audio_feature[idx][:, :audio_feature[idx].shape[1] - 1, :]
            else:
                audio_feature_one = audio_feature[idx]

            # for evaluation, make sure the pred motion length equals gt
            if not generation:
                if audio_feature_one.shape[1] > batch['motion'][0].shape[1] * 2:
                    print("Shape checking:", audio_feature_one.shape, batch['motion'][0].shape)
                    audio_feature_one = audio_feature_one[:, :batch['motion'][0].shape[1] * 2, :]

            audio_feature_one = torch.reshape(audio_feature_one,
                                              (1, audio_feature_one.shape[1] // 2, audio_feature_one.shape[2] * 2))
            resample_audio_feature.append(audio_feature_one)

        # Pad all sequences to max length across the batch
        max_len = max(a.shape[1] for a in resample_audio_feature)
        audio_dim_size = resample_audio_feature[0].shape[2]
        padded_audio = []
        for a in resample_audio_feature:
            if a.shape[1] < max_len:
                pad = torch.zeros(1, max_len - a.shape[1], audio_dim_size,
                                  dtype=a.dtype, device=a.device)
                a = torch.cat([a, pad], dim=1)
            padded_audio.append(a)
        batch['audio'] = torch.cat(padded_audio, dim=0)
        batch['prosody'] = torch.stack(prosody_feature, dim=0)

        # Use flow matching ODE sampling for inference
        motion_out = self.sample(batch['audio'], style_one_hot.to(self.device),
                                 sampling_steps=self.sampling_steps,
                                 prosody=batch['prosody'].to(self.device),
                                 quantizer=self.motion_prior.quantize)
        return motion_out

    @torch.no_grad()
    def forward_multi_sample(self, batch, num_samples=10, generation=False):
        """Generate multiple samples with a single HuBERT pass (much faster)."""
        self.feature_predictor.to(self.device)
        self.feature_extractor.to(self.device)
        self.motion_prior.to(self.device)
        self.diffusion_mlp_head.to(self.device)

        style_one_hot = create_one_hot(keyids=batch["keyid"],
                                       IDs_list=self.all_identity_list,
                                       IDs_labels=self.all_identity_onehot,
                                       one_hot_dim=self.hparams.one_hot_dim)

        # Extract audio features ONCE
        audio_feature = self.feature_extractor(batch['audio'], False)
        resample_audio_feature = []
        prosody_feature = []
        for idx in range(len(audio_feature)):
            prosody_feature.append(batch['prosody'][idx])
            if audio_feature[idx].shape[1] % 2 != 0:
                audio_feature_one = audio_feature[idx][:, :audio_feature[idx].shape[1] - 1, :]
            else:
                audio_feature_one = audio_feature[idx]
            if not generation:
                if audio_feature_one.shape[1] > batch['motion'][0].shape[1] * 2:
                    audio_feature_one = audio_feature_one[:, :batch['motion'][0].shape[1] * 2, :]
            audio_feature_one = torch.reshape(audio_feature_one,
                                              (1, audio_feature_one.shape[1] // 2, audio_feature_one.shape[2] * 2))
            resample_audio_feature.append(audio_feature_one)

        # Pad all sequences to max length across the batch
        max_len = max(a.shape[1] for a in resample_audio_feature)
        audio_dim_size = resample_audio_feature[0].shape[2]
        padded_audio = []
        for a in resample_audio_feature:
            if a.shape[1] < max_len:
                pad = torch.zeros(1, max_len - a.shape[1], audio_dim_size,
                                  dtype=a.dtype, device=a.device)
                a = torch.cat([a, pad], dim=1)
            padded_audio.append(a)
        audio = torch.cat(padded_audio, dim=0)
        prosody = torch.stack(prosody_feature, dim=0).to(self.device)
        style = style_one_hot.to(self.device)

        # Encode condition ONCE (with Gumbel-Softmax quantized token fusion)
        cond, audio_tokens = self.feature_predictor.encode_condition(
            audio, style, prosody,
            quantizer=self.motion_prior.quantize,
            gumbel_temperature=self.gumbel_temperature
        )

        # Sample multiple times (only diffusion sampling varies)
        results = []
        for _ in range(num_samples):
            motion_out = self.sample(audio, style, sampling_steps=self.sampling_steps, cond=cond, prosody=prosody,
                                     quantizer=self.motion_prior.quantize, audio_tokens=audio_tokens)
            results.append(motion_out)
        return results

    @torch.no_grad()
    def sample(self, audio, style, sampling_steps=None, cond=None, prosody=None, quantizer=None, audio_tokens=None):
        """Flow matching Euler ODE sampling: integrate from t=1 (noise) to t=0 (data)."""
        if sampling_steps is None:
            sampling_steps = self.sampling_steps

        if cond is None:
            cond, audio_tokens = self.feature_predictor.encode_condition(
                audio, style, prosody,
                quantizer=quantizer,
                gumbel_temperature=self.gumbel_temperature
            )  # cond: [B, T, 256], audio_tokens: [B, T, 256]
        B, T, D = cond.shape
        z = torch.randn(B, T, D, device=audio.device)  # start from pure noise at t=1

        # Euler integration from t=1 to t=0
        dt = 1.0 / sampling_steps
        for i in range(sampling_steps):
            t_val = 1.0 - i * dt  # current time, going from 1 → 0
            t = torch.full((B,), t_val, device=audio.device, dtype=torch.float32)
            v_hat = self.diffusion_mlp_head(z, t, cond, audio_tokens=audio_tokens)  # predicted velocity
            z = z - dt * v_hat  # Euler step toward data

        # z is now approximately z0 (clean latent)
        z0_hat = z
        z0_hat_q, _, _ = self.motion_prior.quantize(z0_hat)
        # Inference: chunked causal attn avoids OOM on long sequences
        motion_pred = self.motion_prior.motion_decoder(z0_hat_q, use_chunked_attn=True)
        return motion_pred

    # Called during training
    def allsplit_step(self, split: str, batch, batch_idx):
        # extract audio features
        audio_feature = self.feature_extractor(batch['audio'], False)       # list of [1, Ts, 768]
        # style embedding
        style_one_hot = create_one_hot(keyids=batch["keyid"],
                                       IDs_list=self.all_identity_list,
                                       IDs_labels=self.all_identity_onehot,
                                       one_hot_dim=self.hparams.one_hot_dim)

        resample_audio_feature = []
        resample_motion_feature = []
        prosody_feature = []
        for idx in range(len(audio_feature)):
            resample_audio, resample_motion = resample_input(audio_feature[idx], batch['motion'][idx],
                                                             self.feature_extractor.hparams.output_framerate,
                                                             self.hparams.video_framerate)
            resample_audio_feature.append(resample_audio)
            resample_motion_feature.append(resample_motion)
            prosody_feature.append(batch['prosody'][idx])

        if split == "train":
            # Fixed-length random crop for training (no padding needed)
            crop_len = self.crop_length
            cropped_audio = []
            cropped_motion = []
            for a, m in zip(resample_audio_feature, resample_motion_feature):
                T = a.shape[1]
                if T > crop_len:
                    start = random.randint(0, T - crop_len)
                    a = a[:, start:start + crop_len, :]
                    m = m[:, start:start + crop_len, :]
                elif T < crop_len:
                    a_pad = torch.zeros(1, crop_len - T, a.shape[2], dtype=a.dtype, device=a.device)
                    m_pad = torch.zeros(1, crop_len - T, m.shape[2], dtype=m.dtype, device=m.device)
                    a = torch.cat([a, a_pad], dim=1)
                    m = torch.cat([m, m_pad], dim=1)
                cropped_audio.append(a)
                cropped_motion.append(m)
            batch['audio'] = torch.cat(cropped_audio, dim=0)   # [B, crop_len, audio_dim]
            batch['motion'] = torch.cat(cropped_motion, dim=0) # [B, crop_len, 53]
            batch['prosody'] = torch.stack(prosody_feature, dim=0)
            seq_lengths = [min(a.shape[1], crop_len) for a in resample_audio_feature]
        else:
            # batch_size=1 for val/test — pass through as-is
            batch['audio'] = torch.cat(resample_audio_feature, dim=0)
            batch['motion'] = torch.cat(resample_motion_feature, dim=0)
            batch['prosody'] = torch.stack(prosody_feature, dim=0)
            seq_lengths = [a.shape[1] for a in resample_audio_feature]

        audio = batch['audio']       # [B, T, audio_dim]
        motion = batch['motion']     # [B, T, 53]
        style = style_one_hot.to(self.device)  # [B, 43]

        # Shape checks
        assert audio.shape[-1] == self.audio_dim, f"Audio dim mismatch: {audio.shape[-1]} vs expected {self.audio_dim}"
        assert motion.shape[-1] == 53, f"Motion dim mismatch: {motion.shape[-1]}"
        assert style.shape[-1] == sum(self.hparams.one_hot_dim[:]), f"Style dim mismatch: {style.shape[-1]}"

        # --- Get clean target latent z0 from frozen motion prior ---
        with torch.no_grad():
            z_m = self.motion_prior.motion_encoder(motion)    # [B, T, 256]
            z0, _, _ = self.motion_prior.quantize(z_m)        # [B, T, 256]
            z0 = z0.detach()

        # --- Encode condition from audio + style (with Gumbel-Softmax quantized token fusion) ---
        prosody = batch['prosody'].to(self.device)
        cond, audio_tokens = self.feature_predictor.encode_condition(
            audio, style, prosody,
            quantizer=self.motion_prior.quantize,
            gumbel_temperature=self.gumbel_temperature
        )  # cond: [B, T, 256], audio_tokens: [B, T, 256]

        assert z0.shape[-1] == 256, f"z0 dim mismatch: {z0.shape[-1]}"
        assert cond.shape == z0.shape, f"cond/z0 shape mismatch: {cond.shape} vs {z0.shape}"

        # --- Flow matching forward process ---
        B, T, D = z0.shape
        t = torch.rand(B, device=z0.device)                  # t ~ U(0, 1)
        z1 = torch.randn_like(z0)                             # noise sample
        z_t = self.scheduler.interpolate(z0, z1, t)           # [B, T, 256]
        v_target = self.scheduler.target_velocity(z0, z1)     # v = z1 - z0

        assert z_t.shape == z0.shape, f"z_t shape mismatch: {z_t.shape} vs {z0.shape}"

        # --- Predict velocity (with processed audio tokens) ---
        v_hat = self.diffusion_mlp_head(z_t, t, cond, audio_tokens=audio_tokens)  # [B, T, 256]

        assert v_hat.shape == z0.shape, f"v_hat shape mismatch: {v_hat.shape} vs {z0.shape}"

        # --- Create padding mask: [B, T] where 1=valid, 0=padded ---
        mask = torch.zeros(B, T, device=z0.device)
        for i, length in enumerate(seq_lengths):
            mask[i, :length] = 1.0
        mask = mask.unsqueeze(-1)  # [B, T, 1] for broadcasting

        # --- Compute losses with masking ---
        # Flow loss: MSE on velocity, masked
        flow_loss_per_element = F.mse_loss(v_hat, v_target, reduction='none')  # [B, T, D]
        loss_flow = (flow_loss_per_element * mask).sum() / mask.sum()

        # Recover z0 from predicted velocity: z0_hat = z_t - t * v_hat
        t_expand = t.view(B, 1, 1)
        z0_hat = z_t - t_expand * v_hat

        # L1 on recovered latent (differentiable — provides gradient to MLP)
        lat_loss_per_element = F.l1_loss(z0_hat, z0, reduction='none')  # [B, T, D]
        loss_lat = (lat_loss_per_element * mask).sum() / mask.sum()

        # Quantize + decode for motion-space losses (straight-through estimator
        # in VectorQuantizer allows gradients to flow back to diffusion MLP)
        z0_hat_q, _, _ = self.motion_prior.quantize(z0_hat)
        # Train: original full causal attn. Val/test: chunked to avoid OOM on long seqs.
        motion_pred = self.motion_prior.motion_decoder(
            z0_hat_q, use_chunked_attn=(split != "train")
        )  # [B, T, 53]

        # Expression and jaw losses, masked
        exp_loss_per_element = F.l1_loss(motion_pred[..., :50], motion[..., :50], reduction='none')  # [B, T, 50]
        loss_exp = (exp_loss_per_element * mask).sum() / mask.sum()
        jaw_loss_per_element = F.l1_loss(motion_pred[..., 50:53], motion[..., 50:53], reduction='none')  # [B, T, 3]
        loss_jaw = (jaw_loss_per_element * mask).sum() / mask.sum()

        loss = (
            self.lambda_flow * loss_flow
            + self.lambda_lat * loss_lat
            + self.lambda_exp * loss_exp
            + self.lambda_jaw * loss_jaw
        )

        # --- Logging (epoch-level for callbacks/monitors) ---
        self.log(f"flow/{split}", loss_flow.detach(), on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log(f"lat/{split}", loss_lat.detach(), on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log(f"exp/{split}", loss_exp.detach(), on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log(f"jaw/{split}", loss_jaw.detach(), on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log(f"total/{split}", loss.detach(), on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

        # Log latent statistics (step-level for debugging)
        self.log(f"z0_mean/{split}", z0.mean().detach(), on_step=True, on_epoch=False, prog_bar=False, sync_dist=True)
        self.log(f"z0_std/{split}", z0.std().detach(), on_step=True, on_epoch=False, prog_bar=False, sync_dist=True)
        self.log(f"z_t_mean/{split}", z_t.mean().detach(), on_step=True, on_epoch=False, prog_bar=False, sync_dist=True)
        self.log(f"z_t_std/{split}", z_t.std().detach(), on_step=True, on_epoch=False, prog_bar=False, sync_dist=True)
        self.log(f"v_hat_mean/{split}", v_hat.mean().detach(), on_step=True, on_epoch=False, prog_bar=False, sync_dist=True)
        self.log(f"v_hat_std/{split}", v_hat.std().detach(), on_step=True, on_epoch=False, prog_bar=False, sync_dist=True)

        # Compute the metrics
        if split == "val":
            self.metrics.update(motion_pred.detach(),
                                motion.detach(),
                                seq_lengths)

        # Show loss on progress bar
        if split == "train":
            self.log("loss_train", loss.detach(), prog_bar=True, on_step=True, on_epoch=False, sync_dist=True)
        if split == "val":
            self.log("loss_val", loss.detach(), prog_bar=True, on_step=True, on_epoch=False, sync_dist=True)

        return loss

    def allsplit_batch_end(self, split: str, batch_idx: int):
        if split == "val":
            metrics_dict = self.metrics.compute()
            dico = {f"Metrics/{metric}": value for metric, value in metrics_dict.items()}
            dico.update({"epoch": float(self.trainer.current_epoch),
                         "step": float(self.trainer.current_epoch)})
            self.log_dict(dico, sync_dist=True)

    def training_step(self, batch, batch_idx):
        loss = self.allsplit_step("train", batch, batch_idx)
        self.allsplit_batch_end("train", batch_idx)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.allsplit_step("val", batch, batch_idx)
        self.allsplit_batch_end("val", batch_idx)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.allsplit_step("test", batch, batch_idx)
        self.allsplit_batch_end("test", batch_idx)
        return loss

    def configure_optimizers(self):
        return {"optimizer": self.optimizer}

    def on_train_start(self):
        if self.resumed_training:
            self.resumed_training = False

        for param_group in self.trainer.optimizers[0].param_groups:
            if param_group['lr'] != self.hparams.optim.lr:
                param_group['lr'] = self.hparams.optim.lr
                print("New learning rate:", self.hparams.optim.lr)

        print("Current learning rate:", self.optimizer.param_groups[0]['lr'])
