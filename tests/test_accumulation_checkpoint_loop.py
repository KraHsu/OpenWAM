"""Exercise the actual trainer loop with tiny CPU autograd and epoch tails."""
from contextlib import contextmanager
from types import SimpleNamespace

import torch
from omegaconf import OmegaConf

from openwam.train.openwam_trainer import OpenWAMTrainer


def test_checkpoint_waits_for_update_and_resume_uses_saved_opt_count(monkeypatch, tmp_path):
    import openwam.train.openwam_trainer as module

    class Accelerator:
        num_processes = 1
        is_main_process = True
        sync_gradients = False
        step = 0

        @contextmanager
        def accumulate(self, model):
            self.step += 1
            position = (self.step-1) % 5+1
            self.sync_gradients = position in (3, 5)
            yield

        def backward(self, loss):
            (loss/3).backward()

    trainer = OpenWAMTrainer.__new__(OpenWAMTrainer)
    trainer.cfg = OmegaConf.create({'training': {
        'num_epochs': 2, 'max_steps': None, 'batch_size': 1, 'gradient_accumulation_steps': 3,
        'save_steps': 4, 'keep_last_k_ckpts': 2, 'save_full_states_for_resume': True,
    }})
    trainer.accelerator = Accelerator()
    trainer.architecture = torch.nn.Linear(1, 1)
    trainer.dataset = list(range(5))
    trainer._run_seed = None
    trainer.build_optimizer = lambda: torch.optim.SGD(trainer.architecture.parameters(), lr=.1)
    trainer.build_dataloader = lambda batch_size: trainer.dataset
    trainer.build_lr_scheduler = lambda *args, **kwargs: None
    trainer.setup_output_dir = lambda *args: (str(tmp_path), None)
    trainer.prepare_accelerate = lambda opt, dl, sched: (opt, dl, sched)
    def compute(_):
        loss = trainer.architecture(torch.ones(1, 1)).square().mean()
        return {'total': loss, 'video': loss.detach(), 'action': loss.detach()}
    trainer.compute_loss = compute
    trainer.log_step = lambda **kwargs: None
    trainer.finish_training = lambda *args: None
    monkeypatch.setattr(module, 'init_wandb', lambda cfg: None)
    monkeypatch.setattr(module, 'save_weights', lambda *args, **kwargs: None)
    monkeypatch.setattr(module, 'manage_checkpoints', lambda *args: None)
    saves = []
    def save(acc, output, step, opt_step, epoch):
        assert acc.sync_gradients
        saves.append((step, opt_step, epoch))
    monkeypatch.setattr(module, 'save_full_state', save)
    trainer.train()
    assert saves == [(5, 2, 0), (8, 3, 1)]
    monkeypatch.setattr(module, 'load_full_state', lambda *args: {'global_step': 10, 'opt_step': 4})
    assert trainer.resume_if_configured('state', trainer.dataset, 3) == (10, 4, 2, 0)
