import torch
import numpy as np

from tqdm import tqdm
from torch.utils.data import DataLoader

from ocr.dataset.ca_syn_dataset import CASyntheticDataset
from ocr.net.cafcn import get_cafcn
from ocr.utils.converters import CAFCNTokenizer
from ocr.metric.cafcn_metric import accuracy
from ocr.trainer.base_trainer import BaseTrainer
from ocr.lr_scheduler.lr_range_test import LRRangeTest
from ocr.loss.cafcn_loss import CAFCNLoss


class CAFCNTrainer(BaseTrainer):

    def _build_optimizer(self):
        opt = self.opt
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)

    def _build_model(self):
        self.model = get_cafcn(num_classes=len(self.opt.vocab), weights_file=self.opt.pretrained)
        self.model = self.model.to(self.device)

    def _build_converter(self):
        opt = self.opt
        vocab = opt.vocab
        self.converter = CAFCNTokenizer(vocab)

    def _build_dataloader(self):
        opt = self.opt
        tokenizer = self.converter
        train_dataset = CASyntheticDataset(
            tokenizer,
            opt.train_data,
            opt.data_dir,
            opt.img_w,
            opt.img_h,
            train=True,
        )
        val_dataset = CASyntheticDataset(
            tokenizer,
            opt.valid_data,
            opt.data_dir,
            opt.img_w,
            opt.img_h,
            train=False,
        )

        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=opt.batch_size,
            num_workers=opt.num_workers,
            shuffle=True,
            pin_memory=True
        )

        self.val_dataloader = DataLoader(
            val_dataset, batch_size=1, shuffle=False, num_workers=opt.num_workers,
            pin_memory=True
        )

    def _build_criterion(self):
        self.criterion = CAFCNLoss()

    def train_batch(self, step, batch_data):
        imgs, targets = batch_data
        imgs = imgs.to(self.device)
        for k in targets:
            if isinstance(targets[k], torch.Tensor):
                targets[k] = targets[k].to(self.device)

        outputs = self.model(imgs)
        loss, loss_stats = self.criterion(outputs, targets)

        # Calculate loss
        if self.opt.use_accum:
            loss = loss / self.opt.accum_steps
            loss.backward()
            if ((step + 1) % self.opt.accum_steps) == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
                if not self.opt.fixed_lr:
                    self.scheduler.step(step)
        else:
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            if not self.opt.fixed_lr:
                self.scheduler.step(step)

        return loss_stats

    def train(self):
        opt = self.opt
        if opt.resume:
            self.resume()
        iters_per_epoch = len(self.train_dataloader)
        passed_iter = step = self.client_state['step']
        if passed_iter >= iters_per_epoch:
            start_epoch = self.client_state['epoch'] + 1
            passed_iter = passed_iter - iters_per_epoch * (step // iters_per_epoch)
        else:
            start_epoch = self.client_state['epoch']

        self.logger.info(f'Train from epoch{start_epoch} iter{passed_iter}')
        best_accuracy = 0
        for epoch in range(start_epoch, opt.num_epochs):
            self.model.train()
            self.client_state['epoch'] = epoch

            num_iters = iters_per_epoch - passed_iter
            self.logger.info(f'Epoch: {epoch} - {num_iters} iteritions')
            bar = tqdm(total=num_iters)
            for iter_id, batch_data in enumerate(self.train_dataloader):
                self.client_state['step'] = step
                loss_stats = self.train_batch(step, batch_data)

                # Log
                if (step + 1) % opt.log_interval == 0:
                    bar_desc = f'Train {epoch}/{opt.num_epochs}'
                    for k, v in loss_stats.items():
                        self.writer.add_scalar(k, v, step)
                        bar_desc += f' {k}: {v:.4f}'
                    if not opt.fixed_lr:
                        cur_lr = self.scheduler.get_last_lr()
                        cur_lr = cur_lr[0] if isinstance(cur_lr, list) else cur_lr
                        self.writer.add_scalar('lr', cur_lr, step)
                    bar.set_description(bar_desc)
                bar.update()

                # if (step + 1) % opt.val_interval == 0:
                #     val_loss, accuracy = self.validate()
                #     self.writer.add_scalar('val_loss', val_loss, step)
                #     self.writer.add_scalar('accuracy', accuracy, step)

                # Save every opt.save_interval steps
                if (step + 1) % opt.save_interval == 0:
                    self.save(epoch, step, 'last')
                step += 1
                if iter_id == num_iters - 1:
                    break
            passed_iter = 0
            self.save(epoch, step, 'last')
            val_loss, accuracy = self.validate()
            self.writer.add_scalar('val_loss', val_loss, step)
            self.writer.add_scalar('accuracy', accuracy, step)

            self.logger.info(f"Val loss: {val_loss}, accuracy: {accuracy}")
            if accuracy >= best_accuracy:
                best_accuracy = accuracy
                self.save(epoch, step, 'best')

    def validate(self):
        self.model.eval()
        losses = []
        scores = []
        for imgs, targets in tqdm(self.val_dataloader, desc='Validate'):
            imgs = imgs.to(self.device)
            for k in targets:
                if isinstance(targets[k], torch.Tensor):
                    targets[k] = targets[k].to(self.device)

            # Calculate loss
            with torch.no_grad():
                outputs = self.model(imgs)
                _, loss_stats = self.criterion(outputs, targets)
                losses.append(loss_stats['loss'])
                scores.append(accuracy(outputs, targets['labels']))
        loss = np.mean(losses)
        score = np.mean(scores)
        self.model.train()
        return loss, score

    def find_lr(self, config, num_iters):
        self.scheduler = LRRangeTest(self.optimizer,
                                     lr_range_test_min_lr=config['lr_range_test_min_lr'],
                                     lr_range_test_step_size=config['lr_range_test_step_size'],
                                     lr_range_test_step_rate=config['lr_range_test_step_rate'],
                                     lr_range_test_staircase=config['lr_range_test_staircase'])
        bar = tqdm(total=num_iters)
        for step, batch_data in enumerate(self.train_dataloader):
            if step >= num_iters:
                break

            imgs, targets = batch_data
            imgs = imgs.to(self.device)
            for k in targets:
                if isinstance(targets[k], torch.Tensor):
                    targets[k] = targets[k].to(self.device)

            outputs = self.model(imgs)
            loss, loss_stats = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            self.writer.add_scalar('Loss', loss_stats['loss'], step)
            if step != 0:
                lr = self.scheduler.get_last_lr()[0]
                self.writer.add_scalar('LR', lr, step)
            self.scheduler.step(step)
            bar.update()
