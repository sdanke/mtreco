""""施工中"""
import torch
import numpy as np
import gc
import torch.nn.functional as F

from tqdm import tqdm
from torch.utils.data import DataLoader
from nltk.metrics.distance import edit_distance

from ocr.dataset.synthetic_dataset import SyntheticDataset
from ocr.net.model import Model
from ocr.utils import Averager
from ocr.utils.io import load_model, load_json_file
from ocr.utils.converters import AttnLabelConverter, CTCLabelConverterForBaiduWarpctc, CTCLabelConverter
from ocr.utils.logger import get_logger
from ocr.lr_scheduler import get_lr_schedule
from ocr.lr_scheduler.lr_range_test import LRRangeTest
from ocr.trainer.base_trainer import BaseTrainer


class DTRTrainer(BaseTrainer):
    def __init__(self, opt):
        self.opt = opt
        if opt.with_cuda:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device('cpu')
        self.logger = get_logger(opt.log)
        self.arch_config = opt.arch_config
        self._build_model(self.opt, self.arch_config)
        self._build_optimizer(self.opt)
        self._build_criterion(self.opt, self.arch_config)
        self._build_converter(self.opt, self.arch_config)
        self._build_dataloader(self.opt)
        self._build_summary_writer(self.opt)
        self.client_state = {'epoch': 0, 'step': 0}
        self.scheduler = get_lr_schedule(self.optimizer, load_json_file(opt.lr_scheduler_config))

    def _build_optimizer(self):
        opt = self.opt
        filtered_parameters = []
        params_num = []
        for p in filter(lambda p: p.requires_grad, self.model.parameters()):
            filtered_parameters.append(p)
            params_num.append(np.prod(p.size()))
        self.logger.info(f'Trainable params num: {sum(params_num)}')
        self.optimizer = torch.optim.Adam(filtered_parameters, lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizer = self.optimizer

    def _build_model(self):
        opt = self.opt
        config = self.arch_config
        self.model = Model(**config)
        if opt.pretrained is not None:
            load_model(self.model, opt.pretrained)
        self.model = self.model.to(self.device)

    def _build_converter(self):
        opt = self.opt
        config = self.arch_config
        vocab = opt.vocab
        if 'CTC' in config['prediction']:
            if self.opt.baiduCTC:
                converter = CTCLabelConverterForBaiduWarpctc(vocab)
            else:
                converter = CTCLabelConverter(vocab)
        else:
            converter = AttnLabelConverter(vocab)
        self.converter = converter

    def _build_dataloader(self):
        opt = self.opt
        train_ds = SyntheticDataset(opt.train_data)
        self.train_dataloader = DataLoader(train_ds, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers, pin_memory=True, drop_last=True)

        val_ds = SyntheticDataset(opt.valid_data)
        self.val_dataloader = DataLoader(val_ds, batch_size=opt.batch_size, shuffle=False, num_workers=0, pin_memory=True, drop_last=False)

    def _build_criterion(self):
        opt = self.opt
        config = self.arch_config
        if 'CTC' in config['prediction']:
            if opt.baiduCTC:
                # need to install warpctc. see our guideline.
                from warpctc_pytorch import CTCLoss
                criterion = CTCLoss()
            else:
                criterion = torch.nn.CTCLoss(zero_infinity=True)
        else:
            criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
        self.criterion = criterion
        self.criterion = self.criterion.to(self.device)

    def train_batch(self, step, batch_data):
        imgs, labels = batch_data
        imgs = imgs.to(self.device)
        labels, length = self.converter.encode(labels)
        labels = labels.to(self.device)
        length = length.to(self.device)

        # Calculate loss
        batch_size = imgs.size(0)
        if 'CTC' in self.arch_config['prediction']:
            preds = self.model(imgs, labels)
            preds_size = torch.IntTensor([preds.size(1)] * batch_size)
            if self.opt.baiduCTC:
                preds = preds.permute(1, 0, 2)  # to use CTCLoss format
                loss = self.criterion(preds, labels, preds_size, length) / batch_size
            else:
                preds = preds.log_softmax(2).permute(1, 0, 2)
                loss = self.criterion(preds, labels, preds_size, length)

        else:
            preds = self.model(imgs, labels[:, :-1])  # align with Attention.forward
            target = labels[:, 1:]  # without [GO] Symbol
            loss = self.criterion(preds.view(-1, preds.shape[-1]), target.contiguous().view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.opt.grad_clip)
        self.optimizer.step()
        self.optimizer.zero_grad()

        return loss

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

        best_accuracy = 0
        best_norm_ED = 0
        for epoch in range(start_epoch, opt.num_epochs):
            self.model.train()
            self.client_state['epoch'] = epoch

            train_loss_avg = Averager()
            num_iters = iters_per_epoch - passed_iter
            bar = tqdm(total=num_iters)
            for iter_id, batch_data in enumerate(self.train_dataloader):
                self.client_state['step'] = step
                loss = self.train_batch(batch_data)
                self.scheduler.step(step)

                # Log
                if (step + 1) % opt.log_interval == 0:
                    train_loss_avg.add(loss)
                    loss = loss.item()
                    self.writer.add_scalar('Loss', loss, step)
                    bar.set_description(f'Train {epoch}/{opt.num_epochs} Loss: {loss:.4f}')
                bar.update()

                # Save every opt.save_interval steps
                if (step + 1) % opt.save_interval == 0:
                    self.save(epoch, step, 'last')
                    gc.collect()
                if iter_id == num_iters - 1:
                    break
            train_loss = train_loss_avg.val()
            passed_iter = 0
            self.logger.info(f'Train loss: {train_loss}')
            self.save(epoch, step, 'last')

            val_loss, accuracy, norm_ED = self.validate()
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                self.save(epoch, step, 'best_accuracy')
            if norm_ED > best_norm_ED:
                best_norm_ED = norm_ED
                self.save(epoch, step,'best_norm_ED')

    def validate(self):
        opt = self.opt
        n_correct = 0
        norm_ED = 0
        length_of_data = 0
        valid_loss_avg = Averager()

        self.model.eval()
        for imgs, labels in tqdm(self.val_dataloader, desc='Validate'):
            batch_size = imgs.size(0)
            length_of_data += batch_size
            # For max length prediction
            length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(self.device)
            text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(self.device)

            labels, length = self.converter.encode(labels)
            imgs = imgs.to(self.device)
            labels = labels.to(self.device)
            length = length.to(self.device)

            # Calculate loss
            with torch.no_grad():
                if 'CTC' in self.arch_config['prediction']:
                    preds = self.model(imgs, text_for_pred)
                    preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                    if self.opt.baiduCTC:
                        loss = self.criterion(preds.permute(1, 0, 2), labels, preds_size, length) / batch_size
                    else:
                        loss = self.criterion(preds.log_softmax(2).permute(1, 0, 2), labels, preds_size, length)

                    if opt.baiduCTC:
                        _, preds_index = preds.max(2)
                        preds_index = preds_index.view(-1)
                    else:
                        _, preds_index = preds.max(2)
                    preds_str = self.converter.decode(preds_index.data, preds_size.data)
                else:
                    preds = self.model(imgs, text_for_pred, is_train=False)
                    preds = preds[:, :labels.shape[1] - 1, :]
                    target = labels[:, 1:]  # without [GO] Symbol
                    loss = self.criterion(preds.contiguous().view(-1, preds.shape[-1]), target.contiguous().view(-1))

                    _, preds_index = preds.max(2)
                    preds_str = self.converter.decode(preds_index, length_for_pred)
                    labels = self.converter.decode(labels[:, 1:], length)

                valid_loss_avg.add(loss)

                # calculate accuracy & confidence score
                preds_prob = F.softmax(preds, dim=2)
                preds_max_prob, _ = preds_prob.max(dim=2)
                confidence_score_list = []
                for gt, pred, pred_max_prob in zip(labels, preds_str, preds_max_prob):
                    if 'Attn' in self.arch_config['prediction']:
                        gt = gt[:gt.find('[s]')]
                        pred_EOS = pred.find('[s]')
                        pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                        pred_max_prob = pred_max_prob[:pred_EOS]

                    if pred == gt:
                        n_correct += 1

                    '''
                    (old version) ICDAR2017 DOST Normalized Edit Distance https://rrc.cvc.uab.es/?ch=7&com=tasks
                    "For each word we calculate the normalized edit distance to the length of the ground truth transcription."
                    if len(gt) == 0:
                        norm_ED += 1
                    else:
                        norm_ED += edit_distance(pred, gt) / len(gt)
                    '''

                    # ICDAR2019 Normalized Edit Distance
                    if len(gt) == 0 or len(pred) == 0:
                        norm_ED += 0
                    elif len(gt) > len(pred):
                        norm_ED += 1 - edit_distance(pred, gt) / len(gt)
                    else:
                        norm_ED += 1 - edit_distance(pred, gt) / len(pred)

                    # calculate confidence score (= multiply of pred_max_prob)
                    try:
                        confidence_score = pred_max_prob.cumprod(dim=0)[-1]
                    except Exception:
                        confidence_score = 0  # for empty pred case, when prune after "end of sentence" token ([s])
                    confidence_score_list.append(confidence_score)
        accuracy = n_correct / float(length_of_data) * 100
        norm_ED = norm_ED / float(length_of_data)  # ICDAR2019 Normalized Edit Distance
        val_loss = valid_loss_avg.val()
        self.logger.info(f'Val results - Loss: {val_loss} accuracy: {accuracy}, norm_ED: {norm_ED}')
        return val_loss, accuracy, norm_ED

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
            loss = self.train_batch(batch_data)
            self.writer.add_scalar('Loss', loss.item(), step)
            if step != 0:
                lr = self.scheduler.get_last_lr()[0]
                self.writer.add_scalar('LR', lr, step)
            self.scheduler.step(step)
            bar.update()