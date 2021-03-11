import argparse
import os
import torch.backends.cudnn as cudnn

from ocr.utils.io import load_cafcn_vocab, load_json_file
from ocr.utils.seed import apply_seed
from ocr.trainer.cafcn_trainer import CAFCNTrainer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', help='Where to store logs and models')
    parser.add_argument('--train_data', required=True, help='path to training dataset')
    parser.add_argument('--valid_data', required=True, help='path to validation dataset')
    parser.add_argument('--seed', type=int, default=123454321, help='for random seed setting')
    parser.add_argument('--num_workers', type=int, help='number of data loading workers', default=0)
    parser.add_argument('--batch_size', type=int, default=512, help='input batch size')
    parser.add_argument('--num_epochs', type=int, default=6, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--fixed_lr', action='store_true', help='whether use fixed lr')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.9')
    parser.add_argument('--with_cuda', action='store_true', help='whether use cuda')
    parser.add_argument('--img_h', type=int, default=32, help='the height of the input image')
    parser.add_argument('--img_w', type=int, default=256, help='the width of the input image')
    parser.add_argument('--data_dir', type=str, default='F:/workspace/dataset/synthetic_line_images_with_boxes')
    parser.add_argument('--vocab', type=str, default='data/corpus/manga_wiki_ja/vocab.txt', help='vocab file')
    parser.add_argument('--pretrained', type=str, default='exps/cafcn/best/model_best.pth', help='pretrained model file')
    parser.add_argument('--save_interval', type=int, default=1000, help='save the model every save_interval steps')
    parser.add_argument('--val_interval', type=int, default=10000, help='validate the model every val_interval steps')
    parser.add_argument('--log_interval', type=int, default=100, help='save the model every save_interval steps')
    parser.add_argument('--lr_scheduler_config', type=str, default='config/cafcn_cyclic_config.json', help='lr_scheduler_config file')

    parser.add_argument('--val', action='store_true', help='whether only apply val')
    parser.add_argument('--resume', action='store_true', help='whether resume from model_last.pth')
    parser.add_argument('--use_accum', action='store_true', help='whether apply accumulation gradients')
    parser.add_argument(
        '--accum_steps',
        type=int,
        default=16,
        help="Num of updates steps to accumulate before performing a backward/update pass"
    )
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=1e-3,
        help="weight decay (L2 penalty).")

    parser.add_argument('--find_lr', action='store_true', help='whether apply accumulation gradients')
    parser.add_argument('--lr_finder_conf', type=str, default='config/lr_find_config.json', help='learning rate finder config file')
    parser.add_argument('--num_iters', type=int, default=4000, help='number of iterations for lr finder')

    opt = parser.parse_args()

    exp_dir = os.path.join('exps', opt.exp_name)
    opt.exp_dir = exp_dir
    os.makedirs(opt.exp_dir, exist_ok=True)
    opt.log = os.path.join(exp_dir, 'log.txt')
    opt.vocab = load_cafcn_vocab(opt.vocab)
    if opt.use_accum:
        opt.batch_size = int(opt.batch_size / opt.accum_steps)

    apply_seed(opt.seed)
    cudnn.benchmark = True
    cudnn.deterministic = True

    trainer = CAFCNTrainer(opt)
    if opt.find_lr:
        lr_finder_conf = load_json_file(opt.lr_finder_conf)
        trainer.find_lr(lr_finder_conf, opt.num_iters)
    elif opt.val:
        loss, score = trainer.validate()
        print(f'Val loss:{loss}, accuracy:{score}')
    else:
        trainer.train()
