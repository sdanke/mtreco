import json
import torch
import yaml
import cv2
from PIL import Image


def load_yaml(yaml_file):
    with open(yaml_file, 'r', encoding='utf-8') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    return data


def load_vocab(vocab_file):
    with open(vocab_file, 'r', encoding='utf-8') as f:
        data = f.read().replace('\n', '')
    return data


def load_cafcn_vocab2(vocab_file):
    with open(vocab_file, 'r', encoding='utf-8') as f:
        vocab_list = json.load(f)
    # vocab_list = load_vocab(vocab_file)
    vocab_dict = {}
    vocab_dict['BG_TOKEN'] = 0
    vocab_dict['OOV_TOKEN'] = 1
    for i, item in enumerate(vocab_list):
        vocab_dict[item] = i + 2
    return vocab_dict


def load_cafcn_vocab(vocab_file):
    with open(vocab_file, 'r', encoding='utf-8') as f:
        data = f.read().replace('\n', '')
    vocab_dict = {}
    vocab_dict['BG_TOKEN'] = 0
    vocab_dict['OOV_TOKEN'] = 1
    for i, item in enumerate(data):
        vocab_dict[item] = i + 2
    return vocab_dict


def load_json_file(file):
    try:
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception:
        return None


def save_json_file(data, file):
    with open(file, 'w', encoding='utf-8') as f:
        json.dump(data,f)


def load_text_source(source_file):
    with open(source_file, 'r', encoding='utf-8') as f:
        source = [line.rstrip() for line in f]
    return source


def load_image(img_file):
    image = Image.open(img_file)
    return image.convert('L')


def load_cvimage(img_file):

    # image = Image.open(img_file)
    image = cv2.imread(img_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # if image is None:
    #     print(img_file)
    # image = image[:, :, ::-1]
    return image


def load_checkpoint(model, checkpoint, show_log=True):
    if "state_dict" in checkpoint:
        state_dict_ = checkpoint["state_dict"]
    elif "model" in checkpoint:
        state_dict_ = checkpoint["model"]
    else:
        state_dict_ = checkpoint

    state_dict = {}

    # convert data_parallal to model
    for k in state_dict_:
        if k.startswith("module") and not k.startswith("module_list"):
            state_dict[k[7:]] = state_dict_[k]
        else:
            state_dict[k] = state_dict_[k]
    model_state_dict = model.state_dict()

    # check loaded parameters and created model parameters
    for k in state_dict:
        if k in model_state_dict:
            if state_dict[k].shape != model_state_dict[k].shape:
                print(
                    "Skip loading parameter {}, required shape{}, loaded shape{}.".format(
                        k, model_state_dict[k].shape, state_dict[k].shape
                    )
                ) if show_log else None
                state_dict[k] = model_state_dict[k]
        else:
            print("Drop parameter {}.".format(k)) if show_log else None
    for k in model_state_dict:
        if not (k in state_dict):
            print("No param {}.".format(k)) if show_log else None
            state_dict[k] = model_state_dict[k]
    model.load_state_dict(state_dict, strict=False)


def load_model(model, model_path, optimizer=None, lr_scheduler=None, resume=False):
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    load_checkpoint(model, checkpoint)
    if "epoch" in checkpoint:
        epoch = checkpoint["epoch"]
        print(f"loaded {model_path}, epoch {epoch}")

    # resume optimizer parameters
    if (
        lr_scheduler and "lr_scheduler" in checkpoint
        and checkpoint["lr_scheduler"] is not None
    ):
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
    if optimizer is not None and resume:
        if "optimizer" in checkpoint:
            # optimizer.
            optimizer.load_state_dict(checkpoint["optimizer"])

        else:
            print("No optimizer parameters in checkpoint.")
    if 'client_state' in checkpoint:
        return checkpoint['client_state']
    # if optimizer is not None:
    #     return model, optimizer
    # else:
    #     return model


def save_model(path, model, client_state, optimizer=None, lr_scheduler=None):
    # model = self.model_with_loss.model
    if isinstance(model, torch.nn.DataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    data = {
        "state_dict": state_dict,
        "client_state": client_state
    }
    if lr_scheduler is not None:
        data["lr_scheduler"] = lr_scheduler.state_dict()
    if optimizer is not None:
        data["optimizer"] = optimizer.state_dict()
    torch.save(data, path)
