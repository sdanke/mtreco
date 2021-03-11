from ocr.utils.cafcn_postprocess import post_process


def accuracy(outputs, labels):
    cls_ids, scores = post_process(outputs[0])
    # labels = targets['labels']
    n_correct = 0
    n_total = len(labels)
    for pred, target in zip(cls_ids, labels):
        if pred == target:
            n_correct += 1

    return n_correct / float(n_total)
