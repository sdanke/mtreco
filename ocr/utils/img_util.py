from PIL import Image, ImageOps


def rotate_resize_padding_image(image, input_size):
    w, h = image.size
    if w < h:
        image = image.rotate(90, expand=True)
        w, h = image.size
    
    target_w, target_h = input_size
    ratio = target_h / float(h)
    new_w, new_h = int(w * ratio), target_h

    if new_w < target_w:
        # add pad
        image = image.resize((new_w, new_h), Image.ANTIALIAS)
        border = 0, 0, target_w - new_w, 0  # left, top, right ,bottom
        image = ImageOps.expand(image, border, fill="white")

    else:
        image = image.resize((target_w, target_h), Image.ANTIALIAS)

    return image
