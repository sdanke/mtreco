import sys
import numpy as np
import os
import random as rnd
import argparse
import cv2
import json
import colorsys

from tqdm import tqdm
from uuid import uuid1
from numpy.random import randint, randn
from trdg import background_generator
from PIL import Image, ImageFilter, ImageDraw

# sys.path.insert(0, 'F:/workspace/code/projects/mtreco')
sys.path.insert(0, f'{sys.path[0]}/..')

from ocr.utils.corpus import get_corpus
from ocr.utils.cafcn_img_generator import generate
from ocr.utils.box_util import bbox_resize, bbox_margin, bboxes_rotate, bbox_rotate


class TextGenerator:
    def __init__(self, total, max_len=12):
        self.manga_corpus = get_corpus('manga')
        self.wiki_ja_corpus = get_corpus('wiki_ja')
        self.characters = list(set(self.manga_corpus['vocab'] + self.wiki_ja_corpus['vocab']))

        self.num_manga_texts = len(self.manga_corpus['corpus'])
        self.num_wikija_texts = len(self.wiki_ja_corpus['corpus'])
        self.num_chars = len(self.characters)
        self.total = total
        self.max_len = max_len

    def __len__(self):
        return self.total

    def __getitem__(self, index):
        return self.next()

    def next(self):
        if randn() <= 0.4:
            # Manga text
            return self.text_from_manga()
        elif randn() > 0.4 and randn() <= 0.6:
            # wiki text
            return self.text_from_wiki()
        else:
            # sample from vocab
            return self.text_from_random()

    def text_with_limited_len(self, text):
        diff = len(text) - self.max_len
        if diff <= 0:
            return text
        start = randint(0, diff)
        return text[start:self.max_len]

    def text_from_manga(self):
        random_idx = randint(0, self.num_manga_texts)
        text = self.manga_corpus['corpus'][random_idx]
        return self.text_with_limited_len(text)

    def text_from_wiki(self):
        random_idx = randint(0, self.num_wikija_texts)
        text = self.wiki_ja_corpus['corpus'][random_idx]
        return self.text_with_limited_len(text)

    def text_from_random(self):
        random_len = randint(8, self.max_len + 1)
        indices = randint(0, self.num_chars, random_len)
        text = ''.join([self.characters[i] for i in indices])
        return self.text_with_limited_len(text)


def rgb2hex(rgb):
    return '#%02x%02x%02x' % tuple(rgb)


def hex2rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))


def get_oppsite_rgb(rgb):
    r, g, b = rgb
    return 255 - r, 255 - g, 255 - b


def get_n_hls_colors(num):
    hls_colors = []
    i = 0
    step = 360.0 / num
    while i < 360:
        h = i
        s = 90 + rnd.random() * 10
        lv = 50 + rnd.random() * 10
        _hlsc = [h / 360.0, lv / 100.0, s / 100.0]
        hls_colors.append(_hlsc)
        i += step

    return hls_colors


def ncolors(num):
    rgb_colors = []
    if num < 1:
        return rgb_colors
    hls_colors = get_n_hls_colors(num)
    for hlsc in hls_colors:
        _r, _g, _b = colorsys.hls_to_rgb(hlsc[0], hlsc[1], hlsc[2])
        r, g, b = [int(x * 255.0) for x in (_r, _g, _b)]
        rgb_colors.append([r, g, b])
    return rgb_colors


def gaussian_noise(height, width, bg_color):
    """
        Create a background with Gaussian noise (to mimic paper)
    """

    # We create an all white image
    image = np.ones((height, width)) * 255

    # We add gaussian noise
    cv2.randn(image, 235, 10)
    # cv2.randn()

    return Image.fromarray(image).convert("RGBA")


def get_random_font(text, char_avaliable_fonts, indices_font_map):
    elements = []
    ilegal_chars = []
    for c in list(text):
        if c not in char_avaliable_fonts:
            ilegal_chars.append(c)
            continue
        elements.append(c)
    elements = list(set(elements))
    
    for ic in ilegal_chars:
        text = text.replace(ic, '')
    if len(elements) == 0:
        return None
    elif len(elements) == 1:
        avaliable_indices = char_avaliable_fonts[elements[0]]
    else:
        avaliable_indices = list(set(char_avaliable_fonts[elements[0]]).intersection(*[char_avaliable_fonts[i] for i in elements[1:]]))
    if len(avaliable_indices) == 0:
        return None
    random_idx = np.random.choice(avaliable_indices)
    return text, indices_font_map[random_idx]


def generate_img(
    index,
    text,
    font,
    out_dir,
    size,
    extension,
    skewing_angle,
    random_skew,
    blur,
    random_blur,
    background_type,
    distorsion_type,
    distorsion_orientation,
    is_handwritten,
    name_format,
    width,
    alignment,
    text_color,
    orientation,
    space_width,
    character_spacing,
    margins,
    fit,
    output_mask,
    word_split,
    image_dir,
    stroke_width=0,
    stroke_fill="#282828",
    image_mode="RGB",
):
    image = None

    margin_top, margin_left, margin_bottom, margin_right = margins
    horizontal_margin = margin_left + margin_right
    vertical_margin = margin_top + margin_bottom

    ##########################
    # Create picture of text #
    ##########################
    image, mask, bboxes = generate(
        text,
        font,
        text_color,
        size,
        orientation,
        space_width,
        character_spacing,
        fit,
        word_split,
        stroke_width,
        stroke_fill,
    )

    #############################
    # Apply distorsion to image #
    #############################
    if distorsion_type == 0:
        distorted_img = image  # Mind = blown
        distorted_mask = mask
    else:
        raise('Not use distorsion')

    ##################################
    # Resize image to desired format #
    ##################################

    # Horizontal text
    if orientation == 0:
        ratio = float(size - vertical_margin) / float(image.size[1])
        bboxes = bbox_resize(bboxes, (ratio, ratio))
        new_width = int(
            distorted_img.size[0]
            * (float(size - vertical_margin) / float(distorted_img.size[1]))
        )
        resized_img = distorted_img.resize(
            (new_width, size - vertical_margin), Image.ANTIALIAS
        )
        resized_mask = distorted_mask.resize((new_width, size - vertical_margin), Image.NEAREST)
        background_width = width if width > 0 else new_width + horizontal_margin
        background_height = size
    # Vertical text
    elif orientation == 1:
        ratio = (float(size - horizontal_margin) / float(image.size[0]))
        bboxes = bbox_resize(bboxes, (ratio, ratio))
        new_height = int(
            float(distorted_img.size[1])
            * (float(size - horizontal_margin) / float(distorted_img.size[0]))
        )
        resized_img = distorted_img.resize(
            (size - horizontal_margin, new_height), Image.ANTIALIAS
        )
        resized_mask = distorted_mask.resize(
            (size - horizontal_margin, new_height), Image.NEAREST
        )
        background_width = size
        background_height = new_height + vertical_margin
    else:
        raise ValueError("Invalid orientation")

    #############################
    # Generate background image #
    #############################
    if background_type == 0:
        background_img = background_generator.gaussian_noise(
            background_height, background_width
        )
    elif background_type == 1:
        background_img = background_generator.plain_white(
            background_height, background_width
        )
    elif background_type == 2:
        background_img = Image.new(
            "RGB", (background_width, background_height), stroke_fill
        )
    else:
        background_img = background_generator.image(
            background_height, background_width, image_dir
        )
    background_mask = Image.new(
        "RGB", (background_width, background_height), (0, 0, 0)
    )

    #############################
    # Place text with alignment #
    #############################

    new_text_width, _ = resized_img.size

    if alignment == 0 or width == -1:
        background_img.paste(resized_img, (margin_left, margin_top), resized_img)
        background_mask.paste(resized_mask, (margin_left, margin_top))
        bboxes = bbox_margin(bboxes, (margin_left, margin_top))
    elif alignment == 1:
        background_img.paste(
            resized_img,
            (int(background_width / 2 - new_text_width / 2), margin_top),
            resized_img,
        )
        background_mask.paste(
            resized_mask,
            (int(background_width / 2 - new_text_width / 2), margin_top),
        )
    else:
        background_img.paste(
            resized_img,
            (background_width - new_text_width - margin_right, margin_top),
            resized_img,
        )
        background_mask.paste(
            resized_mask,
            (background_width - new_text_width - margin_right, margin_top),
        )

    #######################
    # Apply gaussian blur #
    #######################

    gaussian_filter = ImageFilter.GaussianBlur(
        radius=blur if not random_blur else rnd.randint(0, blur)
    )
    final_image = background_img.filter(gaussian_filter)
    final_mask = background_mask.filter(gaussian_filter)

    ############################################
    # Change image mode (RGB, grayscale, etc.) #
    ############################################

    final_image = final_image.convert(image_mode)
    final_mask = final_mask.convert(image_mode)
    if output_mask == 1:
        return final_image, final_mask
    return final_image, bboxes


def step(idx, text_dataset, char_avaliable_fonts, indices_font_map, args):
    text = text_dataset[idx]
    if text.strip() == '':
        return

    # Font
    ret = get_random_font(text, char_avaliable_fonts, indices_font_map)
    if ret is None:
        return
    text, font = ret

    font_size = randint(24, 48)

    skew_angle = 0
    random_skew = False

    # Blur
    blur = 0
    random_blur = False

    # Background & text color
    bg_rate = randn()
    if bg_rate <= 0.2:
        # add noise
        background_type = 0
        text_color = "#000000,#404040"
        stroke_width = 0
        stroke_fill = "#000000"
    elif bg_rate > 0.2 and bg_rate <= 0.6:
        # white
        background_type = 1
        text_color = "#000000,#404040"
        stroke_width = 0
        stroke_fill = "#000000"
    elif bg_rate > 0.6 and bg_rate <= 0.8:
        # random color
        background_type = 2
        # random_rgb = list(np.random.choice(range(256), size=3))
        text_color, stroke_fill = ncolors(2)
        text_color = rgb2hex(text_color)
        stroke_fill = rgb2hex(stroke_fill)
        stroke_width = 0
    else:
        # random img
        background_type = 3
        # random_rgb = list(np.random.choice(range(256), size=3))
        text_color, stroke_fill = ncolors(2)
        text_color = rgb2hex(text_color)
        stroke_fill = rgb2hex(stroke_fill)
        stroke_width = randint(1, 3)

    image_mode = "L"
    image_dir = r'F:\workspace\dataset\manga109\cleaned2\images'

    distorsion_type = 0
    distorsion_orientation = 0
    is_handwritten = False
    width = -1
    alignment = 1

    orientation = 1 if randn() <= 0.7 else 0
    space_width = 1.0
    character_spacing = randint(0, 11)
    if orientation == 1:
        margins = (randint(0, 25), 5, randint(0, 25), 5)
    else:
        margins = (5, randint(0, 25), 5, randint(0, 25))
    fit = False
    output_mask = False
    word_split = False

    generated_img, bboxes = generate_img(
        idx,
        text,
        font,
        None,
        font_size,
        None,
        skew_angle,
        random_skew,
        blur,
        random_blur,
        background_type,
        distorsion_type,
        distorsion_orientation,
        is_handwritten,
        0,
        width,
        alignment,
        text_color,
        orientation,
        space_width,
        character_spacing,
        margins,
        fit,
        output_mask,
        word_split,
        image_dir,
        stroke_width=stroke_width,
        stroke_fill=stroke_fill,
        image_mode=image_mode,
    )

    img_id = uuid1()
    generated_img.save(f'{args.output_dir}/imgs/{img_id}.jpg')
    with open(f'{args.output_dir}/labels/{img_id}.json', 'w', encoding='utf-8') as f:
        json.dump({
            'text': text,
            'bboxes': bboxes
        }, f)


parser = argparse.ArgumentParser(
    description="Generate synthetic text data for text recognition."
)
parser.add_argument(
    "--total", type=int, nargs="?", help="The num of imgs u want to generate", default=1000
)
parser.add_argument(
    "--output_dir", type=str, nargs="?", help="The output directory", default="data/genetated"
)
parser.add_argument(
    "--font_meta", type=str, help="The fonts meta info", required=True
)

args = parser.parse_args()


if __name__ == '__main__':
    if not os.path.exists(f'{args.output_dir}/imgs'):
        os.makedirs(f'{args.output_dir}/imgs')
    if not os.path.exists(f'{args.output_dir}/labels'):
        os.makedirs(f'{args.output_dir}/labels')

    text_dataset = TextGenerator(total=args.total)
    # font_files = font_list = glob('E:/Data/jpfonts/*.*')

    with open(args.font_meta, 'r', encoding='utf-8') as f:
        meta = json.load(f)
        indices_font_map = {v: k for k, v in meta['fonts_indices_map'].items()}
        char_avaliable_fonts = meta['char_avaliable_fonts']

    for i in tqdm(range(args.total)):
        step(i, text_dataset, char_avaliable_fonts, indices_font_map, args)
