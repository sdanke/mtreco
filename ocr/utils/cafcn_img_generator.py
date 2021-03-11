import os
import random as rnd

from PIL import Image, ImageColor, ImageFont, ImageDraw, ImageFilter
from .box_util import bbox_resize, bbox_margin, bbox_xywh_rotate
from trdg import (
    background_generator,
)


# def generate(text, font, text_color, font_size, orientation, space_width):
#     if orientation == 0:
#         return _generate_horizontal_text(
#             text, font, text_color, font_size, space_width
#         )
#     elif orientation == 1:
#         return _generate_vertical_text(
#             text, font, text_color, font_size, space_width
#         )
#     else:
#         raise ValueError("Unknown orientation " + str(orientation))


# def _generate_horizontal_text(text, font, text_color, font_size, space_width):
#     image_font = ImageFont.truetype(font=font, size=font_size)
#     words = text.replace(" ", "")
#     space_width = image_font.getsize(" ")[0] * space_width

#     words_width = [image_font.getsize(w)[0] for w in words]
#     text_width = sum(words_width) + int(space_width) * (len(words) - 1)
#     text_height = max([image_font.getsize(w)[1] for w in words])

#     txt_img = Image.new("RGBA", (text_width, text_height), (0, 0, 0, 0))

#     txt_draw = ImageDraw.Draw(txt_img)

#     colors = [ImageColor.getrgb(c) for c in text_color.split(",")]
#     c1, c2 = colors[0], colors[-1]

#     fill = (
#         rnd.randint(min(c1[0], c2[0]), max(c1[0], c2[0])),
#         rnd.randint(min(c1[1], c2[1]), max(c1[1], c2[1])),
#         rnd.randint(min(c1[2], c2[2]), max(c1[2], c2[2])),
#     )

#     boxes = []
#     for i, c in enumerate(words):
#         x, y = sum(words_width[0:i]) + i * int(space_width), 0
#         w, h = image_font.getsize(c)
#         boxes.append([x, y, x + w, y + h])
#         txt_draw.text(
#             (x, y),
#             c,
#             fill=fill,
#             font=image_font,
#         )

#     return txt_img, boxes


# def _generate_vertical_text(text, font, text_color, font_size, space_width):
#     image_font = ImageFont.truetype(font=font, size=font_size)

#     space_height = int(image_font.getsize(" ")[1] * space_width)

#     char_heights = [
#         image_font.getsize(c)[1] if c != " " else space_height for c in text
#     ]
#     text_width = max([image_font.getsize(c)[0] for c in text])
#     text_height = sum(char_heights)

#     txt_img = Image.new("RGBA", (text_width, text_height), (0, 0, 0, 0))

#     txt_draw = ImageDraw.Draw(txt_img)

#     colors = [ImageColor.getrgb(c) for c in text_color.split(",")]
#     c1, c2 = colors[0], colors[-1]

#     fill = (
#         rnd.randint(c1[0], c2[0]),
#         rnd.randint(c1[1], c2[1]),
#         rnd.randint(c1[2], c2[2]),
#     )

#     boxes = []
#     for i, c in enumerate(text):
#         x, y = 0, sum(char_heights[0:i])
#         w, h = text_width, char_heights[i]
#         boxes.append([x, y, x + w, y + h])
#         txt_draw.text((x, y), c, fill=fill, font=image_font, align='Center')

#     return txt_img, boxes


def generate(
    text,
    font,
    text_color,
    font_size,
    orientation,
    space_width,
    character_spacing,
    fit,
    word_split,
    stroke_width=0,
    stroke_fill="#282828",
    angle=0,
):
    if orientation == 0:
        return _generate_horizontal_text(
            text,
            font,
            text_color,
            font_size,
            space_width,
            character_spacing,
            fit,
            word_split,
            stroke_width,
            stroke_fill,
            angle=angle
        )
    elif orientation == 1:
        return _generate_vertical_text(
            text, font, text_color, font_size, space_width, character_spacing, fit,
            stroke_width, stroke_fill, angle=angle
        )
    else:
        raise ValueError("Unknown orientation " + str(orientation))


def _generate_horizontal_text(
    text, font, text_color, font_size, space_width, character_spacing, fit, word_split,
    stroke_width=0, stroke_fill="#282828", angle=0
):
    image_font = ImageFont.truetype(font=font, size=font_size)

    space_width = int(image_font.getsize(" ")[0] * space_width)

    if word_split:
        splitted_text = []
        for w in text.split(" "):
            splitted_text.append(w)
            splitted_text.append(" ")
        splitted_text.pop()
    else:
        splitted_text = text

    piece_widths = [
        image_font.getsize(p)[0] if p != " " else space_width for p in splitted_text
    ]
    text_width = sum(piece_widths)
    if not word_split:
        text_width += character_spacing * (len(text) - 1)

    def get_pos(i):
        return sum(piece_widths[0:i]) + i * character_spacing * int(not word_split), 0

    # bboxes = [
    #     bbox_xywh_rotate([
    #         get_pos(i)[0],
    #         get_pos(i)[1],
    #         image_font.getsize(p)[0],
    #         image_font.getsize(p)[1]
    #     ], -angle) for i, p in enumerate(splitted_text)
    # ]

    text_height = max([image_font.getsize(p)[1] for p in splitted_text])
    # text_height = max([box[3] for box in bboxes])

    txt_img = Image.new("RGBA", (text_width, text_height), (0, 0, 0, 0))
    txt_mask = Image.new("RGB", (text_width, text_height), (0, 0, 0))

    txt_img_draw = ImageDraw.Draw(txt_img)
    txt_mask_draw = ImageDraw.Draw(txt_mask, mode="RGB")
    txt_mask_draw.fontmode = "1"

    colors = [ImageColor.getrgb(c) for c in text_color.split(",")]
    c1, c2 = colors[0], colors[-1]

    fill = (
        rnd.randint(min(c1[0], c2[0]), max(c1[0], c2[0])),
        rnd.randint(min(c1[1], c2[1]), max(c1[1], c2[1])),
        rnd.randint(min(c1[2], c2[2]), max(c1[2], c2[2])),
    )

    stroke_colors = [ImageColor.getrgb(c) for c in stroke_fill.split(",")]
    stroke_c1, stroke_c2 = stroke_colors[0], stroke_colors[-1]

    stroke_fill = (
        rnd.randint(min(stroke_c1[0], stroke_c2[0]), max(stroke_c1[0], stroke_c2[0])),
        rnd.randint(min(stroke_c1[1], stroke_c2[1]), max(stroke_c1[1], stroke_c2[1])),
        rnd.randint(min(stroke_c1[2], stroke_c2[2]), max(stroke_c1[2], stroke_c2[2])),
    )

    bboxes = []
    for i, p in enumerate(splitted_text):
        x, y = sum(piece_widths[0:i]) + i * character_spacing * int(not word_split), 0
        w, h = image_font.getsize(p)
        # x, y, _, _ = bboxes[i]
        bboxes.append([x, y, x + w, y + h])
        txt_img_draw.text(
            (x, y),
            p,
            fill=fill,
            font=image_font,
            stroke_width=stroke_width,
            stroke_fill=stroke_fill,
        )
        txt_mask_draw.text(
            (x, y),
            p,
            fill=((i + 1) // (255 * 255), (i + 1) // 255, (i + 1) % 255),
            font=image_font,
            stroke_width=stroke_width,
            stroke_fill=stroke_fill,
        )

    if fit:
        return txt_img.crop(txt_img.getbbox()), txt_mask.crop(txt_img.getbbox()), bboxes
    else:
        return txt_img, txt_mask, bboxes


def _generate_vertical_text(
    text, font, text_color, font_size, space_width, character_spacing, fit,
    stroke_width=0, stroke_fill="#282828", angle=0
):
    image_font = ImageFont.truetype(font=font, size=font_size)

    space_height = int(image_font.getsize(" ")[1] * space_width)

    char_heights = [
        image_font.getsize(c)[1] if c != " " else space_height for c in text
    ]
    # text_width = max([image_font.getsize(c)[0] for c in text])

    def get_pos(i):
        return 0, sum(char_heights[0:i]) + i * character_spacing

    # bboxes = [
    #     bbox_xywh_rotate([
    #         get_pos(i)[0],
    #         get_pos(i)[1],
    #         image_font.getsize(p)[0],
    #         image_font.getsize(p)[1]
    #     ], -angle) for i, p in enumerate(text)
    # ]
    text_width = max([image_font.getsize(c)[0] for c in text])
    # text_width = max([box[2] for box in bboxes])
    text_height = sum(char_heights) + character_spacing * len(text)

    txt_img = Image.new("RGBA", (text_width, text_height), (0, 0, 0, 0))
    txt_mask = Image.new("RGBA", (text_width, text_height), (0, 0, 0, 0))

    txt_img_draw = ImageDraw.Draw(txt_img)
    txt_mask_draw = ImageDraw.Draw(txt_mask)

    colors = [ImageColor.getrgb(c) for c in text_color.split(",")]
    c1, c2 = colors[0], colors[-1]

    fill = (
        rnd.randint(c1[0], c2[0]),
        rnd.randint(c1[1], c2[1]),
        rnd.randint(c1[2], c2[2]),
    )

    stroke_colors = [ImageColor.getrgb(c) for c in stroke_fill.split(",")]
    stroke_c1, stroke_c2 = stroke_colors[0], stroke_colors[-1]

    stroke_fill = (
        rnd.randint(stroke_c1[0], stroke_c2[0]),
        rnd.randint(stroke_c1[1], stroke_c2[1]),
        rnd.randint(stroke_c1[2], stroke_c2[2]),
    )

    bboxes = []
    for i, c in enumerate(text):
        x, y = 0, sum(char_heights[0:i]) + i * character_spacing
        w, h = text_width, char_heights[i]
        # x, y, _, _ = bboxes[i]
        bboxes.append([x, y, x + w, y + h])
        txt_img_draw.text(
            (x, y),
            c,
            fill=fill,
            font=image_font,
            stroke_width=stroke_width,
            stroke_fill=stroke_fill,
        )
        txt_mask_draw.text(
            (x, y),
            c,
            fill=(i // (255 * 255), i // 255, i % 255),
            font=image_font,
            stroke_width=stroke_width,
            stroke_fill=stroke_fill,
        )

    if fit:
        return txt_img.crop(txt_img.getbbox()), txt_mask.crop(txt_img.getbbox()), bboxes
    else:
        return txt_img, txt_mask, bboxes


class CustomTextDataGenerator(object):
    @classmethod
    def generate_from_tuple(cls, t):
        """
            Same as generate, but takes all parameters as one tuple
        """

        cls.generate(*t)

    @classmethod
    def generate(
        cls,
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
        character_spacing=0,
        margins=(5, 5, 5, 5),
        fit=False,
        output_mask=False,
        word_split=False,
        image_dir=None,
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
        image, _, bboxes = generate(
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

        ##################################
        # Resize image to desired format #
        ##################################

        # Horizontal text
        if orientation == 0:
            ratio = float(size - vertical_margin) / float(image.size[1])
            bboxes = bbox_resize(bboxes, (ratio, ratio))
            # ratio_y = size - vertical_margin / size
            new_width = int(image.size[0] * ratio)
            resized_img = image.resize(
                (new_width, size - vertical_margin), Image.ANTIALIAS
            )
            background_width = width if width > 0 else new_width + horizontal_margin
            background_height = size
        # Vertical text
        elif orientation == 1:
            ratio = (float(size - horizontal_margin) / float(image.size[0]))
            bboxes = bbox_resize(bboxes, (ratio, ratio))
            new_height = int(
                float(image.size[1]) * ratio
            )
            resized_img = image.resize(
                (size - horizontal_margin, new_height), Image.ANTIALIAS
            )
            background_width = size
            background_height = new_height + vertical_margin
        else:
            raise ValueError("Invalid orientation")

        #############################
        # Generate background image #
        #############################
        if background_type == 0:
            background = background_generator.gaussian_noise(
                background_height, background_width
            )
        elif background_type == 1:
            background = background_generator.plain_white(
                background_height, background_width
            )
        elif background_type == 2:
            background = background_generator.quasicrystal(
                background_height, background_width
            )
        else:
            background = background_generator.picture(
                background_height, background_width
            )

        #############################
        # Place text with alignment #
        #############################

        new_text_width, _ = resized_img.size

        if alignment == 0 or width == -1:
            background.paste(resized_img, (margin_left, margin_top), resized_img)
            bboxes = bbox_margin(bboxes, (margin_left, margin_top))
        elif alignment == 1:
            background.paste(
                resized_img,
                (int(background_width / 2 - new_text_width / 2), margin_top),
                resized_img,
            )
        else:
            background.paste(
                resized_img,
                (background_width - new_text_width - margin_right, margin_top),
                resized_img,
            )

        ##################################
        # Apply gaussian blur #
        ##################################

        final_image = background.filter(
            ImageFilter.GaussianBlur(
                radius=(blur if not random_blur else rnd.randint(0, blur))
            )
        )

        #####################################
        # Generate name for resulting image #
        #####################################
        if name_format == 0:
            image_name = "{}_{}.{}".format(text, str(index), extension)
        elif name_format == 1:
            image_name = "{}_{}.{}".format(str(index), text, extension)
        elif name_format == 2:
            image_name = "{}.{}".format(str(index), extension)
        else:
            print("{} is not a valid name format. Using default.".format(name_format))
            image_name = "{}_{}.{}".format(text, str(index), extension)

        # Save the image
        if out_dir is not None:
            final_image.convert("RGB").save(os.path.join(out_dir, image_name))
        else:
            return final_image.convert("RGB"), bboxes
