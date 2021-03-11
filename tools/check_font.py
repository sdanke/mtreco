import json
import sys
import argparse

from tqdm import tqdm
from glob import glob
from fontTools.ttLib import TTFont

sys.path.insert(0, f'{sys.path[0]}/..')
from ocr.utils.corpus import get_corpus


def get_fonts(font_files):
    fonts = []
    for font_path in font_files:
        font = TTFont(font_path, fontNumber=0)
        fonts.append(font)
    return fonts


def check_otf_font(unicode_char, font):
    for cmap in font['cmap'].tables:
        if cmap.isUnicode():
            if ord(unicode_char) in cmap.cmap:
                return True
    return False


def check_font(character, font, font_file):
    try:
        # font = TTFont(font_file, fontNumber=0)
        if font_file.split('.')[-1] == 'otf':
            return check_otf_font(character, font)
        # cmap = font.getBestCmap()
        # glyphSet = font.getGlyphSet()
        glyphSet = font['glyf']
        found = False
        for cmap in font['cmap'].tables:
            if cmap.isUnicode():
                if ord(character) in cmap.cmap:
                    glyphName = cmap.cmap[ord(character)]
                    # if glyphName in glyphSet._glyphs:
                    #     glyph = glyphSet._glyphs[glyphName]
                    if glyphName in glyphSet:
                        glyph = glyphSet[glyphName]
                        found = glyph.numberOfContours != 0
                        del glyph
                    else:
                        found = False
                    del glyphName
                else:
                    found = False
        del glyphSet
        return found
    except BaseException:
        return False


def find_avaliable_fonts(character, return_dict, font, font_path, fonts_indices_map):
    # for font_path in font_files:
    if check_font(character, font, font_path):
        if character in return_dict:
            return_dict[character].append(fonts_indices_map[font_path])
            # value = f'{return_dict[character]},{fonts_indices_map[font_path]}'
        else:
            return_dict[character] = [fonts_indices_map[font_path]]
            # value = fonts_indices_map[font_path]
        # return_dict[character] = value
        # del value
    return character, return_dict


# def update(args):
#     character, return_dict, need_save = args
#     pbar.update()
#     if need_save:
#         with open('check_fonts/resume.json', 'w', encoding='utf-8') as f:
#             json.dump(dict(return_dict), f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Check fonts for line image generator."
    )

    parser.add_argument(
        "--fonts_dir", type=str, help="The fonts file directory", default="E:/Data/jpfonts"
    )

    parser.add_argument(
        "--output", type=str, nargs="?", help="The output directory", default="../data/font_meta.json"
    )

    args = parser.parse_args()

    # placeholder = np.array(Image.open('temp/placeholder.jpg'))
    # with open('fonts_indices_map.json', 'r', encoding='utf-8') as f:
    #     fonts_indices_map = json.load(f)
    # with open('check_fonts/resume.json', 'r', encoding='utf-8') as f:
    #     resume = json.load(f)
    corpus = get_corpus('manga_wiki_ja')
    vocab = corpus['vocab']
    font_list = glob(f'{args.fonts_dir}/*.*')[:100]
    fonts = get_fonts(font_list)
    fonts_indices_map = {value: idx for idx, value in enumerate(font_list)}

    num_fonts = len(font_list)
    num_chars = len(vocab)
    return_dict = {}
    pbar = tqdm(total=num_chars * num_fonts)

    for character in vocab:
        # if character in resume:
        #     continue
        for f_idx in range(num_fonts):
            find_avaliable_fonts(character, return_dict, fonts[f_idx], font_list[f_idx], fonts_indices_map)
            pbar.update()

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump({
            'fonts_indices_map': fonts_indices_map,
            'char_avaliable_fonts': return_dict
        }, f)
