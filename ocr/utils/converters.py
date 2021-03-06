import torch
import collections


class CTCLabelConverter(object):
    """ Convert between text-label and text-index """

    def __init__(self, character):
        # character (str): set of the possible characters.
        dict_character = list(character)

        self.dict = {}
        for i, char in enumerate(dict_character):
            # NOTE: 0 is reserved for 'CTCblank' token required by CTCLoss
            self.dict[char] = i + 1

        self.character = ['[CTCblank]'] + dict_character  # dummy '[CTCblank]' token for CTCLoss (index 0)

    def encode(self, text, batch_max_length=25):
        """convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]
            batch_max_length: max length of text label in the batch. 25 by default

        output:
            text: text index for CTCLoss. [batch_size, batch_max_length]
            length: length of each text. [batch_size]
        """
        length = [len(s) for s in text]

        # The index used for padding (=0) would not affect the CTC loss calculation.
        batch_text = torch.LongTensor(len(text), batch_max_length).fill_(0)
        for i, t in enumerate(text):
            text = list(t)
            text = [self.dict[char] for char in text]
            batch_text[i][:len(text)] = torch.LongTensor(text)
        return (batch_text, torch.IntTensor(length))

    def decode(self, text_index, length):
        """ convert text-index into text-label. """
        texts = []
        for index, l in enumerate(length):
            t = text_index[index, :]

            char_list = []
            for i in range(l):
                if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):  # removing repeated characters and blank.
                    char_list.append(self.character[t[i]])
            text = ''.join(char_list)

            texts.append(text)
        return texts


class CTCLabelConverterForBaiduWarpctc(object):
    """ Convert between text-label and text-index for baidu warpctc """

    def __init__(self, character):
        # character (str): set of the possible characters.
        dict_character = list(character)

        self.dict = {}
        for i, char in enumerate(dict_character):
            # NOTE: 0 is reserved for 'CTCblank' token required by CTCLoss
            self.dict[char] = i + 1

        self.character = ['[CTCblank]'] + dict_character  # dummy '[CTCblank]' token for CTCLoss (index 0)

    def encode(self, text, batch_max_length=25):
        """convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]
        output:
            text: concatenated text index for CTCLoss.
                    [sum(text_lengths)] = [text_index_0 + text_index_1 + ... + text_index_(n - 1)]
            length: length of each text. [batch_size]
        """
        length = [len(s) for s in text]
        text = ''.join(text)
        text = [self.dict[char] for char in text]

        return (torch.IntTensor(text), torch.IntTensor(length))

    def decode(self, text_index, length):
        """ convert text-index into text-label. """
        texts = []
        index = 0
        for x in length:
            t = text_index[index:index + x]

            char_list = []
            for i in range(x):
                if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):  # removing repeated characters and blank.
                    char_list.append(self.character[t[i]])
            text = ''.join(char_list)

            texts.append(text)
            index += x
        return texts


class AttnLabelConverter(object):
    """ Convert between text-label and text-index """

    def __init__(self, character):
        # character (str): set of the possible characters.
        # [GO] for the start token of the attention decoder. [s] for end-of-sentence token.
        list_token = ['[GO]', '[s]']  # ['[s]','[UNK]','[PAD]','[GO]']
        list_character = list(character)
        self.character = list_token + list_character

        self.dict = {}
        for i, char in enumerate(self.character):
            # print(i, char)
            self.dict[char] = i

    def encode(self, text, batch_max_length=25):
        """ convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]
            batch_max_length: max length of text label in the batch. 25 by default

        output:
            text : the input of attention decoder. [batch_size x (max_length+2)] +1 for [GO] token and +1 for [s] token.
                text[:, 0] is [GO] token and text is padded with [GO] token after [s] token.
            length : the length of output of attention decoder, which count [s] token also. [3, 7, ....] [batch_size]
        """
        length = [len(s) + 1 for s in text]  # +1 for [s] at end of sentence.
        # batch_max_length = max(length) # this is not allowed for multi-gpu setting
        batch_max_length += 1
        # additional +1 for [GO] at first step. batch_text is padded with [GO] token after [s] token.
        batch_text = torch.LongTensor(len(text), batch_max_length + 1).fill_(0)
        for i, t in enumerate(text):
            text = list(t)
            text.append('[s]')
            text = [self.dict[char] for char in text]
            batch_text[i][1:1 + len(text)] = torch.LongTensor(text)  # batch_text[:, 0] = [GO] token
        return (batch_text, torch.IntTensor(length))

    def decode(self, text_index, length):
        """ convert text-index into text-label. """
        texts = []
        for index, l in enumerate(length):
            text = ''.join([self.character[i] for i in text_index[index, :]])
            texts.append(text)
        return texts


class CAFCNTokenizer(object):
    FH_SPACE = ((u"???", u" "),)
    FH_NUM = (
        (u"???", u"0"), (u"???", u"1"), (u"???", u"2"), (u"???", u"3"), (u"???", u"4"),
        (u"???", u"5"), (u"???", u"6"), (u"???", u"7"), (u"???", u"8"), (u"???", u"9"),
    )
    FH_ALPHA = (
        (u"???", u"a"), (u"???", u"b"), (u"???", u"c"), (u"???", u"d"), (u"???", u"e"),
        (u"???", u"f"), (u"???", u"g"), (u"???", u"h"), (u"???", u"i"), (u"???", u"j"),
        (u"???", u"k"), (u"???", u"l"), (u"???", u"m"), (u"???", u"n"), (u"???", u"o"),
        (u"???", u"p"), (u"???", u"q"), (u"???", u"r"), (u"???", u"s"), (u"???", u"t"),
        (u"???", u"u"), (u"???", u"v"), (u"???", u"w"), (u"???", u"x"), (u"???", u"y"), (u"???", u"z"),
        (u"???", u"A"), (u"???", u"B"), (u"???", u"C"), (u"???", u"D"), (u"???", u"E"),
        (u"???", u"F"), (u"???", u"G"), (u"???", u"H"), (u"???", u"I"), (u"???", u"J"),
        (u"???", u"K"), (u"???", u"L"), (u"???", u"M"), (u"???", u"N"), (u"???", u"O"),
        (u"???", u"P"), (u"???", u"Q"), (u"???", u"R"), (u"???", u"S"), (u"???", u"T"),
        (u"???", u"U"), (u"???", u"V"), (u"???", u"W"), (u"???", u"X"), (u"???", u"Y"), (u"???", u"Z"),
    )
    FH_PUNCTUATION = (
        (u"???", u"."), (u"???", u","), (u"???", u"!"), (u"???", u"?"), (u"???", u'"'),
        (u"???", u"'"), (u"???", u"`"), (u"???", u"@"), (u"???", u"_"), (u"???", u":"),
        (u"???", u";"), (u"???", u"#"), (u"???", u"$"), (u"???", u"%"), (u"???", u"&"),
        (u"???", u"("), (u"???", u")"), (u"???", u"-"), (u"???", u"="), (u"???", u"*"),
        (u"???", u"+"), (u"???", u"-"), (u"???", u"/"), (u"???", u"<"), (u"???", u">"),
        (u"???", u"["), (u"???", u"]"), (u"???", u"^"), (u"???", u"{"),
        (u"???", u"|"), (u"???", u"}"), (u"???", u"~"), (u"???", u"."),
    )

    def __init__(self, vocab_dict):
        self.dict = vocab_dict
        self.inv_dict = {v: k for k, v in vocab_dict.items()}
        self.fu_map = dict(self.FH_ALPHA + self.FH_NUM + self.FH_PUNCTUATION + self.FH_SPACE)

    def format_str(self, s):
        return self.fu_map[s] if s in self.fu_map else s

    def encode(self, text):
        if isinstance(text, str):
            text = [self.dict[self.format_str(item)] if self.format_str(item) in self.dict else self.dict['OOV_TOKEN'] for item in text]
        elif isinstance(text, collections.Iterable):
            text = [self.encode(s) for s in text]
        return torch.LongTensor(text)

    def decode(self, code):
        if isinstance(code, collections.Iterable):
            return ''.join([self.decode(c) for c in code])
        else:
            return self.inv_dict[code] if code != self.dict['OOV_TOKEN'] and code != self.dict['BG_TOKEN'] else ''
