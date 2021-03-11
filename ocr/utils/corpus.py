import os
import unicodedata
import json


def build_manga_corpus():
    manga_corpus = []
    with open('F:/workspace/dataset/manga109/corpus.txt', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            manga_corpus.append(unicodedata.normalize('NFKC', line).replace('\n', ''))
    return manga_corpus


def build_wiki_corpus(postfix='ja'):
    wiki_dir = 'F:/workspace/dataset/Corpus/wiki_cn_jap/wiki_zh_ja_tallip2015/wiki_zh_ja_tallip2015/'
    dev_file = f'{wiki_dir}/dev.{postfix}'
    fragment_file = f'{wiki_dir}/fragment.{postfix}'
    sentence_file = f'{wiki_dir}/sentence.{postfix}'
    test_file = f'{wiki_dir}/test.{postfix}'

    wiki_corpus = []
    for file_path in [dev_file, fragment_file, sentence_file, test_file]:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                wiki_corpus.append(unicodedata.normalize('NFKC', line).replace('\n', ''))
    return wiki_corpus


def save_corpus(corpus, vocab, name):
    corpus_dir = f'F:/workspace/code/projects/mtreco/data/corpus/{name}'
    if not os.path.exists(corpus_dir):
        os.makedirs(corpus_dir)
    with open(f'{corpus_dir}/full.json', 'w', encoding='utf-8') as f:
        json.dump({
            'corpus': corpus,
            'vocab': list(vocab)
        }, f)
    with open(f'{corpus_dir}/corpus.txt', 'w', encoding='utf-8') as f:
        for text in corpus:
            f.write(f'{text}\n')
    with open(f'{corpus_dir}/vocab.txt', 'w', encoding='utf-8') as f:
        for text in vocab:
            f.write(f'{text}')


def get_corpus(name):
    corpus_dir = f'F:/workspace/code/projects/mtreco/data/corpus/{name}'
    with open(f'{corpus_dir}/full.json', 'r', encoding='utf-8') as f:
        corpus = json.load(f)
    return corpus


if __name__ == '__main__':
    manga_corpus = build_manga_corpus()
    wiki_corpus_ja = build_wiki_corpus('ja')
    wiki_corpus_zh = build_wiki_corpus('zh')
    wiki_corpus_full = wiki_corpus_ja + wiki_corpus_zh
    manga_wiki_corpus_ja = manga_corpus + wiki_corpus_ja
    corpus_full = manga_corpus + wiki_corpus_full

    manga_corpus_vocab = set(''.join(manga_corpus))
    wiki_corpus_ja_vocab = set(''.join(wiki_corpus_ja))
    wiki_corpus_zh_vocab = set(''.join(wiki_corpus_zh))
    wiki_corpus_full_vocab = set(''.join(wiki_corpus_full))
    manga_wiki_corpus_ja_vocab = set(''.join(manga_wiki_corpus_ja))
    corpus_full_vocab = set(''.join(corpus_full))

    # save_corpus(manga_corpus, manga_corpus_vocab, 'manga')
    # save_corpus(wiki_corpus_ja, wiki_corpus_ja_vocab, 'wiki_ja')
    # save_corpus(wiki_corpus_zh, wiki_corpus_zh_vocab, 'wiki_zh')
    # save_corpus(wiki_corpus_full, wiki_corpus_full_vocab, 'wiki_ja_zh')
    save_corpus(manga_wiki_corpus_ja, manga_wiki_corpus_ja_vocab, 'manga_wiki_ja')
    # save_corpus(corpus_full, corpus_full_vocab, 'manga_wiki_ja_zh')
    # print(len(wiki_corpus_ja))
