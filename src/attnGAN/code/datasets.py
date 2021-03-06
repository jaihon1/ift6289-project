from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


from nltk.tokenize import RegexpTokenizer
from collections import defaultdict
from miscc.config import cfg
from miscc.vqa import VQA

import torch
import torch.utils.data as data
from torch.autograd import Variable
import torchvision.transforms as transforms

import os
import sys
import numpy as np
import pandas as pd
from PIL import Image
import numpy.random as random

import vqa.utils as vqa_utils
import vqa.config as vqa_config
from vqa.data import VQA as VQA_Dataset

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle


def prepare_data(data):
    imgs, captions, captions_lens, class_ids, keys, qas_gan, qas_len_gan, q_vqa, ans_vqa, item, q_len_vqa = data

    # sort data by the length in a decreasing order
    sorted_cap_lens, sorted_cap_indices = \
        torch.sort(captions_lens, 0, True)

    real_imgs = []
    for i in range(len(imgs)):
        imgs[i] = imgs[i][sorted_cap_indices]
        if cfg.CUDA:
            real_imgs.append(Variable(imgs[i]).to('cuda:0'))
        else:
            real_imgs.append(Variable(imgs[i]))

    captions = captions[sorted_cap_indices].squeeze()
    class_ids = class_ids[sorted_cap_indices].numpy()
    qas_gan = qas_gan[sorted_cap_indices]
    qas_len_gan = qas_len_gan[sorted_cap_indices]
    q_vqa = q_vqa[sorted_cap_indices]
    ans_vqa = ans_vqa[sorted_cap_indices]
    q_len_vqa = q_len_vqa[sorted_cap_indices]
    # sent_indices = sent_indices[sorted_cap_indices]
    keys = [keys[i] for i in sorted_cap_indices.numpy()]
    # print('keys', type(keys), keys[-1])  # list
    if cfg.CUDA:
        captions = Variable(captions).to('cuda:0')
        sorted_cap_lens = Variable(sorted_cap_lens).to('cuda:0')
        qas_gan = Variable(qas_gan).to('cuda:0')
        qas_len_gan = Variable(qas_len_gan).to('cuda:0')
        q_vqa = Variable(q_vqa).to('cuda:0')
        ans_vqa = Variable(ans_vqa).to('cuda:0')
        q_len_vqa = Variable(q_len_vqa).to('cuda:0')
    else:
        captions = Variable(captions)
        sorted_cap_lens = Variable(sorted_cap_lens)
        qas_gan = Variable(qas_gan)
        qas_len_gan = Variable(qas_len_gan)
        q_vqa = Variable(q_vqa)
        ans_vqa = Variable(ans_vqa)
        q_len_vqa = Variable(q_len_vqa)
    return [real_imgs, captions, sorted_cap_lens,
            class_ids, keys, qas_gan, qas_len_gan, q_vqa, ans_vqa, item, q_len_vqa]


def prepare_data_valid(data):
    imgs, captions, captions_lens, class_ids, keys, _, _, _, _, _, _ = data

    # sort data by the length in a decreasing order
    sorted_cap_lens, sorted_cap_indices = \
        torch.sort(captions_lens, 0, True)

    real_imgs = []
    for i in range(len(imgs)):
        imgs[i] = imgs[i][sorted_cap_indices]
        if cfg.CUDA:
            real_imgs.append(Variable(imgs[i]).to('cuda:0'))
        else:
            real_imgs.append(Variable(imgs[i]))

    captions = captions[sorted_cap_indices].squeeze()
    class_ids = class_ids[sorted_cap_indices].numpy()
    # sent_indices = sent_indices[sorted_cap_indices]
    keys = [keys[i] for i in sorted_cap_indices.numpy()]
    # print('keys', type(keys), keys[-1])  # list
    if cfg.CUDA:
        captions = Variable(captions).to('cuda:0')
        sorted_cap_lens = Variable(sorted_cap_lens).to('cuda:0')
    else:
        captions = Variable(captions)
        sorted_cap_lens = Variable(sorted_cap_lens)

    return [real_imgs, captions, sorted_cap_lens, class_ids, keys]


def get_imgs(img_path, imsize, bbox=None,
             transform=None, normalize=None):
    img = Image.open(img_path).convert('RGB')
    width, height = img.size
    if bbox is not None:
        r = int(np.maximum(bbox[2], bbox[3]) * 0.75)
        center_x = int((2 * bbox[0] + bbox[2]) / 2)
        center_y = int((2 * bbox[1] + bbox[3]) / 2)
        y1 = np.maximum(0, center_y - r)
        y2 = np.minimum(height, center_y + r)
        x1 = np.maximum(0, center_x - r)
        x2 = np.minimum(width, center_x + r)
        img = img.crop([x1, y1, x2, y2])

    if transform is not None:
        img = transform(img)

    ret = []
    if cfg.GAN.B_DCGAN:
        ret = [normalize(img)]
    else:
        for i in range(cfg.TREE.BRANCH_NUM):
            # print(imsize[i])
            if i < (cfg.TREE.BRANCH_NUM - 1):
                re_img = transforms.Resize(imsize[i])(img)
            else:
                re_img = img
            ret.append(normalize(re_img))

    return ret


class TextDataset(data.Dataset):
    def __init__(self, data_dir,
                 split='train',
                 base_size=64,
                 transform=None, target_transform=None):
        self.transform = transform
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.target_transform = target_transform
        self.embeddings_num = cfg.TEXT.CAPTIONS_PER_IMAGE

        self.imsize = []
        for i in range(cfg.TREE.BRANCH_NUM):
            self.imsize.append(base_size)
            base_size = base_size * 2

        self.data = []
        self.data_dir = data_dir
        if data_dir.find('birds') != -1:
            self.bbox = self.load_bbox()
        else:
            self.bbox = None
        split_dir = os.path.join(data_dir, split)

        self.filenames, self.captions, self.qas, self.ixtoword, \
            self.wordtoix, self.n_words = self.load_text_data(data_dir, split)

        self.class_id = self.load_class_id(split_dir, len(self.filenames))
        self.number_example = len(self.filenames)
        self.split = split

        if split=='train':
            train = True
            val = False
        elif split=='val':
            train = False
            val = True
        self.VQA_data = VQA_Dataset(
            vqa_utils.path_for(train=train, val=val, question=True),
            vqa_utils.path_for(train=train, val=val, answer=True),
            vqa_config.preprocessed_path,
            answerable_only=False,
        )
        self.mask_answerable, self.q_ids = self.build_mask_answerable(data_dir, split)

    def load_bbox(self):
        data_dir = self.data_dir
        bbox_path = os.path.join(data_dir, 'CUB_200_2011/bounding_boxes.txt')
        df_bounding_boxes = pd.read_csv(bbox_path,
                                        delim_whitespace=True,
                                        header=None).astype(int)
        #
        filepath = os.path.join(data_dir, 'CUB_200_2011/images.txt')
        df_filenames = \
            pd.read_csv(filepath, delim_whitespace=True, header=None)
        filenames = df_filenames[1].tolist()
        print('Total filenames: ', len(filenames), filenames[0])
        #
        filename_bbox = {img_file[:-4]: [] for img_file in filenames}
        numImgs = len(filenames)
        for i in range(0, numImgs):
            # bbox = [x-left, y-top, width, height]
            bbox = df_bounding_boxes.iloc[i][1:].tolist()

            key = filenames[i][:-4]
            filename_bbox[key] = bbox
        #
        return filename_bbox

    def load_captions(self, data_dir, filenames):
        all_captions = dict()
        for i in range(len(filenames)):
            cap_path = '%s/text/%s.txt' % (data_dir, filenames[i])
            img_id = int(filenames[i].split('_')[-1])
            all_captions[img_id] = []
            with open(cap_path, "r", encoding='utf-8') as f:
                captions = f.read().split('\n')
                cnt = 0
                for cap in captions:
                    if len(cap) == 0:
                        continue
                    cap = cap.replace("\ufffd\ufffd", " ")
                    # picks out sequences of alphanumeric characters as tokens
                    # and drops everything else
                    tokenizer = RegexpTokenizer(r'\w+')
                    tokens = tokenizer.tokenize(cap.lower())
                    # print('tokens', tokens)
                    if len(tokens) == 0:
                        print('cap', cap)
                        continue

                    tokens_new = []
                    for t in tokens:
                        t = t.encode('ascii', 'ignore').decode('ascii')
                        if len(t) > 0:
                            tokens_new.append(t)
                    all_captions[img_id].append(tokens_new)
                    cnt += 1
                    if cnt == self.embeddings_num:
                        break
                if cnt < self.embeddings_num:
                    print('ERROR: the captions for %s less than %d'
                          % (filenames[i], cnt))
        return all_captions

    def load_qa(self, data_dir, filenames, split, answerable):
        ann_file = os.path.join(data_dir,'mscoco_%s2014_annotations.json' % split)
        ques_file = os.path.join(data_dir,'OpenEnded_mscoco_%s2014_questions.json' % split)
        if os.path.isfile(ann_file) and os.path.isfile(ques_file):
            vqa = VQA(ann_file, ques_file)

        image_ids = [int(name.split('_')[-1]) for name in filenames]
        all_qa = {id: [] for id in image_ids}

        for id in image_ids:
            question_ids = vqa.getQuesIds(imgIds=[id])
            for q_id in question_ids:
                qa = vqa.loadQ(q_id)[0]['question']
                # qa += ' ' + vqa.loadQA(q_id)[0]['multiple_choice_answer']
                answer = vqa.loadQA(q_id)[0]
                qa += ' ' + answer['multiple_choice_answer']

                if len(qa) == 0:
                    continue
                qa = qa.replace("\ufffd\ufffd", " ")
                # picks out sequences of alphanumeric characters as tokens
                # and drops everything else
                tokenizer = RegexpTokenizer(r'\w+')
                tokens = tokenizer.tokenize(qa.lower())
                # print('tokens', tokens)
                if len(tokens) == 0:
                    print('qa', qa)
                    continue

                tokens_new = []
                for tok in tokens:
                    tok = tok.encode('ascii', 'ignore').decode('ascii')
                    if len(tok) > 0:
                        tokens_new.append(tok)
                all_qa[id].append(tokens_new)
        return all_qa

    def build_dictionary(self, train_captions, test_captions, train_qa, test_qa):
        word_counts = defaultdict(float)
        # captions = train_captions + test_captions + sum(train_qa.values(), []) + sum(test_qa.values(), [])
        captions = sum(train_captions.values(), []) + sum(test_captions.values(), [])
        for sent in captions:
            for word in sent:
                word_counts[word] += 1

        vocab = [w for w in word_counts if word_counts[w] >= 0]

        ixtoword = dict()
        ixtoword[0] = '<end>'
        wordtoix = dict()
        wordtoix['<end>'] = 0
        ix = 1
        for w in vocab:
            wordtoix[w] = ix
            ixtoword[ix] = w
            ix += 1

        train_captions_new = {id: [] for id in train_captions.keys()}
        for k,v in train_captions.items():
            for t in v:
                rev = []
                for w in t:
                    if w in wordtoix:
                        rev.append(wordtoix[w])
                # rev.append(0)  # do not need '<end>' token
                train_captions_new[k].append(rev)

        test_captions_new = {id: [] for id in test_captions.keys()}
        for k,v in test_captions.items():
            for t in v:
                rev = []
                for w in t:
                    if w in wordtoix:
                        rev.append(wordtoix[w])
                # rev.append(0)  # do not need '<end>' token
                test_captions_new[k].append(rev)

        train_qa_new = {id: [] for id in train_qa.keys()}
        for k,v in train_qa.items():
            for t in v:
                rev = []
                for w in t:
                    if w in wordtoix:
                        rev.append(wordtoix[w])
                train_qa_new[k].append(rev)

        test_qa_new = {id: [] for id in test_qa.keys()}
        for k,v in test_qa.items():
            for t in v:
                rev = []
                for w in t:
                    if w in wordtoix:
                        rev.append(wordtoix[w])
                test_qa_new[k].append(rev)

        return [train_captions_new, test_captions_new, train_qa_new, test_qa_new,
                ixtoword, wordtoix, len(ixtoword)]

    def load_text_data(self, data_dir, split):
        filepath = os.path.join(data_dir, 'captions.pickle')
        train_names = self.load_filenames(data_dir, 'train')
        test_names = self.load_filenames(data_dir, 'test')
        if not os.path.isfile(filepath):
            train_captions = self.load_captions(data_dir, train_names)
            test_captions = self.load_captions(data_dir, test_names)
            train_qa = self.load_qa(data_dir, train_names, 'train', True)
            test_qa = self.load_qa(data_dir, test_names, 'val', False)

            train_captions, test_captions, train_qa, test_qa, ixtoword, wordtoix, n_words = \
                self.build_dictionary(train_captions, test_captions, train_qa, test_qa)
            with open(filepath, 'wb') as f:
                pickle.dump([train_captions, test_captions, train_qa, test_qa,
                             ixtoword, wordtoix], f, protocol=2)
                print('number of words', n_words)
                print('Save to: ', filepath)
        else:
            with open(filepath, 'rb') as f:
                x = pickle.load(f)
                train_captions, test_captions = x[0], x[1]
                train_qa, test_qa = x[2], x[3]
                ixtoword, wordtoix = x[4], x[5]
                del x
                n_words = len(ixtoword)
                print('Load from: ', filepath)
        if split == 'train':
            # a list of list: each list contains
            # the indices of words in a sentence
            captions = train_captions
            qas = train_qa
            filenames = train_names
            self.answerable_only = True
        else:  # split=='test'
            captions = test_captions
            qas = test_qa
            filenames = test_names
            self.answerable_only = False
        return filenames, captions, qas, ixtoword, wordtoix, n_words

    def build_mask_answerable(self, data_dir, split):
        ann_file = os.path.join(data_dir,'mscoco_%s2014_annotations.json' % split)
        ques_file = os.path.join(data_dir,'OpenEnded_mscoco_%s2014_questions.json' % split)
        if os.path.isfile(ann_file) and os.path.isfile(ques_file):
            vqa = VQA(ann_file, ques_file)
        image_ids = [int(name.split('_')[-1]) for name in self.filenames]
        mask = {id: [] for id in image_ids}
        q_ids = {id: [] for id in image_ids}
        for id in image_ids:
            question_ids = vqa.getQuesIds(imgIds=[id])
            for q_id in question_ids:
                answer = vqa.loadQA(q_id)[0]
                if self.answerable_only and len(answer['answers'])<=0:
                    mask[id].append(False)
                else:
                    mask[id].append(True)
                q_ids[id].append(q_id)
        return mask, q_ids

    def load_class_id(self, data_dir, total_num):
        if os.path.isfile(data_dir + '/class_info.pickle'):
            with open(data_dir + '/class_info.pickle', 'rb') as f:
                class_id = pickle.load(f)
        else:
            class_id = np.arange(total_num)
        return class_id

    def load_filenames(self, data_dir, split):
        filepath = '%s/%s/filenames.pickle' % (data_dir, split)
        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                filenames = pickle.load(f)
            print('Load filenames from: %s (%d)' % (filepath, len(filenames)))
        else:
            filenames = []
        return filenames

    # def get_caption(self, sent_ix):
    def get_caption(self, img_id, sent_ix):
        # a list of indices for a sentence
        sent_caption = np.asarray(self.captions[img_id][sent_ix]).astype('int64')
        if (sent_caption == 0).sum() > 0:
            print('ERROR: do not need END (0) token', sent_caption)
        num_words = len(sent_caption)
        # pad with 0s (i.e., '<end>')
        x = np.zeros((cfg.TEXT.WORDS_NUM, 1), dtype='int64')
        x_len = num_words
        if num_words <= cfg.TEXT.WORDS_NUM:
            x[:num_words, 0] = sent_caption
        else:
            ix = list(np.arange(num_words))  # 1, 2, 3,..., maxNum
            np.random.shuffle(ix)
            ix = ix[:cfg.TEXT.WORDS_NUM]
            ix = np.sort(ix)
            x[:, 0] = sent_caption[ix]
            x_len = cfg.TEXT.WORDS_NUM
        return x, x_len

    def get_qa(self, img_id, sent_ix, mask):
        # a list of indices for a sentence
        sent_caption = np.asarray(np.asarray(self.qas[img_id], dtype=object)[mask][sent_ix]).astype('int64')
        if (sent_caption == 0).sum() > 0:
            print('ERROR: do not need END (0) token', sent_caption)
        num_words = len(sent_caption)
        # pad with 0s (i.e., '<end>')
        x = np.zeros((cfg.TEXT.WORDS_NUM, 1), dtype='int64')
        x_len = num_words
        if num_words <= cfg.TEXT.WORDS_NUM:
            x[:num_words, 0] = sent_caption
        else:
            ix = list(np.arange(num_words))  # 1, 2, 3,..., maxNum
            np.random.shuffle(ix)
            ix = ix[:cfg.TEXT.WORDS_NUM]
            ix = np.sort(ix)
            x[:, 0] = sent_caption[ix]
            x_len = cfg.TEXT.WORDS_NUM
        return x, x_len

    def __getitem__(self, index):
        #
        key = self.filenames[index]
        cls_id = self.class_id[index]
        #
        if self.bbox is not None:
            bbox = self.bbox[key]
            data_dir = '%s/CUB_200_2011' % self.data_dir
        else:
            bbox = None
            data_dir = self.data_dir
        #
        img_name = '%s/images_%s/%s.jpg' % (data_dir, self.split, key)
        imgs = get_imgs(img_name, self.imsize,
                        bbox, self.transform, normalize=self.norm)
        img_id = int(key.split('_')[-1])

        # random select a sentence
        sent_ix = random.randint(0, self.embeddings_num)
        # new_sent_ix = index * self.embeddings_num + sent_ix
        caps, cap_len = self.get_caption(img_id, sent_ix)

        # random select of question-answer pair
        mask = self.mask_answerable[img_id]
        sent_ix = random.randint(0, sum(mask))
        qa, qa_len = self.get_qa(img_id, sent_ix, mask)
        q_id = np.asarray(self.q_ids[img_id])[mask][sent_ix]

        #  get question and answer for VQA model
        item = self.VQA_data.img_id_to_index_for_qa[img_id][q_id]
        v, q, a, item, q_length = self.VQA_data[item]

        return imgs, caps, cap_len, cls_id, key, qa, qa_len, q, a, item, q_length


    def __len__(self):
        return len(self.filenames)
