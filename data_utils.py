# @Author  : tiancn
# @Time    : 2022/9/18 11:39
import pandas as pd
from transformers import BertTokenizer
from torch.utils.data import Dataset,DataLoader
import json
import torch
import pickle
import transformers
from tqdm import tqdm
import numpy as np
from transformers import ViTFeatureExtractor
transformers.logging.set_verbosity_error()


feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")



class MyTokenizer:
    def __init__(self,max_len):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_len = max_len

    def text_to_sequence(self, text,sentiment_targets,tran=False):
        text_left_right = [s.strip() for s in text.split("$T$")]
        target_mask = len(text_left_right[0].split()) * [0] \
                      + len(sentiment_targets.split()) * [1] +\
                      len(text_left_right[1].split()) * [0]
        target_mask = torch.tensor([0] + target_mask + (self.max_len - 1-len(target_mask)) * [0])
        words = (text_left_right[0] + ' ' + sentiment_targets + ' ' + text_left_right[1]).split()
        words_len = len(words)
        trans = []
        realwords = []

        for word in words:
            wordpieces = self.tokenizer.tokenize(word)
            tmplen = len(realwords)

            realwords.extend(wordpieces)
            trans.append([tmplen, len(realwords)])
        #        unknownidx = 1_convert_token_to_id
        #print(realwords)
        #text_pair_words = self.tokenizer.tokenize(text_pair)

        #扩充trans
        start_num = len(realwords)
        start_len = len(trans)
        for i in range(start_len,self.max_len):
            trans.append([start_num,start_num+1])
            start_num = start_num + 1

        #for
        sequence = [self.tokenizer._convert_token_to_id('[CLS]')] \
                   + [self.tokenizer._convert_token_to_id(w) for w in realwords] \
                   + [self.tokenizer._convert_token_to_id('[SEP]')]
        # if len(sequence) == 0:
        #     sequence = [0]
        if len(sequence) <= self.max_len:
            sequence_re = torch.tensor(sequence + (self.max_len - len(sequence)) * [0])
            attention_mask_self = torch.tensor(len(sequence) * [1] + (self.max_len - len(sequence)) * [0])
        else:
            sequence_re = torch.tensor(sequence)[0:self.max_len]
            attention_mask_self = torch.tensor(self.max_len * [1])

        text_length = len(sequence)
        if text_length > 60:
            print('.............................................................................')
        if tran: return sequence_re,attention_mask_self,text_length, target_mask,words_len,trans
        return sequence_re,attention_mask_self,text_length,target_mask,words_len


class Twitter(Dataset):
    def __init__(self,df,max_len,image_captions,vit_features,ori_adjs):
        self.datas = []
        #tweet content
        self.tweets = df.tweet_content.to_numpy()
        #label
        self.labels = df.sentiment.to_numpy()
        #aspect
        self.sentiment_targets = df.target.to_numpy()
        #img_name
        self.image_ids = df.image_id.to_numpy()
        #caption
        self.image_captions = image_captions
        #bert tokenizer
        self.max_len = max_len
        #image feature
        self.vit_features = vit_features
        #adj
        self.ori_adjs = ori_adjs

        self.tokenizer_self = MyTokenizer(max_len)

        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        print('dataset:'+str(len(self.tweets)))

        #vit input
        img_dir = '/media/xdu/disk/tiancn/publicdata/twitter2015_images'

        for i in tqdm(range(len(self.tweets))):
            tweet = self.tweets[i]
            label = self.labels[i]
            sentiment_target = self.sentiment_targets[i]
            image_id = self.image_ids[i]
            # img_path = os.path.join(img_dir, image_id)
            # img = cv.imread(img_path)
            # transf = transforms.ToTensor()
            # img_tensor = transf(img)
            # vit_inputs = feature_extractor(img_tensor, 'pt')['pixel_values']


            #print(image_id)
            try:
                caption = self.image_captions[image_id]
            except KeyError:  # A couple of the images have no content.
                caption = ""

            vit_feature = self.vit_features[image_id]


            encoding = self.tokenizer.encode_plus(
                tweet.replace("$T$",sentiment_target),
                #=sentiment_target + "." + caption,
                text_pair=caption,
                add_special_tokens=True,
                max_length=80,
                return_token_type_ids=False,
                padding="max_length",
                return_attention_mask=True,
                return_tensors="pt",
                truncation=True,
            )
            target_encoding = self.tokenizer.encode_plus(
                sentiment_target,
                #text_pair=sentiment_target + "." + caption,
                #text_pair=sentiment_target,
                add_special_tokens=True,
                max_length=5,
                return_token_type_ids=False,
                padding="max_length",
                return_attention_mask=True,
                return_tensors="pt",
                truncation=True,
            )
            #tokenizer
            # my_input_ids,my_attention_mask,text_lenght,trans = \
            #     self.tokenizer_self.text_to_sequence(tweet,sentiment_target + "." + caption,True)
            my_input_ids, my_attention_mask, text_length,target_mask, word_len,trans = \
                self.tokenizer_self.text_to_sequence(tweet,sentiment_target ,True)


            input_ids = encoding["input_ids"].flatten()
            attention_mask = encoding["attention_mask"].flatten()
            transformer_mask = (attention_mask == 0).numpy().tolist() + (attention_mask == 0).numpy().tolist()
            #position = [0.0] * 80 + [1.0] * 197

            #target_bert
            target_input_ids = target_encoding["input_ids"].flatten()
            target_attention_mask = target_encoding["attention_mask"].flatten()
            #target_mult_mask = (attention_mask == 0).numpy().tolist()

            # pad adj
            ori_adj = self.ori_adjs[i]
            context_asp_adj_matrix = np.zeros(
                (self.max_len, self.max_len)).astype('float32')
            context_asp_adj_matrix[1:word_len + 1, 1:word_len + 1] = ori_adj

            data = {
                "review_text": tweet,
                "sentiment_targets": sentiment_target,
                "caption": caption,
                "input_ids": my_input_ids,
                "attention_mask": my_attention_mask,
                "targets": torch.tensor(label, dtype=torch.long),
                "vit_feature":vit_feature,
                'transformer_mask':torch.tensor(transformer_mask),
               # 'position':torch.tensor(position,dtype=torch.float32),
                'target_input_ids':target_input_ids,
                'target_attention_mask':target_attention_mask,
                'target_mask':target_mask,
                "text_length":torch.tensor(text_length),
                "word_length":torch.tensor(word_len),
                "tran_indices":torch.tensor(trans),
                "context_asp_adj_matrix":torch.tensor(context_asp_adj_matrix),
                "globel_input_id":input_ids,
                "globel_mask":attention_mask
            }
            self.datas.append(data)
    def __len__(self):
        return len(self.tweets)

    def __getitem__(self, index):
        return self.datas[index]

def creatr_data_loader(fname,type,max_len,batch_size):
    # read data use pd
    df = pd.read_csv(fname, sep="\t")
    # re name
    if type == 'train' or type == 'dev':
        df = df.rename(
            {
                "#1 Label": "sentiment",
                "#2 ImageID": "image_id",
                "#3 String": "tweet_content",
                "#3 String.1": "target",
            },
            axis=1,
        ).drop(["index"], axis=1)
    if type == 'test':
        df = df.rename(
            {
                "index": "sentiment",
                "#1 ImageID": "image_id",
                "#2 String": "tweet_content",
                "#2 String.1": "target",
            },
            axis=1,
        )
    # caption filepath
    captions_json = ""
    # Load the image captions.
    with open(captions_json, "r") as f:
        image_captions = json.load(f)

    # image feature filepath
    img_data_path = ''
    with open(img_data_path, "rb") as f:
        vit_feature = pickle.load(f)

    #adj maritx file path
    adj_file = ''
    with open(adj_file,"rb") as f:
        ori_adj = pickle.load(f)

    ds = Twitter(df,max_len,image_captions,vit_feature,ori_adj)
    return DataLoader(ds,batch_size = batch_size,num_workers=0)

if __name__ == '__main__':
    train_data_loader=creatr_data_loader('/media/xdu/disk/tiancn/publicdata/twitter2015/train.tsv','train',60,1)
    one_batch = next(iter(train_data_loader))
    print(one_batch)