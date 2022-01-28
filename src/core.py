from dataclasses import dataclass
from collections import deque, Iterable
from functools import partial
from typing import List
from multiprocessing import Pool, cpu_count
import difflib
import math
from random import sample

import numpy as np
import pandas as pd
from pandas import DataFrame
from tqdm import tqdm
import tensorflow as tf

INPUT_SIZE = 256
NUM_PREDICT = 128

fail = 0
success = 0
CUSTOMER_TOKENS = [12967, 30]
SALES_TOKENS = [4925, 30]


@dataclass
class Columns:
    is_staff = "speaker"
    c_id = "customer_id"
    s_id = "sales_id"
    msg_time = "timestamp"
    text = "text"
    mass_mask = "mass_mask"

    @property
    def data_type(self):
        return {
            self.is_staff: np.bool8,
            self.s_id: "string",
            self.c_id: "string",
            self.msg_time: np.uint32,
            self.text: "string",
            self.mass_mask: np.bool8,
        }


def grouped_map_parallel(df_grouped, func):
    with Pool(cpu_count()) as p:
        ret_list = p.map(func, [group for name, group in tqdm(df_grouped)])
    return ret_list


def string_similar(s1, s2):
    return difflib.SequenceMatcher(None, s1, s2).quick_ratio()


@dataclass
class Conversation:
    name: str
    id: str
    num_c_msg: int
    num_msg: int

    @property
    def num_s_msg(self):
        return self.num_msg - self.num_c_msg

    @property
    def percent_c_msg(self):
        return self.num_c_msg / self.num_msg


def get_doc(texts, nlp_):
    doc_list = []
    for text in tqdm(texts, mininterval=5):
        doc_list.append(nlp_(text))
    return doc_list


def _int64_feature(values):
    if isinstance(values[0], Iterable):
        values = [i for value in values for i in value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

def _float_mat_feature(values):
    values = [i for value in values for i in value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=values))


def get_feature(input_ids, target_indices, alternative_indices, max_length=INPUT_SIZE):
    global fail, success
    n = len(input_ids)
    target_indices = [x for x in target_indices if x < max_length]
    alternative_indices = [x for x in alternative_indices if x < max_length]
    m = len(target_indices)
    o = len(alternative_indices)
    if m + o < NUM_PREDICT:
        fail += 1
        return
    if m < NUM_PREDICT:
        target_indices = target_indices + sample(alternative_indices, NUM_PREDICT - m)
    else:
        target_indices = sample(target_indices, NUM_PREDICT)
    target_indices.sort()
    # perm_mask = np.zeros((NUM_PREDICT, max_length))
    target_mapping = np.zeros((NUM_PREDICT, max_length))
    labels = []

    for i, ind in enumerate(target_indices):
        # perm_mask[i][ind:] = 1
        target_mapping[i][ind] = 1
        labels.append(input_ids[ind])

    gap = max_length - n
    token_type_ids = np.zeros(max_length, dtype=int)
    if n < max_length:
        input_ids.extend([5] * gap)
        token_type_ids[n:] = 3

    feature = {
        "input_ids": _int64_feature(input_ids[:max_length]),
        "labels": _int64_feature(labels),
        "token_type_ids": _int64_feature(token_type_ids),
        # "perm_mask": _float_mat_feature(perm_mask),
        "target_mapping": _float_mat_feature(target_mapping),
    }
    success += 1
    return feature


def get_inputs(df: DataFrame, tokenizer, cols=Columns()):
    df = df.sort_values(by=[cols.msg_time])
    print(df)
    iter_rows = zip(df[cols.is_staff], df[cols.text])
    features = []
    split_index = 0
    input_ids = []
    target_indices = []
    alternative_indices = []

    def clear(is_staff=None, remain_indices: Iterable = None, force=False):
        nonlocal input_ids, target_indices, alternative_indices, split_index
        target_indices = []
        alternative_indices = []
        if force:
            split_index = 0
            input_ids = []
            return
        input_ids = input_ids[split_index:]
        if remain_indices and (is_staff is not None):
            remain_indices = [x - split_index for x in remain_indices]
            if is_staff:
                target_indices = remain_indices
            else:
                alternative_indices = remain_indices

    for is_staff, text in tqdm(iter_rows, total=len(df), mininterval=10):
        prefix = SALES_TOKENS if is_staff else CUSTOMER_TOKENS
        temp = tokenizer.encode(text, add_special_tokens=False)
        n = len(temp)
        temp = prefix + temp
        input_ids.extend(temp)
        m = len(input_ids)
        content_indices = range(m - n, min(INPUT_SIZE, m))
        if is_staff:
            target_indices.extend(content_indices)
        else:
            alternative_indices.extend(content_indices)
        if m > INPUT_SIZE:
            remain_indices = range(INPUT_SIZE, m)
            feature = get_feature(input_ids.copy(), target_indices, alternative_indices)
            if feature:
                features.append(feature)
            clear(is_staff=is_staff, remain_indices=remain_indices)
        # clear之后 input_ids一然很长 说明这是一个很长很长的句子
        # 所以再预测一次之后 强制clear
        split_index = len(input_ids)
        if split_index >= INPUT_SIZE:
            feature = get_feature(input_ids.copy(), target_indices, alternative_indices)
            if feature:
                features.append(feature)
            clear(force=True)
    return features


def mass_mask(df: DataFrame, cols=Columns()):
    df = df.loc[df[cols.is_staff] == True]
    df = df.sort_values(by=[cols.msg_time])
    que = deque(maxlen=5)
    mass_i = set()
    iter_rows = zip(df.index, df[cols.text])
    for i, text in tqdm(iter_rows, total=len(df)):
        if len(text) > 20:
            for q_i, q_text in que:
                if string_similar(q_text, text) > 0.95:
                    mass_i.add(i)
                    mass_i.add(q_i)
            que.append((i, text))
    return mass_i


class ConversationManager:
    def __init__(self, df, cols=None) -> None:
        self.df: DataFrame = df
        self.cols = cols if cols is not None else Columns()

    @classmethod
    def from_csv(cls, path, cols=None, **kwages):
        if cols is None:
            cols = Columns()
        df = pd.read_csv(path, dtype=cols.data_type, **kwages)
        # df=df.head(500000)
        # df.to_csv('sample.csv', index=False)
        df[cols.text].fillna("<<unk>>", inplace=True)
        return cls(df, cols)

    def mark_mass_msg(self):
        import time

        s = time.time()
        cols = self.cols
        s_group_df = self.df.groupby(cols.s_id)

        reduce_mass_i = []
        res = grouped_map_parallel(s_group_df, mass_mask)
        for mass_i in res:
            reduce_mass_i.extend(mass_i)
        self.df[cols.mass_mask] = False
        self.df.loc[reduce_mass_i, cols.mass_mask] = True
        e = time.time()
        print(e - s)

    def create_docs(self):
        import spacy
        from spacy.tokens import DocBin

        cols = self.cols

        if cols.mass_mask in self.df:
            df = self.df.loc[self.df[cols.mass_mask] == False]
        else:
            df = self.df
        data = df["text"].values
        nlp = spacy.load("zh_core_web_sm")

        n = cpu_count()
        with Pool(n) as p:
            step_len = int(math.ceil(len(data) / float(n)))
            res = p.map(
                partial(get_doc, nlp_=nlp),
                [data[i : i + step_len] for i in range(0, len(data), step_len)],
            )
        doc_list = [item for sublist in res for item in sublist]

        DocBin(docs=doc_list).to_disk("res.spacy")

    def get_pretraing_data(self):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-xlnet-base")
        cols = self.cols
        df = self.df.loc[self.df[cols.mass_mask] == False]
        df = df.loc[~df[cols.text].str.startswith("<<")]
        grouped_df = df.groupby(cols.c_id)
        res = grouped_map_parallel(grouped_df, partial(get_inputs, tokenizer=tokenizer))
        record_writer = tf.compat.v1.python_io.TFRecordWriter("/datafile/kaixuan/nlg/s0125.tfrecords")
        for features in res:
            for feature in features:
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                record_writer.write(example.SerializeToString())
        record_writer.close()


if __name__ == "__main__":
    # cm = ConversationManager.from_csv("/home/yangkaixuan/datafile/airflow/nlg_preprocess/2021-11-30/sample.csv")
    # cm = ConversationManager.from_csv(
    #     "/home/yangkaixuan/download/all_message1130.tsv", sep="\t"
    # )
    cm = ConversationManager.from_csv("/datafile/kaixuan/nlg/all_data.csv")
    # cm.mark_mass_msg()
    cm.get_pretraing_data()
    # cm.df.to_csv("result.csv", index=False)
