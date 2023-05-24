import sys
import torch
import torch.utils.data as data
import random
import pickle
import os
from nltk import word_tokenize
from vocab import VocabEntry
import numpy as np
import re
from tqdm import tqdm
from scipy import sparse
import math
import json
dmap = {
    'Math':{0: 2, 1: 3, 2: 10, 3: 14, 4: 15, 5: 17, 6: 23, 7: 24, 8: 25, 9: 26, 10: 28, 11: 35, 12: 42, 13: 45, 14: 46, 15: 48, 16: 50, 17: 53, 18: 54, 19: 55, 20: 60, 21: 63, 22: 64, 23: 70, 24: 75, 25: 79, 26: 87, 27: 88, 28: 89, 29: 92, 30: 94, 31: 96, 32: 97, 33: 101, 34: 103, 35: 104, 36: 105, 37: 106},
    'Lang': {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11, 11: 12, 12: 13, 13: 14, 14: 15, 15: 16, 16: 17, 17: 18, 18: 19, 19: 20, 20: 21, 21: 22, 22: 24, 23: 26, 24: 27, 25: 28, 26: 29, 27: 30, 28: 31, 29: 32, 30: 33, 31: 34, 32: 35, 33: 36, 34: 37, 35: 38, 36: 39, 37: 40, 38: 41, 39: 42, 40: 43, 41: 44, 42: 45, 43: 46, 44: 47, 45: 48, 46: 49, 47: 50, 48: 51, 49: 52, 50: 53, 51: 54, 52: 55, 53: 57, 54: 58, 55: 59, 56: 60, 57: 61, 58: 62, 59: 63, 60: 64, 61: 65},
    'Chart':{0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11, 11: 12, 12: 13, 13: 14, 14: 15, 15: 16, 16: 17, 17: 18, 18: 19, 19: 20, 20: 21, 21: 22, 22: 24, 23: 25, 24: 26},
    'Time':{0: 1, 1: 8, 2: 9},
    'Mockito':{0: 1, 1: 3, 2: 6},
    'Cli':{0: 8, 1: 9, 2: 13, 3: 16}
}
class SumDataset(data.Dataset):
    def __init__(self, config, dataName="train", proj="Math", testid=0, lst=[]):
        self.train_path = proj + ".pkl"
        self.val_path = "ndev.txt"
        self.test_path = "ntest.txt"
        self.proj = proj
        self.SentenceLen = config.SentenceLen
        self.Nl_Voc = {"pad": 0, "Unknown": 1}
        self.Code_Voc = {"pad": 0, "Unknown": 1}
        self.Char_Voc = {"pad": 0, "Unknown": 1}
        self.Nl_Voc['Method'] = len(self.Nl_Voc)
        self.Nl_Voc['Test'] = len(self.Nl_Voc)
        self.Nl_Voc['Line'] = len(self.Nl_Voc)
        self.Nl_Voc['RTest'] = len(self.Nl_Voc)
        self.Nl_Len = config.NlLen
        self.Code_Len = config.CodeLen
        self.Char_Len = config.WoLen
        self.batch_size = config.batch_size
        self.PAD_token = 0
        self.data = None
        self.dataName = dataName
        self.Codes = []
        self.ids = []
        self.Nls = []
        if os.path.exists("nl_voc.pkl"):
            self.Load_Voc()
        else:
            self.init_dic()
        print(self.Nl_Voc)
        if not os.path.exists(self.proj + 'data.pkl'):
            data = self.preProcessData(open(self.train_path, "rb"))
        else:
            data = pickle.load(open(self.proj + 'data.pkl', 'rb'))
        self.data = []
        if dataName == "train":
            for i in range(len(data)):
                tmp = []
                for j in range(len(data[i])):
                    if j in lst:
                        continue
                    tmp.append(data[i][j])
                self.data.append(tmp)
        elif dataName == 'test':
            testnum = 0
            ids = []
            while len(ids) < testnum:
                rid = random.randint(0, len(data[0]) - 1)
                if rid == testid or rid in ids or rid == 51:
                    continue
                ids.append(rid)
            self.ids = ids
            for i in range(len(data)):
                tmp = []
                for x in self.ids:
                    tmp.append(data[i][x])
                self.data.append(tmp)
        else:
            testnum = 1
            ids = []
            for i in range(len(data)): 
                tmp = []
                for x in range(testnum * testid, testnum * testid + testnum):
                    if x < len(data[i]):
                        if i == 0:
                            ids.append(x)
                        tmp.append(data[i][x])
                self.data.append(tmp)
            self.ids = ids

    def Load_Voc(self):
        if os.path.exists("nl_voc.pkl"):
            self.Nl_Voc = pickle.load(open("nl_voc.pkl", "rb"))
        if os.path.exists("code_voc.pkl"):
            self.Code_Voc = pickle.load(open("code_voc.pkl", "rb"))
        if os.path.exists("char_voc.pkl"):
            self.Char_Voc = pickle.load(open("char_voc.pkl", "rb"))
    def splitCamel(self, token):
        ans = []
        tmp = ""
        for i, x in enumerate(token):
            if i != 0 and x.isupper() and token[i - 1].islower() or x in '$.' or token[i - 1] in '.$':
                ans.append(tmp)
                tmp = x.lower()
            else:
                tmp += x.lower()
        ans.append(tmp)
        return ans
    def init_dic(self):
        print("initVoc")
        f = open(self.p + '.pkl', 'rb')
        data = pickle.load(f)
        maxNlLen = 0
        maxCodeLen = 0
        maxCharLen = 0
        Nls = []
        Codes = []
        for x in data:
            for s in x['methods']:
                s = s[:s.index('(')]
                if len(s.split(":")) > 1:
                    tokens = ".".join(s.split(":")[0].split('.')[-2:] + [s.split(":")[1]])
                else:
                    tokens = ".".join(s.split(":")[0].split('.')[-2:])
                Codes.append(self.splitCamel(tokens))
                print(Codes[-1])
            for s in x['ftest']:
                if len(s.split(":")) > 1:
                    tokens = ".".join(s.split(":")[0].split('.')[-2:] + [s.split(":")[1]])
                else:
                    tokens = ".".join(s.split(":")[0].split('.')[-2:])
                Codes.append(self.splitCamel(tokens))
        code_voc = VocabEntry.from_corpus(Codes, size=50000, freq_cutoff = 0)
        self.Code_Voc = code_voc.word2id
        open("code_voc.pkl", "wb").write(pickle.dumps(self.Code_Voc))
    def Get_Em(self, WordList, voc):
        ans = []
        for x in WordList:
            if x not in voc:
                ans.append(1)
            else:
                ans.append(voc[x])
        return ans
    def pad_seq(self, seq, maxlen):
        act_len = len(seq)
        if len(seq) < maxlen:
            seq = seq + [self.PAD_token] * maxlen
            seq = seq[:maxlen]
        else:
            seq = seq[:maxlen]
            act_len = maxlen
        return seq
    def pad_list(self,seq, maxlen1, maxlen2):
        if len(seq) < maxlen1:
            seq = seq + [[self.PAD_token] * maxlen2] * maxlen1
            seq = seq[:maxlen1]
        else:
            seq = seq[:maxlen1]
        return seq
    def getoverlap(self, a, b):
        ans = []
        for x in a:
            maxl = 0
            for y in b:
                tmp = 0
                for xm in x:
                    if xm in y:
                        tmp += 1
                maxl = max(maxl, tmp)
            ans.append(int(100 * maxl / len(x)) + 1)
        return ans
    def preProcessData(self, dataFile):
        path_stacktrace = os.path.join('../FLocalization/stacktrace', self.proj)    
        lines = pickle.load(dataFile)
        Nodes = []
        Types = []
        LineNodes = []
        LineTypes = []
        LineMus = []
        Res = []
        inputText = []
        inputNlad = []
        maxl = 0
        maxl2 = 0
        error = 0
        error1 = 0
        error2 = 0
        correct = 0
        for k in range(len(lines)):
            x = lines[k]
            if os.path.exists(path_stacktrace + '/%d.json'%dmap[self.proj][k]):
                stack_info = json.load(open(path_stacktrace + '/%d.json'%dmap[self.proj][k]))
                if x['ftest'].keys() != stack_info.keys():
                    with open("problem_stack",'a') as f:
                        f.write("{} {} no!\n".format(k, dmap[self.proj][k]))
                        f.write(str(x['ftest'].keys()) + '\n')
                        f.write(str(stack_info.keys()) + '\n')
                    for error_trace in x['ftest'].keys():
                        if error_trace not in stack_info.keys():
                            error += 1
                        else:
                            correct += 1
                    
            nodes = []
            types = []
            res = []
            nladrow = []
            nladcol = []
            nladval = []
            texta = []
            textb = []
            linenodes = []
            linetypes = []
            methodnum = len(x['methods'])
            rrdict = {}
            for s in x['methods']:
                rrdict[x['methods'][s]] = s[:s.index('(')]
            for i in range(methodnum):
                nodes.append('Method')
                if len(rrdict[i].split(":")) > 1:
                    tokens = ".".join(rrdict[i].split(":")[0].split('.')[-2:] + [rrdict[i].split(":")[1]]) 
                else:
                    tokens = ".".join(rrdict[i].split(":")[0].split('.')[-2:]) 
                ans = self.splitCamel(tokens)
                ans.remove('.')
                texta.append(ans)
                if i not in x['correctnum']:
                    types.append(1)
                else:
                    types.append(x['correctnum'][i] + 1)
                if i in x['ans']:
                    res.append(1)
                else:
                    res.append(0)
            rrdic = {}
            for s in x['ftest']:
                rrdic[x['ftest'][s]] = s
            try:
                for i in range(len(x['ftest'])):
                    nodes.append('Test')
                    types.append(0)
                    if len(rrdic[i].split(":")) > 1:
                        tokens = ".".join(rrdic[i].split(":")[0].split('.')[-2:] + [rrdic[i].split(":")[1]])
                    else:
                        tokens = ".".join(rrdic[i].split(":")[0].split('.')[-2:])
                    ans = self.splitCamel(tokens)
                    ans.remove('.')
                    textb.append(ans)
            except:
                print(ans)
            rrdic = {}
            for i in range(len(x['rtest'])):
                nodes.append('RTest')
                types.append(0)

            mus = []
            for i in range(len(x['lines'])):
                if i not in x['ltype']:
                    x['ltype'][i] = 'Empty'
                if x['ltype'][i] not in self.Nl_Voc:
                    self.Nl_Voc[x['ltype'][i]] = len(self.Nl_Voc)
                linenodes.append(x['ltype'][i])
                if i in x['lcorrectnum']:
                    linetypes.append(x['lcorrectnum'][i])
                else:
                    linetypes.append(1)
            maxl = max(maxl, len(nodes))
            maxl2 = max(maxl2, len(linenodes))
            ed = {}

            line2method = {}
            for e in x['edge2']:
                line2method[e[1]] = e[0]
                a = e[0]
                b = e[1] + self.Nl_Len
                if (a, b) not in ed:
                    ed[(a, b)] = 1
                else:
                    print(a, b)
                    assert(0)
                if (b, a) not in ed:
                    ed[(b, a)] = 1
                else:
                    print(a, b)
                    assert(0)
                nladrow.append(a)
                nladcol.append(b)
                nladval.append(1)
                nladrow.append(b)
                nladcol.append(a)
                nladval.append(1)
            for e in x['edge10']:
                if e[0] not in line2method:
                    error1 += 1
                a = e[0] + self.Nl_Len
                b = e[1] + methodnum + len(x['ftest'])
                nladrow.append(a)
                nladcol.append(b)
                if (a, b) not in ed:
                    ed[(a, b)] = 1
                else:
                    pass
                if (b, a) not in ed:
                    ed[(b, a)] = 1
                else:
                    pass
                nladval.append(1)
                nladrow.append(b)
                nladcol.append(a)
                nladval.append(1)
            for e in x['edge']:
                if e[0] not in line2method:
                    error2 += 1
                a = e[0] + self.Nl_Len
                b = e[1] + methodnum
                nladrow.append(a)
                nladcol.append(b)
                if (a, b) not in ed:
                    ed[(a, b)] = 1
                else:
                    print(e[0])
                    print(a, b)
                    assert(0)
                if (b, a) not in ed:
                    ed[(b, a)] = 1
                else:
                    print(a, b)
                    assert(0)
                nladval.append(1)
                nladrow.append(b)
                nladcol.append(a)
                nladval.append(1)
            overlap = self.getoverlap(texta, textb)

            Nodes.append(self.pad_seq(self.Get_Em(nodes, self.Nl_Voc), self.Nl_Len))
            Types.append(self.pad_seq(types, self.Nl_Len))
            Res.append(self.pad_seq(res, self.Nl_Len))
            LineMus.append(self.pad_list(mus, self.Code_Len, 3))
            inputText.append(self.pad_seq(overlap, self.Nl_Len))
            LineNodes.append(self.pad_seq(self.Get_Em(linenodes, self.Nl_Voc), self.Code_Len))
            LineTypes.append(self.pad_seq(linetypes, self.Code_Len))
            row = {}
            col = {}
            for i  in range(len(nladrow)):
                if nladrow[i] not in row:
                    row[nladrow[i]] = 0
                row[nladrow[i]] += 1
                if nladcol[i] not in col:
                    col[nladcol[i]] = 0
                col[nladcol[i]] += 1
            for i in range(len(nladrow)):
                nladval[i] = 1 / math.sqrt(row[nladrow[i]]) * 1 / math.sqrt(col[nladcol[i]])
            nlad = sparse.coo_matrix((nladval, (nladrow, nladcol)), shape=(self.Nl_Len + self.Code_Len, self.Nl_Len + self.Code_Len))
            inputNlad.append(nlad)
        print("max1: %d max2: %d"%(maxl, maxl2))
        print("correct: %d error: %d"%(correct, error))
        print("error1: %d error2: %d"%(error1, error2))

        batchs = [Nodes, Types, inputNlad, Res, inputText, LineNodes, LineTypes, LineMus]
        self.data = batchs
        open(self.proj + "data.pkl", "wb").write(pickle.dumps(batchs, protocol=4))
        return batchs

    def __getitem__(self, offset):
        ans = []
        if True:
            for i in range(len(self.data)):
                if i == 2:
                    ans.append(self.data[i][offset].toarray())
                else:
                    ans.append(np.array(self.data[i][offset]))
        return ans
    def __len__(self):
        return len(self.data[0])
    def Get_Train(self, batch_size):
        data = self.data
        loaddata = data
        batch_nums = int(len(data[0]) / batch_size)
        if True:
            if self.dataName == 'train':
                shuffle = np.random.permutation(range(len(loaddata[0])))
            else:
                shuffle = np.arange(len(loaddata[0]))
            for i in range(batch_nums):
                ans = []
                for j in range(len(data)):
                    if j != 2:
                        tmpd = np.array(data[j])[shuffle[batch_size * i: batch_size * (i + 1)]]
                        ans.append(torch.from_numpy(np.array(tmpd)))
                    else:
                        ids = []
                        v = []
                        for idx in range(batch_size * i, batch_size * (i + 1)):
                            for p in range(len(data[j][shuffle[idx]].row)):
                                ids.append([idx - batch_size * i, data[j][shuffle[idx]].row[p], data[j][shuffle[idx]].col[p]])
                                v.append(data[j][shuffle[idx]].data[p])
                        ans.append(torch.sparse.FloatTensor(torch.LongTensor(ids).t(), torch.FloatTensor(v), torch.Size([batch_size, self.Nl_Len + self.Code_Len, self.Nl_Len + self.Code_Len])))
                yield ans
            if batch_nums * batch_size < len(data[0]):
                ans = []
                for j in range(len(data)):
                    if j != 2:
                        tmpd = np.array(data[j])[shuffle[batch_nums * batch_size: ]]
                        ans.append(torch.from_numpy(np.array(tmpd)))
                    else:
                        ids = []
                        v = []
                        for idx in range(batch_size * batch_nums, len(data[0])):
                            for p in range(len(data[j][shuffle[idx]].row)):
                                ids.append([idx - batch_size * batch_nums, data[j][shuffle[idx]].row[p], data[j][shuffle[idx]].col[p]])
                                v.append(data[j][shuffle[idx]].data[p])
                        ans.append(torch.sparse.FloatTensor(torch.LongTensor(ids).t(), torch.FloatTensor(v), torch.Size([len(data[0]) - batch_size * batch_nums, self.Nl_Len + self.Code_Len, self.Nl_Len + self.Code_Len])))
                yield ans
            
class node:
    def __init__(self, name):
        self.name = name
        self.father = None
        self.child = []
        self.id = -1