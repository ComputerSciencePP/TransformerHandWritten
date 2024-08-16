import torch
import math
from torch.autograd import Variable


class Embeddings(torch.nn.Module):
    def __init__(self, vocab_size, d_model):
        # vocab_size: 词表的大小
        # d_model: 词嵌入的维度
        super().__init__()
        self.lut = torch.nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    
    def forward(self, x):
        # x: 输入到模型的文本通过词表映射后的张量
        return self.lut(x) * math.sqrt(self.d_model)



if __name__ == '__main__':
    embedding = torch.nn.Embedding(10, 3) # 将输入张量中的每个元素都变成3维向量
    input1 = torch.LongTensor([[1, 0, 3], [4, 5, 6]])
    print(embedding(input1))

    embedding = torch.nn.Embedding(10, 3, padding_idx=0) # 输入张量中0元素扩展成3维元素全为0的向量
    print(embedding(input1))

    embedding = Embeddings(1000, 512)
    r = embedding(input1)
    print(r)
    print(r.shape)

