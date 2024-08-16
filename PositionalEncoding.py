import torch

class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, dropout, max_len = 5000):
        """
        d_model: 词嵌入维度
        dropout: 置0比例
        max_len: 每个句子的长度
        """
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)
        # 初始化位置编码矩阵，每个句子的单子被扩展成d_model维度嵌入
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1) # torch.Size([max_len]) -> torch.Size([max_len, 1]), 实现了转置
        div_term = torch.exp(torch.arange(0, d_model, 2) * - (math.log(1000.0) / d_model)) # - (math.log(1000.0) / d_model)是一个负的放缩因子
        pe[:, 0::2] = torch.sin(position * div_term) # position是max_lenX1， div_term是1Xd_modle / 2
        pe[:, 1::2] = torch.cos(position * div_term)




if __name__ == '__main__':
    positionalEncoding = PositionalEncoding()