import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F

from .rnn import MultiLayerLSTMCells
from .rnn import lstm_encoder
from .util import sequence_mean, len_mask
from .attention import prob_normalize

INI = 1e-2

class ConvSentEncoder(nn.Module):
    """
    Convolutional word-level sentence encoder
    w/ max-over-time pooling, [3, 4, 5] kernel sizes, ReLU activation
    """
    def __init__(self, vocab_size, emb_dim, n_hidden, dropout):
        super().__init__()
        self._embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self._convs = nn.ModuleList([nn.Conv1d(emb_dim, n_hidden, i)
                                     for i in range(3, 6)])
        self._dropout = dropout
        self._grad_handle = None

    def forward(self, input_):
        emb_input = self._embedding(input_)
        conv_in = F.dropout(emb_input.transpose(1, 2),
                            self._dropout, training=self.training)
        output = torch.cat([F.relu(conv(conv_in)).max(dim=2)[0]
                            for conv in self._convs], dim=1)
        return output

    def set_embedding(self, embedding):
        """embedding is the weight matrix"""
        assert self._embedding.weight.size() == embedding.size()
        self._embedding.weight.data.copy_(embedding)


class LSTMEncoder(nn.Module):
    def __init__(self, input_dim, n_hidden, n_layer, dropout, bidirectional):
        super().__init__()
        self._init_h = nn.Parameter(
            torch.Tensor(n_layer*(2 if bidirectional else 1), n_hidden))
        self._init_c = nn.Parameter(
            torch.Tensor(n_layer*(2 if bidirectional else 1), n_hidden))
        init.uniform_(self._init_h, -INI, INI)
        init.uniform_(self._init_c, -INI, INI)
        self._lstm = nn.LSTM(input_dim, n_hidden, n_layer,
                             dropout=dropout, bidirectional=bidirectional)

    def forward(self, input_, in_lens=None):
        """ [batch_size, max_num_sent, input_dim] Tensor"""
        size = (self._init_h.size(0), input_.size(0), self._init_h.size(1))
        init_states = (self._init_h.unsqueeze(1).expand(*size),
                       self._init_c.unsqueeze(1).expand(*size))
        lstm_out, _ = lstm_encoder(
            input_, self._lstm, in_lens, init_states)
        return lstm_out.transpose(0, 1)

    @property
    def input_size(self):
        return self._lstm.input_size

    @property
    def hidden_size(self):
        return self._lstm.hidden_size

    @property
    def num_layers(self):
        return self._lstm.num_layers

    @property
    def bidirectional(self):
        return self._lstm.bidirectional


class LSTMPointerNet(nn.Module):
    """Pointer network as in Vinyals et al """
    def __init__(self, input_dim, n_hidden, n_layer,
                 dropout, n_hop):
        super().__init__()
        self._init_h = nn.Parameter(torch.Tensor(n_layer, n_hidden))
        self._init_c = nn.Parameter(torch.Tensor(n_layer, n_hidden))
        self._init_i = nn.Parameter(torch.Tensor(input_dim))
        init.uniform_(self._init_h, -INI, INI)
        init.uniform_(self._init_c, -INI, INI)
        init.uniform_(self._init_i, -0.1, 0.1)
        self._lstm = nn.LSTM(
            input_dim, n_hidden, n_layer,
            bidirectional=False, dropout=dropout
        )
        self._lstm_cell = None

        # attention parameters
        self._attn_wm = nn.Parameter(torch.Tensor(input_dim, n_hidden))
        self._attn_wq = nn.Parameter(torch.Tensor(n_hidden, n_hidden))
        self._attn_v = nn.Parameter(torch.Tensor(n_hidden))
        init.xavier_normal_(self._attn_wm)
        init.xavier_normal_(self._attn_wq)
        init.uniform_(self._attn_v, -INI, INI)

        # hop parameters
        self._hop_wm = nn.Parameter(torch.Tensor(input_dim, n_hidden))
        self._hop_wq = nn.Parameter(torch.Tensor(n_hidden, n_hidden))
        self._hop_v = nn.Parameter(torch.Tensor(n_hidden))
        init.xavier_normal_(self._hop_wm)
        init.xavier_normal_(self._hop_wq)
        init.uniform_(self._hop_v, -INI, INI)
        self._n_hop = n_hop

    def forward(self, attn_mem, mem_sizes, lstm_in):
        """
        atten_mem: Tensor of size [batch_size, max_sent_num, sent_emb_dim] = batch of vector representations of the sentences of each report
        mem_sizes = number of sentences in each report
        lstm_in: Tensor of size [batch_size, max_golden_sent_num, sent_emb_dim] = batch of ground truth extracted sentences
        """
        attn_feat, hop_feat, lstm_states, init_i = self._prepare(attn_mem)
        lstm_in = torch.cat([init_i, lstm_in], dim=1).transpose(0, 1) # the input is the concatenation of the initialized input and of the ground truth sentences
        # it should be noted that this happens only in training, in fact during training we apply "teacher forcing", which consists in forcing as input the actual ground truth at each time step of the LSTM instead of the previously predicted output
        query, final_states = self._lstm(lstm_in, lstm_states)  # query: Tensor of size [number of sentences extracted, batch size, hidden size]
        query = query.transpose(0, 1)
        for _ in range(self._n_hop):
            query = LSTMPointerNet.attention(
                hop_feat, query, self._hop_v, self._hop_wq, mem_sizes)
        output = LSTMPointerNet.attention_score(
            attn_feat, query, self._attn_v, self._attn_wq)
        return output  # unormalized extraction logit

    def extract(self, attn_mem, mem_sizes, k):
        """extract k sentences, decode only, batch_size==1"""
        attn_feat, hop_feat, lstm_states, lstm_in = self._prepare(attn_mem)
        lstm_in = lstm_in.squeeze(1)
        if self._lstm_cell is None:
            self._lstm_cell = MultiLayerLSTMCells.convert(
                self._lstm).to(attn_mem.device)
        extracts = []
        for _ in range(k):
            h, c = self._lstm_cell(lstm_in, lstm_states)
            query = h[-1] # take only the last layer
            for _ in range(self._n_hop):
                query = LSTMPointerNet.attention(
                    hop_feat, query, self._hop_v, self._hop_wq, mem_sizes)
            score = LSTMPointerNet.attention_score(
                attn_feat, query, self._attn_v, self._attn_wq)
            score = score.squeeze()
            for e in extracts:
                score[e] = -1e6 # put the probability of extracting the same sentence to 0, this cant be done during the pointer net training because this is not differentiable
            ext = score.max(dim=0)[1].item() # extract the sentence with the maximum score (i.e. the maximum "attention")
            extracts.append(ext)
            lstm_states = (h, c) # different from the training, in this case there is no "teacher forcing", so as input to the next step using the current states
            lstm_in = attn_mem[:, ext, :] # also in this case use previous output as input
        return extracts

    def _prepare(self, attn_mem):
        # atten_mem: Tensor of size [batch_size, max_sent_num, sent_emb_dim] = batch of vector representations of the sentences of each report
        
        attn_feat = torch.matmul(attn_mem, self._attn_wm.unsqueeze(0))
        hop_feat = torch.matmul(attn_mem, self._hop_wm.unsqueeze(0))
        bs = attn_mem.size(0) # batch size
        n_l, d = self._init_h.size() # number of layer and hidden size
        size = (n_l, bs, d)
        lstm_states = (self._init_h.unsqueeze(1).expand(*size).contiguous(),
                       self._init_c.unsqueeze(1).expand(*size).contiguous())
        d = self._init_i.size(0) # embedding dimension
        init_i = self._init_i.unsqueeze(0).unsqueeze(1).expand(bs, 1, d)
        return attn_feat, hop_feat, lstm_states, init_i

    @staticmethod
    def attention_score(attention, query, v, w):
        """ unnormalized attention score"""
        sum_ = attention.unsqueeze(1) + torch.matmul(
            query, w.unsqueeze(0)
        ).unsqueeze(2)  # [B, Nq, Ns, D]
        score = torch.matmul(
            F.tanh(sum_), v.unsqueeze(0).unsqueeze(1).unsqueeze(3)
        ).squeeze(3)  # [B, Nq, Ns]
        return score

    @staticmethod
    def attention(attention, query, v, w, mem_sizes):
        """ attention context vector"""
        score = LSTMPointerNet.attention_score(attention, query, v, w)
        if mem_sizes is None:
            norm_score = F.softmax(score, dim=-1)
        else:
            mask = len_mask(mem_sizes, score.device).unsqueeze(-2)
            norm_score = prob_normalize(score, mask)
        output = torch.matmul(norm_score, attention)
        return output


class PtrExtractSumm(nn.Module):
    """ rnn-ext"""
    def __init__(self, emb_dim, vocab_size, conv_hidden,
                 lstm_hidden, lstm_layer, bidirectional,
                 n_hop=1, dropout=0.0):
        super().__init__()
        self._sent_enc = ConvSentEncoder(
            vocab_size, emb_dim, conv_hidden, dropout)
        self._art_enc = LSTMEncoder(
            3*conv_hidden, lstm_hidden, lstm_layer,
            dropout=dropout, bidirectional=bidirectional
        )
        enc_out_dim = lstm_hidden * (2 if bidirectional else 1)
        self._extractor = LSTMPointerNet(
            enc_out_dim, lstm_hidden, lstm_layer,
            dropout, n_hop
        )

    def forward(self, article_sents, sent_nums, target):
        '''
        article sents = batch of reports -> [batch_size, number of sentences, number of words, w2v embedding dimension]
        sent_nums = number of sentence for each report in the batch
        target = ground truth indices of the sentences to extract
        '''
        enc_out = self._encode(article_sents, sent_nums)
        # at this point in enc_out the sentences have gone through the conv and an LSTM
        # in enc_out we have the vector representation of each sentence starting from the word embeddings contained in each sentence
        bs, nt = target.size()
        d = enc_out.size(2)
        ptr_in = torch.gather(
            enc_out, dim=1, index=target.unsqueeze(2).expand(bs, nt, d)
        ) # select the sentences corresponding to the indices of the ground truth extracted sentences
        output = self._extractor(enc_out, sent_nums, ptr_in) # _extractor is the PointerNet
        return output

    def extract(self, article_sents, sent_nums=None, k=4):
        enc_out = self._encode(article_sents, sent_nums)
        output = self._extractor.extract(enc_out, sent_nums, k)
        return output

    def _encode(self, article_sents, sent_nums):
        if sent_nums is None:  # test-time excode only
            enc_sent = self._sent_enc(article_sents[0]).unsqueeze(0)
        else:
            max_n = max(sent_nums)
            enc_sents = [self._sent_enc(art_sent) # create a vector representation for each sentence starting from its word embeddings
                         for art_sent in article_sents]
            def zero(n, device):
                z = torch.zeros(n, self._art_enc.input_size).to(device)
                return z
            enc_sent = torch.stack(
                [torch.cat([s, zero(max_n-n, s.device)], dim=0)
                   if n != max_n
                 else s
                 for s, n in zip(enc_sents, sent_nums)],
                dim=0
            )
            # in enc_sent we pad with zero-vectors (vectors made of all zeros) so that all the reports in the batch have the same size
        lstm_out = self._art_enc(enc_sent, sent_nums) # to further encode semantic dependencies between sentences we use a BiLSTM
        return lstm_out

    def set_embedding(self, embedding):
        self._sent_enc.set_embedding(embedding)
