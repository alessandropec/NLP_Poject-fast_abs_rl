import torch
from torch import nn
from torch.nn import init

from .rnn import lstm_encoder
from .rnn import MultiLayerLSTMCells
from .attention import step_attention
from .util import sequence_mean, len_mask


INIT = 1e-2


class Seq2SeqSumm(nn.Module):
    def __init__(self, vocab_size, emb_dim,
                 n_hidden, bidirectional, n_layer, dropout=0.0):
        super().__init__()
        # embedding weight parameter is shared between encoder, decoder,
        # and used as final projection layer to vocab logit
        # can initialize with pretrained word vectors
        self._embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self._enc_lstm = nn.LSTM(
            emb_dim, n_hidden, n_layer,
            bidirectional=bidirectional, dropout=dropout
        )
        # initial encoder LSTM states are learned parameters
        state_layer = n_layer * (2 if bidirectional else 1)
        self._init_enc_h = nn.Parameter(
            torch.Tensor(state_layer, n_hidden)
        )
        self._init_enc_c = nn.Parameter(
            torch.Tensor(state_layer, n_hidden)
        )
        init.uniform_(self._init_enc_h, -INIT, INIT)
        init.uniform_(self._init_enc_c, -INIT, INIT)

        # vanillat lstm / LNlstm
        self._dec_lstm = MultiLayerLSTMCells(
            2*emb_dim, n_hidden, n_layer, dropout=dropout
        )
        # project encoder final states to decoder initial states
        enc_out_dim = n_hidden * (2 if bidirectional else 1)
        self._dec_h = nn.Linear(enc_out_dim, n_hidden, bias=False)
        self._dec_c = nn.Linear(enc_out_dim, n_hidden, bias=False)
        # multiplicative attention
        self._attn_wm = nn.Parameter(torch.Tensor(enc_out_dim, n_hidden))
        self._attn_wq = nn.Parameter(torch.Tensor(n_hidden, n_hidden))
        init.xavier_normal_(self._attn_wm)
        init.xavier_normal_(self._attn_wq)
        # project decoder output to emb_dim, then
        # apply weight matrix from embedding layer
        self._projection = nn.Sequential(
            nn.Linear(2*n_hidden, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, emb_dim, bias=False)
        )
        # functional object for easier usage
        self._decoder = AttentionalLSTMDecoder(
            self._embedding, self._dec_lstm,
            self._attn_wq, self._projection
        )

    def forward(self, article, art_lens, abstract):
        attention, init_dec_states = self.encode(article, art_lens)
        mask = len_mask(art_lens, attention.device).unsqueeze(-2)
        logit = self._decoder((attention, mask), init_dec_states, abstract)
        return logit

    def encode(self, article, art_lens=None):
        '''
        article: [batch_size, num_words, emb_size]
        art_lens = number of words for each sentence
        '''
        size = (
            self._init_enc_h.size(0), # n_layer (*2 if bidir)
            len(art_lens) if art_lens else 1, # batch_size
            self._init_enc_h.size(1) # hidden size
        )
        init_enc_states = (
            self._init_enc_h.unsqueeze(1).expand(*size),
            self._init_enc_c.unsqueeze(1).expand(*size)
        )
        enc_art, final_states = lstm_encoder(
            article, self._enc_lstm, art_lens,
            init_enc_states, self._embedding
        )
        # enc_art: [num words, batch size, hidden size (*2 if bidir)] contains hidden vector at each time step of the last layer
        # final_states: tuple of [num_layers(*2 if bidir), batch size, hidden size]
        if self._enc_lstm.bidirectional:
            h, c = final_states
            final_states = (
                torch.cat(h.chunk(2, dim=0), dim=2),
                torch.cat(c.chunk(2, dim=0), dim=2)
            )
            # separate the two directions with chunk and then concatenate them
            # final_states: tuple of [num_layers, batch size, hidden size (*2 if bidir)]
        init_h = torch.stack([self._dec_h(s)
                              for s in final_states[0]], dim=0)
        init_c = torch.stack([self._dec_c(s)
                              for s in final_states[1]], dim=0)
        # _dec_h and _dec_c are NNs to remap the final encoder states to the decoder
        # init_h and init_c : [num_layers, batch_size, hidden size] 
        init_dec_states = (init_h, init_c)
        attention = torch.matmul(enc_art, self._attn_wm).transpose(0, 1)
        # attention: [batch_size, num words, hidden size] alignment vector a_t, source hidden states (after encoder so aware of context)
        init_attn_out = self._projection(torch.cat(
            [init_h[-1], sequence_mean(attention, art_lens, dim=1)], dim=1
        ))
        # sequence_mean(attention, art_lens, dim=1) is the initial context vector, for now not weighted avg but normal avg
        # concatenate last layer of h ([batch size, hidden size]) with sequence mean of attention (where we sum all the words in a sentence and divide by the length of the sentence)
        # init_attn_out: [batch size, embedding size]
        # init_attn_out represents the initial attentional hidden state
        return attention, (init_dec_states, init_attn_out)

    def batch_decode(self, article, art_lens, go, eos, max_len):
        """ greedy decode support batching"""
        batch_size = len(art_lens)
        attention, init_dec_states = self.encode(article, art_lens)
        mask = len_mask(art_lens, attention.device).unsqueeze(-2)
        attention = (attention, mask)
        tok = torch.LongTensor([go]*batch_size).to(article.device)
        outputs = []
        attns = []
        states = init_dec_states
        for i in range(max_len):
            tok, states, attn_score = self._decoder.decode_step(
                tok, states, attention)
            outputs.append(tok[:, 0])
            attns.append(attn_score)
        return outputs, attns

    def decode(self, article, go, eos, max_len):
        attention, init_dec_states = self.encode(article)
        attention = (attention, None)
        tok = torch.LongTensor([go]).to(article.device)
        outputs = []
        attns = []
        states = init_dec_states
        for i in range(max_len):
            tok, states, attn_score = self._decoder.decode_step(
                tok, states, attention)
            if tok[0, 0].item() == eos:
                break
            outputs.append(tok[0, 0].item())
            attns.append(attn_score.squeeze(0))
        return outputs, attns

    def set_embedding(self, embedding):
        """embedding is the weight matrix"""
        assert self._embedding.weight.size() == embedding.size()
        self._embedding.weight.data.copy_(embedding)


class AttentionalLSTMDecoder(object):
    def __init__(self, embedding, lstm, attn_w, projection):
        super().__init__()
        self._embedding = embedding
        self._lstm = lstm
        self._attn_w = attn_w
        self._projection = projection

    def __call__(self, attention, init_states, target):
        # attention = (attention, mask, extend_art, extend_vsize)
        # init_states = init_dec_states (tuple of [num_layers, batch_size, hidden size],[batch size, embedding size])
        # target = abstract
        max_len = target.size(1) # max number of words in the target
        states = init_states
        logits = []
        for i in range(max_len):
            tok = target[:, i:i+1] # teacher forcing, as input provide the ground truth at each step
            logit, states, _ = self._step(tok, states, attention)
            # logit = probability distribution over two vocabs
            # states = tuple of:
                # decoder states at current time step (h,c)
                # current attentional hidden state, that will be concatenated to the current ground truth token to predict next token
            logits.append(logit)
        logit = torch.stack(logits, dim=1)
        return logit

    def _step(self, tok, states, attention):
        prev_states, prev_out = states
        lstm_in = torch.cat(
            [self._embedding(tok).squeeze(1), prev_out],
            dim=1
        )
        states = self._lstm(lstm_in, prev_states)
        lstm_out = states[0][-1]
        query = torch.mm(lstm_out, self._attn_w)
        attention, attn_mask = attention
        context, score = step_attention(
            query, attention, attention, attn_mask)
        dec_out = self._projection(torch.cat([lstm_out, context], dim=1))
        states = (states, dec_out)
        logit = torch.mm(dec_out, self._embedding.weight.t())
        return logit, states, score

    def decode_step(self, tok, states, attention):
        logit, states, score = self._step(tok, states, attention)
        out = torch.max(logit, dim=1, keepdim=True)[1]
        return out, states, score
