import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F

from .attention import step_attention
from .util import len_mask
from .summ import Seq2SeqSumm, AttentionalLSTMDecoder
from . import beam_search as bs


INIT = 1e-2


class _CopyLinear(nn.Module):
    def __init__(self, context_dim, state_dim, input_dim, bias=True):
        super().__init__()
        self._v_c = nn.Parameter(torch.Tensor(context_dim))
        self._v_s = nn.Parameter(torch.Tensor(state_dim))
        self._v_i = nn.Parameter(torch.Tensor(input_dim))
        init.uniform_(self._v_c, -INIT, INIT)
        init.uniform_(self._v_s, -INIT, INIT)
        init.uniform_(self._v_i, -INIT, INIT)
        if bias:
            self._b = nn.Parameter(torch.zeros(1))
        else:
            self.register_parameter(None, '_b')

    def forward(self, context, state, input_):
        # context = context vector
        # state = current last layer of the hidden state of the decoder
        # input_ = concatenation of previous prediction and previous attentional hidden state
        output = (torch.matmul(context, self._v_c.unsqueeze(1))
                  + torch.matmul(state, self._v_s.unsqueeze(1))
                  + torch.matmul(input_, self._v_i.unsqueeze(1)))
        if self._b is not None:
            output = output + self._b.unsqueeze(0)
        return output


class CopySumm(Seq2SeqSumm):
    def __init__(self, vocab_size, emb_dim,
                 n_hidden, bidirectional, n_layer, dropout=0.0):
        super().__init__(vocab_size, emb_dim,
                         n_hidden, bidirectional, n_layer, dropout)
        self._copy = _CopyLinear(n_hidden, n_hidden, 2*emb_dim)
        self._decoder = CopyLSTMDecoder(
            self._copy, self._embedding, self._dec_lstm,
            self._attn_wq, self._projection
        )

    def forward(self, article, art_lens, abstract, extend_art, extend_vsize):
        '''
        article: [batch_size, num_words, emb_size]
        art_lens = number of words for each sentence
        '''
        attention, init_dec_states = self.encode(article, art_lens)
        # attention: [batch size, num words, hidden size] alignment vector, source hidden states
        # init_dec_states: 
            # 1) tuple of [num_layers, batch_size, hidden size] with initial decoder states (h,c)
            # 2) [batch size, embedding size] initial attentional hidden state
        mask = len_mask(art_lens, attention.device).unsqueeze(-2)
        logit = self._decoder(
            (attention, mask, extend_art, extend_vsize),
            init_dec_states, abstract
        )
        return logit

    def batch_decode(self, article, art_lens, extend_art, extend_vsize,
                     go, eos, unk, max_len):
        """ greedy decode support batching"""
        batch_size = len(art_lens)
        vsize = self._embedding.num_embeddings
        attention, init_dec_states = self.encode(article, art_lens)
        mask = len_mask(art_lens, attention.device).unsqueeze(-2)
        attention = (attention, mask, extend_art, extend_vsize)
        tok = torch.LongTensor([go]*batch_size).to(article.device)
        outputs = []
        attns = []
        states = init_dec_states
        for i in range(max_len):
            tok, states, attn_score = self._decoder.decode_step(
                tok, states, attention)
            attns.append(attn_score)
            outputs.append(tok[:, 0].clone())
            tok.masked_fill_(tok >= vsize, unk)
        return outputs, attns

    def decode(self, article, extend_art, extend_vsize, go, eos, unk, max_len):
        vsize = self._embedding.num_embeddings
        attention, init_dec_states = self.encode(article)
        attention = (attention, None, extend_art, extend_vsize)
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
            if tok[0, 0].item() >= vsize:
                tok[0, 0] = unk
        return outputs, attns



class CopyLSTMDecoder(AttentionalLSTMDecoder):
    def __init__(self, copy, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._copy = copy

    def _step(self, tok, states, attention):
        # tok = previous ground truth
        # states = (tuple of [num_layers, batch_size, hidden size]),[batch size, embedding size]
        # attention = (attention, mask, extend_art, extend_vsize)

        prev_states, prev_out = states 
        # prev_state: tuple of [num_layers, batch_size, hidden size]
        # prev_out: [batch size, embedding size]
        lstm_in = torch.cat(
            [self._embedding(tok).squeeze(1), prev_out],
            dim=1
        )
        # lstm_in is the concatenation of the previous token (during training is the previous ground truth, during val it is the previous predicted token) and of the previous attentional hidden state
        states = self._lstm(lstm_in, prev_states)
        # _lstm has parameters (2*emb_dim, n_hidden, n_layer)
        # states = tuple:
            # new_h: [num layers, batch size, hidden size]
            # new_c: [num layers, batch size, hidden size]

        lstm_out = states[0][-1] # last layer of the new_h
        query = torch.mm(lstm_out, self._attn_w)
        # query: [batch size, hidden size]
        attention, attn_mask, extend_src, extend_vsize = attention
        context, score = step_attention(
            query, attention, attention, attn_mask)
        # calculate the context vector as the weighted sum of the encoder hidden states, according to attention distribution calculated with the decoder output as query
        dec_out = self._projection(torch.cat([lstm_out, context], dim=1))
        # dec_out: [batch size, embedding size] attentional hidden state, used for prediction of the next word

        # extend the generation probability to include also words of the extended vocabulary
        gen_prob = self._compute_gen_prob(dec_out, extend_vsize)

        # compute the probabilty of copying
        copy_prob = torch.sigmoid(self._copy(context, states[0][-1], lstm_in))
        
        # we add the two distributions, probability of copying or generating
        lp = torch.log(
            ((-copy_prob + 1) * gen_prob
            ).scatter_add(
                dim=1,
                index=extend_src.expand_as(score),
                src=score * copy_prob
        ) + 1e-8)  # numerical stability for log
        return lp, (states, dec_out), score


    def topk_step(self, tok, states, attention, k):
        """tok:[BB, B], states ([L, BB, B, D]*2, [BB, B, D])"""
        (h, c), prev_out = states

        # lstm is not bemable
        nl, _, _, d = h.size()
        beam, batch = tok.size()
        lstm_in_beamable = torch.cat(
            [self._embedding(tok), prev_out], dim=-1)
        lstm_in = lstm_in_beamable.contiguous().view(beam*batch, -1)
        prev_states = (h.contiguous().view(nl, -1, d),
                       c.contiguous().view(nl, -1, d))
        h, c = self._lstm(lstm_in, prev_states)
        states = (h.contiguous().view(nl, beam, batch, -1),
                  c.contiguous().view(nl, beam, batch, -1))
        lstm_out = states[0][-1]

        # attention is beamable
        query = torch.matmul(lstm_out, self._attn_w)
        attention, attn_mask, extend_src, extend_vsize = attention
        context, score = step_attention(
            query, attention, attention, attn_mask)
        dec_out = self._projection(torch.cat([lstm_out, context], dim=-1))

        # copy mechanism is not beamable
        gen_prob = self._compute_gen_prob(
            dec_out.contiguous().view(batch*beam, -1), extend_vsize)
        copy_prob = torch.sigmoid(
            self._copy(context, lstm_out, lstm_in_beamable)
        ).contiguous().view(-1, 1)
        lp = torch.log(
            ((-copy_prob + 1) * gen_prob
            ).scatter_add(
                dim=1,
                index=extend_src.expand_as(score).contiguous().view(
                    beam*batch, -1),
                source=score.contiguous().view(beam*batch, -1) * copy_prob
        ) + 1e-8).contiguous().view(beam, batch, -1)

        k_lp, k_tok = lp.topk(k=k, dim=-1)
        return k_tok, k_lp, (states, dec_out), score

    def _compute_gen_prob(self, dec_out, extend_vsize, eps=1e-6):
        # dec_out: [batch size, embedding size] attentional hidden state
        # extend_vsize = size of the extended vocabulary, w2v vocab + OOV
        
        logit = torch.mm(dec_out, self._embedding.weight.t())
        # this logit will be used as probability distribution on the w2v vocabulary words

        bsize, vsize = logit.size()
        if extend_vsize > vsize:
            ext_logit = torch.Tensor(bsize, extend_vsize-vsize
                                    ).to(logit.device)
            ext_logit.fill_(eps) # OOV words probability distribution filled with arbitrarily small number
            gen_logit = torch.cat([logit, ext_logit], dim=1)
        else:
            gen_logit = logit
        gen_prob = F.softmax(gen_logit, dim=-1)
        # gen_prob is the actual probability distribution over the whole vocab (w2v + OOV)
        return gen_prob

    def _compute_copy_activation(self, context, state, input_, score):
        copy = self._copy(context, state, input_) * score
        return copy
