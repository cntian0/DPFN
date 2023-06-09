# @Author  : tiancn
# @Time    : 2022/9/18 11:54
import torch
from torch import nn
from  transformers import BertModel
import numpy as np
from pdb import set_trace as stop
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1, dropout2=False, attn_type='softmax'):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        if dropout2:
            # self.dropout2 = nn.Dropout(dropout2)
            self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5), attn_type=attn_type,
                                                       dropout=dropout2)
        else:
            self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5), attn_type=attn_type,
                                                       dropout=dropout)

        self.dropout = nn.Dropout(dropout)

        self.layer_norm = nn.LayerNorm(d_model)

        if n_head > 1:
            self.fc = nn.Linear(n_head * d_v, d_model, bias=False)
            nn.init.xavier_normal_(self.fc.weight)

    def forward(self, q, k, v, attn_mask=None, dec_self=False):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        if hasattr(self, 'dropout2'):
            q = self.dropout2(q)

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv

        if attn_mask is not None:
            attn_mask = attn_mask.repeat(n_head, 1, 1)  # (n*b) x .. x ..

        output, attn = self.attention(q, k, v, attn_mask=attn_mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)  # b x lq x (n*dv)

        if hasattr(self, 'fc'):
            output = self.fc(output)

        if hasattr(self, 'dropout'):
            output = self.dropout(output)

        if dec_self:
            output = self.layer_norm(output + residual)
        else:
            output = self.layer_norm(output + residual)

        return output, attn


class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, dropout=0.1, attn_type='softmax'):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(dropout)
        if attn_type == 'softmax':
            self.attn_type = nn.Softmax(dim=2)
            # self.softmax = BottleSoftmax()
        else:
            self.attn_type = nn.Sigmoid()

    def forward(self, q, k, v, attn_mask=None, stop_sig=False):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if attn_mask is not None:
            # attn = attn.masked_fill(attn_mask, -np.inf)
            attn = attn.masked_fill(attn_mask, -1e6)

        if stop_sig:
            print('**')
            stop()

        attn = self.attn_type(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn

def flatten(x):
    if len(x.size()) == 2:
        batch_size = x.size()[0]
        seq_length = x.size()[1]
        return x.view([batch_size * seq_length])
    elif len(x.size()) == 3:
        batch_size = x.size()[0]
        seq_length = x.size()[1]
        hidden_size = x.size()[2]
        return x.view([batch_size * seq_length, hidden_size])
    else:
        raise Exception()


class SimpleBertModel(nn.Module):
    def __init__(self,opt):
        super(SimpleBertModel, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.caption_text_bert = BertModel.from_pretrained("bert-base-uncased")

        self.max_len = opt.MAX_LEN
        self.layers = 2

        self.layernorm = nn.LayerNorm(self.bert.config.hidden_size)

        self.mulitHead_text_img = MultiHeadAttention(4, self.bert.config.hidden_size, self.bert.config.hidden_size,
                                            self.bert.config.hidden_size)

        self.W1 = nn.ModuleList()
        for layer in range(self.layers):
            input_dim = self.bert.config.hidden_size
            self.W1.append(nn.Linear(input_dim, input_dim))

        self.W2 = nn.ModuleList()
        for layer in range(self.layers):
            input_dim = self.bert.config.hidden_size
            self.W2.append(nn.Linear(input_dim, input_dim))


        self.bert_drop = nn.Dropout(0.1)
        self.gcn_drop = nn.Dropout(0.1)

        self.linear_global = nn.Linear(768 * 2,768)
        self.linear_local = nn.Linear(768 * 2, 768)

        #outMLP
        self.outMLP = nn.Linear(768 * 2,opt.NUM_CLASSES)

    def forward(self,inputs):
        input_ids, attention_mask,vit_feature,transformer_mask,target_input_ids,target_attention_mask,target_mask,text_length,word_length,tran_indices,context_asp_adj_matrix,globel_input_id,globel_mask= inputs

        #text modility
        outputs = self.bert(input_ids = input_ids,attention_mask = attention_mask)
        #pooler_output
        text_feat =  outputs.last_hidden_state
        text_feat = self.layernorm(text_feat)
        text_feat = self.bert_drop(text_feat)

        #fusion
        text_include_img = self.mulitHead_text_img(text_feat, vit_feature, vit_feature)[0]

        #cal cos
        similarity = torch.cosine_similarity(text_include_img.unsqueeze(2), text_include_img.unsqueeze(1), dim=-1)

        global_bert = self.caption_text_bert(input_ids = globel_input_id,attention_mask = globel_mask).pooler_output
        global_bert = self.layernorm(global_bert)
        global_bert = self.bert_drop(global_bert)

        #GCN
        gcn_text_feat = text_feat[:,1:,:]
        gcn_text_img_feat = text_include_img[:, 1:, :]
        tmps = torch.zeros((input_ids.size(0), self.max_len, 768),dtype=torch.float32).to(input_ids.device)
        tmps_text_img = torch.zeros((input_ids.size(0), self.max_len, 768), dtype=torch.float32).to(input_ids.device)
        for i, spans in enumerate(tran_indices):
            true_len = word_length[i]
            nums = 0
            for j, span in enumerate(spans):
                if nums == true_len:
                    break
                nums += 1
                tmps[i, j + 1] = torch.sum(gcn_text_feat[i, span[0]:span[1]], 0)
                tmps_text_img[i, j + 1] = torch.sum(gcn_text_img_feat[i, span[0]:span[1]], 0)

        context_asp_adj_matrix_text_img = torch.mul(context_asp_adj_matrix,similarity)
        denom_dep = context_asp_adj_matrix.sum(2).unsqueeze(2) + 1
        denom_dep_text_img = context_asp_adj_matrix_text_img.sum(2).unsqueeze(2) + 1
        for l in range(self.layers):
            # ************GCN*************
            Ax_dep = context_asp_adj_matrix.bmm(tmps)
            AxW_dep = self.W1[l](Ax_dep)
            AxW_dep = AxW_dep / denom_dep
            gAxW_dep = F.relu(AxW_dep)

            # ************GCN_text_img*************
            Ax_dep_text_img = context_asp_adj_matrix_text_img.bmm(tmps_text_img)
            AxW_dep_text_img = self.W2[l](Ax_dep_text_img)
            AxW_dep_text_img = AxW_dep_text_img / denom_dep_text_img
            gAxW_dep_text_img = F.relu(AxW_dep_text_img)

        #outputs_dep = self.gcn_drop(gAxW_dep)
        outputs_dep = gAxW_dep
        outputs_dep_text_img = gAxW_dep_text_img

        asp_wn = target_mask.sum(dim=1).unsqueeze(-1)
        aspect_mask = target_mask.unsqueeze(-1).repeat(1, 1, 768)
        gcn_out_text = (outputs_dep * aspect_mask).sum(dim=1) / asp_wn
        gcn_out_text_img = (outputs_dep_text_img * aspect_mask).sum(dim=1) / asp_wn

        #target
        local_feat = self.linear_local(torch.cat([gcn_out_text, gcn_out_text_img], dim=1))
        local_globa_feat = torch.cat([global_bert,local_feat],dim=1)

        return self.outMLP(local_globa_feat)





