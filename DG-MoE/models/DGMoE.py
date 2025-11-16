from sympy import true
import torch
from torch import nn
from .basic_layers import Transformer, GradientReversalLayer
from .bert import BertTextEncoder
from einops import rearrange, repeat
from models.transformers_encoder.transformer import TransformerEncoder
import torch.nn.functional as F
from scipy.spatial.distance import squareform
import scipy.linalg
import numpy as np
from torch_geometric.nn import GATConv

gpu_id =2
USE_CUDA = torch.cuda.is_available()
device = torch.device(f"cuda:{gpu_id}" if USE_CUDA else "cpu")

def edge_perms(l, window_past, window_future):

    all_perms = set()
    array = np.arange(l)
    for j in range(l):
        perms = set()
        
        if window_past == -1 and window_future == -1:
            eff_array = array
        elif window_past == -1:
            eff_array = array[:min(l, j+window_future+1)]
        elif window_future == -1:
            eff_array = array[max(0, j-window_past):]
        else:
            eff_array = array[max(0, j-window_past):min(l, j+window_future+1)]
        
        for item in eff_array:
            perms.add((j, item))
        all_perms = all_perms.union(perms)
    return list(all_perms)

def batch_graphify(features, qmask, lengths, window_past, window_future, no_cuda):
    """
    Method to prepare the data format required for the GCN network. Pytorch geometric puts all nodes for classification 
    in one single graph. Following this, we create a single graph for a mini-batch of dialogue instances. This method 
    ensures that the various graph indexing is properly carried out so as to make sure that, utterances (nodes) from 
    each dialogue instance will have edges with utterances in that same dialogue instance, but not with utternaces 
    from any other dialogue instances in that mini-batch.
    """
    edge_index, edge_type, node_features = [], [], []
    edge_index_modal = []
    batch_size = features.size(0)
    length_sum = 0
    edge_index_lengths = []   

    for j in range(batch_size):
        node_features.append(features[j,:lengths[j].item(), :])
        perms1 = edge_perms(lengths[j].item(), window_past, window_future)
        perms2 = [(item[0]+length_sum, item[1]+length_sum) for item in perms1]
        length_sum += lengths[j].item()
        edge_index_lengths.append(len(perms1))
        for item1, item2 in zip(perms1, perms2):
            edge_index.append(torch.tensor([item2[0], item2[1]]))

    node_features = torch.cat(node_features, dim=0)
    edge_index = torch.stack(edge_index).transpose(0, 1)

    edge_index_ =  torch.stack([edge_index[0] + node_features.shape[0], edge_index[1] + node_features.shape[0]],dim=0)
    for i in range(node_features.shape[0]):
        edge_index_modal.append(torch.tensor([i, i+node_features.shape[0]]))
        edge_index_modal.append(torch.tensor([i+node_features.shape[0], i]))
    edge_index_modal_ = torch.stack(edge_index_modal).transpose(0, 1)

    edge_index1 = torch.cat([edge_index,edge_index_,edge_index_modal_], dim=-1)

    if not no_cuda:
        node_features = node_features.cuda()
        edge_index = edge_index.cuda()
        edge_index1 = edge_index1.cuda()

    return node_features, edge_index, edge_index_lengths, edge_index1

def simple_batch_graphify(features, lengths, no_cuda):
    node_features = []
    batch_size = features.size(0)

    for j in range(batch_size):
        node_features.append(features[j,:lengths[j].item(), :])

    node_features = torch.cat(node_features, dim=0)

    if not no_cuda:
        node_features = node_features.cuda()

    return node_features

class MyMultimodal(nn.Module):
    def __init__(self, args):

        super(MyMultimodal, self).__init__()

        self.h_pl = nn.Parameter(torch.ones(1, args['model']['feature_extractor']['token_length'][0], 128))
        self.h_pa = nn.Parameter(torch.ones(1, args['model']['feature_extractor']['token_length'][0], 128))
        self.h_pv = nn.Parameter(torch.ones(1, args['model']['feature_extractor']['token_length'][0], 128))

        self.bertmodel = BertTextEncoder(use_finetune=True, transformers='bert', pretrained=args['model']['feature_extractor']['bert_pretrained'])

        self.proj_l = nn.Sequential(
            nn.Linear(args['model']['feature_extractor']['input_dims'][0], args['model']['feature_extractor']['hidden_dims'][0]),
            Transformer(num_frames=args['model']['feature_extractor']['input_length'][0], 
                        save_hidden=False, 
                        token_len=args['model']['feature_extractor']['token_length'][0], 
                        dim=args['model']['feature_extractor']['hidden_dims'][0], 
                        depth=args['model']['feature_extractor']['depth'], 
                        heads=args['model']['feature_extractor']['heads'], 
                        mlp_dim=args['model']['feature_extractor']['hidden_dims'][0])
        )

        self.proj_a = nn.Sequential(
            nn.Linear(args['model']['feature_extractor']['input_dims'][2], args['model']['feature_extractor']['hidden_dims'][2]),
            Transformer(num_frames=args['model']['feature_extractor']['input_length'][2], 
                        save_hidden=False, 
                        token_len=args['model']['feature_extractor']['token_length'][2], 
                        dim=args['model']['feature_extractor']['hidden_dims'][2], 
                        depth=args['model']['feature_extractor']['depth'], 
                        heads=args['model']['feature_extractor']['heads'], 
                        mlp_dim=args['model']['feature_extractor']['hidden_dims'][2])
        )
        
        self.proj_v = nn.Sequential(
            nn.Linear(args['model']['feature_extractor']['input_dims'][1], args['model']['feature_extractor']['hidden_dims'][1]),
            Transformer(num_frames=args['model']['feature_extractor']['input_length'][1], 
                        save_hidden=False, 
                        token_len=args['model']['feature_extractor']['token_length'][1], 
                        dim=args['model']['feature_extractor']['hidden_dims'][1], 
                        depth=args['model']['feature_extractor']['depth'], 
                        heads=args['model']['feature_extractor']['heads'], 
                        mlp_dim=args['model']['feature_extractor']['hidden_dims'][1])
        )
        
        self.proxy_dominate_modality_generator = Transformer(
            num_frames=args['model']['dmc']['proxy_dominant_feature_generator']['input_length'], 
            save_hidden=False, 
            token_len=args['model']['dmc']['proxy_dominant_feature_generator']['token_length'], 
            dim=args['model']['dmc']['proxy_dominant_feature_generator']['input_dim'], 
            depth=args['model']['dmc']['proxy_dominant_feature_generator']['depth'], 
            heads=args['model']['dmc']['proxy_dominant_feature_generator']['heads'], 
            mlp_dim=args['model']['dmc']['proxy_dominant_feature_generator']['hidden_dim'])
        
        self.GRL = GradientReversalLayer(alpha=1.0)

        self.effective_discriminator = nn.Sequential(
            nn.Linear(args['model']['dmc']['effectiveness_discriminator']['input_dim'], 
                      args['model']['dmc']['effectiveness_discriminator']['hidden_dim']),
            nn.LeakyReLU(0.1),
            nn.Linear(args['model']['dmc']['effectiveness_discriminator']['hidden_dim'], 
                      args['model']['dmc']['effectiveness_discriminator']['out_dim']),
        )

        self.completeness_check = nn.ModuleList([
            Transformer(num_frames=args['model']['dmc']['completeness_check']['input_length'], 
                        save_hidden=False, 
                        token_len=args['model']['dmc']['completeness_check']['token_length'], 
                        dim=args['model']['dmc']['completeness_check']['input_dim'], 
                        depth=args['model']['dmc']['completeness_check']['depth'], 
                        heads=args['model']['dmc']['completeness_check']['heads'], 
                        mlp_dim=args['model']['dmc']['completeness_check']['hidden_dim']),

            nn.Sequential(
                nn.Linear(args['model']['dmc']['completeness_check']['hidden_dim'], int(args['model']['dmc']['completeness_check']['hidden_dim']/2)),
                nn.LeakyReLU(0.1),
                nn.Linear(int(args['model']['dmc']['completeness_check']['hidden_dim']/2), 1),
                nn.Sigmoid()),
        ])

        self.text_lstm = nn.LSTM(128, 128, batch_first=True)
        self.audio_lstm = nn.LSTM(128, 128, batch_first=True)
        self.video_lstm = nn.LSTM(128, 128, batch_first=True)

        self.dmml = nn.ModuleList([
            nn.Linear(args['model']['dmml']['regression']['input_dim'], args['model']['dmml']['regression']['out_dim'])
        ])

        self.encoder_s_l = self.get_network(self_type='l', layers = 2)       
        self.encoder_s_v = self.get_network(self_type='v', layers = 2)
        self.encoder_s_a = self.get_network(self_type='a', layers = 2)

        self.encoder_c = self.get_network(self_type='l', layers = 2)   

        self.decoder_l = nn.Conv1d(16, 8, kernel_size=1, padding=0, bias=False)     
        self.decoder_v = nn.Conv1d(16, 8, kernel_size=1, padding=0, bias=False)
        self.decoder_a = nn.Conv1d(16, 8, kernel_size=1, padding=0, bias=False)

        self.trans_l_with_a = self.get_network(self_type='la', layers = 2)  
        self.trans_l_with_v = self.get_network(self_type='lv', layers = 2) 
        self.trans_a_with_l = self.get_network(self_type='al')
        self.trans_a_with_v = self.get_network(self_type='av')
        self.trans_v_with_l = self.get_network(self_type='vl')
        self.trans_v_with_a = self.get_network(self_type='va')
        self.trans_l_mem = self.get_network(self_type='l_mem', layers=2)
        self.trans_a_mem = self.get_network(self_type='a_mem', layers=3)
        self.trans_v_mem = self.get_network(self_type='v_mem', layers=3)

        self.linear_dis_share = nn.Sequential(nn.ReLU(), nn.Linear(128, 128))

        self.linear_dis_spec = nn.ModuleList([])
        for i in range(3):
            self.linear_dis_spec.append(nn.Sequential(nn.ReLU(), nn.Linear(128, 128)))
        
        self.linear_dis_sup = nn.ModuleList([])
        for i in range(3):
            self.linear_dis_sup.append(nn.Sequential(nn.ReLU(), nn.Linear(128, 128)))
        
        self.moe_model = MoE(input_dim=128, num_experts=3)
        
        self.mse_loss = nn.MSELoss(reduction='mean')

        self.graph = GAT(nfeat=384,
                         nhid=8,
                         nclass=128,  # 最后的类别 相当于下一层的输入
                         dropout=0.6,
                         nheads=3,  # 之前是8个
                         alpha=0.2
                         )
        
        improvedgatlayer_av = ImprovedGATLayer(200, dropout=0, num_heads=4, use_residual=True, no_cuda=False)
        self.improvedgat_av = ImprovedGAT(improvedgatlayer_av, num_layers=5, hidesize=200)

        improvedgatlayer_al = ImprovedGATLayer(200, dropout=0, num_heads=4, use_residual=True, no_cuda=False)
        self.improvedgat_al = ImprovedGAT(improvedgatlayer_al, num_layers=5, hidesize=200)

        improvedgatlayer_vl = ImprovedGATLayer(200, dropout=0, num_heads=4, use_residual=True, no_cuda=False)
        self.improvedgat_vl = ImprovedGAT(improvedgatlayer_vl, num_layers=5, hidesize=200)

        self.fusion_avl = ConcatFusion(len('avl'), input_dim=3*200, output_dim=128)

        self.fc_out = nn.Linear(200, 128)

    def get_network(self, self_type='l', layers=-1):
        if self_type in ['l', 'al', 'vl']:
            embed_dim, attn_dropout = 128, 0.3
        elif self_type in ['a', 'la', 'va']:
            embed_dim, attn_dropout = 128, 0.2
        elif self_type in ['v', 'lv', 'av']:
            embed_dim, attn_dropout = 128, 0.0    
        elif self_type == 'l_mem':
            embed_dim, attn_dropout = 128, 0.3
        elif self_type == 'a_mem':
            embed_dim, attn_dropout = 128, 0.3
        elif self_type == 'v_mem':
            embed_dim, attn_dropout = 128, 0.3
        else:
            raise ValueError("Unknown network type")

        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=8,
                                  layers=max(2, layers),
                                  attn_dropout=attn_dropout,
                                  relu_dropout=0.0,
                                  res_dropout=0.0,
                                  embed_dropout=0.2,
                                  attn_mask=true)
    
    def sim_operation_between_3(self, x0, x1, x2):
        return self.calc_sim(x0, x1) + self.calc_sim(x0, x2) + self.calc_sim(x1, x2)
    
    def calc_sim(self, x1, x2):
        return self.mse_loss(x1,x2)
    
    def calc_diff_loss(self, diff_list):
        for i in range(len(diff_list)):
            for j in range(i+1,len(diff_list)):
                if i == 0 and j == 1:
                    loss = torch.mean(torch.abs(torch.cosine_similarity(diff_list[i], diff_list[j], dim=-1)))
                else:
                    loss = loss + torch.mean(torch.abs(torch.cosine_similarity(diff_list[i], diff_list[j], dim=-1)))
        return loss

    def forward(self, incomplete_input):

        vision_m, audio_m, language_m = incomplete_input

        h_1_l = self.proj_l(self.bertmodel(language_m))[:, :8]
        h_1_a = self.proj_a(audio_m)[:, :8]
        h_1_v = self.proj_v(vision_m)[:, :8]

        c_l = self.encoder_c(h_1_l)
        c_a = self.encoder_c(h_1_a)
        c_v = self.encoder_c(h_1_v)

        s_l = self.encoder_s_l(h_1_l)    
        s_a = self.encoder_s_a(h_1_a)
        s_v = self.encoder_s_v(h_1_v)

        batch_size,seq_len,_=c_l.shape
        umask = torch.ones((batch_size, seq_len), dtype=torch.float32).to(device)
        lengths0 = []
        for j, umask_ in enumerate(umask):
            lengths0.append((umask[j] == 1).nonzero()[-1][0] + 1)
        seq_lengths = torch.stack(lengths0)

        features_c_l, edge_index_c_l, _, edge_index1_c_l = batch_graphify(c_l, None, seq_lengths, 16, 16, False)
        features_c_v = simple_batch_graphify(c_a, seq_lengths, False)
        features_c_a = simple_batch_graphify(c_v, seq_lengths, False)
        features_s_l = simple_batch_graphify(s_l, seq_lengths, False)
        features_s_a = simple_batch_graphify(s_a, seq_lengths, False)
        features_s_v = simple_batch_graphify(s_v, seq_lengths, False)

        features_single_cav = torch.cat([features_c_a, features_c_v], dim=0)
        features_cross_cav = self.improvedgat_av(features_single_cav, edge_index1_c_l)
        features_cross_ca1, features_cross_cv1 = torch.chunk(features_cross_cav, 2, dim=0)

        features_single_cal = torch.cat([features_c_a, features_c_l], dim=0)
        features_cross_cal = self.improvedgat_al(features_single_cal, edge_index1_c_l)
        features_cross_ca2, features_cross_cl1 = torch.chunk(features_cross_cal, 2, dim=0)

        features_single_cvl = torch.cat([features_c_v, features_c_l], dim=0)
        features_cross_cvl = self.improvedgat_vl(features_single_cvl, edge_index1_c_l)
        features_cross_cv2, features_cross_cl2 = torch.chunk(features_cross_cvl, 2, dim=0)

        features_single_sav = torch.cat([features_s_a, features_s_v], dim=0)
        features_cross_sav = self.improvedgat_av(features_single_sav, edge_index1_c_l)
        features_cross_sa1, features_cross_sv1 = torch.chunk(features_cross_sav, 2, dim=0)

        features_single_sal = torch.cat([features_s_a, features_s_l], dim=0)
        features_cross_sal = self.improvedgat_al(features_single_sal, edge_index1_c_l)
        features_cross_sa2, features_cross_sl1 = torch.chunk(features_cross_sal, 2, dim=0)

        features_single_svl = torch.cat([features_s_v, features_s_l], dim=0)
        features_cross_svl = self.improvedgat_vl(features_single_svl, edge_index1_c_l)
        features_cross_sv2, features_cross_sl2 = torch.chunk(features_cross_svl, 2, dim=0)

        shared_l = features_cross_cl1 + features_cross_cl2
        shared_a = features_cross_ca1 + features_cross_ca2
        shared_v = features_cross_cv1 + features_cross_cv2
        
        specil_l = features_cross_sl1 + features_cross_sl2
        specil_a = features_cross_sa1 + features_cross_sa2
        specil_v = features_cross_sv1 + features_cross_sv2

        shared_l = self.fc_out(shared_l)
        shared_l = shared_l.view(batch_size, 8, 128)
        shared_a = self.fc_out(shared_a)
        shared_a = shared_a.view(batch_size, 8, 128)
        shared_v = self.fc_out(shared_v)
        shared_v = shared_v.view(batch_size, 8, 128)
        specil_l = self.fc_out(specil_l)
        specil_l = specil_l.view(batch_size, 8, 128)
        specil_a = self.fc_out(specil_a)
        specil_a = specil_a.view(batch_size, 8, 128)
        specil_v = self.fc_out(specil_v)
        specil_v = specil_v.view(batch_size, 8, 128)

        recon_l = shared_l + specil_l
        recon_a = shared_a + specil_a
        recon_v = shared_v + specil_v

        c_l_r = self.encoder_c(recon_l)
        c_a_r = self.encoder_c(recon_a)
        c_v_r = self.encoder_c(recon_v)

        s_l_r = self.encoder_s_l(recon_l)    
        s_v_r = self.encoder_s_v(recon_a)
        s_a_r = self.encoder_s_a(recon_v)

        output = self.moe_model([recon_l,recon_a,recon_v])
    
        weighted_c_l = output['gating_weights'][:, 0].unsqueeze(1).unsqueeze(2) * recon_l
        weighted_c_a = output['gating_weights'][:, 1].unsqueeze(1).unsqueeze(2) * recon_a  
        weighted_c_v = output['gating_weights'][:, 2].unsqueeze(1).unsqueeze(2) * recon_v
       
        x_fusion = weighted_c_l + weighted_c_a + weighted_c_v
        
        out_prediction = self.dmml[0](x_fusion)
        out_prediction = out_prediction.mean(dim=1) 

        share_sim_loss = self.sim_operation_between_3(c_l,c_a,c_v)
        sup_sim_loss = (self.calc_sim(c_l, c_l_r) + \
                                self.calc_sim(c_a, c_a_r) + \
                                self.calc_sim(c_v, c_v_r)+ \
                                self.calc_sim(s_l, s_l_r) + \
                                self.calc_sim(s_a, s_a_r) + \
                                self.calc_sim(s_v, s_v_r)) 
        diff_list = [c_l,c_a,c_v,s_l,s_a, s_v]
        diff_loss = self.calc_diff_loss(diff_list)

        return {'sentiment_preds': out_prediction, 
                'share_sim_loss': share_sim_loss,
                'sup_sim_loss': sup_sim_loss,
                'diff_loss': diff_loss}
    
class ConcatFusion(nn.Module):
    def __init__(self, len_modals, input_dim=1024, output_dim=100):
        super(ConcatFusion, self).__init__()
        self.fc_out = nn.Linear(input_dim, output_dim)
        self.len_modals = len_modals
    def forward(self, x, y, z):
        if self.len_modals ==2:
            output = torch.cat((x, y), dim=1)
            output = self.fc_out(output)
            return output
        if self.len_modals ==3:
            output = torch.cat((x, y, z), dim=1)
            output = self.fc_out(output)
            return output

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))  # concat(V,NeigV)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        a_input = self._prepare_attentional_mechanism_input(Wh)  # 每一个节点和所有节点，特征。(Vall, Vall, feature)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        # 之前计算的是一个节点和所有节点的attention，其实需要的是连接的节点的attention系数
        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)    # 将邻接矩阵中小于0的变成负无穷
        attention = F.softmax(attention, dim=1)  # 按行求softmax。 sum(axis=1) === 1
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)   # 聚合邻居函数

        if self.concat:
            return F.elu(h_prime)   # elu-激活函数
        else:
            return h_prime
        
    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size()[0] # number of nodes

        # Below, two matrices are created that contain embeddings in their rows in different orders.
        # (e stands for embedding)
        # These are the rows of the first matrix (Wh_repeated_in_chunks): 
        # e1, e1, ..., e1,            e2, e2, ..., e2,            ..., eN, eN, ..., eN
        # '-------------' -> N times  '-------------' -> N times       '-------------' -> N times
        # 
        # These are the rows of the second matrix (Wh_repeated_alternating): 
        # e1, e2, ..., eN, e1, e2, ..., eN, ..., e1, e2, ..., eN 
        # '----------------------------------------------------' -> N times
        # 
        
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)  # 复制
        Wh_repeated_alternating = Wh.repeat(N, 1)
        # Wh_repeated_in_chunks.shape == Wh_repeated_alternating.shape == (N * N, out_features)

        # The all_combination_matrix, created below, will look like this (|| denotes concatenation):
        # e1 || e1
        # e1 || e2
        # e1 || e3
        # ...
        # e1 || eN
        # e2 || e1
        # e2 || e2
        # e2 || e3
        # ...
        # e2 || eN
        # ...
        # eN || e1
        # eN || e2
        # eN || e3
        # ...
        # eN || eN

        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
        # all_combinations_matrix.shape == (N * N, 2 * out_features)

        return all_combinations_matrix.view(N, N, 2 * self.out_features)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
        
class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in
                           range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha,
                                           concat=False)  # 第二层(最后一层)的attention layer

    def forward(self, x, adj):
        #x = F.dropout(x, self.dropout, training=self.training)
        print(x)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)  # 将每层attention拼接
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))  # 第二层的attention layer
        return F.log_softmax(x, dim=1)
    
class MoE(nn.Module):
    def __init__(self, input_dim, num_experts):
        super(MoE, self).__init__()
        self.num_experts = num_experts
        self.gating_type = 'importance'
        self.moe_fusion_type = 'concat'

        self.experts = nn.ModuleList()
        for _ in range(num_experts):
            self.experts.append(nn.Sequential(nn.ReLU(), nn.Linear(input_dim, input_dim)))
        if self.gating_type == 'concat':
            self.gating_fc = nn.Sequential(nn.ReLU(),
                                            nn.Linear(input_dim * num_experts, input_dim),
                                            nn.ReLU(),
                                            nn.Linear(input_dim, num_experts))
        elif self.gating_type == 'importance':
            self.gating_fc = nn.Sequential(nn.ReLU(),
                                           nn.Linear(input_dim * num_experts, input_dim))

        self.flatten = nn.Flatten(start_dim=1)
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def gating_network(self, x):
        if self.gating_type == 'concat':
            weights = self.softmax(self.gating_fc(self.flatten(x))) # [bs, num_experts]
        elif self.gating_type == 'importance':
            sample_fusion = torch.unsqueeze(self.gating_fc(self.flatten(x)), -1)  # [bs, input_dim, 1]
            weights = self.softmax(torch.squeeze(torch.bmm(x, sample_fusion), dim=-1))  # [bs, num_experts]
        elif self.gating_type == 'correlation':
            input = self.sigmoid(x)
            corrs = torch.bmm(input,torch.transpose(input,-1,-2)) # [bs, num_experts, num_experts]
            weights = self.softmax(torch.sum(corrs, dim=-1))
        return weights

    def forward(self, x):
        gating_input = torch.stack([x[i].mean(dim=1) for i in range(self.num_experts)], dim=1)  # [bs, 7, 128]
        gating_weights = self.gating_network(gating_input) # [bs, num_experts]
        expert_outputs = torch.stack([self.experts[i](x[i].mean(dim=1)) for i in range(self.num_experts)], dim=1)  # [bs, num_experts, input_dim]

        if self.moe_fusion_type == 'sum':
            moe_output = torch.sum(expert_outputs * torch.unsqueeze(gating_weights, dim=-1), dim=1)  # [bs, input_dim]
        elif self.moe_fusion_type == 'concat':
            moe_output = self.flatten(expert_outputs * torch.unsqueeze(gating_weights, dim=-1))  # [bs, input_dim * num_experts]

        return {'moe_output': moe_output,
                'gating_weights': gating_weights}
    
class ImprovedGATLayer(torch.nn.Module):
    def __init__(self, hidesize, dropout=0.5, num_heads=5, use_residual=True, no_cuda=False):
        super(ImprovedGATLayer, self).__init__()
        self.no_cuda = no_cuda
        self.use_residual = use_residual
        self.convs = GATConv(hidesize, hidesize, heads=num_heads, add_self_loops=True, concat=False)

    def forward(self, features, edge_index):
        x = features
        if self.use_residual:
            x = x + self.convs(x, edge_index)
        else:
            x = self.convs(x, edge_index)

        return x

class ImprovedGAT(torch.nn.Module):
    def __init__(self, encoder_layer, num_layers, hidesize):
        super(ImprovedGAT, self).__init__()
        layer = []
        for l in range(num_layers):
            layer.append(encoder_layer)
        self.layers = nn.ModuleList(layer)
        self.out_mlp = nn.Linear((num_layers+1)*hidesize, hidesize)
        self.input_proj = nn.Linear(128, 200)

    def forward(self, features, edge_index):
        features = self.input_proj(features)
        out = features
        output = [out]
        for mod in self.layers:
            out = mod(out, edge_index)
            output.append(out)
        output_ = torch.cat(output, dim=-1)
        output_ = self.out_mlp(output_)
        return output_
    
def build_model(args):
    return MyMultimodal(args)