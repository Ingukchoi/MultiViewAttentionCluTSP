import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv
import math
from Sublayer import MultiHeadAttention
from torch_geometric.data import Batch
import torch.nn.functional as F
from Data_generation import create_graph_data


########################################
# CluTSPSolver
######################################## 

class CluTSPSolver(nn.Module):
    def __init__(self, **model_prarams):
        super().__init__()

        self.encoder = CluTSP_Encoder(**model_prarams)
        self.decoder  = CluTSP_Decoder(**model_prarams)

    def forward(self, pos_input, batch_size, num_node, sampling_size, return_pi=False):

        # Get graph data
        ########################################
        CluTSP_data, Intra_clu_data = create_graph_data(pos_input)
        CluTSP_data=Batch.from_data_list(CluTSP_data)
        Intra_clu_data=Batch.from_data_list(Intra_clu_data)
        
        node_embeddings = self.encoder(CluTSP_data.x, CluTSP_data.edge_index, CluTSP_data.edge_attr, Intra_clu_data.x, Intra_clu_data.edge_index, batch_size, num_node)
        depot_embeddings = node_embeddings[:, 0:1, :]
        node_embeddings = node_embeddings[:, 1:, :]
        
        cord_input = pos_input[:, :, 0:2]
        clu_input = pos_input[:, :, -1:]
        input_cord = torch.cat([cord_input, clu_input], dim=2)
        
        input_cord = input_cord[:, 1:, :]
        depot_cord = cord_input[:,0:1,:]

        return self.decoder(input_cord, depot_cord, depot_embeddings, node_embeddings, sampling_size, return_pi)

########################################
# ENCODER
########################################    
    
class CluTSP_Encoder(nn.Module):
  def __init__(self, **model_params):
    super().__init__()
    self.model_params = model_params
    self.embedding_dim = self.model_params['embedding_dim']
    self.ff_hidden_dim = self.model_params['ff_hidden_dim']
    self.num_Encoder_layer = self.model_params['encoder_layer_num']
    self.input_dim = self.model_params['input_dimension']

    self.Initial_embeddings = nn.Linear(self.input_dim, self.embedding_dim, bias=True)
    self.encoder_layer = nn.ModuleList([CluTSP_Encoder_Layer(**model_params) for _ in range(self.num_Encoder_layer)])

  def forward (self, x, edge_index, edge_attr, clu_x, clu_x_edge_index, batch_size, num_nodes):
    embeddings=self.Initial_embeddings(x)

    for e_layer in self.encoder_layer:
        embeddings = e_layer(embeddings, edge_index, edge_attr, clu_x_edge_index)
    embeddings = embeddings.view(batch_size, num_nodes, self.embedding_dim)
    return embeddings
  
class CluTSP_Encoder_Layer(nn.Module):
  def __init__ (self,  **model_params):
      super().__init__()
      self.model_params = model_params

      embedding_dim = self.model_params['embedding_dim']
      ff_hidden_dim = self.model_params['ff_hidden_dim']
      head_num = self.model_params['head_num']
      qkv_dim = self.model_params['qkv_dim']

      self.GATv2 = GATv2Conv(embedding_dim, qkv_dim, heads=head_num, concat=True, dropout=0.2, edge_dim=1)
      self.Multi_head_combine = nn.Linear(embedding_dim, embedding_dim)
      self.BatchNorm = nn.BatchNorm1d(embedding_dim)
      
      self.FF = nn.Linear(embedding_dim, ff_hidden_dim)
      self.FF_2 = nn.Linear(ff_hidden_dim, embedding_dim)
      self.BN = nn.BatchNorm1d(embedding_dim)
      
      self.GATv2_clu = GATv2Conv(embedding_dim, qkv_dim, heads=head_num, concat=True, dropout=0.2)
      self.Multi_head_combine_clu = nn.Linear(embedding_dim, embedding_dim)
      self.BatchNorm_clu = nn.BatchNorm1d(embedding_dim)
      
      self.FF_clu = nn.Linear(embedding_dim, ff_hidden_dim)
      self.FF_2_clu = nn.Linear(ff_hidden_dim, embedding_dim)
      self.BN_clu = nn.BatchNorm1d(embedding_dim)
      
      self.embedding_cat = nn.Linear(2*embedding_dim, embedding_dim)
      
  def forward(self, embeddings, edge_index, edge_attr, clu_edge_index):
      residual = embeddings
      
      x=self.GATv2(embeddings, edge_index, edge_attr)
      x=self.Multi_head_combine(x)
      GAT_x=self.BatchNorm(x+residual)
      
      x = self.FF_2(F.elu(self.FF(GAT_x)))
      x = self.BN(x+GAT_x)
      
      x_clu=self.GATv2_clu(embeddings, clu_edge_index)
      x_clu=self.Multi_head_combine_clu(x_clu)
      GAT_x_clu=self.BatchNorm_clu(x_clu+residual)
      
      x_clu = self.FF_2_clu(F.elu(self.FF_clu(GAT_x_clu)))
      x_clu = self.BN_clu(x_clu+GAT_x_clu)
      
      embeddings = self.embedding_cat(torch.cat([x, x_clu], dim=1))
      return embeddings
      

########################################
# DECODER
########################################

class CluTSP_Decoder(nn.Module):
  def __init__(self, **model_params):
      super().__init__()
      self.model_params = model_params
      self.logit_clipping = self.model_params['logit_clipping']
      embedding_dim = self.model_params['embedding_dim']
      head_num = self.model_params['head_num']
      qkv_dim = self.model_params['qkv_dim']


      self.layers = nn.ModuleList([GlobalGuidingModule(**model_params) for _ in range(1)])
      self.w_cluster = nn.Linear(2*embedding_dim, embedding_dim)

      self.Wk_1 = nn.Linear(embedding_dim, embedding_dim)
      self.Wk_2 = nn.Linear(embedding_dim, embedding_dim)
      self.Wv = nn.Linear(embedding_dim, embedding_dim)
      self.Wg = nn.Linear(embedding_dim, embedding_dim)
      self.MHA = MultiHeadAttention(head_num, embedding_dim, qkv_dim, qkv_dim, False)
      self.w_context=nn.Linear(4*embedding_dim+2, embedding_dim)
      
      self.set_node_select('sampling')
      self.set_cluster_select('sampling')
  
  # util functions
  #######################################################

  def visit_node(self, log_p, mask):
        if self.node_select == 'greedy':
            selected_nodes = torch.argmax(log_p, 1) 
        else:
            selected_nodes = torch.multinomial(log_p.exp(), 1).squeeze(1) 

        mask[torch.arange(mask.size(0)), 0, selected_nodes] = True

        return selected_nodes, mask
  
  def get_cost(self, pi, pos_input, depot_cordinate):

      #assert self.check_all_nodes_visited(pi), "Not all nodes were visited exactly once"
      pos_input = pos_input[:,:,:2]
      visited = pos_input.gather(1, pi.unsqueeze(-1).expand_as(pos_input))

      dist_from_depot_to_first = (visited[:, 0] - depot_cordinate[:, 0]).norm(p=2, dim=1) # depot to first node
      inter_node_distances = (visited[:, 1:] - visited[:, :-1]).norm(p=2, dim=2).sum(1) # tour length
      dist_from_last_to_depot = (depot_cordinate[:, 0] - visited[:, -1]).norm(p=2, dim=1) # last node to depot
      cost = dist_from_depot_to_first + inter_node_distances + dist_from_last_to_depot # total toru length
      return cost

  def cal_log_likelihood(self, log_p, pi):
      log_p = log_p.gather(2, pi.unsqueeze(-1)).squeeze(-1)
      return log_p

  def set_node_select(self, node_select): #Greedy or sampling
      self.node_select = node_select
  
  def set_cluster_select(self, cluster_select): #Greedy or sampling
      self.cluster_select = cluster_select
      
  def cluster_masking_func(self, pos_input, cluster_idx, selected_nodes, cluster_mask, mask, visit_clu_masking):
      batch_size = pos_input.size(0)
      device = pos_input.device

      batch_indices = torch.arange(batch_size, device=device)
      selected_cluster = cluster_idx[batch_indices, selected_nodes]

      cluster_mask = (cluster_idx != selected_cluster.unsqueeze(1)).unsqueeze(1)
      current_cluster_nodes = (cluster_idx == selected_cluster.unsqueeze(1))
      all_visited = torch.all(mask.squeeze(1) | ~current_cluster_nodes, dim=1)

      cluster_mask[all_visited] = False
      cluster_mask[all_visited] |= mask[all_visited]

      visit_clu_masking.scatter_(2, selected_cluster.long().unsqueeze(1).unsqueeze(2), 1)
      return cluster_mask, visit_clu_masking
  
    
  def cluster_comp_check(self, pos_input, mask, selected_nodes, cluster_idx):
      batch_size = pos_input.size(0)
    
      selected_cluster = cluster_idx[torch.arange(batch_size, device=pos_input.device), selected_nodes]
      cluster_nodes = (cluster_idx == selected_cluster.unsqueeze(1))
    
      visited_count = (mask[:, 0, :] & cluster_nodes).sum(dim=1)
      cluster_completion = (visited_count == 1).unsqueeze(1)
    
      return cluster_completion
  
  def get_cluster_embeddings(self, node_embeddings, input_data):
    cluster_indices = input_data[:, :, 2].long()                   
    unique_clusters = torch.unique(cluster_indices)              

    cluster_eq = cluster_indices.unsqueeze(-1) == unique_clusters 
    mask = cluster_eq.unsqueeze(-1).float()                     

    embeddings_exp = node_embeddings.unsqueeze(2)
    cluster_nodes = embeddings_exp * mask  

    cluster_sum  = cluster_nodes.sum(dim=1)                      
    cluster_size = mask.sum(dim=1).clamp(min=1e-8)        
    cluster_mean = cluster_sum / cluster_size       

    neg_inf = torch.finfo(node_embeddings.dtype).min
    cluster_nodes_inf = embeddings_exp.masked_fill(~cluster_eq.unsqueeze(-1), neg_inf)
    cluster_max, _ = cluster_nodes_inf.max(dim=1)

    cluster_embeddings = torch.cat([cluster_mean, cluster_max], dim=-1)
    return cluster_embeddings

  def unvisit_embed(self, node_embedding, mask, num_node):
        expanded_mask = mask.squeeze(1).unsqueeze(-1)
        masked_embeddings = node_embedding.masked_fill(expanded_mask, 0)
        masked_embeddings = torch.where(expanded_mask, torch.zeros_like(node_embedding), node_embedding)
        unvisit_mean_embedding = masked_embeddings.sum(dim=1, keepdim=True) / num_node
        return unvisit_mean_embedding
  
  def forward(self, pos_input, depot_cord, depot_embeddings, embedded_node, sampling_size, return_pi=False):
      
      #multi-start sampling roollout # Greedy ==1
      pos_input = pos_input.repeat_interleave(sampling_size, dim=0)
      embedded_node = embedded_node.repeat_interleave(sampling_size, dim=0)
      depot_embeddings = depot_embeddings.repeat_interleave(sampling_size, dim=0)
      depot_cord = depot_cord.repeat_interleave(sampling_size, dim=0)
      
      clu_embed = self.get_cluster_embeddings(embedded_node, pos_input) 
      clu_embed = self.w_cluster(clu_embed)
      cluster_embedding = torch.cat([depot_embeddings, clu_embed], dim=1)
      
      cluster_indices = pos_input[:, :, 2].long()
      num_cluster = len(torch.unique(cluster_indices))

      Batch_size = pos_input.size(0) 
      problem_size = pos_input.size(1) 
    
      current_embedding = depot_embeddings
    
      mask = torch.zeros((Batch_size, 1, problem_size)).bool()  
      cluster_mask = torch.zeros((Batch_size, 1, problem_size)).bool() 
      visit_clu_mask  = torch.zeros((Batch_size, 1, num_cluster+1)).bool() 
      is_new_cluster = torch.logical_not(torch.zeros(Batch_size, 1, dtype=torch.bool))
      
      aug_context_embedding = None
      cluster_guidance_embeddings = None
      cluster_guidance=None
      
      paths = list()
      cluster_paths = list()
      log_ps = list()
      clu_log_ps = list()

      node_keys = self.Wk_1(embedded_node)
      node_values = self.Wv(embedded_node)
      node_keys_candidates = self.Wk_2(embedded_node)

      # Autoregressive
      for i in range(problem_size):
        for layer in self.layers:
          visited_clu_maksing = visit_clu_mask.clone()
          aug_context_embedding, cluster_guidance_embeddings, cluster_guidance, cluster_log_p = layer(depot_embeddings, cluster_embedding, current_embedding, embedded_node, aug_context_embedding, is_new_cluster, cluster_mask, visited_clu_maksing, mask, cluster_guidance_embeddings, self.cluster_select, cluster_guidance, i)  
          
        ninf_mask = mask|cluster_mask   # mask: visited / cluster_mask: cluster constraint
        visited_mask = mask.squeeze(1) 

        sel_nodes = paths[-1] if i > 0 else torch.zeros(Batch_size, dtype=torch.long, device=pos_input.device)

        unique_clusters = torch.unique(cluster_indices)
        total_clusters = unique_clusters.numel()

        cluster_masks = cluster_indices.unsqueeze(-1) == unique_clusters.unsqueeze(0).unsqueeze(0)
        visited_cluster_masks = visited_mask.unsqueeze(-1) & cluster_masks
        visited_clusters = visited_cluster_masks.any(dim=1).float().sum(dim=1)
        cluster_ratio = visited_clusters / total_clusters 

        sel_cluster = cluster_indices[torch.arange(Batch_size), sel_nodes] 
        cluster_mask = cluster_indices == sel_cluster.unsqueeze(1) 
        total_nodes = cluster_mask.sum(dim=1).float() 
        visited_nodes = (visited_mask & cluster_mask).sum(dim=1).float() 
        node_ratio = visited_nodes / (total_nodes + 1e-8) 

 
        ratio_feat = torch.stack([cluster_ratio, node_ratio], dim=1) 
        ratio_feat = ratio_feat.unsqueeze(1)
        aug_context_embedding = torch.cat([aug_context_embedding, ratio_feat], dim=2)
        #####################################

        new_context_embedding = self.w_context(aug_context_embedding)
        glimpse_query, _ = self.MHA(new_context_embedding, node_keys, node_values, ninf_mask)
        query = self.Wg(glimpse_query)
        logits = torch.matmul(query, node_keys_candidates.permute(0, 2, 1)).squeeze(1) / math.sqrt(query.size(-1))
        logits = torch.tanh(logits)*self.logit_clipping

        logits -= ninf_mask[:, 0, :] * 1e9
        log_p = torch.log_softmax(logits, dim=1)
        selected_nodes, mask = self.visit_node(log_p, mask)
    
        cluster_mask, visit_clu_mask = self.cluster_masking_func(pos_input, cluster_indices, selected_nodes, cluster_mask, mask, visit_clu_mask)

        selected_node_embeddings = embedded_node[torch.arange(embedded_node.size(0)), selected_nodes]
        current_embedding = selected_node_embeddings[:, None,:]
        
        unvisit_mean_embedding = self.unvisit_embed(embedded_node, mask|cluster_mask, problem_size)
        aug_context = [unvisit_mean_embedding, current_embedding, cluster_guidance_embeddings, depot_embeddings]
        aug_context_embedding = torch.cat(aug_context, dim=2) 

        paths.append(selected_nodes)
        log_ps.append(log_p)
        
        update_mask = is_new_cluster.float()
        if update_mask.sum()>0:
            cluster_paths.append(cluster_guidance)
            clu_log_ps.append(cluster_log_p)
        is_new_cluster = self.cluster_comp_check(pos_input, mask, selected_nodes, cluster_indices) 

      ## Done!
      pi = torch.stack(paths, 1)
      log_p = torch.stack(log_ps, 1)
      clu_log_p = torch.stack(clu_log_ps,1)
      clu_pi = torch.stack(cluster_paths, 1)
    
      cost = self.get_cost(pi, pos_input, depot_cord)
      
      likelihood = self.cal_log_likelihood(log_p, pi)
      cluster_likehood = self.cal_log_likelihood(clu_log_p, clu_pi)

      if return_pi:
          return cost, likelihood, cluster_likehood, pi
      return cost, likelihood, cluster_likehood
      
class GlobalGuidingModule(nn.Module):
    def __init__(self,  **model_params):
        super().__init__()

        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']
        logit_clipping = self.model_params['logit_clipping']
        
        self.W_cluster_q = nn.Linear(3 * embedding_dim, embedding_dim, bias=False)
        self.W_cluster_v = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.W_cluster_k = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.W_cluster_k_single = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.MHA = MultiHeadAttention(head_num, embedding_dim, qkv_dim, qkv_dim, False)
        
        self.logit_clipping = logit_clipping

    def forward(self, depot_embedding, cluster_embedding, current_embedding, node_embeddings, aug_context_embedding, is_new_cluster, cluster_mask, visited_cluster_mask, mask, cluster_guidance_embedding, select_mode, cluster_guidance, step):

        cluster_guding_module = is_new_cluster.float()
        batch_size = node_embeddings.size(0)
        num_nodes = node_embeddings.size(1)
        embed_dim = node_embeddings.size(2)
        
        if step == 0:
            init_aug_context_embedding = torch.zeros(batch_size, 1, 4*embed_dim, requires_grad=True)
            init_cluster_guidance_embedding = torch.zeros(batch_size, 1, embed_dim, requires_grad=True)
            init_cluster_guidance = torch.zeros(batch_size, dtype=torch.long)
        else:
            init_aug_context_embedding = aug_context_embedding.clone()
            init_cluster_guidance_embedding = cluster_guidance_embedding.clone()
            init_cluster_guidance = cluster_guidance.clone()
            
        visite_cluster_mask = visited_cluster_mask.clone()

        # Gluster guiding module: select next cluster to visit
        #######################################################
        if cluster_guding_module.sum() > 0:
            new_cluster_batch = torch.where(cluster_guding_module == 1)[0]
            
            n_mask = mask[new_cluster_batch].clone()
            n_cluster_mask = cluster_mask[new_cluster_batch].clone()
            n_embed = node_embeddings[new_cluster_batch].clone()
            
            unvisit_mean_embedding = self.unvisit_embed(n_embed, n_mask, num_nodes)
            context_embedding = torch.cat([unvisit_mean_embedding, current_embedding[new_cluster_batch], depot_embedding[new_cluster_batch]], dim=2)
            
            cluster_query = self.W_cluster_q(context_embedding) # shape: (batch, 1, embedding_dim)
            cluster_keys = self.W_cluster_k(cluster_embedding[new_cluster_batch])
            cluster_values = self.W_cluster_v(cluster_embedding[new_cluster_batch])
            prob_keys = self.W_cluster_k_single(cluster_embedding[new_cluster_batch])
            
            all_clusters_visited = torch.all(visite_cluster_mask[new_cluster_batch, 0, 1:], dim=1, keepdim=True)
            visite_cluster_mask[new_cluster_batch, 0, 0] = ~all_clusters_visited.squeeze(1)
            
            glimpse_query, _ = self.MHA(cluster_query, cluster_keys, cluster_values, visite_cluster_mask[new_cluster_batch])
            
            clu_logit = torch.matmul(glimpse_query, prob_keys.permute(0,2,1)).squeeze(1)/math.sqrt(glimpse_query.size(-1))
            clu_logit = torch.tanh(clu_logit) * self.logit_clipping
            clu_logit = torch.where(visite_cluster_mask[new_cluster_batch, 0, :], torch.tensor(-1e9, device=clu_logit.device), clu_logit)
            
            clu_log_prob = torch.log_softmax(clu_logit, dim=1)
            cluster_guidance = self.visit_cluster(clu_log_prob, select_mode)
            cluster_guidance_embedding = cluster_embedding[new_cluster_batch][torch.arange(len(new_cluster_batch)), cluster_guidance].unsqueeze(dim=1)
            
            unvisit_cluster_embed = self.unvisit_embed(n_embed, n_mask|n_cluster_mask, num_nodes)
            concat_list = [unvisit_cluster_embed, current_embedding[new_cluster_batch], cluster_guidance_embedding, depot_embedding[new_cluster_batch]]
            aug_context_embedding = torch.cat(concat_list, dim=2)
            
            batch_shape = torch.zeros(batch_size, dtype=torch.bool)
            batch_shape[new_cluster_batch] = True
            
            tmp_aug_context_embed = torch.zeros_like(init_aug_context_embedding)
            tmp_aug_context_embed[new_cluster_batch] = aug_context_embedding
            aug_context_embedding = torch.where(batch_shape.unsqueeze(1).unsqueeze(2), tmp_aug_context_embed, init_aug_context_embedding)

            tmp_clu_guidance_embed = torch.zeros_like(init_cluster_guidance_embedding)
            tmp_clu_guidance_embed[new_cluster_batch] = cluster_guidance_embedding
            cluster_guidance_embedding = torch.where(batch_shape.unsqueeze(1).unsqueeze(2), tmp_clu_guidance_embed, init_cluster_guidance_embedding)
            
            tmp_guidance = init_cluster_guidance.clone()
            tmp_guidance[new_cluster_batch] = cluster_guidance
            cluster_guidance = torch.where(batch_shape, tmp_guidance, init_cluster_guidance)
            
            clu_prob = torch.full((batch_size, visite_cluster_mask.size(2)), 0, dtype=torch.float)
            clu_prob[new_cluster_batch] = clu_log_prob
        else:
            clu_prob = torch.full((batch_size, visite_cluster_mask.size(2)), 0)
    
        return aug_context_embedding, cluster_guidance_embedding, cluster_guidance, clu_prob
      
    # util functions
    #######################################################
    def visit_cluster(self, log_p, select_mode):
      if select_mode =='greedy':
        cluster_guidance = torch.argmax(log_p,1)
      else:
        cluster_guidance = torch.multinomial(log_p.exp(),1).squeeze(1)
      return cluster_guidance
    def unvisit_embed(self, node_embedding, mask, num_node):
        expanded_mask = mask.squeeze(1).unsqueeze(-1) # [B, N, 1]
        masked_embeddings = node_embedding.masked_fill(expanded_mask, 0)
        masked_embeddings = torch.where(expanded_mask, torch.zeros_like(node_embedding), node_embedding)
        unvisit_mean_embedding = masked_embeddings.sum(dim=1, keepdim=True) / num_node
        return unvisit_mean_embedding