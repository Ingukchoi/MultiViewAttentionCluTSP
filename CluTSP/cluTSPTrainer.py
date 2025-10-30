import torch
import torch.optim as optim
from MVAttnModel import CluTSPSolver as Model
from Data_generation import TSPDataset
from logging import getLogger
from utils import *
import random

class cluTSPTrainer:
    def __init__(self,
                 env_params,
                 model_params,
                 optimizer_params,
                 trainer_params):

        # Parameters setting
        self.env_params = env_params
        self.model_params = model_params
        self.optimizer_params = optimizer_params
        self.trainer_params = trainer_params
        
        self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder()
        self.result_log = LogData()

        self.problem_size = self.env_params['n_nodes']

        self.min_cluster_size = self.env_params['min_cluster']
        self.max_cluster_size = self.env_params['max_cluster']

        self.multi_start_size=self.env_params['multi_start']
        self.accmulation_step = self.trainer_params['accmulation_step']
        self.type = self.trainer_params['training']

        # Device setting
        USE_CUDA = self.trainer_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.trainer_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')

        # Main Components
        self.start_epoch = 1
        self.model = Model(**self.model_params)
        self.optimizer = optim.Adam(self.model.parameters(), **self.optimizer_params['optimizer'])
        self.time_estimator = TimeEstimator()

    def run(self):
        self.time_estimator.reset(self.start_epoch)
        for epoch in range(self.start_epoch , self.trainer_params['epochs']+1):
            self.logger.info('=================================================================')

            train_score, train_loss = self._train_one_epoch(epoch)

            self.result_log.append('train_score', epoch, train_score)
            self.result_log.append('train_loss', epoch, train_loss)

            # Logs & Checkpoint
            ############################
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(epoch, self.trainer_params['epochs'])
            self.logger.info("Epoch {:3d}/{:3d}: Time Est.: Elapsed[{}], Remain[{}]".format(
                epoch, self.trainer_params['epochs'], elapsed_time_str, remain_time_str))

            # Save parameter
            ############################
            torch.save(self.model.state_dict(), '{}/checkpoint-{}.pt'.format(self.result_folder, epoch))

            # if training done
            ############################
            all_done = (epoch == self.trainer_params['epochs'])
            if all_done:
                self.logger.info(" *** Training Done *** ")
                self.logger.info("Now, printing log array...")
                util_print_log_array(self.logger, self.result_log)
        
    def _train_one_epoch(self, epoch):

        score_AM = AverageMeter()
        loss_AM = AverageMeter()
        train_num_episode = self.trainer_params['train_episodes']
        episode = 0
        loop_cnt=0
        epi_loop=1
        while episode < train_num_episode:

            remaining = train_num_episode-episode
            batch_size = min(self.trainer_params['train_batch_size'], remaining)

            score, loss = self._train_one_batch(batch_size)

            loss=loss/self.accmulation_step
            loss.backward()
            if episode%self.accmulation_step==0 or batch_size<self.trainer_params['train_batch_size']:
                self.optimizer.step()
                self.optimizer.zero_grad()

            score_AM.update(score, batch_size)
            loss_AM.update(loss.item(), batch_size)

            episode += batch_size
            epi_loop+=1
            # Log First 10 Batch, only at the first epoch
            if epoch == self.start_epoch:
                loop_cnt += 1
                if loop_cnt <= 10:
                    self.logger.info('Epoch {:3d}: Train {:3d}/{:3d}({:1.1f}%)  Score: {:.4f},  Loss: {:.4f}'
                                     .format(epoch, episode, train_num_episode, 100. * episode / train_num_episode,
                                             score_AM.avg, loss_AM.avg))
        # Log Once, for each epoch         
        self.logger.info('Epoch {:3d}: Train ({:3.0f}%)  Score: {:.4f},  Loss: {:.4f}'
                         .format(epoch, 100. * episode / train_num_episode,
                                 score_AM.avg, loss_AM.avg))
        
        return score_AM.avg, loss_AM.avg

    def _train_one_batch(self, batch_size):
        self.model.train()
        
        low = self.min_cluster_size
        high = self.max_cluster_size
        self.cluster_size = random.randint(low, high)

        dataset = TSPDataset(self.problem_size, batch_size, self.cluster_size, seed=None)
        cluTSP_data = dataset.data

        r, log_prob, clu_log_prob = self.model(cluTSP_data, batch_size, self.problem_size, self.multi_start_size)

        reward = -r.view(-1, self.multi_start_size)
        log_prob = log_prob.view(batch_size, self.multi_start_size, -1)

        if clu_log_prob is not None:
            clu_log_prob = clu_log_prob.view(batch_size, self.multi_start_size, -1)

        advantage = reward-reward.float().mean(dim=1, keepdims=True)
        sum_log_prob = log_prob.sum(dim=2)
        
        if clu_log_prob is None:
            loss_mean = -advantage * sum_log_prob
            loss = loss_mean.mean()
        else:
            sum_clu_log_prob = clu_log_prob.sum(dim=2)
            loss_node_mean = -advantage * sum_log_prob
            loss_clu_mean = -advantage *sum_clu_log_prob
            loss_mean = loss_node_mean + loss_clu_mean
            loss = loss_mean.mean()

        max_pomo_reward, _ = reward.max(dim=1) 
        score_mean = -max_pomo_reward.float().mean()

        return score_mean.item(), loss