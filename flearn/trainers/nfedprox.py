import numpy as np
from tqdm import trange, tqdm
import tensorflow.compat.v1 as tf

from .fedbase import BaseFedarated
from flearn.optimizer.pgd import PerturbedGradientDescent
from flearn.utils.tf_utils import process_grad, process_sparse_grad

class Server(BaseFedarated):
    def __init__(self, params, learner, dataset):
        print('Using Federated prox to Train')
        self.inner_opt = PerturbedGradientDescent(params['learning_rate'], params['mu'])
        super(Server, self).__init__(params, learner, dataset)

    def get_sigmaDi(self,min_Di,num_Dk,min_Dk,csolns):#self.num_rounds,self.L,self.clients_per_round,self.epsilon,self.delta,min_Di,num_Dk,csolns
        sigma = []
        b = -self.num_rounds/self.epsilon*np.log((np.exp(-self.epsilon/self.num_rounds)-1)/self.clients_per_round+1)
        c = np.sqrt(2*np.log(1.25/self.delta))
        for i in range(len(csolns)):
            Di = csolns[i][0]
            pi = Di/num_Dk
            gamma = -np.log(1-self.clients_per_round+self.clients_per_round*np.e**(-self.epsilon/self.L*(min_Di/min_Dk/np.sqrt(pi))))
            if self.num_rounds > self.epsilon/gamma:
                sigma.append(2*c*self.Clip/self.epsilon*np.sqrt(np.square(self.num_rounds/min_Dk/b)-pi*np.square(self.L/min_Di)))
            else:
                sigma.append(0)
        return sigma

    def _Clip(self):
        #初始模型的范数 确定Clip，因为模型大小与范数直接相关——self.noised_latest_model
        num_weights = len(self.noised_latest_model)
        norm = [np.sqrt(np.sum(np.square(self.noised_latest_model[i]))) for i in range(num_weights)]
        #self.Clip = max(list(np.array(norm)/2 - (self.cn - self.cnk)*0.1))
        self.Clip = 5 - (self.cn - self.cnk)*0.1
        return

    def _clients_per_round(self):
        if self.cnk/self.cn <= 0.5:
            self.clients_per_round = 0.5
        else:
            self.clients_per_round = 0.1
        #print("K/N--------------",self.clients_per_round)
        return 
        
    def train(self):
        '''Train using Federated Proximal'''
        
        #统计dissim--------------------------------------------------------------------------------------------
        model_len = process_grad(self.noised_latest_model).size
        global_grads = np.zeros(model_len)
        client_grads = np.zeros(model_len)
        num_samples = []
        local_grads = []

        for c in self.clients:
            num, client_grad = c.get_grads(model_len)
            local_grads.append(client_grad)
            num_samples.append(num)
            global_grads = np.add(global_grads, client_grad * num)
        global_grads = global_grads * 1.0 / np.sum(np.asarray(num_samples))

        difference = 0
        for idx in range(len(self.clients)):
            difference += np.sum(np.square(global_grads - local_grads[idx]))
        difference = difference * 1.0 / len(self.clients)
        tqdm.write('global gradient difference: {}'.format(difference))

        #确定参数-----------------------------------------------------------------------------------------------
        self._Clip()
        self._clients_per_round()
        #-------------------------------------------------------------------------------------------------------
        print("K/N--------------",self.clients_per_round)
        print("Clip-------------",self.Clip)
        num_clients=int(self.num_clients*self.clients_per_round)
        print('Training with {} workers ---'.format(num_clients))
        
        #计算min_Di,sigmaU is fixed
        _,all_clients = self.select_clients(0, self.num_clients)
        min_Di = 10e10
        min_Dk = 0
        clients_num_samples = []
        for idx, c in enumerate(all_clients.tolist()):
            min_Di = min(min_Di,c.num_samples)
            clients_num_samples.append(c.num_samples)
        clients_num_samples.sort()
        min_Dk = sum(clients_num_samples[:num_clients])
        sigmaU = np.sqrt(2*np.log(1.25/self.delta))*self.L*2*self.Clip/self.epsilon/min_Di
        #print('sigmaU---',sigmaU)
        #print('min_Di---',min_Di)

        
        for i in range(self.num_rounds):
            indices, selected_clients = self.select_clients(i, num_clients)  # uniform sampling
            np.random.seed(i)  # make sure that the stragglers are the same for FedProx and FedAvg
            csolns = [] # buffer for receiving client solutions

            self.inner_opt.set_params(self.noised_latest_model, self.client_model)
            
            num_Dk = 0
            for idx, c in enumerate(selected_clients.tolist()):
                num_Dk += c.num_samples
                
                # communicate the latest model
                c.set_params(self.noised_latest_model)
                #c.set_params(self.latest_model)

                soln, stats = c.solve_inner_noise(num_epochs=self.num_epochs, batch_size=self.batch_size)

                num_weights = len(soln[1])
                norm = [np.sqrt(np.sum(np.square(soln[1][i]))) for i in range(num_weights)]
                #print("norm----------",norm)
                factor = [max(1,norm[i]/self.Clip) for i in range(num_weights)]
                #print("client_weight----------------",soln[1])
                soln[1] = [soln[1][i]/factor[i] for i in range(num_weights)]
                norm = [np.sqrt(np.sum(np.square(soln[1][i]))) for i in range(num_weights)]
                #print("Cliped_norm-------------",norm)
                GaussianNoise = [np.random.normal(loc=0.0, scale=sigmaU, size=soln[1][i].shape) for i in range(num_weights)]
                soln[1] = [soln[1][i] + GaussianNoise[i] for i in range(num_weights)]
                # gather solutions from client
                csolns.append(soln)
        
                # track communication cost
                self.metrics.update(rnd=i, cid=c.id, stats=stats)

            # update models
            #加噪得到self.noised_latest_model，要附带本轮选中client hold数据的信息，
            self.latest_model = self.aggregate(csolns)
            #暂时
            num_weights = len(self.latest_model)
            sigmaDi = self.get_sigmaDi(min_Di,num_Dk,min_Dk,csolns)
            #print("sigmaDi---",sigmaDi)
            for j in range(num_clients):
                GaussianNoise_i = [np.random.normal(loc=0.0, scale=sigmaDi[j], size=self.latest_model[j].shape) for j in range(num_weights)]
                GaussianNoise = [GaussianNoise[j] + csolns[j][0]/num_Dk*GaussianNoise_i[j] for j in range(num_weights)]
            
            #self.noised_latest_model = self.latest_model
            self.noised_latest_model = [self.latest_model[i] + GaussianNoise[i] for i in range(num_weights)]
            self.client_model.set_params(self.noised_latest_model)
            # test model
            if i % self.eval_every == 0:
                stats = self.test() # have set the latest model for all clients
                stats_train = self.train_error_and_loss()

                tqdm.write('At round {} accuracy: {}'.format(i, np.sum(stats[3])*1.0/np.sum(stats[2])))  # testing accuracy
                tqdm.write('At round {} training accuracy: {}'.format(i, np.sum(stats_train[3])*1.0/np.sum(stats_train[2])))
                tqdm.write('At round {} training loss: {}'.format(i, np.dot(stats_train[4], stats_train[2])*1.0/np.sum(stats_train[2])))
            
            #统计dissim------------------------------------------------------------------------------------------
            model_len = process_grad(self.noised_latest_model).size
            global_grads = np.zeros(model_len)
            client_grads = np.zeros(model_len)
            num_samples = []
            local_grads = []

            for c in self.clients:
                num, client_grad = c.get_grads(model_len)
                local_grads.append(client_grad)
                num_samples.append(num)
                global_grads = np.add(global_grads, client_grad * num)
            global_grads = global_grads * 1.0 / np.sum(np.asarray(num_samples))

            difference = 0
            for idx in range(len(self.clients)):
                difference += np.sum(np.square(global_grads - local_grads[idx]))
            difference = difference * 1.0 / len(self.clients)
            tqdm.write('gradient difference: {}'.format(difference))
            #-------------------------------------------------------------------------------------------------------

        # final test model
        stats = self.test()
        #train得到的全局模型给所有用户  用户进行测试
        stats_train = self.train_error_and_loss()
        self.metrics.accuracies.append(stats)
        self.metrics.train_accuracies.append(stats_train)
        tqdm.write('At round {} accuracy: {}'.format(self.num_rounds, np.sum(stats[3])*1.0/np.sum(stats[2])))
        tqdm.write('At round {} training accuracy: {}'.format(self.num_rounds, np.sum(stats_train[3])*1.0/np.sum(stats_train[2])))