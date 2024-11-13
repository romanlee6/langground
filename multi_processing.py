import time
from utils import *
import torch
import torch.multiprocessing as mp
import os, threading

class MultiProcessWorker(mp.Process):
    # TODO: Make environment init threadsafe
    def __init__(self, id, trainer_maker, comm, seed, *args, **kwargs):
        self.id = id
        self.seed = seed
        super(MultiProcessWorker, self).__init__()
        self.trainer = trainer_maker()
        self.comm = comm


    def run(self):
        torch.manual_seed(self.seed + self.id + 1)
        np.random.seed(self.seed + self.id + 1)

        while True:
            task = self.comm.recv()

            if type(task) == list:
                if len(task) == 2:
                    task, epoch = task
                elif len(task) == 3:
                    task, success, num_episodes = task

            if task == 'quit':
                return
            elif task == 'run_batch':
                batch, stat = self.trainer.run_batch(epoch)
                self.trainer.optimizer.zero_grad()

                s = self.trainer.compute_grad(batch, other_stat=stat)

                merge_stat(s, stat)
                self.comm.send(stat)

            elif task == 'send_grads':
                grads = []
                for p in self.trainer.params:
                    if p._grad is not None:
                        grads.append(p._grad.data)

                self.comm.send(grads)



class MultiProcessTrainer(object):
    def __init__(self, args, trainer_maker):
        self.comms = []
        self.trainer = trainer_maker()
        # itself will do the same job as workers
        self.nworkers = args.nprocesses - 1
        for i in range(self.nworkers):
            comm, comm_remote = mp.Pipe()
            self.comms.append(comm)
            worker = MultiProcessWorker(i, trainer_maker, comm_remote, seed=args.seed)
            worker.start()
        self.grads = None
        self.worker_grads = None
        self.is_random = args.random
        self.best_model_reward = -np.inf

    def quit(self):
        for comm in self.comms:
            comm.send('quit')

    def obtain_grad_pointers(self):
        # only need perform this once
        if self.grads is None:
            self.grads = []
            for p in self.trainer.params:
                if p._grad is not None:
                    self.grads.append(p._grad.data)

        if self.worker_grads is None:
            self.worker_grads = []
            for comm in self.comms:
                comm.send('send_grads')
                self.worker_grads.append(comm.recv())


    def train_batch(self, epoch):
        # run workers in parallel
        for comm in self.comms:
            comm.send(['run_batch', epoch])


        # run its own trainer
        st = time.time()
        # print(f"multi process run batch called")
        batch, stat = self.trainer.run_batch(epoch)

        # print("time taken for data collection computation ", time.time() - st)

        grad_st = time.time()
        self.trainer.optimizer.zero_grad()
        s = self.trainer.compute_grad(batch, other_stat=stat)

        # print(f"time for grad {time.time() - grad_st}")

        merge_stat(s, stat)
        #print(f"merge stat completed")

        # check if workers are finished
        for comm in self.comms:
            #print('a')
            s = comm.recv()
            #print('b')
            merge_stat(s, stat)
            #print('c')
        #print(f"other merge stat completed")


        # add gradients of workers
        self.obtain_grad_pointers()
        for i in range(len(self.grads)):
            for g in self.worker_grads:
                self.grads[i] += g[i]
            self.grads[i] /= stat['num_steps']

        # print(f"calling step")
        op_st = time.time()
        self.trainer.optimizer.step()
        if self.trainer.args.scheduleLR:
            self.trainer.scheduler.step()

        stat['learning_rate'] = self.trainer.get_lr(self.trainer.optimizer)
        return stat

    def state_dict(self):
        return self.trainer.state_dict()

    def setup_var_reload(self):
        return self.trainer.setup_var_reload()

    def load_state_dict(self, state):
        self.trainer.load_state_dict(state)

    def load_scheduler(self, start_epoch):
        self.trainer.load_scheduler(start_epoch)
