import logging
import multiprocessing as mp
import time
from queue import Empty


# Worker:
# 1. read data from data queue and build batch
# 2. batch inference 
# 3. write results into result queue
class BaseInferLightWorker:
    def __init__(self, data_queue:mp.Queue, result_queue:mp.Queue, 
                 model_args:dict, 
                 batch_size=16, max_delay=0.1,
                 ready_event=None) -> None:
        self.data_queue = data_queue
        self.result_queue = result_queue
        self.batch_size = batch_size
        self.max_delay = max_delay
        self.logger = logging.getLogger('InferLight-Worker')
        self.logger.setLevel(logging.DEBUG)

        self.load_model(model_args)
        
        # 通知主进程，模型加载完成
        if ready_event:
            ready_event.set()

    def run(self):
        self.logger.info('Worker started!')
        while True:
            data, task_ids = [], []
            since = time.time()
            for i in range(self.batch_size):
                try:
                    d = self.data_queue.get(block=True, timeout=self.max_delay)
                    task_ids.append(d[0])
                    data.append(d[1])
                    self.logger.info('get one new task')
                except Empty:
                    pass
                if time.time()-since>=self.max_delay:
                    break

            # 对批量数据进行处理
            if len(data)>0:
                start = time.perf_counter()
                batch = self.build_batch(data)
                start_token = time.perf_counter()
                results = self.inference(batch)
                end = time.perf_counter()
                token_time_elapsed = (start_token - start)*1000
                infer_time_elapsed = (end - start_token)*1000
                self.logger.info(f'tokenize succeeded. batch size: {len(data)}, time elapsed: {token_time_elapsed:.3f} ms')
                self.logger.info(f'inference succeeded. batch size: {len(data)}, time elapsed: {infer_time_elapsed:.3f} ms')
                for (task_id, result) in zip(task_ids, results):
                    self.result_queue.put((task_id, result))


    def build_batch(self, requests):
        raise NotImplementedError

    def inference(self, batch):
        raise NotImplementedError

    def load_model(self, model_args):
        raise NotImplementedError

    @classmethod
    def start(cls, data_queue:mp.Queue, result_queue:mp.Queue, model_args:dict, batch_size=16, max_delay=0.1,ready_event=None):
        w = cls(data_queue, result_queue, model_args, batch_size, max_delay, ready_event)
        w.run()
