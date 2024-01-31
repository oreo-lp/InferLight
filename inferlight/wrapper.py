import asyncio
import logging
import multiprocessing as mp
import threading
import uuid
from queue import Empty

from cachetools import TTLCache

from .data import InferStatus, InferResponse

# mp.set_start_method("spawn", force= True)

class LightWrapper:

    def __init__(self, worker_class, model_args: dict,
                 batch_size=16, max_delay=0.1) -> None:
        # setup logger
        self.logger = logging.getLogger('InferLight-Wrapper')
        self.logger.setLevel(logging.INFO)

        # 结果缓存对象，用于缓存模型推理的结果
        self.result_cache = TTLCache(maxsize=10000, ttl=5)  # 过去时间为5s

        self.mp = mp.get_context('spawn')
        self.data_queue = self.mp.Queue()
        self.result_queue = self.mp.Queue()

        # setup worker
        self.logger.info('Starting worker...')
        worker_ready_event = self.mp.Event()
        # 开辟多线程进行模型的推理
        self._worker_p = self.mp.Process(target=worker_class.start, args=(
            self.data_queue, self.result_queue, model_args, batch_size, max_delay, worker_ready_event
        ), daemon=True)
        self._worker_p.start()
        is_ready = worker_ready_event.wait(timeout=30)
        if is_ready:
            self.logger.info('Worker started!')
        else:
            self.logger.error('Failed to start worker!')

        # 推理结果收集线程
        self.back_thread = threading.Thread(
            target=self._collect_result, name="thread_collect_result")
        self.back_thread.daemon = True
        self.back_thread.start()

    # 将数据保存到缓存中，一旦缓存中的数据存在时间超过5s
    # 缓存中的数据就会被清空
    def _collect_result(self):
        self.logger.info('Result collecting thread started!')
        while True:
            try:
                msg = self.result_queue.get(block=True, timeout=0.01)
            except Empty:
                msg = None
            if msg is not None:
                (task_id, result) = msg
                self.result_cache[task_id] = result

    async def get_result(self, task_id):
        # 直到有推理结果的时候才返回，不然就卡死
        while task_id not in self.result_cache:
            await asyncio.sleep(0.01)
        return self.result_cache[task_id]

    async def predict(self, input, timeout=2) -> InferResponse:
        # generate unique task_id
        task_id = str(uuid.uuid4())

        # send input to worker process
        self.data_queue.put((task_id, input))
        try:
            result = await asyncio.wait_for(self.get_result(task_id), timeout=timeout)
        except asyncio.TimeoutError:
            return InferResponse(InferStatus.TIMEOUT, None)

        return InferResponse(InferStatus.SUCCEED, result)
