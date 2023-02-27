#!/usr/bin/env python

# vim:ts=4:sw=4:expandtab

import os
import subprocess
import sys
from queue import Queue
from threading import Thread
from time import time


class Job:
    def __init__(self, no, cmd):
        self.no = str(no)
        self.cmd = str(cmd)

    def __str__(self):
        return f'[{self.no}] {self.cmd}'

    def get(self):
        return self.cmd


class WorkerStopToken:  # used to notify the worker to stop
    pass


class Worker(Thread):
    def __init__(self, name, job_queue, result_queue):
        Thread.__init__(self)
        self.name = name
        self.job_queue = job_queue
        self.result_queue = result_queue

    def run(self):
        while 1:
            job = self.job_queue.get()
            if job is WorkerStopToken:
                self.job_queue.put(job)
                # print 'worker %s stop.' % self.name
                break
            try:
                (job1, during) = self.run_one(job.get())
            except:
                # we failed, let others do that and we just quit
                self.job_queue.put(job)
                print(f'worker {self.name} quit.')
                break
            else:
                self.result_queue.put((self.name, job, during))


class SSHWorker(Worker):
    def __init__(self, name, job_queue, result_queue, host):
        Worker.__init__(self, name, job_queue, result_queue)
        self.host = host
        self.cwd = os.getcwd()
        home = os.path.expanduser('~')
        if self.cwd.startswith(home):
            self.cwd = self.cwd[len(home):]
            if self.cwd.startswith(os.sep):
                self.cwd = self.cwd[1:]

    def run_one(self, job):
        start = time()
        job = job.replace('"', '\\"')
        job = job.replace('$', '\\$')
        cmd = (
            f'ssh -x {self.host} "'
            '[[ -f \\$HOME/.bashrc ]] && source \\$HOME/.bashrc; '
            f'cd {self.cwd}; '
            f'{job}'
            '" '
            f'2> >(xargs -I {{}} echo "{self.host}: {{}}" >&2) '
            f'| xargs -I {{}} echo "{self.host}: {{}}"'
        )
        subprocess.call(cmd, shell=True, executable='/bin/bash')
        during = time() - start
        return (job, during)


class Grid:
    def __init__(self, workers, jobs):
        self.workers = workers
        self.jobs = []
        for i in range(len(jobs)):
            self.jobs.append(Job(i, jobs[i]))

        # put jobs in queue
        self.job_queue = Queue(0)
        self.result_queue = Queue(0)

        for i in range(len(self.jobs)):
            self.job_queue.put(self.jobs[i])

        # hack the queue to become a stack --
        # this is important when some thread
        # failed and re-put a job. If we still
        # use FIFO, the job will be put
        # into the end of the queue, and the graph
        # will only be updated in the end

        def _put(self, item):
            if sys.hexversion >= 0x020400A1:
                self.queue.appendleft(item)
            else:
                self.queue.insert(0, item)

        from types import MethodType as instancemethod
        self.job_queue._put = instancemethod(_put, self.job_queue)

    def go(self):
        elapsed_time = 0
        # fire ssh workers
        for host in self.workers:
            SSHWorker(host, self.job_queue, self.result_queue, host).start()

        # gather results

        done_jobs = {}

        for job in self.jobs:
            while job not in done_jobs:
                (worker, job1, during) = self.result_queue.get()
                elapsed_time = elapsed_time + during
                done_jobs[job1] = job1
                # print(f'[{worker}] ({during:.1f}s) {repr(job1.get())}')

        print(f'Elapsed Time (all workers): {elapsed_time:.1f}s')

        self.job_queue.put(WorkerStopToken)
