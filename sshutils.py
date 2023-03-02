import os
import queue
import subprocess
import threading
from typing import Optional


def run_remote(cmd: str, host: str, working_dir: str):
    cmd = cmd.replace('"', '\\"').replace('$', '\\$')
    # ssh -o LogLevel=QUIET -xte none
    # should be the ideal solution to signal SIGHUP,
    # but it messes up the terminal for some reason.
    fullcmd = (
        f'ssh -x {host} "'
        '[[ -f \\$HOME/.bashrc ]] && source \\$HOME/.bashrc; '
        f'cd \\"{working_dir}\\"; '
        f'{cmd}'
        '" '
        f'2> >(xargs -I {{}} echo "{host}: {{}}" >&2) '
        f'| xargs -I {{}} echo "{host}: {{}}"'
    )
    return subprocess.call(fullcmd, shell=True, executable='/bin/bash')


def distribute(cmds: 'list[str]',
               hosts: 'list[str]',
               working_dir: Optional[str] = None):
    if working_dir is None:
        working_dir = home_relative_cwd()

    done = queue.Queue()

    def make_target(i, cmd, host):
        def target():
            run_remote(cmd, host, working_dir)
            done.put(i)
        return target

    pool = [threading.Thread(
        target=make_target(i, cmd, host),
    ) for i, (cmd, host) in enumerate(zip(cmds, hosts))]
    for t in pool:
        t.start()

    cmds = cmds[len(pool):]
    while len(cmds) > 0:
        i = done.get()
        pool[i].join()
        pool[i] = threading.Thread(
            target=make_target(i, cmds[0], hosts[i])
        )
        pool[i].start()
        cmds = cmds[1:]

    for t in pool:
        t.join()


def execute(cmd: str, hosts: 'list[str]', working_dir: Optional[str] = None):
    distribute([cmd]*len(hosts), hosts, working_dir)


def home_relative_cwd() -> str:
    """Returns cwd relative to home if cwd is in home, returns cwd otherwise.
    """
    cwd = os.getcwd()
    home = os.path.expanduser('~')
    if cwd.startswith(home):
        return os.path.relpath(cwd, home)
    return cwd
