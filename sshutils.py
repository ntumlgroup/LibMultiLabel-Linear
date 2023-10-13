from __future__ import annotations

import os
import queue
import shlex
import subprocess
import sys
import threading
import signal
import uuid
from typing import Optional


__all__ = ["distribute", "execute", "home_relative_cwd", "propogate_signal"]

ongoing_groups = {}


def home_relative_cwd() -> str:
    """Returns cwd relative to home if cwd is under home, returns cwd otherwise."""
    cwd = os.getcwd()
    home = os.path.expanduser("~")
    if cwd.startswith(home):
        return os.path.relpath(cwd, home)
    return cwd


def distribute(cmds: list[str], hosts: list[str], working_dir: str = home_relative_cwd()):
    """Distribute cmds on hosts, giving each host a disjoint subset to execute.
    Each host executes its subset sequentially.
    Before each cmd is executed, sources $HOME/.bash_init if it exist.

    Args:
        cmds (list[str]): List of shell commands.
        hosts (list[str]): List of ssh host names.
        working_dir (str, optional): Working directory. Defaults to home_relative_cwd().
    """
    if working_dir is None:
        working_dir = home_relative_cwd()

    done = queue.Queue()

    def make_thread(i, cmd, host):
        def target():
            run_remote(cmd, host, working_dir, gid)
            done.put(i)

        return threading.Thread(target=target)

    pool = [make_thread(i, cmd, host) for i, (cmd, host) in enumerate(zip(cmds, hosts))]

    gid = uuid.uuid4().hex
    try:
        ongoing_groups[gid] = hosts
        for t in pool:
            t.start()

        cmds = cmds[len(pool) :]
        while len(cmds) > 0:
            i = done.get()
            pool[i].join()
            pool[i] = make_thread(i, cmds[0], hosts[i])
            pool[i].start()
            cmds = cmds[1:]

        for t in pool:
            t.join()

    finally:
        ongoing_groups.pop(gid)


def execute(cmd: str, hosts: list[str], working_dir: str = home_relative_cwd()):
    """Executes cmd on every host.
    Before each cmd is executed, sources $HOME/.bash_init if it exist.

    Args:
        cmd (str): Shell command.
        hosts (list[str]): List of ssh host names.
        working_dir (str, optional): Working directory. Defaults to home_relative_cwd().
    """
    distribute([cmd] * len(hosts), hosts, working_dir)


def run_remote_raw(cmd: str, host: str) -> int:
    """Runs cmd on host with correct quotation and prefixes output with host name.

    Args:
        cmd (str): Shell command.
        host (str): ssh host name.

    Returns:
        int: exit status of the local ssh command.
    """
    fullcmd = (
        f"ssh -x {host} {shlex.quote(cmd)} "
        f"2> >(xargs -I {{}} echo {host}: {{}} >&2) "
        f"| xargs -I {{}} echo {host}: {{}}"
    )
    return subprocess.call(fullcmd, shell=True, executable="/bin/bash")


def run_remote(cmd: str, host: str, working_dir: str, gid: str) -> int:
    """Low level function to execute cmd on host. Wraps cmd with housekeeping
    commands which handles gid, .bash_init and changing working directory.

    Args:
        cmd (str): Shell command.
        host (str): ssh host name.
        working_dir (str): Working directory.
        gid (str): gid representing a group of cmds.

    Returns:
        int: exit status of the local ssh command.
    """
    remotecmd = (
        'mkdir -p "$HOME"/.sshutils; '
        f'echo $$ > "$HOME"/.sshutils/{gid}; '
        '[[ -f "$HOME"/.bash_init ]] && . "$HOME"/.bash_init; '
        f'cd "{working_dir}"; '
        f"{cmd}; "
        f'rm -f "$HOME"/.sshutils/{gid}'
    )
    return run_remote_raw(remotecmd, host)


def propogate_signal(handlers: Optional[tuple] = None) -> Optional[tuple]:
    """Sets or unsets signal handlers to propogate signals to remote hosts.
    If argument is None, sets signal propogating handlers and returns previous handlers.
    Pass previously returned handlers as arguments to unset.

    Returns:
        Optional[tuple]: Previous handlers if argument is None.
    """
    if handlers is None:

        def handler(sig, frame):
            kill_remote()
            sys.exit(sig)

        sigint = signal.signal(signal.SIGINT, handler)
        sigterm = signal.signal(signal.SIGTERM, handler)
        sighup = signal.signal(signal.SIGHUP, handler)
        return (sigint, sigterm, sighup)
    else:
        signal.signal(signal.SIGINT, handlers[0])
        signal.signal(signal.SIGTERM, handlers[1])
        signal.signal(signal.SIGHUP, handlers[2])


def kill_remote():
    for gid, hosts in ongoing_groups.items():
        cmd = (
            f'[[ -f "$HOME"/.sshutils/{gid} ]] && '
            f'kill -s {int(signal.SIGHUP)} $(cat "$HOME"/.sshutils/{gid}) && '
            f'rm -f "$HOME"/.sshutils/{gid}'
        )
        for host in hosts:
            run_remote_raw(cmd, host)
