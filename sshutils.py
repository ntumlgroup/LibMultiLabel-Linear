import os
import queue
import subprocess
import threading

# TODO: propogate local signals to remote hosts.


def home_relative_cwd() -> str:
    """Returns cwd relative to home if cwd is under home, returns cwd otherwise.
    """
    cwd = os.getcwd()
    home = os.path.expanduser('~')
    if cwd.startswith(home):
        return os.path.relpath(cwd, home)
    return cwd


def run_remote(cmd: str, host: str, working_dir: str) -> int:
    """Low level function to execute cmd on host.

    Args:
        cmd (str): Shell command.
        host (str): ssh host name.
        working_dir (str): Working directory.

    Returns:
        int: exit status of the local ssh command.
    """
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
               working_dir: str = home_relative_cwd()):
    """Distribute cmds on hosts. Each host executes cmds sequentially.

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
            run_remote(cmd, host, working_dir)
            done.put(i)
        return threading.Thread(target=target)

    pool = [make_thread(i, cmd, host)
            for i, (cmd, host) in enumerate(zip(cmds, hosts))]
    for t in pool:
        t.start()

    cmds = cmds[len(pool):]
    while len(cmds) > 0:
        i = done.get()
        pool[i].join()
        pool[i] = make_thread(i, cmds[0], hosts[i])
        pool[i].start()
        cmds = cmds[1:]

    for t in pool:
        t.join()


def execute(cmd: str, hosts: 'list[str]', working_dir: str = home_relative_cwd()):
    """Executes cmd on every host.

    Args:
        cmd (str): Shell command.
        hosts (list[str]): List of ssh host names.
        working_dir (str, optional): Working directory. Defaults to home_relative_cwd().
    """
    distribute([cmd]*len(hosts), hosts, working_dir)
