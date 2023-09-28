import sys
from subprocess import run
import os
import inotify.adapters
from signal import signal, SIGINT


def handler(signal_received, frame):
    os.system("sudo /opt/lampp/lampp stop")
    exit(0)


if __name__ == '__main__':

    signal(SIGINT, handler)

    notifier = inotify.adapters.Inotify()
    notifier.add_watch('/opt/lampp/htdocs/testing/uploads/')

    os.system("pkill apache2")
    os.system("service mysql stop")
    os.system("sudo /opt/lampp/lampp start")

    for event in notifier.event_gen():
        if event is not None:
            if 'IN_MOVED_TO' in event[1]:
                if str(event[3]).endswith('.wav'):
                    print("file '{0}' created in '{1}'".format(event[3], event[2]))
                    run([sys.executable, "actual_project.py", str(event[2] + event[3])])
                    os.remove(str(event[2] + event[3]))
                else:
                    print("file '{0}' created in '{1}'".format(event[3], event[2]))
                    print("File with different extension detected thus deleting...")
                    os.remove("uploads/" + event[3])
