from __future__ import absolute_import, division, print_function, unicode_literals

# Define possible errors in simulator


def _assert(cand, note=""):
    if not cand:
        raise ValueError(note)


class ModelBuildError(Exception):
    def __init__(self, msg='Not defined'):
        self.msg = 'MIDAP MODEL Build Error: ' + msg

    def __str__(self):
        return self.msg


class ControlManagerError(Exception):
    def __init__(self, msg='Not defined'):
        self.msg = 'MIDAP Control Sequence Generation Error: ' + msg

    def __str__(self):
        return self.msg


class MIDAPError(Exception):
    def __init__(self, msg='Not defined'):
        self.msg = 'MIDAP Hardware simulator Error: ' + msg

    def __str__(self):
        return self.msg


class DataInfoTableError(Exception):
    def __init__(self, msg='Not defined'):
        self.msg = 'MIDAP Data Info Table Setup Error: ' + msg

    def __str__(self):
        return self.msg


class CompilerError(Exception):
    def __init__(self, msg='Not defined'):
        self.msg = 'MIDAP Compiler Error: ' + msg

    def __str__(self):
        return self.msg
