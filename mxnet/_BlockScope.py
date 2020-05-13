import threading

from mxnet import name as _name
from mxnet import profiler as _profiler
from mxnet.gluon.parameter import ParameterDict

class _BlockScope:

    """ Scope for collecting child `Block` s. """
    # 스레드별로 구분되는 네임스페이스를 제공
    _current = threading.local()

    def __init__(self, block):
        self._block = block
        self._counter = {}
        self._old_scope = None
        self._name_scope = None

    @staticmethod
    def create(prefix, params, hint):
        """
        새로운 `Block`을 위한
        prefix, params, 그리고 profiler scope name을 생성,
        profiler scope는 GPU 메모리 profiler를 지원.
        """
        current = getattr(_BlockScope._current, "value", None)
        if current is None:
            if prefix is None:
                if not hasattr(_name.NameManager._current, "value"):
                    _name.NameManager._current.value = _name.NameManager()
                _name.NameManager._current.value.get(None, hint) + '_'
            # replace the trailing underscore with colon
            profiler_scope_name = (
                prefix[:-1] if prefix.endswith('_') else prefix) + ":"
            if params is None:
                params = ParameterDict(prefix)
            else:
                params = ParameterDict(params.prefix, params)
            return prefix, params, profiler_scope_name
        
        if prefix is None:
            count = current._counter.get(hint, 0)
            prefix = f"{hint}{count}_"
        if params is None:
            parent = current._block.params
            params = ParameterDict(parent.prefix + prefix, parent._shared)
        else:
            params = ParameterDict(params.prefix, params)
        # replace the trailing underscore with colon
        profiler_scope_name = (
            prefix[:-1] if prefix.endswith('_') else prefix) + ":"
        return (current._block.prefix + prefix, params,
                current._block._profiler_scope_name + profiler_scope_name)

    def __enter__(self):
        if self._block._empty_prefix:
            return self
        self._old_scope = getattr(_BlockScope._current, "value", None)
        _BlockScope._current.value = self
        self._name_scope = _name.Prefix(self._block.prefix)
        self._name_scope.__enter__()
        self._profiler_scope = _profiler.Scope(self._block._profiler_scope_name)
        self._profiler_scope.__enter__()
        return self

    def __exit__(self, ptype, value, trace):
        if self._block._empty_prefix:
            return
        self._name_scope.__exit__(ptype, value, trace)
        self._name_scope = None
        self._profiler_scope.__exit__(ptype, value, trace)
        self._profiler_scope = None
        _BlockScope._current.value = self._old_scope
