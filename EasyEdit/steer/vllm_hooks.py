_HOOK_ATTR = "_easyedit_activation_hook"
_ORIGINAL_ATTR = "_easyedit_activation_original"
_WRAPPER_ATTR = "_easyedit_activation_wrapper"
_METHODS_KEY = "methods"
_CALLS_KEY = "calls"


class _CallableLayerWrapper:
    def __init__(self, layer):
        self._easyedit_wrapped_layer = layer
        setattr(self, _WRAPPER_ATTR, True)
        setattr(self, _HOOK_ATTR, {_METHODS_KEY: {}})

    def __getattr__(self, name):
        return getattr(self._easyedit_wrapped_layer, name)

    def __call__(self, *args, **kwargs):
        output = self._easyedit_wrapped_layer(*args, **kwargs)
        return _add_configured_vectors(output, getattr(self, _HOOK_ATTR))


class InstallVllmCAAHooks:
    def __init__(self, layer_vectors, multipliers=None):
        self.layer_vectors = layer_vectors
        self.multipliers = multipliers

    def __call__(self, model):
        return install_vllm_caa_hooks(model, self.layer_vectors, self.multipliers)


class InstallVllmActivationHooks:
    def __init__(self, method_name, layer_vectors, multipliers=None):
        self.method_name = method_name
        self.layer_vectors = layer_vectors
        self.multipliers = multipliers

    def __call__(self, model):
        return install_vllm_activation_hooks(
            model, self.method_name, self.layer_vectors, self.multipliers
        )


class ClearVllmCAAHooks:
    def __init__(self, layers=None):
        self.layers = layers

    def __call__(self, model):
        return clear_vllm_caa_hooks(model, self.layers)


class ClearVllmActivationHooks:
    def __init__(self, method_name=None, layers=None):
        self.method_name = method_name
        self.layers = layers

    def __call__(self, model):
        return clear_vllm_activation_hooks(model, self.method_name, self.layers)


class InstallVllmCAAHooksOnWorker:
    def __init__(self, layer_vectors, multipliers=None):
        self.layer_vectors = layer_vectors
        self.multipliers = multipliers

    def __call__(self, worker):
        try:
            return install_vllm_caa_hooks(_worker_model(worker), self.layer_vectors, self.multipliers)
        except Exception:
            return False


class InstallVllmActivationHooksOnWorker:
    def __init__(self, method_name, layer_vectors, multipliers=None):
        self.method_name = method_name
        self.layer_vectors = layer_vectors
        self.multipliers = multipliers

    def __call__(self, worker):
        try:
            return install_vllm_activation_hooks(
                _worker_model(worker),
                self.method_name,
                self.layer_vectors,
                self.multipliers,
            )
        except Exception:
            return False


class ClearVllmCAAHooksOnWorker:
    def __init__(self, layers=None):
        self.layers = layers

    def __call__(self, worker):
        try:
            return clear_vllm_caa_hooks(_worker_model(worker), self.layers)
        except Exception:
            return False


class ClearVllmActivationHooksOnWorker:
    def __init__(self, method_name=None, layers=None):
        self.method_name = method_name
        self.layers = layers

    def __call__(self, worker):
        try:
            return clear_vllm_activation_hooks(
                _worker_model(worker), self.method_name, self.layers
            )
        except Exception:
            return False


class GetVllmCAAHookStatsOnWorker:
    def __call__(self, worker):
        try:
            return get_vllm_caa_hook_stats(_worker_model(worker))
        except Exception:
            return False


class GetVllmActivationHookStatsOnWorker:
    def __call__(self, worker):
        try:
            return get_vllm_activation_hook_stats(_worker_model(worker))
        except Exception:
            return False


def install_vllm_caa_hooks(model, layer_vectors, multipliers=None):
    """Install CAA vector addition on a vLLM-loaded model and return touched layers."""
    return install_vllm_activation_hooks(model, "caa", layer_vectors, multipliers)


def install_vllm_hooks(model, layer_vectors, multipliers=None, method_name="caa"):
    """Install activation additions on a vLLM-loaded model.

    Defaults to CAA for backward compatibility with the original hook tests.
    """
    return install_vllm_activation_hooks(model, method_name, layer_vectors, multipliers)


def install_vllm_activation_hooks(model, method_name, layer_vectors, multipliers=None):
    """Install activation additions on a vLLM-loaded model and return touched layers."""
    layers = _find_layers(model)
    multipliers = multipliers or {}
    installed = []

    for layer_idx, vector in layer_vectors.items():
        try:
            local_idx, layer = _resolve_layer(layers, layer_idx)
        except IndexError:
            continue
        layer = _ensure_layer_hook(layers, local_idx, layer)
        hook_state = getattr(layer, _HOOK_ATTR)
        method_state = _method_state(hook_state, method_name)
        method_state["vector"] = vector
        method_state["multiplier"] = multipliers.get(layer_idx, 1.0)
        method_state[_CALLS_KEY] = 0
        installed.append(layer_idx)

    return installed


def clear_vllm_caa_hooks(model, layers=None):
    """Clear EasyEdit CAA vectors from a vLLM-loaded model and return cleared layers."""
    return clear_vllm_activation_hooks(model, "caa", layers)


def clear_vllm_hooks(model, layers=None, method_name="caa"):
    """Clear EasyEdit activation additions from a vLLM-loaded model."""
    return clear_vllm_activation_hooks(model, method_name, layers)


def clear_vllm_activation_hooks(model, method_name=None, layers=None):
    """Clear EasyEdit activation additions from a vLLM-loaded model and return cleared layers."""
    model_layers = _find_layers(model)
    target_layers = range(len(model_layers)) if layers is None else layers
    cleared = []

    for layer_idx in target_layers:
        try:
            _, layer = _resolve_layer(model_layers, layer_idx)
        except IndexError:
            continue
        if hasattr(layer, _HOOK_ATTR):
            hook_state = getattr(layer, _HOOK_ATTR)
            methods = hook_state.setdefault(_METHODS_KEY, {})
            if method_name is None:
                methods.clear()
            elif method_name in methods:
                calls = methods[method_name].get(_CALLS_KEY, 0)
                methods[method_name].clear()
                methods[method_name][_CALLS_KEY] = calls
            else:
                continue
            cleared.append(layer_idx)

    return cleared


def get_vllm_caa_hook_stats(model):
    """Return call counters for layers touched by EasyEdit CAA hooks."""
    all_stats = get_vllm_activation_hook_stats(model, "caa")
    return {
        layer_idx: method_stats.get("caa", {"calls": 0, "configured": False})
        for layer_idx, method_stats in all_stats.items()
    }


def get_vllm_activation_hook_stats(model, method_name=None):
    """Return call counters for layers touched by EasyEdit activation hooks."""
    model_layers = _find_layers(model)
    stats = {}
    for layer_idx, layer in enumerate(model_layers):
        if hasattr(layer, _HOOK_ATTR):
            hook_state = getattr(layer, _HOOK_ATTR)
            methods = hook_state.setdefault(_METHODS_KEY, {})
            selected = methods if method_name is None else {method_name: methods.get(method_name, {})}
            stats[layer_idx] = {}
            for name, state in selected.items():
                stats[layer_idx][name] = {
                    "calls": state.get(_CALLS_KEY, 0),
                    "configured": "vector" in state,
                }
    return stats


def apply_caa_to_vllm_workers(llm, layer_vectors, multipliers=None):
    """Broadcast CAA hook installation to every vLLM worker through LLM.apply_model."""
    return apply_activation_additions_to_vllm_workers(llm, "caa", layer_vectors, multipliers)


def apply_activation_additions_to_vllm_workers(llm, method_name, layer_vectors, multipliers=None):
    """Broadcast activation hook installation to every vLLM worker through LLM.apply_model."""
    if not hasattr(llm, "apply_model"):
        raise ValueError("vLLM LLM object does not expose apply_model")
    return llm.apply_model(InstallVllmActivationHooks(method_name, layer_vectors, multipliers))


def clear_caa_from_vllm_workers(llm, layers=None):
    """Broadcast CAA hook clearing to every vLLM worker through LLM.apply_model."""
    return clear_activation_additions_from_vllm_workers(llm, "caa", layers)


def clear_activation_additions_from_vllm_workers(llm, method_name=None, layers=None):
    """Broadcast activation hook clearing to every vLLM worker through LLM.apply_model."""
    if not hasattr(llm, "apply_model"):
        raise ValueError("vLLM LLM object does not expose apply_model")
    return llm.apply_model(ClearVllmActivationHooks(method_name, layers))


def get_caa_hook_stats_with_vllm_rpc(llm, require_all=True):
    """Read CAA hook counters through vLLM collective_rpc worker calls."""
    engine = _collective_rpc_engine(llm)
    results = engine.collective_rpc(GetVllmCAAHookStatsOnWorker())
    if require_all:
        _raise_on_worker_failures(results, "stats")
    return results


def get_activation_hook_stats_with_vllm_rpc(llm, require_all=True):
    """Read activation hook counters through vLLM collective_rpc worker calls."""
    engine = _collective_rpc_engine(llm)
    results = engine.collective_rpc(GetVllmActivationHookStatsOnWorker())
    if require_all:
        _raise_on_worker_failures(results, "stats")
    return results


def install_caa_with_vllm_rpc(llm, layer_vectors, multipliers=None, require_all=True):
    """Install CAA hooks through vLLM collective_rpc worker calls."""
    return install_activation_hooks_with_vllm_rpc(
        llm, "caa", layer_vectors, multipliers, require_all
    )


def install_activation_hooks_with_vllm_rpc(
    llm, method_name, layer_vectors, multipliers=None, require_all=True
):
    """Install activation hooks through vLLM collective_rpc worker calls."""
    engine = _collective_rpc_engine(llm)
    results = engine.collective_rpc(
        InstallVllmActivationHooksOnWorker(method_name, layer_vectors, multipliers)
    )
    if require_all:
        _raise_on_worker_failures(results, "install")
    return results


def clear_caa_with_vllm_rpc(llm, layers=None, require_all=True):
    """Clear CAA hook state through vLLM collective_rpc worker calls."""
    return clear_activation_hooks_with_vllm_rpc(llm, "caa", layers, require_all)


def clear_activation_hooks_with_vllm_rpc(llm, method_name=None, layers=None, require_all=True):
    """Clear activation hook state through vLLM collective_rpc worker calls."""
    engine = _collective_rpc_engine(llm)
    results = engine.collective_rpc(ClearVllmActivationHooksOnWorker(method_name, layers))
    if require_all:
        _raise_on_worker_failures(results, "clear")
    return results


def _find_layers(model):
    if hasattr(model, "_decoder_layers") and callable(model._decoder_layers):
        return model._decoder_layers()

    queue = [model]
    seen = set()
    for _ in range(32):
        if not queue:
            break
        current = queue.pop(0)
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))

        layers = getattr(current, "layers", None)
        if layers is not None:
            return layers

        transformer = getattr(current, "transformer", None)
        if transformer is not None and hasattr(transformer, "h"):
            return transformer.h
        if hasattr(current, "h"):
            return current.h

        for attr in ("model", "language_model", "decoder", "transformer"):
            child = getattr(current, attr, None)
            if child is not None:
                queue.append(child)

    raise ValueError(
        "Unsupported vLLM model layout: could not locate decoder layers "
        "(expected model.layers, model.model.layers, transformer.h, or nested language_model layers)"
    )


def _collective_rpc_engine(llm):
    engine = getattr(llm, "llm_engine", llm)
    if not hasattr(engine, "collective_rpc"):
        raise ValueError("vLLM object does not expose llm_engine.collective_rpc")
    return engine


def _worker_model(worker):
    if hasattr(worker, "get_model") and callable(worker.get_model):
        return worker.get_model()
    model_runner = getattr(worker, "model_runner", None)
    if model_runner is not None and hasattr(model_runner, "model"):
        return model_runner.model
    if hasattr(worker, "model"):
        return worker.model
    raise ValueError("Unsupported vLLM worker layout: expected get_model(), model_runner.model, or model")


def _raise_on_worker_failures(results, action):
    failures = [
        index
        for index, result in enumerate(results)
        if isinstance(result, BaseException) or result is False or result is None
    ]
    if failures:
        raise RuntimeError(f"activation hook {action} failed on worker indices {failures}: {results}")


def _resolve_layer(layers, layer_idx):
    for local_idx, layer in enumerate(layers):
        global_idx = _layer_global_idx(layer, local_idx)
        if global_idx == layer_idx:
            return local_idx, layer
    if 0 <= layer_idx < len(layers):
        return layer_idx, layers[layer_idx]
    raise IndexError(f"Layer index {layer_idx} is outside the discovered decoder layers")


def _layer_global_idx(layer, fallback):
    for obj in (layer, getattr(layer, "_easyedit_wrapped_layer", None), getattr(layer, "self_attn", None)):
        if obj is None:
            continue
        for attr in ("layer_id", "layer_idx", "idx", "layer_number"):
            value = getattr(obj, attr, None)
            if isinstance(value, int):
                return value
    return fallback


def _ensure_layer_hook(layers, layer_idx, layer):
    if hasattr(layer, _HOOK_ATTR):
        return layer

    if hasattr(layer, "forward") and callable(layer.forward):
        original_forward = layer.forward

        def hooked_forward(*args, **kwargs):
            output = original_forward(*args, **kwargs)
            return _add_configured_vectors(output, getattr(layer, _HOOK_ATTR))

        setattr(layer, _ORIGINAL_ATTR, original_forward)
        setattr(layer, _HOOK_ATTR, {_METHODS_KEY: {}})
        layer.forward = hooked_forward
        return layer

    wrapped_layer = _CallableLayerWrapper(layer)
    layers[layer_idx] = wrapped_layer
    return wrapped_layer


def _add_configured_vectors(output, hook_state):
    methods = hook_state.setdefault(_METHODS_KEY, {})
    configured = [
        state for state in methods.values()
        if "vector" in state
    ]
    if not configured:
        return output

    hidden_states, tail = _split_layer_output(output)
    original_hidden_states = hidden_states

    for state in configured:
        state[_CALLS_KEY] = state.get(_CALLS_KEY, 0) + 1
        multiplier = state["multiplier"]
        if multiplier == 0:
            continue
        vector = _vector_like_hidden(state["vector"], hidden_states)
        hidden_states = hidden_states + (vector * multiplier)

    if hidden_states is original_hidden_states:
        return output
    if tail is None:
        return hidden_states
    return (hidden_states,) + tail


def _method_state(hook_state, method_name):
    methods = hook_state.setdefault(_METHODS_KEY, {})
    return methods.setdefault(method_name, {})


def _split_layer_output(output):
    if isinstance(output, tuple):
        return output[0], output[1:]
    return output, None


def _vector_like_hidden(vector, hidden_states):
    if hasattr(vector, "to"):
        to_kwargs = {}
        if hasattr(hidden_states, "device"):
            to_kwargs["device"] = hidden_states.device
        if hasattr(hidden_states, "dtype"):
            to_kwargs["dtype"] = hidden_states.dtype
        return vector.to(**to_kwargs)
    return vector
