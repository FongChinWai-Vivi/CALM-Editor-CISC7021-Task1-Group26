def uses_vllm(model):
    return getattr(model, "VLLM_model", None) is not None


def reset_vllm_activation_layers(model, method_name, layers):
    from ..vllm_hooks import clear_activation_hooks_with_vllm_rpc

    if not uses_vllm(model):
        return None
    clear_activation_hooks_with_vllm_rpc(
        model.VLLM_model, method_name=method_name, layers=layers, require_all=True
    )
    registry = getattr(model, "_easyedit_vllm_activation_hooks", None)
    if registry is not None:
        registry.pop(method_name, None)
    return model


def set_vllm_add_activations(model, method_name, layer_vectors):
    from ..vllm_hooks import install_activation_hooks_with_vllm_rpc

    if not uses_vllm(model):
        return None
    prepared = {
        layer: _cpu_vector(vector)
        for layer, vector in layer_vectors.items()
    }
    results = install_activation_hooks_with_vllm_rpc(
        model.VLLM_model,
        method_name=method_name,
        layer_vectors=prepared,
        multipliers=None,
        require_all=True,
    )
    installed_layers = {
        layer
        for worker_result in results
        if isinstance(worker_result, (list, tuple, set))
        for layer in worker_result
    }
    missing_layers = sorted(set(prepared) - installed_layers)
    if missing_layers:
        raise RuntimeError(
            f"vLLM activation hook for {method_name} did not install on layers {missing_layers}; "
            f"worker results were {results}"
        )
    registry = getattr(model, "_easyedit_vllm_activation_hooks", None)
    if registry is None:
        registry = {}
        setattr(model, "_easyedit_vllm_activation_hooks", registry)
    registry[method_name] = prepared
    return model


def clear_all_vllm_activation_hooks(model):
    from ..vllm_hooks import clear_activation_hooks_with_vllm_rpc

    if not uses_vllm(model):
        return {}
    registry = getattr(model, "_easyedit_vllm_activation_hooks", {})
    saved = dict(registry)
    if saved:
        clear_activation_hooks_with_vllm_rpc(
            model.VLLM_model, method_name=None, layers=None, require_all=True
        )
    return saved


def restore_vllm_activation_hooks(model, saved):
    if not uses_vllm(model) or not saved:
        return model
    for method_name, layer_vectors in saved.items():
        set_vllm_add_activations(model, method_name, layer_vectors)
    return model


def _cpu_vector(vector):
    if hasattr(vector, "detach"):
        vector = vector.detach()
    if hasattr(vector, "cpu"):
        vector = vector.cpu()
    return vector
