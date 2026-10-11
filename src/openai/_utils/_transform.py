from __future__ import annotations

import io
import base64
import pathlib
from typing import Any, Union, Mapping, TypeVar, cast
from datetime import date, datetime
from typing_extensions import Literal, get_args, override, get_type_hints as _get_type_hints

import anyio
import pydantic

from ._utils import (
    is_list,
    is_given,
    lru_cache,
    is_mapping,
    is_iterable,
    is_sequence,
)
from .._files import is_base64_file_input
from ._compat import get_origin, is_typeddict
from ._typing import (
    is_list_type,
    is_union_type,
    is_iterable_type,
    is_required_type,
    is_sequence_type,
    is_annotated_type,
    is_not_required_type,
    strip_annotated_type,
)

_T = TypeVar("_T")


# TODO: support for drilling globals() and locals()
# TODO: ensure works correctly with forward references in all cases


PropertyFormat = Literal["iso8601", "base64", "custom"]


class PropertyInfo:
    """Metadata class to be used in Annotated types to provide information about a given type.

    For example:

    class MyParams(TypedDict):
        account_holder_name: Annotated[str, PropertyInfo(alias='accountHolderName')]

    This means that {'account_holder_name': 'Robert'} will be transformed to {'accountHolderName': 'Robert'} before being sent to the API.
    """

    alias: str | None
    format: PropertyFormat | None
    format_template: str | None
    discriminator: str | None

    def __init__(
        self,
        *,
        alias: str | None = None,
        format: PropertyFormat | None = None,
        format_template: str | None = None,
        discriminator: str | None = None,
    ) -> None:
        self.alias = alias
        self.format = format
        self.format_template = format_template
        self.discriminator = discriminator

    @override
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(alias='{self.alias}', format={self.format}, format_template='{self.format_template}', discriminator='{self.discriminator}')"


def maybe_transform(
    data: object,
    expected_type: object,
) -> Any | None:
    """Wrapper over `transform()` that allows `None` to be passed.

    See `transform()` for more details.
    """
    if data is None:
        return None
    return transform(data, expected_type)


# Wrapper over _transform_recursive providing fake types
def transform(
    data: _T,
    expected_type: object,
) -> _T:
    """Transform dictionaries based off of type information from the given type, for example:

    ```py
    class Params(TypedDict, total=False):
        card_id: Required[Annotated[str, PropertyInfo(alias="cardID")]]


    transformed = transform({"card_id": "<my card ID>"}, Params)
    # {'cardID': '<my card ID>'}
    ```

    Any keys / data that does not have type information given will be included as is.

    It should be noted that the transformations that this function does are not represented in the type system.
    """
    transformed = _transform_recursive(data, annotation=cast(type, expected_type))
    return cast(_T, transformed)


@lru_cache(maxsize=8096)
def _get_annotated_type(type_: type) -> type | None:
    """If the given type is an `Annotated` type then it is returned, if not `None` is returned.

    This also unwraps the type when applicable, e.g. `Required[Annotated[T, ...]]`
    """
    if is_required_type(type_) or is_not_required_type(type_):
        # Unwrap Required or NotRequired to expose the Annotated metadata
        type_ = get_args(type_)[0]

    if is_annotated_type(type_):
        return type_

    return None


@lru_cache(maxsize=8096)
def _annotation_format_free(annotation: object) -> bool:
    """Whether the leaf branch could never fire a format conversion for this annotation.

    The leaf branch of `_transform_recursive` inspects `annotation` — not
    `inner_type` — for `PropertyInfo(format=...)`. The two can differ after
    union expansion (the union branch recurses with the outer annotation and
    each subtype as the inner type), so the fast path must additionally verify
    the annotation itself is format-free.
    """
    annotated_type = _get_annotated_type(annotation)
    if annotated_type is None:
        return True
    # ignore the first argument as it is the actual type
    return not any(isinstance(meta, PropertyInfo) and meta.format is not None for meta in get_args(annotated_type)[1:])


def _fast_path_ok(annotation: object, inner_type: object, plan: _TypePlan) -> bool:
    if plan[1]:  # has_property_info
        return False
    if annotation is inner_type:
        return True
    return _annotation_format_free(annotation)


@lru_cache(maxsize=8096)
def _has_property_info(root: object) -> bool:
    """Whether the type subtree contains any `PropertyInfo` with an alias or format set.

    This is computed once per type and cached so the per-request transform walk can
    skip type introspection entirely for the common case of plain data shapes.

    Note: `discriminator` is intentionally ignored here as it is only consumed
    during response parsing (see `_models.py`), never during request transforms.

    The traversal is iterative with a `seen` set so recursive type definitions
    cannot cause infinite recursion.
    """
    seen: set[object] = set()
    stack = [root]
    while stack:
        type_ = stack.pop()
        try:
            if type_ in seen:
                continue
            seen.add(type_)
        except TypeError:
            # Unhashable annotation; be conservative and assume a transform is needed.
            return True

        if is_required_type(type_) or is_not_required_type(type_):
            args = get_args(type_)
            if args:
                stack.append(args[0])
            continue

        if is_annotated_type(type_):
            args = get_args(type_)
            for meta in args[1:]:
                if isinstance(meta, PropertyInfo) and (meta.alias is not None or meta.format is not None):
                    return True
            if args:
                stack.append(args[0])
            continue

        if is_typeddict(type_):
            stack.extend(get_type_hints(type_, include_extras=True).values())
            continue

        if is_union_type(type_):
            stack.extend(get_args(type_))
            continue

        if get_origin(type_) is not None:
            stack.extend(get_args(type_))

    return False


# A precomputed per-type dispatch plan for the transform walk.
# (kind, has_pi, extra, list_flags, subplans). `kind` is one of "typeddict",
# "dict", "list", "union", "leaf", mirroring the branch order in
# `_transform_recursive`. `extra` is the TypedDict hints dict, the dict items
# type, the list inner type, or the union subtypes, depending on `kind`.
# `list_flags` records which list/iterable/sequence predicates matched.
# `subplans` holds already-resolved child plans so the hot loop performs no
# cache lookups or type hashing at all: for "typeddict" a {key: plan} dict,
# for "dict"/"list" a single plan, for "union" a tuple of plans aligned with
# `extra`, else None. `_LAZY` marks a recursive type edge; it falls back to a
# regular dispatcher call (a cache hit by then) at runtime.
_TypePlan = tuple[str, bool, object, tuple[bool, bool, bool] | None, object]

_LAZY: Any = object()


def _is_mergeable_union_member(subtype: object) -> bool:
    """Whether a union subtype can be folded into a merged TypedDict plan.

    Only TypedDicts (key renames are impossible without `PropertyInfo`, so a
    merged key set is exactly equivalent to sequential passes — see `_type_plan`)
    and `NoneType` (contributes no keys) qualify.
    """
    return subtype is type(None) or is_typeddict(strip_annotated_type(subtype))


# Tracks types whose plan is currently being built, so recursive type
# definitions resolve to `_LAZY` instead of recursing forever.
_expanding_plans: set[object] = set()


def _type_plan(inner_type: object) -> _TypePlan:
    """Return the cached dispatch plan for a type.

    The `_LAZY` sentinel is returned (and never cached) for recursive type
    edges; callers fall back to a regular dispatcher call for those.
    """
    try:
        if inner_type in _expanding_plans:
            return _LAZY  # type: ignore[return-value]
    except TypeError:
        # Unhashable annotation; propagate to the caller's slow-path fallback.
        raise
    return _type_plan_cached(inner_type)


@lru_cache(maxsize=8096)
def _type_plan_cached(inner_type: object) -> _TypePlan:
    _expanding_plans.add(inner_type)
    try:
        return _build_plan(inner_type)
    finally:
        _expanding_plans.discard(inner_type)


def _build_plan(inner_type: object) -> _TypePlan:
    """Build the dispatch plan for a type (one typing-introspection pass)."""
    stripped_type = strip_annotated_type(inner_type)
    has_pi = _has_property_info(inner_type)
    if is_typeddict(stripped_type):
        hints = get_type_hints(stripped_type, include_extras=True)
        return ("typeddict", has_pi, hints, None, {key: _type_plan(field_type) for key, field_type in hints.items()})
    if (get_origin(stripped_type) or stripped_type) == dict:
        args = get_args(stripped_type)
        items_type = args[1] if len(args) > 1 else object
        return ("dict", has_pi, items_type, None, _type_plan(items_type))
    list_flags = (is_list_type(stripped_type), is_iterable_type(stripped_type), is_sequence_type(stripped_type))
    if any(list_flags):
        args = get_args(stripped_type)
        inner = args[0] if args else object
        return ("list", has_pi, inner, list_flags, _type_plan(inner))
    if is_union_type(stripped_type):
        subtypes = get_args(stripped_type)
        if not has_pi and subtypes and all(_is_mergeable_union_member(s) for s in subtypes):
            # All members are TypedDicts without `PropertyInfo` metadata (or
            # NoneType). Sequentially applying each member's transform is exactly
            # equivalent to a single pass with the merged key set, because for
            # metadata-free types every pass only performs idempotent data-driven
            # jobs, and a key annotated by several members is processed as
            # `G_Tn(...G_T2(G_T1(v)))` — precisely what the union branch itself
            # does for a merged `Union[T1, ..., Tn]` field type. This collapses
            # the per-value union fan-out that dominates request bodies built
            # from large unions (e.g. response `input` items).
            merged: dict[str, list[object]] = {}
            for subtype in subtypes:
                if subtype is type(None):
                    continue
                for key, field_type in get_type_hints(strip_annotated_type(subtype), include_extras=True).items():
                    merged.setdefault(key, []).append(field_type)
            hints = {key: types[0] if len(types) == 1 else Union[tuple(types)] for key, types in merged.items()}
            return (
                "typeddict",
                False,
                hints,
                None,
                {key: _type_plan(field_type) for key, field_type in hints.items()},
            )
        return ("union", has_pi, subtypes, None, tuple(_type_plan(s) for s in subtypes))
    return ("leaf", has_pi, None, None, None)


def _transform_fast(data: object, plan: _TypePlan, annotation: object) -> object:
    """Fast path for types whose subtree holds no `PropertyInfo` metadata.

    Key renames and format conversions are impossible by construction here; the
    only remaining jobs are data-driven ones also performed by the slow path:
    BaseModel dumping, NOT_GIVEN stripping (TypedDicts) and iterable
    materialization. Unannotated TypedDict keys are kept as-is, mirroring
    `_transform_typeddict`.

    Child plans are threaded through from `plan`, so no cache lookups or type
    hashing happen in the hot loop. `annotation` is only used for `_LAZY`
    (recursive type edge) fallbacks.
    """
    from .._compat import model_dump

    kind, _, extra, list_flags, subplans = plan

    if kind == "typeddict" and is_mapping(data):
        result: dict[str, object] = {}
        for key, value in data.items():
            if not is_given(value):
                continue
            subplan = subplans.get(key)
            if subplan is None:
                result[key] = value
            elif subplan is _LAZY:
                result[key] = _transform_recursive(value, annotation=extra[key])
            else:
                result[key] = _transform_fast(value, subplan, extra[key])
        return result

    if kind == "dict" and is_mapping(data):
        if subplans is _LAZY:
            return {key: _transform_recursive(value, annotation=extra) for key, value in data.items()}
        return {key: _transform_fast(value, subplans, extra) for key, value in data.items()}

    if kind == "list":
        assert list_flags is not None
        is_l, is_i, is_s = list_flags
        if (
            (is_l and is_list(data))
            or (is_i and is_iterable(data) and not isinstance(data, str))
            or (is_s and is_sequence(data) and not isinstance(data, str))
        ):
            # dicts are technically iterable, but it is an iterable on the keys of the dict
            # and is not usually intended as an iterable, so we don't transform it.
            if isinstance(data, dict):
                return data

            inner_type = cast(type, extra)
            if _no_transform_needed(inner_type):
                # for some types there is no need to transform anything, so we can get a small
                # perf boost from skipping that work.
                #
                # but we still need to convert to a list to ensure the data is json-serializable
                if is_list(data):
                    return data
                return list(data)

            if subplans is _LAZY:
                return [_transform_recursive(d, annotation=annotation, inner_type=inner_type) for d in data]
            return [_transform_fast(d, subplans, inner_type) for d in data]
        # fall through to the union / BaseModel / leaf handling below

    if kind == "union":
        for subtype, subplan in zip(cast(tuple[object, ...], extra), subplans, strict=True):
            if subplan is _LAZY:
                data = _transform_recursive(data, annotation=annotation, inner_type=subtype)
            else:
                # The gate held for (annotation, union), so it holds for each
                # metadata-free subtype with the same annotation: bypass it.
                data = _transform_fast(data, subplan, annotation)
        return data

    if isinstance(data, pydantic.BaseModel):
        return model_dump(data, exclude_unset=True, mode="json", exclude=getattr(data, "__api_exclude__", None))

    # Leaf: by construction of the fast path, neither `annotation` nor the
    # inner type can carry `PropertyInfo` metadata here (see `_fast_path_ok`),
    # so the original leaf logic — which only acts on
    # `PropertyInfo(format=...)` — can never fire.
    return data


async def _async_transform_fast(data: object, plan: _TypePlan, annotation: object) -> object:
    """Async variant of `_transform_fast`.

    Mirrors `_async_transform_recursive`, including its quirk of delegating
    `dict` values to the synchronous `_transform_recursive`.
    """
    from .._compat import model_dump

    kind, _, extra, list_flags, subplans = plan

    if kind == "typeddict" and is_mapping(data):
        result: dict[str, object] = {}
        for key, value in data.items():
            if not is_given(value):
                continue
            subplan = subplans.get(key)
            if subplan is None:
                result[key] = value
            elif subplan is _LAZY:
                result[key] = await _async_transform_recursive(value, annotation=extra[key])
            else:
                result[key] = await _async_transform_fast(value, subplan, extra[key])
        return result

    if kind == "dict" and is_mapping(data):
        if subplans is _LAZY:
            return {key: _transform_recursive(value, annotation=extra) for key, value in data.items()}
        # NB: the sync variant is used here to mirror `_async_transform_recursive`.
        return {key: _transform_fast(value, subplans, extra) for key, value in data.items()}

    if kind == "list":
        assert list_flags is not None
        is_l, is_i, is_s = list_flags
        if (
            (is_l and is_list(data))
            or (is_i and is_iterable(data) and not isinstance(data, str))
            or (is_s and is_sequence(data) and not isinstance(data, str))
        ):
            if isinstance(data, dict):
                return data

            inner_type = cast(type, extra)
            if _no_transform_needed(inner_type):
                if is_list(data):
                    return data
                return list(data)

            if subplans is _LAZY:
                return [await _async_transform_recursive(d, annotation=annotation, inner_type=inner_type) for d in data]
            return [await _async_transform_fast(d, subplans, inner_type) for d in data]
        # fall through to the union / BaseModel / leaf handling below

    if kind == "union":
        for subtype, subplan in zip(cast(tuple[object, ...], extra), subplans, strict=True):
            if subplan is _LAZY:
                data = await _async_transform_recursive(data, annotation=annotation, inner_type=subtype)
            else:
                data = await _async_transform_fast(data, subplan, annotation)
        return data

    if isinstance(data, pydantic.BaseModel):
        return model_dump(data, exclude_unset=True, mode="json", exclude=getattr(data, "__api_exclude__", None))

    # Leaf: same invariant as `_transform_fast` — no `PropertyInfo` metadata can
    # be present here, so return data as-is (see `_transform_fast`).
    return data


def _maybe_transform_key(key: str, type_: type) -> str:
    """Transform the given `data` based on the annotations provided in `type_`.

    Note: this function only looks at `Annotated` types that contain `PropertyInfo` metadata.
    """
    annotated_type = _get_annotated_type(type_)
    if annotated_type is None:
        # no `Annotated` definition for this type, no transformation needed
        return key

    # ignore the first argument as it is the actual type
    annotations = get_args(annotated_type)[1:]
    for annotation in annotations:
        if isinstance(annotation, PropertyInfo) and annotation.alias is not None:
            return annotation.alias

    return key


def _no_transform_needed(annotation: type) -> bool:
    return annotation == float or annotation == int


def _transform_recursive(
    data: object,
    *,
    annotation: type,
    inner_type: type | None = None,
) -> object:
    """Transform the given data against the expected type.

    Args:
        annotation: The direct type annotation given to the particular piece of data.
            This may or may not be wrapped in metadata types, e.g. `Required[T]`, `Annotated[T, ...]` etc

        inner_type: If applicable, this is the "inside" type. This is useful in certain cases where the outside type
            is a container type such as `List[T]`. In that case `inner_type` should be set to `T` so that each entry in
            the list can be transformed using the metadata from the container type.

            Defaults to the same value as the `annotation` argument.
    """
    from .._compat import model_dump

    if inner_type is None:
        inner_type = annotation

    try:
        plan = _type_plan(inner_type)
    except TypeError:
        # Unhashable annotation; fall through to the slow path below.
        plan = None
    if plan is not None and plan is not _LAZY and _fast_path_ok(annotation, inner_type, plan):
        return _transform_fast(data, plan, annotation)

    stripped_type = strip_annotated_type(inner_type)
    origin = get_origin(stripped_type) or stripped_type
    if is_typeddict(stripped_type) and is_mapping(data):
        return _transform_typeddict(data, stripped_type)

    if origin == dict and is_mapping(data):
        args = get_args(stripped_type)
        items_type = args[1] if len(args) > 1 else object
        return {key: _transform_recursive(value, annotation=items_type) for key, value in data.items()}

    if (
        # List[T]
        (is_list_type(stripped_type) and is_list(data))
        # Iterable[T]
        or (is_iterable_type(stripped_type) and is_iterable(data) and not isinstance(data, str))
        # Sequence[T]
        or (is_sequence_type(stripped_type) and is_sequence(data) and not isinstance(data, str))
    ):
        # dicts are technically iterable, but it is an iterable on the keys of the dict and is not usually
        # intended as an iterable, so we don't transform it.
        if isinstance(data, dict):
            return cast(object, data)

        args = get_args(stripped_type)
        inner_type = cast(type, args[0]) if args else object
        if _no_transform_needed(inner_type):
            # for some types there is no need to transform anything, so we can get a small
            # perf boost from skipping that work.
            #
            # but we still need to convert to a list to ensure the data is json-serializable
            if is_list(data):
                return data
            return list(data)

        return [_transform_recursive(d, annotation=annotation, inner_type=inner_type) for d in data]

    if is_union_type(stripped_type):
        # For union types we run the transformation against all subtypes to ensure that everything is transformed.
        #
        # TODO: there may be edge cases where the same normalized field name will transform to two different names
        # in different subtypes.
        for subtype in get_args(stripped_type):
            data = _transform_recursive(data, annotation=annotation, inner_type=subtype)
        return data

    if isinstance(data, pydantic.BaseModel):
        return model_dump(data, exclude_unset=True, mode="json", exclude=getattr(data, "__api_exclude__", None))

    annotated_type = _get_annotated_type(annotation)
    if annotated_type is None:
        return data

    # ignore the first argument as it is the actual type
    annotations = get_args(annotated_type)[1:]
    for annotation in annotations:
        if isinstance(annotation, PropertyInfo) and annotation.format is not None:
            return _format_data(data, annotation.format, annotation.format_template)

    return data


def _format_data(data: object, format_: PropertyFormat, format_template: str | None) -> object:
    if isinstance(data, (date, datetime)):
        if format_ == "iso8601":
            return data.isoformat()

        if format_ == "custom" and format_template is not None:
            return data.strftime(format_template)

    if format_ == "base64" and is_base64_file_input(data):
        binary: str | bytes | None = None

        if isinstance(data, pathlib.Path):
            binary = data.read_bytes()
        elif isinstance(data, io.IOBase):
            binary = data.read()

            if isinstance(binary, str):  # type: ignore[unreachable]
                binary = binary.encode()

        if not isinstance(binary, bytes):
            raise RuntimeError(f"Could not read bytes from {data}; Received {type(binary)}")

        return base64.b64encode(binary).decode("ascii")

    return data


def _transform_typeddict(
    data: Mapping[str, object],
    expected_type: type,
) -> Mapping[str, object]:
    result: dict[str, object] = {}
    annotations = get_type_hints(expected_type, include_extras=True)
    for key, value in data.items():
        if not is_given(value):
            # we don't need to include omitted values here as they'll
            # be stripped out before the request is sent anyway
            continue

        type_ = annotations.get(key)
        if type_ is None:
            # we do not have a type annotation for this field, leave it as is
            result[key] = value
        else:
            result[_maybe_transform_key(key, type_)] = _transform_recursive(value, annotation=type_)
    return result


async def async_maybe_transform(
    data: object,
    expected_type: object,
) -> Any | None:
    """Wrapper over `async_transform()` that allows `None` to be passed.

    See `async_transform()` for more details.
    """
    if data is None:
        return None
    return await async_transform(data, expected_type)


async def async_transform(
    data: _T,
    expected_type: object,
) -> _T:
    """Transform dictionaries based off of type information from the given type, for example:

    ```py
    class Params(TypedDict, total=False):
        card_id: Required[Annotated[str, PropertyInfo(alias="cardID")]]


    transformed = transform({"card_id": "<my card ID>"}, Params)
    # {'cardID': '<my card ID>'}
    ```

    Any keys / data that does not have type information given will be included as is.

    It should be noted that the transformations that this function does are not represented in the type system.
    """
    transformed = await _async_transform_recursive(data, annotation=cast(type, expected_type))
    return cast(_T, transformed)


async def _async_transform_recursive(
    data: object,
    *,
    annotation: type,
    inner_type: type | None = None,
) -> object:
    """Transform the given data against the expected type.

    Args:
        annotation: The direct type annotation given to the particular piece of data.
            This may or may not be wrapped in metadata types, e.g. `Required[T]`, `Annotated[T, ...]` etc

        inner_type: If applicable, this is the "inside" type. This is useful in certain cases where the outside type
            is a container type such as `List[T]`. In that case `inner_type` should be set to `T` so that each entry in
            the list can be transformed using the metadata from the container type.

            Defaults to the same value as the `annotation` argument.
    """
    from .._compat import model_dump

    if inner_type is None:
        inner_type = annotation

    try:
        plan = _type_plan(inner_type)
    except TypeError:
        # Unhashable annotation; fall through to the slow path below.
        plan = None
    if plan is not None and plan is not _LAZY and _fast_path_ok(annotation, inner_type, plan):
        return await _async_transform_fast(data, plan, annotation)

    stripped_type = strip_annotated_type(inner_type)
    origin = get_origin(stripped_type) or stripped_type
    if is_typeddict(stripped_type) and is_mapping(data):
        return await _async_transform_typeddict(data, stripped_type)

    if origin == dict and is_mapping(data):
        args = get_args(stripped_type)
        items_type = args[1] if len(args) > 1 else object
        return {key: _transform_recursive(value, annotation=items_type) for key, value in data.items()}

    if (
        # List[T]
        (is_list_type(stripped_type) and is_list(data))
        # Iterable[T]
        or (is_iterable_type(stripped_type) and is_iterable(data) and not isinstance(data, str))
        # Sequence[T]
        or (is_sequence_type(stripped_type) and is_sequence(data) and not isinstance(data, str))
    ):
        # dicts are technically iterable, but it is an iterable on the keys of the dict and is not usually
        # intended as an iterable, so we don't transform it.
        if isinstance(data, dict):
            return cast(object, data)

        args = get_args(stripped_type)
        inner_type = cast(type, args[0]) if args else object
        if _no_transform_needed(inner_type):
            # for some types there is no need to transform anything, so we can get a small
            # perf boost from skipping that work.
            #
            # but we still need to convert to a list to ensure the data is json-serializable
            if is_list(data):
                return data
            return list(data)

        return [await _async_transform_recursive(d, annotation=annotation, inner_type=inner_type) for d in data]

    if is_union_type(stripped_type):
        # For union types we run the transformation against all subtypes to ensure that everything is transformed.
        #
        # TODO: there may be edge cases where the same normalized field name will transform to two different names
        # in different subtypes.
        for subtype in get_args(stripped_type):
            data = await _async_transform_recursive(data, annotation=annotation, inner_type=subtype)
        return data

    if isinstance(data, pydantic.BaseModel):
        return model_dump(data, exclude_unset=True, mode="json", exclude=getattr(data, "__api_exclude__", None))

    annotated_type = _get_annotated_type(annotation)
    if annotated_type is None:
        return data

    # ignore the first argument as it is the actual type
    annotations = get_args(annotated_type)[1:]
    for annotation in annotations:
        if isinstance(annotation, PropertyInfo) and annotation.format is not None:
            return await _async_format_data(data, annotation.format, annotation.format_template)

    return data


async def _async_format_data(data: object, format_: PropertyFormat, format_template: str | None) -> object:
    if isinstance(data, (date, datetime)):
        if format_ == "iso8601":
            return data.isoformat()

        if format_ == "custom" and format_template is not None:
            return data.strftime(format_template)

    if format_ == "base64" and is_base64_file_input(data):
        binary: str | bytes | None = None

        if isinstance(data, pathlib.Path):
            binary = await anyio.Path(data).read_bytes()
        elif isinstance(data, io.IOBase):
            binary = data.read()

            if isinstance(binary, str):  # type: ignore[unreachable]
                binary = binary.encode()

        if not isinstance(binary, bytes):
            raise RuntimeError(f"Could not read bytes from {data}; Received {type(binary)}")

        return base64.b64encode(binary).decode("ascii")

    return data


async def _async_transform_typeddict(
    data: Mapping[str, object],
    expected_type: type,
) -> Mapping[str, object]:
    result: dict[str, object] = {}
    annotations = get_type_hints(expected_type, include_extras=True)
    for key, value in data.items():
        if not is_given(value):
            # we don't need to include omitted values here as they'll
            # be stripped out before the request is sent anyway
            continue

        type_ = annotations.get(key)
        if type_ is None:
            # we do not have a type annotation for this field, leave it as is
            result[key] = value
        else:
            result[_maybe_transform_key(key, type_)] = await _async_transform_recursive(value, annotation=type_)
    return result


@lru_cache(maxsize=8096)
def get_type_hints(
    obj: Any,
    globalns: dict[str, Any] | None = None,
    localns: Mapping[str, Any] | None = None,
    include_extras: bool = False,
) -> dict[str, Any]:
    return _get_type_hints(obj, globalns=globalns, localns=localns, include_extras=include_extras)
