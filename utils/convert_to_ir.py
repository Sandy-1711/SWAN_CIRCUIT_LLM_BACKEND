from typing import Dict, List, Any

def _parse_attrs_from_parts(parts: List[Dict[str, Any]]) -> List[str]:
    """Return lines like: '<part_id> key1:val1 key2:val2' for parts having attrs."""
    lines = []
    for part in parts:
        attrs = part.get("attrs") or {}
        if attrs:
            entries = " ".join(f"{k}:{v}" for k, v in attrs.items())
            lines.append(f"{part.get('id')} {entries}")
    return lines


def _parse_components(parts: List[Dict[str, Any]]) -> List[str]:
    """
    Return lines like: '<id>:<type> <extras>'
    Extras = any fields other than id, type, top, left, attrs
    """
    components = []
    for part in parts:
        id_ = part.get("id")
        type_ = part.get("type")
        # exclude these from the extras (to match your JS)
        exclude = {"id", "type", "top", "left", "attrs"}
        extras_items = [(k, v) for k, v in part.items() if k not in exclude]
        extras = " ".join(f"{k}:{v}" for k, v in extras_items)
        components.append(f"{id_}:{type_}{(' ' + extras) if extras else ''}")
    return components


def _parse_connections(connections: List[List[Any]], breadboard_id: str) -> List[str]:
    """
    Keep only connections where either endpoint is the breadboard.
    Output as '<other_end> <breadboard_end>' (order mirrors your JS).
    Each connection item is like: [from, to, color, ...]
    """
    if not breadboard_id:
        return []

    with_bb = []
    for conn in connections:
        if len(conn) < 2:
            continue
        from_, to = conn[0], conn[1]
        if isinstance(from_, str) and from_.startswith(breadboard_id):
            with_bb.append(f"{to} {from_}")
        elif isinstance(to, str) and to.startswith(breadboard_id):
            with_bb.append(f"{from_} {to}")
    return with_bb


def convert_to_ir(json_obj: Dict[str, Any]) -> str:
    """
    Convert a single Wokwi-style JSON object (with 'parts' and 'connections')
    into the IR string:

    <<=components=>>
    <component lines>
    <<=connections=>>
    <connection lines>
    <<=attrs=>>
    <attr lines>
    """
    if not isinstance(json_obj, dict):
        raise TypeError("json_obj must be a dict")

    parts = json_obj.get("parts")
    connections = json_obj.get("connections")

    if not isinstance(parts, list) or not isinstance(connections, list):
        raise ValueError("Invalid JSON object: missing or invalid 'parts' or 'connections'")

    components = _parse_components(parts)
    breadboard_id = next((p.get("id") for p in parts if p.get("type") == "wokwi-breadboard"), None)
    conns = _parse_connections(connections, breadboard_id or "")
    attrs = _parse_attrs_from_parts(parts)

    ir = (
        "<<=components=>>\n"
        + "\n".join(components)
        + "\n<<=connections=>>\n"
        + "\n".join(conns)
        + "\n<<=attrs=>>\n"
        + "\n".join(attrs)
    )
    return ir

