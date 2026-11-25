"""Build the playable arrow paths from the authored contours.

Geometry stays on a square grid. Every initial arrow receives a two-cell tip
pointing straight at the core. The path builder reserves a legal launch order
before routing tails across other approaches and filling the rest of the figure.
"""

import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
from itertools import pairwise
from json import dumps
from math import ceil, floor
from pathlib import Path
from random import Random
from typing import NamedTuple

from PIL import Image, ImageDraw

from silhouettes import HOUSE, SILHOUETTES


ORTHOGONAL = ((0, -1), (1, 0), (0, 1), (-1, 0))
DIAGONAL = ((1, -1), (1, 1), (-1, 1), (-1, -1))
RAYS = ORTHOGONAL + DIAGONAL
# Zero-based arrow IDs on numbered boards with at least two useful targets after charging.
IMPULSE_ARROWS = {5: 6, 6: 6, 7: 1, 12: 8, 18: 17, 24: 19, 31: 9, 38: 10, 44: 1, 50: 44}


class Dependencies(NamedTuple):
    links: int
    starts: int
    directions: int
    forks: int
    joins: int
    distant: int


def neighbors(point, diagonal=False):
    x, y = point
    steps = DIAGONAL if diagonal else ORTHOGONAL
    return tuple((x + dx, y + dy) for dx, dy in steps)


def rasterize(layers, scale):
    bounds = [point for layer in layers for point in (layer[1] if layer[0] is False else layer)]
    x1 = floor(min(x for x, _ in bounds) * scale) - 1
    x2 = ceil(max(x for x, _ in bounds) * scale) + 1
    y1 = floor(min(y for _, y in bounds) * scale) - 1
    y2 = ceil(max(y for _, y in bounds) * scale) + 1
    # Sample at cell centers with subcell precision around ears, eyes and fins.
    density = 4
    image = Image.new("1", (density * (x2 - x1 + 1), density * (y2 - y1 + 1)))
    pen = ImageDraw.Draw(image)
    for layer in layers:
        solid = layer[0] is not False
        points = layer if solid else layer[1]
        pen.polygon([(round((x * scale - x1) * density), round((y * scale - y1) * density))
                     for x, y in points], fill=int(solid))
    assert image.getpixel((-density * x1, -density * y1)), "Core outside the figure"
    return {(x, y) for y in range(y1, y2 + 1) for x in range(x1, x2 + 1)
            if x * x + y * y >= 8 and image.getpixel((density * (x - x1), density * (y - y1)))}


def tip_slots(mask, rng=None, gap_chance=0):
    slots = []
    extent = max(max(abs(x), abs(y)) for x, y in mask)
    for x, y in RAYS:
        lane = []
        distance = 2 if x and y else 3
        while distance < extent:
            if (x * distance, y * distance) in mask and (x * (distance + 1), y * (distance + 1)) in mask:
                if (rng is not None and (x * (distance + 2), y * (distance + 2)) in mask
                        and rng.random() < gap_chance):
                    distance += 1
                    continue
                lane.append(distance)
                distance += 2
            else:
                distance += 1
        slots.append(lane)
    return slots


def raster_for_level(layers, count, extra_capacity=0):
    # Choose the footprint by capacity while the game's cell size stays fixed.
    raster_step = max(12, round(14 + .61 * count) if count >= 8 else 12) + extra_capacity
    target = count + 1 + count // 6 + extra_capacity
    while True:
        mask = rasterize(layers, raster_step / 100)
        if sum(map(len, tip_slots(mask))) >= target:
            return mask
        raster_step += 1
        if raster_step > 190:
            raise ValueError("The figure cannot fit enough tip cells")


def choose_tips(mask, count, rng):
    for chance in (.32, .25, .18, .10, 0):
        lanes = tip_slots(mask, rng, chance)
        if sum(map(len, lanes)) >= count:
            break
    cardinal_capacity = sum(len(lane) for lane in lanes[:4])
    diagonal_capacity = sum(len(lane) for lane in lanes[4:])
    diagonal_count = min(diagonal_capacity, max(round(count / 3), count - cardinal_capacity))
    per_lane = [0] * len(RAYS)
    for start, size in ((0, count - diagonal_count), (4, diagonal_count)):
        available = [side for side in range(start, start + 4) if lanes[side]]
        for side in rng.sample(available, min(size, len(available))):
            per_lane[side] = 1
        for _ in range(size - min(size, len(available))):
            weights = [len(lanes[side]) - per_lane[side] for side in available]
            side = rng.choices(available, weights)[0]
            per_lane[side] += 1
    queues = []
    for side, (lane, number) in enumerate(zip(lanes, per_lane)):
        if not number:
            continue
        # Leave irregular gaps and some empty approaches close to the sphere.
        chosen = sorted(rng.sample(lane, number))
        queues.append(deque((RAYS[side], distance) for distance in chosen))
    seeds = []
    while queues:
        lane = rng.choice(queues)
        seeds.append(lane.popleft())
        if not lane:
            queues.remove(lane)
    return seeds


def ray(point):
    x, y = point
    return (int(x > 0) - int(x < 0), int(y > 0) - int(y < 0))


def crossing_path(start, gate, index, free, route_order, limit, adjacent):
    seen = {start: None}
    queue = deque([(start, 0)])
    while queue:
        point, distance = queue.popleft()
        if point == gate:
            route = []
            while point != start:
                route.append(point)
                point = seen[point]
            return route[::-1]
        if distance == limit:
            continue
        for step in adjacent[point]:
            if step in free and step not in seen and index < route_order.get(step, index + 1):
                seen[step] = point
                queue.append((step, distance + 1))
    return []


def route_crossings(paths, free, route_order, seeds, neighborhoods, rng):
    """Reserve real tail sections that obstruct a different approach."""
    gates = [point for point in route_order if point in free]
    depth = [0] * len(paths)
    limit = 12 + len(paths) // 2
    rng.shuffle(gates)
    gates.sort(key=route_order.__getitem__)
    for gate in gates:
        if gate not in free:
            continue
        target = route_order[gate]
        gate_ray = ray(gate)
        candidates = []
        for index in range(target):
            if seeds[index][0] == gate_ray:
                continue
            diagonal = seeds[index][0] in DIAGONAL
            if diagonal and (gate[0] + gate[1]) % 2:
                continue
            start = paths[index][-1]
            delta = abs(start[0] - gate[0]), abs(start[1] - gate[1])
            distance = max(delta) if diagonal else sum(delta)
            if distance <= limit:
                candidates.append((distance - 3 * depth[index] + rng.random(), index))
        for _, index in sorted(candidates)[:9]:
            route = crossing_path(paths[index][-1], gate, index, free, route_order, limit, neighborhoods[index])
            if route:
                paths[index].extend(route)
                free.difference_update(route)
                for point in route:
                    if point in route_order and ray(point) != seeds[index][0]:
                        dependency = route_order[point]
                        depth[dependency] = max(depth[dependency], depth[index] + 1)
                break


def grow_paths(mask, seeds, layout, rng):
    paths = []
    route_order = {}
    for index, ((x, y), length) in enumerate(seeds):
        paths.append([(x * length, y * length), (x * (length + 1), y * (length + 1))])
        for step in range(2 if x and y else 3, length):
            route_order.setdefault((x * step, y * step), index)
    reserved = {point for path in paths for point in path}
    free = mask - reserved
    orthogonal, diagonal, edge = layout
    neighborhoods = [diagonal if direction in DIAGONAL else orthogonal for direction, _ in seeds]
    count = len(paths)
    route_crossings(paths, free, route_order, seeds, neighborhoods, rng)

    def available(point, index):
        return point in free and index < route_order.get(point, count)

    def extend(path, index):
        adjacent = neighborhoods[index]
        tail = path[-1]
        options = [point for point in adjacent[tail] if available(point, index)]
        if not options:
            return False
        dx = tail[0] - path[-2][0]
        dy = tail[1] - path[-2][1]

        def weight(point):
            degree = sum(available(p, index) for p in adjacent[point])
            onward = max(.12, 1.1 - abs(degree - 1) * .29)
            straight = 1.32 if (point[0] - tail[0], point[1] - tail[1]) == (dx, dy) else 1
            return onward * straight * (1.30 if point in edge else 1)

        point = rng.choices(options, [weight(p) for p in options])[0]
        path.append(point)
        free.remove(point)
        return True

    active = list(range(count))
    while active:
        rng.shuffle(active)
        active = [index for index in active if extend(paths[index], index)]

    def insert_detours(path, index):
        position = 1
        placed = 0
        while position < len(path) - 1:
            a, b = path[position:position + 2]
            dx, dy = b[0] - a[0], b[1] - a[1]
            for side in rng.sample((-1, 1), 2):
                c = (a[0] - side * dy, a[1] + side * dx)
                d = (b[0] - side * dy, b[1] + side * dx)
                if available(c, index) and available(d, index):
                    free.difference_update((c, d))
                    path[position + 1:position + 1] = (c, d)
                    placed += 2
                    break
            position += 1
        return placed

    def reach_new_tail(path, index):
        # Reversing a path suffix preserves the protected head and opens a new end.
        adjacent = neighborhoods[index]
        tail = path[-1]
        candidates = [position for position in range(1, len(path) - 2)
                      if tail in adjacent[path[position]]
                      and any(available(point, index) for point in adjacent[path[position + 1]])]
        if candidates:
            start = rng.choice(candidates) + 1
            path[start:] = reversed(path[start:])

    def improve_edge(path, index):
        for i in range(2, len(path) - 1):
            a, b, c = path[i - 1:i + 2]
            if not is_turn(a, b, c) or b in edge or b in route_order:
                continue
            corner = (a[0] + c[0] - b[0], a[1] + c[1] - b[1])
            if corner in edge and available(corner, index):
                free.remove(corner)
                free.add(b)
                path[i] = corner

    refinements = list(range(count))
    for iteration in range(5):
        previous_free = len(free)
        rng.shuffle(refinements)
        for index in refinements:
            path = paths[index]
            insert_detours(path, index)
            reach_new_tail(path, index)
            while extend(path, index):
                pass
            if iteration < 2:
                improve_edge(path, index)
        if previous_free == len(free) and iteration >= 2:
            break

    score = (len(mask) - len(free)) / len(mask) + .55 * (len(edge - free) / len(edge))
    short = sum(max(0, 5 - len(path)) for path in paths)
    score -= .12 * short / count
    return score, paths


def is_turn(a, b, c):
    return (a[0] + c[0], a[1] + c[1]) != (2 * b[0], 2 * b[1])


def compact(path):
    turns = [path[0]]
    for a, b, c in zip(path, path[1:], path[2:]):
        if is_turn(a, b, c):
            turns.append(b)
    turns.append(path[-1])
    return " ".join(f"{x},{y}" for x, y in reversed(turns))


def expand(shape):
    points = [tuple(map(int, point.split(","))) for point in shape.split()]
    yield points[0]
    first = (points[1][0] - points[0][0], points[1][1] - points[0][1])
    diagonal = bool(first[0] and first[1])
    # The two possible diagonals in any unit square have opposite parity.
    # Using even parity makes two paths intersect only at shared grid vertices.
    assert not diagonal or not sum(points[0]) % 2
    for (x, y), (xx, yy) in pairwise(points):
        dx, dy = abs(xx - x), abs(yy - y)
        assert dx == dy if diagonal else not (dx and dy)
        steps = max(dx, dy)
        assert steps > 0
        for step in range(1, steps + 1):
            yield (x + (xx - x) * step // steps, y + (yy - y) * step // steps)


def dependency_profile(paths):
    owners = {}
    for index, cells in enumerate(paths):
        for x, y in cells:
            assert (x, y) not in owners and x * x + y * y >= 8
            owners[x, y] = index

    routes = []
    for index, cells in enumerate(paths):
        (previous_x, previous_y), (x, y) = cells[-2:]
        dx, dy = x - previous_x, y - previous_y
        assert x * dy == y * dx and x * dx + y * dy < 0
        blocking = set()
        for step in range(1, max(abs(x), abs(y))):
            owner = owners.get((x + step * dx, y + step * dy))
            if owner is not None:
                blocking.add(owner)
        assert index not in blocking
        routes.append(blocking)

    approaches = [ray(cells[-1]) for cells in paths]
    starts = sum(not route for route in routes)
    remaining = set(range(len(paths)))
    ancestors = [set() for _ in paths]
    dependents = [0] * len(paths)
    links, joins, distant = 0, 0, 0
    while remaining:
        clear = {index for index in remaining if not routes[index] & remaining}
        assert clear, "The board has a cycle"
        for index in clear:
            inherited = set().union(*(ancestors[blocker] for blocker in routes[index]))
            direct = routes[index] - inherited
            ancestors[index] = routes[index] | inherited
            joins += len(direct) > 1
            x, y = paths[index][-1]
            for blocker in direct:
                dependents[blocker] += 1
                if approaches[blocker] != approaches[index]:
                    links += 1
                    xx, yy = paths[blocker][-1]
                    distant += abs(x - xx) + abs(y - yy) >= max(7, len(paths) // 4)
        remaining -= clear
    return Dependencies(links, starts, len(set(approaches)), sum(value > 1 for value in dependents), joins, distant)


def validate(shapes):
    return dependency_profile([list(expand(shape)) for shape in shapes])


def palette(color):
    channels = [int(color[i:i + 2], 16) for i in (1, 3, 5)]
    blends = ("#" + "".join(f"{round(c + (255 - c) * f):02x}" for c in channels) for f in (.14, 0, .31))
    return list(blends)


def make_level(job):
    index, (title, color, layers) = job
    count = index + 4
    minimum_links = max(2, (count - 1) // 5)
    minimum_directions = 4 if count < 8 else 5 if count < 14 else 6 if count < 25 else 7
    minimum_forks = max(1, count // 10)
    minimum_joins = count // 10
    attempts = 100 if index < 10 else 85 if index < 25 else 70
    best_score = -1
    best_paths = []
    best_fill = 0
    best_profile = None
    for extra_capacity in (0, 2, 4, 6):
        mask = raster_for_level(layers, count, extra_capacity)
        orthogonal = {point: neighbors(point) for point in mask}
        diagonal = {point: neighbors(point, True) for point in mask if not sum(point) % 2}
        edge = {point for point, nearby in orthogonal.items() if any(p not in mask for p in nearby)}
        layout = orthogonal, diagonal, edge
        direction_floor = min(minimum_directions, sum(bool(lane) for lane in tip_slots(mask)))
        for attempt in range(attempts):
            rng = Random(89653 * (index + 1) + 3109 * attempt + 131 * extra_capacity)
            score, paths = grow_paths(mask, choose_tips(mask, count, rng), layout, rng)
            profile = dependency_profile([path[::-1] for path in paths])
            if best_profile is None or profile.forks + profile.joins > best_profile.forks + best_profile.joins:
                best_profile = profile
            if (profile.links < minimum_links or not 2 <= profile.starts <= 5
                    or profile.directions < direction_floor or profile.forks < minimum_forks
                    or profile.joins < minimum_joins):
                continue
            score += .012 * min(profile.links, minimum_links + 6)
            score += .016 * (min(profile.forks, minimum_forks + 3) + min(profile.joins, minimum_joins + 3))
            score += .016 * min(profile.distant, 2 + count // 5)
            score -= .025 * max(0, profile.starts - 4)
            if score > best_score:
                best_score, best_paths = score, paths
                best_fill = sum(map(len, paths)) / len(mask)
        if best_fill >= .90:
            break
    if not best_paths:
        raise ValueError(f"{title}: could not fit enough branches, found {best_profile}")
    rng = Random(index + 43)
    rng.shuffle(best_paths)
    shapes = [compact(path) for path in best_paths]
    profile = validate(shapes)
    difficulty = "easy" if index < 7 else "medium" if index < 20 else "hard" if index < 37 else "veryHard"
    level = {"title": title, "difficulty": difficulty, "palette": palette(color), "shapes": shapes}
    if index + 1 in IMPULSE_ARROWS:
        level["intro"] = "goldIntro"
        level["impulse"] = IMPULSE_ARROWS[index + 1]
    return level, best_fill, profile


def save_levels(levels, destination):
    assert len(levels) == 50
    assert [len(level["shapes"]) for level in levels] == list(range(4, 54))
    fingerprints = {sha256(" ".join(sorted(level["shapes"])).encode()).hexdigest() for level in levels}
    assert len(fingerprints) == len(levels)
    bodies = ",\n".join(dumps(level, ensure_ascii=False, indent=2) for level in levels)
    destination.write_text("// Generated by tools/build_levels.py from original silhouettes.\n"
                           "const levels = [\n" + bodies + "\n];\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "assets/levels.js")
    args = parser.parse_args()
    assert validate(HOUSE) == Dependencies(2, 2, 4, 1, 0, 2)
    levels = [{"title": "Домик", "difficulty": "easy", "palette": palette("#d9c1f7"), "shapes": HOUSE,
               "intro": "houseIntro"}]
    with ProcessPoolExecutor(max_workers=6) as pool:
        for level, fill, profile in pool.map(make_level, enumerate(SILHOUETTES, start=1)):
            levels.append(level)
            print(f"{len(levels):2} {level['title']:16} {len(level['shapes']):2} arrows   {fill:.0%} mask   {profile}", flush=True)
    save_levels(levels, args.output)


if __name__ == "__main__":
    main()
