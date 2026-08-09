# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The Rubix cube's *state*: which colour is on which facelet.

Split out from the authoring script because two very different things need it and they must
agree exactly:

* ``eva_rl/scripts/author_rubiks_cube_usd.py`` bakes a pattern into a USD at build time;
* ``mdp/events.py:randomize_cube_pattern`` re-assigns the facelets of each cloned env at
  startup, so no two envs show the same cube.

The second only works because both use the **same facelet ordering**. If the authoring script
and the randomiser disagreed about which quad is which facelet, the randomiser would scatter
colours over a cube that no longer corresponds to any real state, and nothing would fail --
it would just quietly render nonsense. So the order is defined once, here, by ``facelets()``.

Deliberately free of ``isaaclab`` and ``pxr``: the authoring script runs on plain numpy so a
cube can be built without starting Kit.

Why a cubie model rather than a colour shuffle
----------------------------------------------
A shuffle that keeps nine of each colour still produces cubes that cannot exist -- white
opposite yellow on the same corner piece, an edge with two faces of the same colour. Here the
cube is 27 cubies, each carrying an integer position and an integer 3x3 rotation, and a face
turn rotates the position *and* the orientation of every cubie in the layer. The sticker
triples on a corner therefore move together, exactly as they are glued together on the real
object, and every reachable state is a state a real cube can be in.
"""

from __future__ import annotations

import numpy as np

#: Face normals in the body frame, and the colour each carries on a SOLVED cube. Opposite
#: faces are opposite axes, which is what preserves the real cube's white/yellow, red/orange,
#: blue/green pairings.
FACES: dict[tuple[int, int, int], str] = {
    (0, 0, 1): "white",
    (0, 0, -1): "yellow",
    (1, 0, 0): "red",
    (-1, 0, 0): "orange",
    (0, 1, 0): "blue",
    (0, -1, 0): "green",
}

COLOR_NAMES = tuple(FACES.values())

#: Linear-space diffuse colours. Near-pure hues with a small floor on purpose: the render
#: path raises the minimum channel, and saturation is ``(max - min) / max``, so a colour that
#: starts at 0.2 in its off-channels arrives washed out (HANDOFF §5.1).
COLORS: dict[str, tuple[float, float, float]] = {
    "white": (0.90, 0.90, 0.90),
    "yellow": (0.92, 0.72, 0.02),
    "red": (0.72, 0.02, 0.02),
    "orange": (0.88, 0.24, 0.01),
    "blue": (0.02, 0.10, 0.62),
    "green": (0.02, 0.46, 0.09),
}
BODY_COLOR = (0.015, 0.015, 0.015)

_R90 = {
    0: np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]),
    1: np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]),
    2: np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]),
}


def face_axes(normal) -> tuple[np.ndarray, np.ndarray]:
    """The two in-face axes for ``normal``, ordered so ``(u, v, n)`` is right-handed.

    Used by the authoring script to wind each quad so its front face points outward, and by
    :func:`facelets` to walk the nine cells of a face. One definition, so the geometry and the
    facelet order cannot drift apart.
    """
    n = np.asarray(normal)
    a = int(np.argmax(np.abs(n)))
    u = np.zeros(3, dtype=int)
    u[(a + 1) % 3] = 1
    return u, np.cross(n, u)


def facelets():
    """⭐ THE canonical facelet order: yields ``(normal, u_index, v_index, cell)``, 54 of them.

    Faces in ``FACES`` order, nine per face, ``iu`` outer and ``iv`` inner. The N-th item is
    the N-th quad in the authored mesh, and that correspondence is the whole contract between
    the authoring script and the runtime randomiser.
    """
    for normal in FACES:
        n = np.asarray(normal)
        u, v = face_axes(normal)
        for iu in (-1, 0, 1):
            for iv in (-1, 0, 1):
                yield normal, iu, iv, n + u * iu + v * iv


class Cube:
    """27 cubies, each with an integer position and an integer orientation matrix.

    ``ori`` maps a direction in the cubie's own frame to one in the cube's frame, so the
    colour showing on world-face ``n`` is the solved colour of the local direction
    ``ori.T @ n``.
    """

    def __init__(self):
        self.pos = [np.array(p) - 1 for p in np.ndindex(3, 3, 3)]
        self.ori = [np.eye(3, dtype=int) for _ in self.pos]

    def turn(self, axis: int, layer: int, quarters: int = 1) -> "Cube":
        R = np.linalg.matrix_power(_R90[axis], quarters % 4)
        for i, p in enumerate(self.pos):
            if p[axis] == layer:
                self.pos[i] = R @ p
                self.ori[i] = R @ self.ori[i]
        return self

    def scramble(self, rng: np.random.Generator, moves: int = 25) -> "Cube":
        for _ in range(moves):
            self.turn(int(rng.integers(3)), int(rng.choice([-1, 1])), int(rng.integers(1, 4)))
        return self

    def color_at(self, cell: np.ndarray, normal) -> str:
        """Colour showing on face ``normal`` of the cubie now sitting at ``cell``."""
        i = next(i for i, p in enumerate(self.pos) if np.array_equal(p, cell))
        return FACES[tuple(int(v) for v in (self.ori[i].T @ np.asarray(normal)))]

    def face_indices(self) -> dict[str, list[int]]:
        """``{colour: [facelet index, ...]}`` over :func:`facelets`' ordering.

        This is exactly the ``GeomSubset`` index array each colour's subset needs, which is
        why both the authoring script and the runtime randomiser call it rather than deriving
        their own.
        """
        out: dict[str, list[int]] = {c: [] for c in COLOR_NAMES}
        for k, (normal, _, _, cell) in enumerate(facelets()):
            out[self.color_at(cell, normal)].append(k)
        return out


def pattern(seed: int | None) -> Cube:
    """``None`` -> solved; an int -> that scramble. Deterministic in the seed."""
    c = Cube()
    return c if seed is None else c.scramble(np.random.default_rng(seed))
