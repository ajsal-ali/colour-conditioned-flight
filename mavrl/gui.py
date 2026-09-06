#!/usr/bin/env python3
"""Two windows shared by the scripted collector and the manual teleop.

* `ViewerWindow` -- MuJoCo's passive viewer, third-person, following the drone.
  It holds direct references to `model`/`data`, so it must be re-attached after
  `env.set_layout()` rebuilds them; `attach()` does that.
* `FeedWindow`   -- a pygame window showing exactly what the policy would see:
  RGB, depth and (in collect mode) segmentation, all at the downsampled 128 px
  the encoder gets, not the 512 px raster. It also reports which keys are
  currently **held**, which is what the teleop needs and what a press-event
  callback cannot give.

pygame is the dependency rather than OpenCV plus a keyboard library because it
supplies both halves: `key.get_pressed()` is a state poll, not an event queue, so
"hold to move, release to stop" is exact rather than reconstructed from OS key
repeat. Focus-scoped too -- keys only register while the feed window is focused,
unlike a global hotkey listener.

Both imports are lazy: headless training must never fail because a GUI library
is missing.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from mavrl import config as C

PANEL = 384          # on-screen size of each 128 px feed
HUD_H = 96


def _require(module: str, extra: str):
    try:
        return __import__(module)
    except ImportError as exc:                   # pragma: no cover - env dependent
        raise SystemExit(
            f"--gui needs {module!r}: pip install {extra}\n"
            f"(headless collection does not need it; drop --gui)") from exc


# --------------------------------------------------------------------------
# depth colouring
# --------------------------------------------------------------------------

#: near -> far. Hand-rolled so the feed window needs no matplotlib.
_ANCHORS = np.array([
    (255, 255, 220), (255, 176, 60), (214, 60, 90),
    (90, 40, 140), (20, 20, 60),
], dtype=np.float32)


def depth_rgb(depth_m: np.ndarray, dmax: float = C.DEPTH_MAX) -> np.ndarray:
    """(H,W) metres -> (H,W,3) uint8, bright near, dark far."""
    t = np.clip(np.asarray(depth_m, dtype=np.float32) / dmax, 0.0, 1.0)
    scaled = t * (len(_ANCHORS) - 1)
    i = np.clip(scaled.astype(np.int32), 0, len(_ANCHORS) - 2)
    f = (scaled - i)[..., None]
    return (_ANCHORS[i] * (1.0 - f) + _ANCHORS[i + 1] * f).astype(np.uint8)


#: One colour per SemClass, in SemClass order.
_SEG_COLORS = np.array([
    (30, 30, 34),      # FREE
    (150, 150, 155),   # WALL
    (220, 60, 60),     # RED_WINDOW
    (60, 110, 230),    # BLUE_WINDOW
    (255, 120, 90),    # RED_BAR
    (110, 190, 255),   # BLUE_BAR
    (90, 80, 60),      # FLOOR
], dtype=np.uint8)


def seg_rgb(seg: np.ndarray) -> np.ndarray:
    idx = np.clip(np.asarray(seg), 0, len(_SEG_COLORS) - 1)
    return _SEG_COLORS[idx]


# --------------------------------------------------------------------------
# 3-D viewer
# --------------------------------------------------------------------------

class ViewerWindow:
    """Third-person MuJoCo viewer that follows the drone down the corridor."""

    def __init__(self, env, distance: float = 6.5, elevation: float = -14.0,
                 azimuth: float = 90.0):
        import mujoco.viewer                     # noqa: F401  (lazy on purpose)
        self._viewer_mod = mujoco.viewer
        self._mujoco = _require("mujoco", "mujoco")
        self.distance, self.elevation, self.azimuth = distance, elevation, azimuth
        self.handle = None
        self.attach(env)

    def attach(self, env) -> None:
        """(Re)bind to the env's current model/data.

        `set_layout` builds a whole new MjModel, so a viewer opened against the
        old one would render a course that no longer exists.
        """
        self.close()
        self.handle = self._viewer_mod.launch_passive(
            env.model, env.data, show_left_ui=False, show_right_ui=False)
        cam = self.handle.cam
        cam.type = self._mujoco.mjtCamera.mjCAMERA_FREE
        cam.distance, cam.elevation, cam.azimuth = (
            self.distance, self.elevation, self.azimuth)

    @property
    def running(self) -> bool:
        return self.handle is not None and self.handle.is_running()

    def sync(self, env) -> None:
        if not self.running:
            return
        pos = env.pos[0]
        # Look slightly ahead of the drone so the next gate stays in frame.
        self.handle.cam.lookat[:] = [0.0, pos[1] + 1.5, max(1.2, pos[2])]
        self.handle.sync()

    def close(self) -> None:
        if self.handle is not None:
            try:
                self.handle.close()
            except Exception:
                pass
            self.handle = None


# --------------------------------------------------------------------------
# camera-feed window
# --------------------------------------------------------------------------

class FeedWindow:
    """Live RGB / depth / segmentation panels, plus held-key state."""

    def __init__(self, title: str = "mavrl feed", show_seg: bool = True,
                 panel: int = PANEL):
        self.pygame = _require("pygame", "pygame")
        self.panel = panel
        self.show_seg = show_seg
        self.n_panels = 3 if show_seg else 2
        self.pygame.init()
        self.pygame.display.set_caption(title)
        self.screen = self.pygame.display.set_mode(
            (self.panel * self.n_panels, self.panel + HUD_H))
        self.font = self.pygame.font.SysFont("consolas", 15)
        self.big = self.pygame.font.SysFont("consolas", 17, bold=True)
        self.clock = self.pygame.time.Clock()
        self.alive = True
        self._pressed_once = set()

    # -- input ---------------------------------------------------------------

    def poll(self):
        """Pump events. Returns (held_keycodes, tapped_keycodes)."""
        self._pressed_once.clear()
        for ev in self.pygame.event.get():
            if ev.type == self.pygame.QUIT:
                self.alive = False
            elif ev.type == self.pygame.KEYDOWN:
                self._pressed_once.add(ev.key)
        held = self.pygame.key.get_pressed()
        return held, set(self._pressed_once)

    def tick(self, fps: float = C.POLICY_FREQ) -> None:
        """Cap to the policy rate so teleop runs at wall-clock speed."""
        self.clock.tick(fps)

    # -- drawing -------------------------------------------------------------

    def _blit(self, arr: np.ndarray, col: int, label: str) -> None:
        # pygame surfaces are (W,H,3); numpy images are (H,W,3).
        surf = self.pygame.surfarray.make_surface(
            np.ascontiguousarray(np.transpose(arr, (1, 0, 2))))
        surf = self.pygame.transform.scale(surf, (self.panel, self.panel))
        self.screen.blit(surf, (col * self.panel, 0))
        self.screen.blit(self.font.render(label, True, (235, 235, 235)),
                         (col * self.panel + 8, 6))

    def draw(self, obs: dict, lines=()) -> None:
        if not self.alive:
            return
        self.screen.fill((16, 16, 20))
        self._blit(obs["image"][..., :3], 0, "RGB (encoder input)")

        if "depth_m" in obs:
            depth, dlabel = obs["depth_m"], "depth (m, clean)"
        else:
            # Fall back to the packed uint8 channel when not in collect mode.
            depth = obs["image"][..., 3].astype(np.float32) / 255.0 * C.DEPTH_MAX
            dlabel = "depth (from obs)"
        self._blit(depth_rgb(depth), 1, dlabel)

        if self.show_seg and "seg" in obs:
            self._blit(seg_rgb(obs["seg"]), 2, "segmentation")

        for i, line in enumerate(lines):
            font = self.big if i == 0 else self.font
            self.screen.blit(font.render(line, True, (225, 225, 230)),
                             (10, self.panel + 8 + i * 19))
        self.pygame.display.flip()

    def message(self, text: str, colour=(255, 210, 120)) -> None:
        """Overlay one line at the top of the HUD (episode result, etc.)."""
        if not self.alive:
            return
        self.screen.blit(self.big.render(text, True, colour),
                         (10, self.panel + 8))
        self.pygame.display.flip()

    def close(self) -> None:
        if self.alive:
            self.alive = False
        try:
            self.pygame.quit()
        except Exception:
            pass


class CvFeedWindow:
    """OpenCV twin of `FeedWindow`: same RGB / depth / seg panels, no pygame.

    Exists because the pygame feed shares a process with MuJoCo's glfw passive
    viewer and the env's offscreen camera render, and three GL consumers in one
    glfw process is fragile (segfaults on teardown on some drivers). cv2's
    highgui is a separate Qt window, so `--cv2` (optionally with `--no-viewer`)
    gives a stable "what the drone sees" feed.

    Drop-in: it exposes the same surface `fly.py` uses -- `.alive`, `.poll()`
    returning `(held, tapped)`, `.draw()`, `.tick()`, `.message()`, `.close()`,
    and a `.pygame` shim carrying the K_* keycodes the caller compares against.
    `held` is always empty: cv2 has no held-key poll, and the only consumer
    (teleop) never uses this class.
    """

    _FONT = 0  # cv2.FONT_HERSHEY_SIMPLEX, spelled out so the import stays lazy

    def __init__(self, title: str = "mavrl feed", show_seg: bool = True,
                 panel: int = PANEL):
        self.cv2 = _require("cv2", "opencv-python")
        from types import SimpleNamespace
        # cv2.waitKey returns ASCII; map the keys fly.py checks onto those codes
        # so it can compare `pg.K_n in tapped` unchanged.
        self.pygame = SimpleNamespace(
            K_ESCAPE=27, K_SPACE=32, K_n=ord("n"), K_r=ord("r"), K_y=ord("y"))
        self.title = title
        self.panel = panel
        self.show_seg = show_seg
        self.alive = True
        self._last = None
        self._wait_ms = max(1, int(1000 / C.POLICY_FREQ))
        self.cv2.namedWindow(self.title, self.cv2.WINDOW_AUTOSIZE)

    # -- input ---------------------------------------------------------------

    def poll(self):
        """Pump the GUI, pace to the policy rate, and read one key.

        The wait here is what holds the loop to real time -- `tick()` is a
        no-op. Returns `(held, tapped)`; held is always empty (see class doc).
        """
        key = self.cv2.waitKey(self._wait_ms) & 0xFF
        try:
            if self.cv2.getWindowProperty(
                    self.title, self.cv2.WND_PROP_VISIBLE) < 1:
                self.alive = False          # user clicked the window's X
        except self.cv2.error:
            pass
        tapped = set() if key == 255 else {key}
        return set(), tapped

    def tick(self, fps: float = C.POLICY_FREQ) -> None:
        """No-op: `poll()`'s waitKey already paces the loop."""

    # -- drawing -------------------------------------------------------------

    def _panel(self, rgb: np.ndarray, label: str) -> np.ndarray:
        """(H,W,3) RGB -> a `panel`x`panel` BGR tile with a caption."""
        tile = self.cv2.resize(np.ascontiguousarray(rgb),
                               (self.panel, self.panel),
                               interpolation=self.cv2.INTER_NEAREST)
        tile = self.cv2.cvtColor(tile, self.cv2.COLOR_RGB2BGR)
        self.cv2.putText(tile, label, (8, 22), self._FONT, 0.5,
                         (235, 235, 235), 1, self.cv2.LINE_AA)
        return tile

    def draw(self, obs: dict, lines=()) -> None:
        if not self.alive:
            return
        tiles = [self._panel(obs["image"][..., :3], "RGB (encoder input)")]

        if "depth_m" in obs:
            depth, dlabel = obs["depth_m"], "depth (m, clean)"
        else:
            depth = obs["image"][..., 3].astype(np.float32) / 255.0 * C.DEPTH_MAX
            dlabel = "depth (from obs)"
        tiles.append(self._panel(depth_rgb(depth), dlabel))

        if self.show_seg and "seg" in obs:
            tiles.append(self._panel(seg_rgb(obs["seg"]), "segmentation"))

        strip = np.hstack(tiles)
        hud = np.full((HUD_H, strip.shape[1], 3), (20, 16, 16), np.uint8)
        for i, line in enumerate(lines):
            self.cv2.putText(hud, line, (10, 22 + i * 19), self._FONT,
                             0.45, (230, 225, 225), 1, self.cv2.LINE_AA)
        self._last = np.vstack([strip, hud])
        self.cv2.imshow(self.title, self._last)
        self.cv2.waitKey(1)                 # force the frame to paint now

    def message(self, text: str, colour=(255, 210, 120)) -> None:
        """Overlay one line in the HUD band (episode result). colour is RGB."""
        if not self.alive or self._last is None:
            return
        img = self._last.copy()
        self.cv2.putText(img, text, (10, self.panel + 26), self._FONT, 0.6,
                         tuple(reversed(colour)), 2, self.cv2.LINE_AA)
        self.cv2.imshow(self.title, img)
        self.cv2.waitKey(1)

    def close(self) -> None:
        if self.alive:
            self.alive = False
        try:
            self.cv2.destroyWindow(self.title)
            self.cv2.waitKey(1)
        except Exception:
            pass


def hud_lines(env, extra: str = "") -> list:
    """Two status lines both collectors show."""
    pos, vel = env.pos[0], env.vel[0]
    speed = float(np.linalg.norm(vel))
    gate = env.gates.current
    tgt = "done" if gate is None else f"y={gate.y:.1f} z={gate.center[2]:.2f}"
    return [
        extra,
        f"pos ({pos[0]:+.2f} {pos[1]:+.2f} {pos[2]:.2f})   "
        f"speed {speed:.2f} m/s   yaw {math.degrees(env.rpy[0, 2]):+.0f} deg",
        f"gates {env.gates_cleared}/{env.gates.n_gates}   missed "
        f"{env.missed_gates}   next gate {tgt}   t "
        f"{env.step_counter / env.SIM_FREQ:.1f}s",
    ]
