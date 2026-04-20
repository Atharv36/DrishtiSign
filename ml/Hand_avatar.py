"""
hand_avatar_3d.py
─────────────────────────────────────────────────────────────────
3D Human-Like Hand Avatar using PyOpenGL + pygame.
Renders into an offscreen surface and returns a numpy BGR frame
that OpenCV can composite into the main DrishtiSign window.

Drop-in replacement for HandAnimator — same API:
    animator = HandAvatar3D()
    animator.load_gesture("A")
    animator.draw(canvas)          # canvas is a numpy BGR array
"""

import os, json, time
import numpy as np
import cv2
import pygame
from pygame.locals import DOUBLEBUF, OPENGL, NOFRAME
from OpenGL.GL  import *
from OpenGL.GLU import *

# ── MediaPipe hand topology ───────────────────────────────────
BONES = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle
    (0, 9), (9, 10), (10, 11), (11, 12),
    # Ring
    (0, 13), (13, 14), (14, 15), (15, 16),
    # Pinky
    (0, 17), (17, 18), (18, 19), (19, 20),
    # Palm knuckles
    (5, 9), (9, 13), (13, 17), (0, 17),
]

# Joint radius (normalized units)
JOINT_R = {
    0:  0.038,   # Wrist  — biggest
    1:  0.028, 2: 0.026, 3: 0.024, 4: 0.020,  # Thumb
    5:  0.030, 6: 0.025, 7: 0.022, 8: 0.018,  # Index
    9:  0.030, 10:0.025, 11:0.022,12:0.018,   # Middle
    13: 0.028, 14:0.024, 15:0.021,16:0.018,   # Ring
    17: 0.026, 18:0.022, 19:0.019,20:0.016,   # Pinky
}

# Bone radius
BONE_R = 0.016

# Skin color (R G B A) in 0-1
SKIN      = (0.92, 0.75, 0.60, 1.0)
SKIN_DARK = (0.70, 0.52, 0.38, 1.0)   # slightly darker for bones
NAIL      = (0.95, 0.80, 0.80, 1.0)   # fingernail tint on tips

GESTURE_DIR = os.path.dirname(os.path.abspath(__file__))


# ── Maths helpers ─────────────────────────────────────────────
def cosine_interp(p1, p2, t):
    s = (1 - np.cos(t * np.pi)) / 2
    return p1 * (1 - s) + p2 * s


def look_at(eye, center, up):
    f = center - eye;  f /= np.linalg.norm(f)
    r = np.cross(f, up); r /= np.linalg.norm(r)
    u = np.cross(r, f)
    M = np.eye(4)
    M[0,:3]=r; M[1,:3]=u; M[2,:3]=-f
    T = np.eye(4)
    T[:3,3] = -eye
    return (M @ T).T          # column-major for OpenGL


# ── OpenGL draw primitives ────────────────────────────────────
def _set_material(color):
    r, g, b, a = color
    glMaterialfv(GL_FRONT, GL_AMBIENT,  [r*0.4, g*0.4, b*0.4, a])
    glMaterialfv(GL_FRONT, GL_DIFFUSE,  [r,     g,     b,     a])
    glMaterialfv(GL_FRONT, GL_SPECULAR, [0.6,   0.5,   0.5,   1.0])
    glMaterialf (GL_FRONT, GL_SHININESS, 40.0)


def draw_sphere(x, y, z, r, color):
    _set_material(color)
    glPushMatrix()
    glTranslatef(x, y, z)
    q = gluNewQuadric()
    gluQuadricNormals(q, GLU_SMOOTH)
    gluSphere(q, r, 20, 20)
    gluDeleteQuadric(q)
    glPopMatrix()


def draw_cylinder(p1, p2, r, color):
    """Draw a smooth cylinder between two 3-D points."""
    _set_material(color)
    d    = p2 - p1
    length = np.linalg.norm(d)
    if length < 1e-6:
        return

    # Align Z-axis to direction d
    z    = d / length
    perp = np.array([1, 0, 0]) if abs(z[0]) < 0.9 else np.array([0, 1, 0])
    x    = np.cross(perp, z); x /= np.linalg.norm(x)
    y    = np.cross(z, x)

    M = np.eye(4, dtype=np.float32)
    M[:3,0]=x; M[:3,1]=y; M[:3,2]=z; M[:3,3]=p1

    glPushMatrix()
    glMultMatrixf(M)
    q = gluNewQuadric()
    gluQuadricNormals(q, GLU_SMOOTH)
    gluCylinder(q, r, r * 0.85, length, 16, 2)   # slight taper
    gluDeleteQuadric(q)
    glPopMatrix()


# ── Landmark processing ───────────────────────────────────────
def landmarks_to_3d(frame_data):
    """
    Convert stored [x, y] (or [x, y, z]) landmark list to a
    centred, normalised (21, 3) numpy array ready for rendering.
    """
    pts = np.array([[p[0], p[1], p[2] if len(p) > 2 else 0.0]
                    for p in frame_data], dtype=np.float32)

    # Flip Y — MediaPipe Y is top-down, OpenGL Y is up
    pts[:, 1] = 1.0 - pts[:, 1]

    # Centre on wrist
    pts -= pts[0]

    # Scale so the hand fills ~0.8 units
    span = np.max(np.linalg.norm(pts, axis=1))
    if span > 1e-6:
        pts = pts / span * 0.8

    # Shift hand to a nice view position (wrist at origin, palm faces camera)
    pts[:, 2] -= 0.1    # push slightly back for depth feel

    return pts


# ── Main avatar class ─────────────────────────────────────────
class HandAvatar3D:
    """
    3D skinned hand avatar.  Renders into a pygame/OpenGL offscreen
    surface and copies the result to a numpy BGR canvas for OpenCV.
    """

    def __init__(self, width=400, height=400):
        self.W = width
        self.H = height

        self.frames   = []
        self.current  = ""
        self.frame_idx = 0
        self.step      = 0
        self.steps_per_frame = 8
        self.last_time = time.time()
        self.speed     = 0.025

        # Rotation state (mouse / auto-rotate)
        self.rot_x  = -15.0
        self.rot_y  =  20.0
        self.auto_rotate = True

        self._init_gl()

    # ── OpenGL init ───────────────────────────────────────────
    def _init_gl(self):
        if not pygame.get_init():
            pygame.init()

        # Hidden offscreen OpenGL window
        self._screen = pygame.display.set_mode(
            (self.W, self.H), DOUBLEBUF | OPENGL | NOFRAME)
        pygame.display.set_caption("Hand Avatar")

        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_LIGHT1)
        glEnable(GL_COLOR_MATERIAL)
        glEnable(GL_NORMALIZE)
        glShadeModel(GL_SMOOTH)

        # Main light — warm front light
        glLightfv(GL_LIGHT0, GL_POSITION, [0.5, 1.0, 1.5, 0.0])
        glLightfv(GL_LIGHT0, GL_DIFFUSE,  [1.0, 0.95, 0.85, 1.0])
        glLightfv(GL_LIGHT0, GL_AMBIENT,  [0.25, 0.22, 0.20, 1.0])
        glLightfv(GL_LIGHT0, GL_SPECULAR, [0.8, 0.7, 0.6, 1.0])

        # Fill light — cool side light
        glLightfv(GL_LIGHT1, GL_POSITION, [-1.5, 0.5, 0.5, 0.0])
        glLightfv(GL_LIGHT1, GL_DIFFUSE,  [0.3, 0.35, 0.5, 1.0])
        glLightfv(GL_LIGHT1, GL_AMBIENT,  [0.0, 0.0, 0.0, 1.0])
        glLightfv(GL_LIGHT1, GL_SPECULAR, [0.0, 0.0, 0.0, 1.0])

        # Projection
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(40.0, self.W / self.H, 0.01, 100.0)
        glMatrixMode(GL_MODELVIEW)

    # ── Gesture loading ───────────────────────────────────────
    def load_gesture(self, name):
        if self.current == name:
            return
        for fname in [f"{name}.json", f"{name.lower()}.json"]:
            fpath = os.path.join(GESTURE_DIR, fname)
            if os.path.exists(fpath):
                with open(fpath) as f:
                    raw = json.load(f)
                self.frames    = [landmarks_to_3d(frame) for frame in raw]
                self.frame_idx = 0
                self.step      = 0
                self.current   = name
                print(f"[HandAvatar3D] Loaded '{name}' — {len(self.frames)} frames")
                return
        print(f"[HandAvatar3D] '{name}.json' not found in {GESTURE_DIR}")
        self.frames  = []
        self.current = ""

    # ── Render one frame ──────────────────────────────────────
    def _render_pose(self, pts):
        """Render all joints and bones for a single pose."""

        # ── Bones (cylinders) ─────────────────────────────────
        for a, b in BONES:
            draw_cylinder(pts[a], pts[b], BONE_R, SKIN_DARK)

        # ── Joints (spheres) ──────────────────────────────────
        for i, pt in enumerate(pts):
            color = NAIL if i in (4, 8, 12, 16, 20) else SKIN
            draw_sphere(pt[0], pt[1], pt[2], JOINT_R.get(i, 0.022), color)

    # ── Main draw call ────────────────────────────────────────
    def draw(self, canvas):
        """
        Render the 3D avatar and composite it into the numpy BGR canvas.
        canvas: numpy array (H, W, 3) BGR — the panel slice from OpenCV.
        """
        H_c, W_c = canvas.shape[:2]

        # ── Advance animation ─────────────────────────────────
        now = time.time()
        if len(self.frames) >= 2 and now - self.last_time >= self.speed:
            self.last_time = now
            self.step += 1
            if self.step >= self.steps_per_frame:
                self.step = 0
                self.frame_idx = (self.frame_idx + 1) % (len(self.frames) - 1)

        if self.auto_rotate:
            self.rot_y = (self.rot_y + 0.4) % 360

        # ── Handle pygame events (keeps window responsive) ────
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN:
                self.auto_rotate = not self.auto_rotate

        # ── OpenGL render ─────────────────────────────────────
        glViewport(0, 0, self.W, self.H)
        glClearColor(0.08, 0.08, 0.12, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glLoadIdentity()
        gluLookAt(0, 0, 2.2,   # camera position
                  0, 0, 0,     # look at origin
                  0, 1, 0)     # up vector

        # Apply rotation
        glRotatef(self.rot_x, 1, 0, 0)
        glRotatef(self.rot_y, 0, 1, 0)

        # Draw pose
        if len(self.frames) >= 2:
            f1 = self.frames[self.frame_idx]
            f2 = self.frames[self.frame_idx + 1]
            t  = self.step / self.steps_per_frame
            pose = cosine_interp(f1, f2, t)
            self._render_pose(pose)
        elif len(self.frames) == 1:
            self._render_pose(self.frames[0])

        pygame.display.flip()

        # ── Grab framebuffer → numpy ──────────────────────────
        glPixelStorei(GL_PACK_ALIGNMENT, 1)
        raw = glReadPixels(0, 0, self.W, self.H, GL_RGB, GL_UNSIGNED_BYTE)
        gl_img = np.frombuffer(raw, dtype=np.uint8).reshape(self.H, self.W, 3)
        gl_img = np.flipud(gl_img)                        # OpenGL is bottom-up
        bgr    = cv2.cvtColor(gl_img, cv2.COLOR_RGB2BGR)

        # ── Resize to fit panel ───────────────────────────────
        if bgr.shape[:2] != (H_c, W_c):
            bgr = cv2.resize(bgr, (W_c, H_c), interpolation=cv2.INTER_LINEAR)

        # ── Overlay label ─────────────────────────────────────
        label = self.current or "No gesture"
        cv2.putText(bgr, label,
                    (15, H_c - 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2, cv2.LINE_AA)

        # Playback progress bar
        if len(self.frames) >= 2:
            total   = (len(self.frames) - 1) * self.steps_per_frame
            current = self.frame_idx * self.steps_per_frame + self.step
            bar_w   = W_c - 30
            fill    = int(bar_w * current / max(total, 1))
            cv2.rectangle(bgr, (15, H_c-7), (15+bar_w, H_c-3), (40,40,40), -1)
            cv2.rectangle(bgr, (15, H_c-7), (15+fill,  H_c-3), (0,200,120), -1)

        # "Auto-rotate ON/OFF" hint
        hint = "Click avatar: toggle rotate"
        cv2.putText(bgr, hint, (15, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (140, 140, 140), 1, cv2.LINE_AA)

        canvas[:] = bgr

    def close(self):
        pygame.quit()