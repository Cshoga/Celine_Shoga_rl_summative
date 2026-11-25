"""pygame-ce visualisation for GovMate.

Shows the citizen's journey through the agencies (step chain), live counters
(days, fees, patience), the document inventory (what the agent knows vs. the
hidden truth), and a rolling log of the agent's actions and rewards.
"""
from __future__ import annotations

import sys

import numpy as np
import pygame

W, H = 1180, 740

BG = (18, 22, 30)
PANEL = (28, 34, 46)
TEXT = (230, 235, 245)
MUTED = (140, 150, 170)
GREEN = (52, 168, 105)
AMBER = (240, 170, 40)
RED = (214, 72, 72)
BLUE = (70, 130, 220)
GREY = (62, 70, 88)

OUTCOME_COLOR = {"success": GREEN, "rejected": RED, "delayed": AMBER, "n/a": MUTED}
END_BANNER = {
    "goal_complete": ("GOAL COMPLETE", GREEN),
    "abandoned": ("CITIZEN ABANDONED THE PROCESS", RED),
    "closed_early": ("SESSION CLOSED EARLY", RED),
    "timeout": ("TIME / STEP LIMIT REACHED", AMBER),
}
GOAL_TITLES = {
    "business_registration": "Business registration",
    "driving_licence": "Definitive driving licence",
    "land_tax": "Land tax payment",
    "passport": "Passport",
    "building_permit": "Building permit",
}


class GovMateRenderer:
    def __init__(self, env, mode: str = "human"):
        self.env = env
        self.mode = mode
        pygame.font.init()
        self.fonts = {
            "title": self._font(30, bold=True),
            "h": self._font(21, bold=True),
            "body": self._font(18),
            "small": self._font(15),
        }
        self.surface = pygame.Surface((W, H))
        self.screen = None
        self.clock = None
        if mode == "human":
            pygame.init()
            pygame.display.set_caption("GovMate - RL agent guiding citizens through government services")
            self.screen = pygame.display.set_mode((W, H))
            self.clock = pygame.time.Clock()

    # ---------------------------------------------------------------- utils
    @staticmethod
    def _font(size, bold=False):
        f = pygame.font.SysFont("segoeui,arial,dejavusans,helvetica", size, bold=bold)
        return f

    def _text(self, txt, font="body", color=TEXT, pos=(0, 0), center=False, right=False):
        img = self.fonts[font].render(str(txt), True, color)
        r = img.get_rect()
        if center:
            r.center = pos
        elif right:
            r.topright = pos
        else:
            r.topleft = pos
        self.surface.blit(img, r)
        return r

    def _wrap(self, txt, font, max_w):
        words, lines, cur = txt.split(), [], ""
        for w in words:
            test = (cur + " " + w).strip()
            if self.fonts[font].size(test)[0] <= max_w:
                cur = test
            else:
                lines.append(cur)
                cur = w
        if cur:
            lines.append(cur)
        return lines

    def _card(self, rect, color=PANEL, border=None, width=2, radius=10):
        pygame.draw.rect(self.surface, color, rect, border_radius=radius)
        if border:
            pygame.draw.rect(self.surface, border, rect, width, border_radius=radius)

    def _bar(self, x, y, w, h, frac, color):
        pygame.draw.rect(self.surface, GREY, (x, y, w, h), border_radius=6)
        pygame.draw.rect(self.surface, color, (x, y, max(0, int(w * min(max(frac, 0), 1))), h),
                         border_radius=6)

    # ---------------------------------------------------------------- panels
    def _header(self):
        env = self.env
        self._text("GovMate", "title", TEXT, (30, 18))
        title = GOAL_TITLES.get(env.goal, env.goal)
        belief = "agent knows the goal" if env.goal_known else "agent does NOT know the goal yet"
        self._text(f"Citizen's goal: {title}", "h", AMBER, (190, 24))
        self._text(f"({belief})", "small", GREEN if env.goal_known else MUTED, (190, 50))
        self._text(f"step {env.steps}/{env.meta['max_steps']}", "body", MUTED, (W - 30, 26), right=True)
        if env.end_reason:
            msg, col = END_BANNER.get(env.end_reason, (env.end_reason, MUTED))
            r = pygame.Rect(0, 0, 400, 38)
            r.center = (830, 46)
            self._card(r, col, radius=8)
            self._text(msg, "h", (255, 255, 255), r.center, center=True)

    def _journey(self):
        env = self.env
        steps = env.goals[env.goal]
        nxt = env.next_step()
        cols, bw, bh, gx, gy = 4, 172, 84, 40, 44
        x0, y0 = 30, 125
        self._text("Journey through agencies", "h", MUTED, (x0, y0 - 28))
        centers = []
        for i, s in enumerate(steps):
            r_i, c_i = divmod(i, cols)
            rect = pygame.Rect(x0 + c_i * (bw + gx), y0 + r_i * (bh + gy), bw, bh)
            centers.append(rect)
        # connectors first (so boxes draw over them)
        for i in range(len(steps) - 1):
            a, b = centers[i], centers[i + 1]
            col = GREEN if steps[i]["id"] in env.completed else GREY
            if a.y == b.y:
                pygame.draw.line(self.surface, col, a.midright, b.midleft, 3)
                pygame.draw.polygon(self.surface, col, [(b.left, b.centery),
                                    (b.left - 9, b.centery - 6), (b.left - 9, b.centery + 6)])
            else:
                mid_y = a.bottom + gy // 2
                pygame.draw.lines(self.surface, col, False,
                                  [a.midbottom, (a.centerx, mid_y), (b.centerx, mid_y), b.midtop], 3)
        for i, s in enumerate(steps):
            rect = centers[i]
            done = s["id"] in env.completed
            is_next = nxt is not None and s["id"] == nxt["id"]
            fill = (30, 70, 52) if done else (70, 56, 24) if is_next else PANEL
            border = GREEN if done else AMBER if is_next else GREY
            self._card(rect, fill, border, 3 if is_next else 2)
            lines = self._wrap(f"{i + 1}. {s['name']}", "small", bw - 16)[:3]
            for j, ln in enumerate(lines):
                self._text(ln, "small", TEXT, (rect.x + 8, rect.y + 6 + j * 17))
            tag = f"{s['kind']} | {'in person' if s['in_person'] else 'online'}"
            self._text(tag, "small", MUTED, (rect.x + 8, rect.bottom - 22))
            if done:
                self._text("done", "small", GREEN, (rect.right - 8, rect.bottom - 22), right=True)
            if is_next:   # citizen token
                cx, cy = rect.centerx, rect.y - 12
                pygame.draw.circle(self.surface, AMBER, (cx, cy), 10)
                pygame.draw.circle(self.surface, (255, 255, 255), (cx, cy), 10, 2)
                if not env.goal_known:
                    self._text("?", "small", (0, 0, 0), (cx, cy), center=True)
        # agency list for this goal
        agencies = []
        for s in steps:
            if s["agency"] not in agencies:
                agencies.append(s["agency"])
        self._text("Agencies: " + "  |  ".join(agencies), "small", MUTED, (x0, y0 + 2 * bh + gy + 8))

    def _side(self):
        env = self.env
        rect = pygame.Rect(W - 330, 100, 300, 395)
        self._card(rect)
        x, y = rect.x + 16, rect.y + 12
        self._text("Citizen status", "h", TEXT, (x, y))
        y += 36
        frac_d = env.days / env.meta["max_days"]
        self._text(f"Days elapsed: {env.days:.0f} / {env.meta['max_days']}", "body", TEXT, (x, y))
        self._bar(x, y + 24, 268, 12, frac_d, BLUE)
        y += 52
        self._text(f"Fees spent: {env.fees:,.0f} RWF", "body", TEXT, (x, y))
        self._bar(x, y + 24, 268, 12, env.fees / env.meta["max_fees_rwf"], AMBER)
        y += 52
        pf = max(env.patience, 0) / env.patience_max
        pcol = GREEN if pf > 0.5 else AMBER if pf > 0.25 else RED
        self._text(f"Patience: {max(env.patience, 0):.1f} / {env.patience_max:.0f}", "body", TEXT, (x, y))
        self._bar(x, y + 24, 268, 12, pf, pcol)
        y += 56
        dig = "yes" if env.digital else "no"
        known = "" if env.digital_known else "  (not asked yet)"
        self._text(f"Digital access: {dig}{known}", "body", TEXT, (x, y))
        y += 28
        self._text(f"Alternative channel: {'in use' if env.alt_channel else 'no'}", "body", TEXT, (x, y))
        y += 28
        self._text(f"Human escalations: {env.escalations} / 2", "body", TEXT, (x, y))
        y += 34
        total = len(env.goals[env.goal])
        self._text(f"Progress: {len(env.completed)} / {total} steps", "body", GREEN, (x, y))

    def _docs(self):
        env = self.env
        x0, y0 = 30, 440
        self._text("Documents  (fill = what the agent knows;  dot = true state)", "h", MUTED, (x0, y0 - 30))
        cw, ch, gap = 190, 46, 12
        for i, d in enumerate(env.docs):
            r_i, c_i = divmod(i, 4)
            rect = pygame.Rect(x0 + c_i * (cw + gap), y0 + r_i * (ch + gap), cw, ch)
            k = env.docs_known[d]
            fill = (30, 70, 52) if k == 1.0 else (86, 38, 42) if k == 0.0 else PANEL
            border = GREEN if k == 1.0 else RED if k == 0.0 else GREY
            self._card(rect, fill, border, 2, radius=8)
            self._text(d, "small", TEXT, (rect.x + 10, rect.centery - 8))
            dot = GREEN if env.docs_true[d] else RED
            pygame.draw.circle(self.surface, dot, (rect.right - 14, rect.centery), 6)

    def _log(self):
        env = self.env
        rect = pygame.Rect(30, 590, 1120, 130)
        self._card(rect)
        self._text("Agent actions", "h", MUTED, (rect.x + 14, rect.y + 6))
        rows = env.action_log[-5:]
        for j, (st, name, outcome, rew) in enumerate(rows):
            y = rect.y + 34 + j * 18
            self._text(f"{st:>3}", "small", MUTED, (rect.x + 14, y))
            self._text(name, "small", TEXT, (rect.x + 60, y))
            self._text(outcome, "small", OUTCOME_COLOR.get(outcome, MUTED), (rect.x + 330, y))
            self._text(f"{rew:+.2f}", "small", GREEN if rew >= 0 else RED, (rect.x + 430, y))
        self._text(f"Episode return: {env.episode_return:+.2f}", "h", TEXT, (rect.right - 20, rect.y + 12), right=True)
        self._text(f"Last reward: {env.last_reward:+.2f}", "body", MUTED, (rect.right - 20, rect.y + 44), right=True)

    # ------------------------------------------------------------------ draw
    def draw(self):
        self.surface.fill(BG)
        self._header()
        self._journey()
        self._side()
        self._docs()
        self._log()
        if self.mode == "human":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    sys.exit(0)
            self.screen.blit(self.surface, (0, 0))
            pygame.display.flip()
            self.clock.tick(self.env.metadata["render_fps"])
            return None
        return np.transpose(pygame.surfarray.array3d(self.surface), (1, 0, 2))

    def close(self):
        if self.mode == "human" and pygame.get_init():
            pygame.display.quit()
            pygame.quit()
