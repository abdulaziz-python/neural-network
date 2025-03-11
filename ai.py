# bu brainfuckerni yaratishdan maqsad neuro tarmoqni o'rganish hamda miyyamni azoblash
# 3/10/2025 00:00 da boshladim

import numpy as np
import pygame
import sys
from pygame import Surface
from scipy.ndimage import gaussian_filter

class NN:
    def __init__(self, in_size, hid_sizes, out_size, lr=0.001):
        self.sizes = [in_size] + hid_sizes + [out_size]


        self.w = [np.random.randn(m, n) * np.sqrt(2.0/(m+n)) for m, n in zip(self.sizes[:-1], self.sizes[1:])]
        self.b = [np.zeros((1, n)) for n in self.sizes[1:]]


        self.a = [np.zeros((1, size)) for size in self.sizes]
        self.z = [None] * (len(self.sizes) - 1)

        self.m_w = [np.zeros_like(w) for w in self.w]
        self.v_w = [np.zeros_like(w) for w in self.w]
        self.m_b = [np.zeros_like(b) for b in self.b]
        self.v_b = [np.zeros_like(b) for b in self.b]
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8
        self.lr = lr
        self.t = 1

    def lrelu(self, x, alpha=0.01):
        return np.where(x > 0, x, alpha * x)

    def lrelu_d(self, x, alpha=0.01):
        return np.where(x > 0, 1.0, alpha)

    def softmax(self, x):
        x_max = np.max(x, axis=1, keepdims=True)
        e_x = np.exp(x - x_max)
        return e_x / np.sum(e_x, axis=1, keepdims=True)

    def forward(self, x):
        self.a[0] = x

        for i in range(len(self.w)):
            z = np.dot(self.a[i], self.w[i]) + self.b[i]
            self.z[i] = z

            if i == len(self.w) - 1:
                self.a[i+1] = self.softmax(z)
            else:
                self.a[i+1] = self.lrelu(z)

        return self.a[-1]

    def train(self, x, y, bs=32, epochs=15):
        n = x.shape[0]

        for _ in range(epochs):
            idx = np.random.permutation(n)
            x_s = x[idx]
            y_s = y[idx]

            for i in range(0, n, bs):
                end = min(i + bs, n)
                self._train_batch(x_s[i:end], y_s[i:end])

    def _train_batch(self, x, y):

        self.forward(x)

        batch_size = x.shape[0]
        delta = self.a[-1] - y

        for i in range(len(self.w) - 1, -1, -1):
            dw = np.dot(self.a[i].T, delta) / batch_size
            db = np.mean(delta, axis=0, keepdims=True)

            self.m_w[i] = self.beta1 * self.m_w[i] + (1 - self.beta1) * dw
            self.v_w[i] = self.beta2 * self.v_w[i] + (1 - self.beta2) * (dw**2)

            self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * db
            self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * (db**2)

            m_w_hat = self.m_w[i] / (1 - self.beta1**self.t)
            v_w_hat = self.v_w[i] / (1 - self.beta2**self.t)

            m_b_hat = self.m_b[i] / (1 - self.beta1**self.t)
            v_b_hat = self.v_b[i] / (1 - self.beta2**self.t)

            self.w[i] -= self.lr * m_w_hat / (np.sqrt(v_w_hat) + self.eps)
            self.b[i] -= self.lr * m_b_hat / (np.sqrt(v_b_hat) + self.eps)

            if i > 0:
                delta = np.dot(delta, self.w[i].T) * self.lrelu_d(self.z[i-1])

        self.t += 1

class ShapeApp:
    def __init__(self):
        pygame.init()
        self.w, self.h = 1000, 600
        self.scr = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption("Optimized Shape Classifier")

        self.cs = 280
        self.cx = 40
        self.cy = (self.h - self.cs) // 2
        self.canvas = Surface((self.cs, self.cs))
        self.canvas.fill((255, 255, 255))

        self.drawing = False
        self.last = None
        self.bsize = 12
        self.bcolor = (0, 0, 0)

        self.fsize = 256
        self.hsizes = [128, 64]
        self.osize = 3
        self.net = NN(self.fsize, self.hsizes, self.osize)

        self.labels = ["Square", "Triangle", "Other"]

        self.bg = (240, 240, 245)
        self.tc = (30, 30, 30)

        self.font = pygame.font.SysFont("Arial", 18)
        self.tfont = pygame.font.SysFont("Arial", 22, bold=True)

        self.gs = 16
        self.cw = self.cs // self.gs
        self.ch = self.cs // self.gs

        self.init_net()

    def init_net(self):
        n = 300

        indata = []
        outdata = []

        for _ in range(n):
            ts = Surface((self.cs, self.cs))
            ts.fill((255, 255, 255))

            size = np.random.randint(80, 180)
            x = np.random.randint(40, self.cs - size - 40)
            y = np.random.randint(40, self.cs - size - 40)
            t = np.random.randint(0, 10)

            if np.random.random() < 0.3:
                a = np.random.randint(5, 45)
                pts = self._rot_rect(x, y, size, size, a)
                pygame.draw.polygon(ts, (0, 0, 0), pts, t)
            else:
                pygame.draw.rect(ts, (0, 0, 0), (x, y, size, size), t)

            arr = pygame.surfarray.array3d(ts)
            gray = np.mean(arr, axis=2)
            bin_img = (gray < 128).astype(np.float32)

            feat = self._extract_feat(bin_img)
            indata.append(feat)
            outdata.append([1, 0, 0])

        for _ in range(n):
            ts = Surface((self.cs, self.cs))
            ts.fill((255, 255, 255))

            x1 = np.random.randint(40, self.cs - 40)
            y1 = np.random.randint(40, 80)

            x2 = np.random.randint(20, x1 - 10) if x1 > 30 else x1 + 20
            y2 = np.random.randint(y1 + 50, self.cs - 40)

            x3 = np.random.randint(x1 + 10, self.cs - 20) if x1 < self.cs - 30 else x1 - 20
            y3 = np.random.randint(y1 + 50, self.cs - 40)

            t = np.random.randint(0, 8)

            if np.random.random() < 0.3:
                cx = (x1 + x2 + x3) // 3
                cy = (y1 + y2 + y3) // 3
                a = np.random.randint(5, 45)

                x1, y1 = self._rot_pt(x1, y1, cx, cy, a)
                x2, y2 = self._rot_pt(x2, y2, cx, cy, a)
                x3, y3 = self._rot_pt(x3, y3, cx, cy, a)

            pygame.draw.polygon(ts, (0, 0, 0), [(x1, y1), (x2, y2), (x3, y3)], t)

            arr = pygame.surfarray.array3d(ts)
            gray = np.mean(arr, axis=2)
            bin_img = (gray < 128).astype(np.float32)

            feat = self._extract_feat(bin_img)
            indata.append(feat)
            outdata.append([0, 1, 0])

        for _ in range(n):
            ts = Surface((self.cs, self.cs))
            ts.fill((255, 255, 255))

            shape_type = np.random.choice(['circle', 'polygon'])

            if shape_type == 'circle':
                r = np.random.randint(40, 90)
                cx = np.random.randint(r + 20, self.cs - r - 20)
                cy = np.random.randint(r + 20, self.cs - r - 20)
                t = np.random.randint(0, 8)

                pygame.draw.circle(ts, (0, 0, 0), (cx, cy), r, t)

            else:
                num_sides = np.random.randint(5, 8)
                r = np.random.randint(40, 90)
                cx = np.random.randint(r + 20, self.cs - r - 20)
                cy = np.random.randint(r + 20, self.cs - r - 20)

                pts = []
                for i in range(num_sides):
                    angle = 2 * np.pi * i / num_sides
                    x = cx + r * np.cos(angle)
                    y = cy + r * np.sin(angle)
                    pts.append((int(x), int(y)))

                t = np.random.randint(0, 8)
                pygame.draw.polygon(ts, (0, 0, 0), pts, t)

            arr = pygame.surfarray.array3d(ts)
            gray = np.mean(arr, axis=2)
            bin_img = (gray < 128).astype(np.float32)

            feat = self._extract_feat(bin_img)
            indata.append(feat)
            outdata.append([0, 0, 1])

        inarr = np.array(indata, dtype=np.float32)
        outarr = np.array(outdata, dtype=np.float32)

        self.net.train(inarr, outarr, bs=32, epochs=15)

    def _rot_pt(self, x, y, cx, cy, a):

        ar = np.radians(a)
        cos_a, sin_a = np.cos(ar), np.sin(ar)

        x_shifted, y_shifted = x - cx, y - cy
        x_rot = x_shifted * cos_a - y_shifted * sin_a
        y_rot = x_shifted * sin_a + y_shifted * cos_a

        return int(x_rot + cx), int(y_rot + cy)

    def _rot_rect(self, x, y, w, h, a):

        pts = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]

        cx, cy = x + w // 2, y + h // 2
        return [self._rot_pt(px, py, cx, cy, a) for px, py in pts]

    def _extract_feat(self, img):
        features = []

        img_smooth = gaussian_filter(img, sigma=1.0)


        grid_features = []
        for i in range(self.gs):
            for j in range(self.gs):
                cell = img_smooth[i*self.ch:(i+1)*self.ch, j*self.cw:(j+1)*self.cw]
                grid_features.append(np.mean(cell))

        features = np.array(grid_features, dtype=np.float32)

        if len(features) < self.fsize:
            features = np.pad(features, (0, self.fsize - len(features)))
        elif len(features) > self.fsize:
            features = features[:self.fsize]

        features = np.clip(features, 0, 1)

        return features

    def draw_net(self):
        vx, vy = 360, 100
        vw, vh = 600, 350

        layer_x = [vx + 50, vx + vw//3, vx + 2*vw//3, vx + vw - 50]

        display_neurons = [min(16, self.fsize), min(12, self.hsizes[0]),
                          min(8, self.hsizes[1]), self.osize]

        for l in range(3):
            src_neurons = []
            tgt_neurons = []


            for i in range(display_neurons[l]):
                y = vy + (i+1) * vh / (display_neurons[l]+1)
                src_neurons.append((layer_x[l], y))

            for i in range(display_neurons[l+1]):
                y = vy + (i+1) * vh / (display_neurons[l+1]+1)
                tgt_neurons.append((layer_x[l+1], y))


            for i, (sx, sy) in enumerate(src_neurons):
                for j, (tx, ty) in enumerate(tgt_neurons):

                    if (i * j) % (4 if l == 0 else 2) != 0:
                        continue


                    pygame.draw.line(self.scr, (200, 200, 200), (sx, sy), (tx, ty), 1)

            for i, (x, y) in enumerate(src_neurons):
                pygame.draw.circle(self.scr, (100, 100, 255), (x, y), 5)
                pygame.draw.circle(self.scr, (0, 0, 0), (x, y), 5, 1)

        for i in range(self.osize):
            x = layer_x[3]
            y = vy + (i+1) * vh / (self.osize+1)

            act = float(self.net.a[3][0, i])
            color = (int(255*act), int(100*act), int(100*act))

            pygame.draw.circle(self.scr, color, (x, y), 10)
            pygame.draw.circle(self.scr, (0, 0, 0), (x, y), 10, 1)

            lbl = self.labels[i] + f": {act:.2f}"
            txt = self.font.render(lbl, True, self.tc)
            self.scr.blit(txt, (x + 15, y - 10))

        labels = ["Input", "Hidden 1", "Hidden 2", "Output"]
        for i, x in enumerate(layer_x):
            lbl = self.font.render(labels[i], True, self.tc)
            self.scr.blit(lbl, (x - 20, vy + vh + 10))

    def classify(self):
        try:
            arr = pygame.surfarray.array3d(self.canvas)
            gray = np.mean(arr, axis=2)
            bin_img = (gray < 128).astype(np.float32)

            feat = self._extract_feat(bin_img)

            pred = self.net.forward(feat.reshape(1, -1))
            return pred[0]
        except Exception:
            return np.array([0.33, 0.33, 0.34])

    def draw_pred(self, pred):
        cx, cy = 40, 480
        cw, ch = 280, 25

        pred = np.clip(pred, 0, 1)
        midx = np.argmax(pred)

        for i, p in enumerate(pred):
            bw = int(p * cw)
            color = (100, 200, 100) if i == midx else (200, 200, 200)
            pygame.draw.rect(self.scr, color, (cx, cy + i*35, bw, ch))
            pygame.draw.rect(self.scr, (0, 0, 0), (cx, cy + i*35, cw, ch), 1)

            lbl = self.labels[i]
            txt = self.font.render(f"{lbl}: {p:.2f}", True, self.tc)
            self.scr.blit(txt, (cx + cw + 10, cy + i*35 + 5))

        if np.max(pred) > 0.5:
            res = self.labels[midx]
            rtxt = self.tfont.render(f"Prediction: {res}", True, (50, 50, 50))
            self.scr.blit(rtxt, (cx, cy - 35))
        else:
            rtxt = self.tfont.render("Prediction: Uncertain", True, (50, 50, 50))
            self.scr.blit(rtxt, (cx, cy - 35))

    def clear(self):
        self.canvas.fill((255, 255, 255))

    def run(self):
        clock = pygame.time.Clock()

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_c:
                        self.clear()

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        mx, my = event.pos
                        if (self.cx <= mx <= self.cx + self.cs and
                            self.cy <= my <= self.cy + self.cs):
                            self.drawing = True
                            self.last = (mx - self.cx, my - self.cy)

                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:
                        self.drawing = False
                        self.last = None

                elif event.type == pygame.MOUSEMOTION:
                    if self.drawing:
                        mx, my = event.pos
                        if (self.cx <= mx <= self.cx + self.cs and
                            self.cy <= my <= self.cy + self.cs):
                            curr = (mx - self.cx, my - self.cy)
                            if self.last:
                                pygame.draw.line(
                                    self.canvas,
                                    self.bcolor,
                                    self.last,
                                    curr,
                                    self.bsize
                                )
                            self.last = curr
            self.scr.fill(self.bg)

            pygame.draw.rect(
                self.scr,
                (100, 100, 100),
                (self.cx - 2, self.cy - 2, self.cs + 4, self.cs + 4),
                2
            )
            self.scr.blit(self.canvas, (self.cx, self.cy))

            pred = self.classify()

            self.draw_net()
            self.draw_pred(pred)

            instr = [
                "Draw a shape in the canvas (square, triangle, or other)",
                "Press 'C' to clear the canvas",
            ]

            for i, ins in enumerate(instr):
                txt = self.font.render(ins, True, self.tc)
                self.scr.blit(txt, (40, 30 + i * 25))

            pygame.display.flip()
            clock.tick(60)

if __name__ == "__main__":
    app = ShapeApp()
    try:
        app.run()
    except Exception:
        pygame.quit()


# nihoyat brainfuckerni yozib tugatdim hamda o'qitdim.
# 3/11/2025 3:53 AM da tugatdim 440 qator kod yozish oson lekin bunaqa neuro tarmoq yaratish azob.
# buni o'qiyotganingizda mehnatimni qadirlaysiz deb umid qilaman.
