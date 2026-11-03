"""Two plain images for the X post. Numbers come from the repo's RESULTS.md and run r35. Run: python3 make_images.py"""
import subprocess

Y, B, G, O, RED, INK, MUTE = ("#d9a900", "#fff6cc"), ("#2a78d6", "#dfeafa"), ("#1baf7a", "#dff5ec"), ("#f39a4d", "#fdeedd"), "#d8342c", "#111111", "#55544f"


class Fig:
    def __init__(self): self.s = ['<svg xmlns="http://www.w3.org/2000/svg" width="1080" height="1080" viewBox="0 0 1080 1080"><defs><marker id="h" markerWidth="7" markerHeight="7" refX="6.5" refY="3.5" orient="auto"><path d="M0 0 L7 3.5 L0 7 z" fill="#111111"/></marker></defs><rect width="1080" height="1080" fill="white"/>']
    def text(self, x, y, s, size=34, weight="normal", fill=INK, anchor="middle"): self.s.append(f'<text x="{x}" y="{y}" font-family="Helvetica, Arial, sans-serif" font-size="{size}" font-weight="{weight}" fill="{fill}" text-anchor="{anchor}">{s}</text>')
    def box(self, x, y, w, h, role, stroke=None, sw=2, r=18): self.s.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{role[1]}" stroke="{stroke or role[0]}" stroke-width="{sw}"/>')
    def arrow(self, a, b): self.s.append(f'<path d="M{a[0]} {a[1]} L{b[0]} {b[1]}" fill="none" stroke="{INK}" stroke-width="2.4" marker-end="url(#h)"/>')
    def save(self, name): self.s.append("</svg>"); open(name + ".svg", "w").write("\n".join(self.s)); subprocess.run(["rsvg-convert", "-z", "2", name + ".svg", "-o", name + ".png"], check=True); subprocess.run(["rm", name + ".svg"], check=True)


f = Fig()
f.text(540, 80, "Opus beat normal training", 56, "bold")
f.box(240, 125, 600, 110, Y); f.text(540, 172, "Same training data", 38, "bold"); f.text(540, 215, "70% of the examples", 30, fill=MUTE)
f.arrow((400, 235), (285, 305)); f.arrow((680, 235), (795, 305))
f.box(50, 305, 470, 190, B); f.text(285, 370, "Normal training", 40, "bold"); f.text(285, 420, "thousands of", 32); f.text(285, 460, "small adjustments", 32)
f.box(560, 305, 470, 190, G, RED, 4); f.text(795, 370, "Opus 5.5", 40, "bold"); f.text(795, 420, "writes the weight numbers", 32); f.text(795, 460, "itself, no training", 32)
f.arrow((285, 495), (285, 555)); f.arrow((795, 495), (795, 555))
f.box(50, 555, 980, 300, ("#111111", "#ffffff"), sw=2, r=10); f.text(540, 610, "Correct on the hidden 30% neither one saw", 36, "bold")
f.text(285, 750, "83%", 120, "bold", B[0]); f.text(795, 750, "91%", 120, "bold", "#12875e"); f.text(285, 815, "best training result", 30, fill=MUTE); f.text(795, 815, "Opus", 30, fill=MUTE)
f.text(540, 920, "Average of 16 toy tasks. Opus won on 14 of 16.", 36, "bold"); f.text(540, 975, "A tiny network with 8,751 weights.", 32, fill=MUTE); f.text(540, 1020, "These tasks all have a rule. On handwritten digits, a harder task, training won.", 30, fill=MUTE)
f.save("charts/result")

f = Fig()
f.text(540, 60, "One task: find the hidden rule", 50, "bold")
W = ("#c9c8c2", "#ffffff")
f.box(50, 85, 980, 265, Y); f.text(85, 128, "1. Opus reads the examples", 34, "bold", anchor="start")
for i, t in enumerate(("0 and 0 → 14", "0 and 1 → 7", "2 and 1 → 5")): f.box(85 + i * 310, 145, 290, 58, W, r=10); f.text(230 + i * 310, 186, t, 31)
f.text(85, 240, "How a number enters the network: 32 on/off switches, one on", 29, "bold", anchor="start")
for y, name, on in ((252, "first number 0", 0), (290, "second number 1", 1)):
    f.text(85, y + 25, name, 27, anchor="start")
    for k in range(12): f.box(330 + k * 38, y, 30, 30, ("#111111", "#111111" if k == on else "#ffffff"), sw=1.2, r=4)
    f.text(800, y + 25, "… up to 31", 27, fill=MUTE, anchor="start")
f.arrow((540, 350), (540, 375))
f.box(50, 375, 980, 300, G); f.text(85, 418, "2. Opus finds the rule", 34, "bold", anchor="start"); f.text(85, 460, "Each number has a secret code. The answer is code OR code.", 29, anchor="start")
for y, t in ((480, "first 0 = code 6"), (550, "second 0 = code 8")): f.box(85, y, 330, 58, W, r=10); f.text(250, y + 40, t, 30)
for r, (name, bits, end) in enumerate((("6", "0110", ""), ("8", "1000", ""), ("OR", "1110", "= 14"))):
    y = 476 + r * 46; bold = "bold" if r == 2 else "normal"; f.text(555, y + 32, name, 30, bold, anchor="end")
    for k, b in enumerate(bits): f.box(575 + k * 48, y, 44, 42, ("#c9c8c2", "#ffffff") if r < 2 else ("#12875e", "#ffffff"), sw=1.4 if r < 2 else 2.4, r=5); f.text(597 + k * 48, y + 32, b, 30, bold)
    if end: f.text(785, y + 32, end, 32, "bold", anchor="start")
f.text(540, 652, "OR compares binary digits: the result is 1 if either digit is 1.", 29, "bold")
f.arrow((540, 675), (540, 700))
f.box(50, 700, 980, 125, O); f.text(85, 743, "3. Opus calculates every weight", 34, "bold", anchor="start"); f.text(85, 790, "It writes all 8,751 numbers so the network computes that rule.", 29, anchor="start")
f.arrow((540, 825), (540, 850))
f.box(50, 850, 980, 180, B); f.text(85, 893, "4. Test on examples Opus never saw", 34, "bold", anchor="start"); f.text(85, 942, "0 and 4 was hidden. The rule gives 6 OR 5 = 7.", 30, anchor="start"); f.text(85, 995, "This rule got 100% of its hidden examples right.", 32, "bold", anchor="start")
f.save("charts/example")
