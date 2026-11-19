#!/usr/bin/env python3
"""finish_film.py: sound design + titles + subtitles + encode for a 3jsvideo film.
  finish_film.py frames/ out.mp4 --tts tts/ [--fps 24] [--size 1920x1080] [--no-subs]
frames/_film.json (written by render.js) gives the timeline: segments {id, kind, spk, text, t0, t1, set, world, shot}.
Narration wavs come from tts/<id>.wav (24 kHz). Everything else (ambience, SFX, drone) is synthesized here."""
import argparse, json, os, glob, subprocess, wave
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import soundfile as sf

SR = 48000
ap = argparse.ArgumentParser(); ap.add_argument('frames'); ap.add_argument('out'); ap.add_argument('--tts', default='tts')
ap.add_argument('--fps', type=int, default=24); ap.add_argument('--size', default='1920x1080'); ap.add_argument('--no-subs', action='store_true')
ap.add_argument('--audio-only', action='store_true')
ap.add_argument('--start', type=int, default=None, help='chunk mode: index of the first frame in frames/ (video only, no audio)')
a = ap.parse_args()
F = json.load(open(os.path.join(a.frames, '_film.json'))); SEG = F['segments']; LEN = F['len']; N = int(LEN * SR)
rng = np.random.default_rng(3); t = np.arange(N) / SR
HERE = os.path.dirname(os.path.abspath(__file__))

# ------------------------------------------------------------------ dsp helpers
def band(x, lo, hi):
    X = np.fft.rfft(x); f = np.fft.rfftfreq(len(x), 1 / SR); X[(f < lo) | (f > hi)] = 0; return np.fft.irfft(X, len(x))
def conv(x, ir): n = len(x) + len(ir); return np.fft.irfft(np.fft.rfft(x, n) * np.fft.rfft(ir, n), n)[:len(x)]
def room(sec, decay): L = int(sec * SR); ir = rng.standard_normal(L) * np.exp(-np.arange(L) / (decay * SR)); ir[:int(0.01 * SR)] *= 0.2; ir[0] = 1; return ir / np.abs(ir).sum() * 6
def movavg(x, k):
    k = max(1, int(k)); c = np.cumsum(np.concatenate([[0.0], x])); y = (c[k:] - c[:-k]) / k
    pad = len(x) - len(y); return np.concatenate([np.full(pad // 2, y[0]), y, np.full(pad - pad // 2, y[-1])])
def seg_mask(pred, fade=0.6):
    m = np.zeros(N)
    for s in SEG:
        if pred(s): m[int(s['t0'] * SR):int(s['t1'] * SR)] = 1
    return movavg(m, fade * SR)
def place(dst, x, at, gain=1.0):
    i = int(at * SR); n = min(len(x), N - i)
    if n > 0 and i >= 0: dst[i:i + n] += x[:n] * gain
def resample(x, sr):
    if sr == SR: return x
    idx = np.arange(int(len(x) * SR / sr)) * sr / SR; return np.interp(idx, np.arange(len(x)), x)
def env(L, a, d): e = np.ones(L); ia = int(a * SR); id_ = int(d * SR); e[:ia] = np.linspace(0, 1, ia) if ia else 1; e[-id_:] *= np.linspace(1, 0, id_) if id_ else 1; return e
def A(shot, last=False): xs = [s for s in SEG if s['shot'] == shot]; return (xs[-1] if last else xs[0])['t0'] if xs else None

# ------------------------------------------------------------------ narration with per-voice treatment
voice = np.zeros(N)
verb_small = room(0.6, 0.12); verb_train = room(1.0, 0.25)
for s in SEG:
    if s['kind'] != 'line': continue
    p = os.path.join(a.tts, s['id'] + '.wav')
    if not os.path.exists(p): continue
    x, sr = sf.read(p); x = resample(x if x.ndim == 1 else x.mean(1), sr); x = x / (np.abs(x).max() + 1e-6) * 0.8
    if s['spk'] == 'PHONE':
        x = band(x, 320, 3200); x = np.tanh(x * 2.2) * 0.6; x += band(rng.standard_normal(len(x)), 1500, 7000) * 0.025
    elif s['spk'] == 'PA':
        x = band(x, 450, 3800); x = np.tanh(x * 3) * 0.55; x = x + conv(x, verb_train) * 0.5; x += band(rng.standard_normal(len(x)), 2000, 9000) * 0.03
    elif s['spk'] == 'COAT':
        x = x + conv(x, verb_small) * 0.25; x = band(x, 60, 9000) * 1.1
    elif s['spk'] == 'HEY':
        x = x * 0.9 + conv(x, room(0.8, 0.2)) * 0.3
    elif s['spk'] == 'OLD':
        x = x + conv(x, verb_small) * 0.15
    world_train = s['world'] == 'train'
    if world_train and s['spk'] in ('ME', 'COAT', 'DOUBLE'): x = x + conv(x, verb_train) * 0.12
    place(voice, x, s['t0'] + 0.12)
print('narration placed')

# ------------------------------------------------------------------ ambience beds per set
def noise_bed(lo, hi, seed): r = np.random.default_rng(seed).standard_normal(N); b = band(r, lo, hi); return b / (np.abs(b).max() + 1e-9)
city = noise_bed(60, 900, 1) * 0.5 + noise_bed(300, 2500, 2) * 0.25 * (0.6 + 0.4 * np.sin(2 * np.pi * 0.07 * t))
cars = np.zeros(N)
for k in range(int(LEN / 4)):
    c = rng.uniform(0, LEN); L = int(SR * rng.uniform(2.5, 5)); e = np.sin(np.linspace(0, np.pi, L)) ** 2
    place(cars, band(rng.standard_normal(L), 80, 700) * e * 0.35, c)
honk = np.zeros(N)
for k in range(int(LEN / 25)):
    c = rng.uniform(0, LEN); L = int(SR * 0.35); tt = np.arange(L) / SR; f0 = rng.choice([392, 440, 466])
    place(honk, (np.sign(np.sin(2 * np.pi * f0 * tt)) * 0.5 + np.sin(2 * np.pi * f0 * 1.26 * tt) * 0.5) * env(L, 0.01, 0.05) * 0.06, c)
busk = np.zeros(N)   # street performer: plucked chords
for k in range(int(LEN * 1.6)):
    c = k / 1.6; f0 = [196, 247, 294, 220, 262, 330][k % 6]; L = int(SR * 1.2); tt = np.arange(L) / SR
    place(busk, sum(np.sin(2 * np.pi * f0 * h * tt) / h for h in (1, 2, 3)) * np.exp(-tt * 3) * 0.05, c)
crowd = noise_bed(250, 1800, 3)
murmur = crowd * (0.7 + 0.3 * np.sin(2 * np.pi * 0.21 * t) * np.sin(2 * np.pi * 0.05 * t + 1))
station = noise_bed(40, 300, 4) * 0.6 + np.sin(2 * np.pi * 60 * t) * 0.05
rumble = noise_bed(25, 220, 5) * (0.8 + 0.2 * np.sin(2 * np.pi * 1.6 * t)) + noise_bed(800, 3000, 6) * 0.08
roomtone = noise_bed(80, 400, 7) * 0.25
m_city = seg_mask(lambda s: s['world'] == 'street', 1.0)
m_mall = seg_mask(lambda s: s['set'] in ('mall', 'foodcourt'), 1.0)
m_store = seg_mask(lambda s: s['set'] == 'storefront', 1.0)
m_station = seg_mask(lambda s: s['set'] == 'station', 1.0)
m_train = seg_mask(lambda s: s['world'] == 'train' and s['set'] != 'station', 1.0)
m_bed = seg_mask(lambda s: s['world'] == 'bedroom', 1.0)
# the crowd falls silent in chapter 2, tinnitus tone instead
t_quiet = A('crowd_frozen_wide'); t_run2 = A('narr_run_away_wide')
quiet = np.ones(N)
if t_quiet: qi = int(t_quiet * SR); quiet[qi:] = np.exp(-np.arange(N - qi) / (0.6 * SR)) * 0.95 + 0.05
if t_run2: ri = int((t_run2 + 0.5) * SR); quiet[ri:] = np.minimum(1, quiet[ri:] + np.linspace(0, 1, N - ri) * 4)
tinn = np.sin(2 * np.pi * 7400 * t) * 0.012 * (1 - quiet) * m_city
amb = (city * 0.16 + cars + honk + busk * 0.6 + murmur * 0.22) * m_city * quiet + tinn
amb += (murmur * 0.42 + city * 0.06 + busk * 0.3) * m_mall + murmur * 0.12 * m_store
amb += station * 0.22 * m_station + roomtone * 0.5 * m_bed
t_close, t_home, t_black = A('doors_close'), A('doors_open_bedroom'), A('blackout')
mv = np.zeros(N)
if t_close and t_home: i0, i1 = int((t_close + 1.2) * SR), int((t_home - 1.5) * SR); mv[i0:i1] = 1; mv = movavg(mv, SR)
amb += rumble * (0.12 + 0.3 * mv) * m_train

# ------------------------------------------------------------------ sound effects
sfx = np.zeros(N)
def steps(t0, t1, rate, gain, seed=0):
    r = np.random.default_rng(seed); k = t0
    while k < t1:
        L = int(0.07 * SR); x = band(r.standard_normal(L), 80, 2500) * np.exp(-np.arange(L) / (0.012 * SR)); place(sfx, x * gain, k); k += 1 / rate * r.uniform(0.92, 1.08)
def hb(t0, t1, bpm, gain):
    k = t0
    while k < t1:
        for o, g in ((0, 1), (0.22, 0.7)):
            L = int(0.18 * SR); tt = np.arange(L) / SR; place(sfx, np.sin(2 * np.pi * 52 * tt) * np.exp(-tt * 22) * gain * g, k + o)
        k += 60 / bpm
def ring(t0, dur):
    k = t0
    while k < t0 + dur:
        L = int(1.0 * SR); tt = np.arange(L) / SR; x = (np.sin(2 * np.pi * 1400 * tt) + np.sin(2 * np.pi * 1750 * tt)) * (np.sin(2 * np.pi * 20 * tt) > 0) * 0.08
        place(sfx, x * env(L, 0.01, 0.05), k); k += 2.6
def knock(at, gain=0.9, wood=True):
    for i in range(3):
        L = int(0.25 * SR); tt = np.arange(L) / SR
        x = (np.sin(2 * np.pi * (180 if wood else 900) * tt) * np.exp(-tt * 30) + band(rng.standard_normal(L), 200, 3000) * np.exp(-tt * 60) * 0.6)
        place(sfx, x * gain, at + i * 0.62)
def static(at, dur, gain): L = int(dur * SR); place(sfx, band(rng.standard_normal(L), 800, 9000) * gain * env(L, 0.05, 0.2), at)
def whoosh(at, dur, gain): L = int(dur * SR); tt = np.linspace(0, 1, L); place(sfx, band(rng.standard_normal(L), 200, 3000) * np.sin(np.pi * tt) ** 2 * gain, at)
def boom(at, gain): L = int(2.5 * SR); tt = np.arange(L) / SR; place(sfx, np.sin(2 * np.pi * 38 * tt) * np.exp(-tt * 1.6) * gain, at)
def buzz(at, dur, gain): L = int(dur * SR); tt = np.arange(L) / SR; place(sfx, np.sign(np.sin(2 * np.pi * 120 * tt)) * band(rng.standard_normal(L), 100, 4000) * 0.3 * gain * env(L, 0.005, 0.02), at)
def doors(at, gain): L = int(1.2 * SR); tt = np.arange(L) / SR; place(sfx, (band(rng.standard_normal(L), 300, 2000) * 0.5 + np.sin(2 * np.pi * 90 * tt) * 0.2) * np.sin(np.pi * tt / 1.2) * gain, at)
def chime(at):
    for i, f in enumerate((660, 520)): L = int(0.9 * SR); tt = np.arange(L) / SR; place(sfx, np.sin(2 * np.pi * f * tt) * np.exp(-tt * 3) * 0.12, at + i * 0.45)
def vib(at, gain): L = int(0.9 * SR); tt = np.arange(L) / SR; place(sfx, np.sin(2 * np.pi * 180 * tt) * ((tt % 0.4) < 0.25) * gain, at)
# walking / running footsteps
S = lambda shot: A(shot)
for (a0, a1, rate, g) in ((S('street_wide_drift'), S('whip_turn'), 1.9, 0.08), (S('narr_walk_front'), S('narr_push_handheld'), 1.9, 0.07),
                          (S('narr_push_handheld'), S('narr_run_front'), 2.6, 0.09), (S('narr_run_front'), S('narr_gasp'), 3.6, 0.13), (S('narr_run_away_wide'), S('stairs_down'), 3.6, 0.12),
                          (S('stairs_down'), S('jump_inside'), 3.8, 0.14)):
    if a0 and a1 and a1 > a0: steps(a0, a1, rate, g, int(a0))
for shot, g in (('whisper_black', 0.0), ('whip_turn', 0.25), ('coat_eyes_extreme', 0.0)):
    for s in [x for x in SEG if x['shot'] == shot and shot == 'whip_turn']: whoosh(s['t0'] - 0.05, 0.5, g)
if S('whisper_black'): boom(S('whisper_black'), 0.5)
if S('coat_across_street_push'): hb(S('coat_across_street_push'), S('coat_gone') or S('coat_across_street_push') + 8, 72, 0.25)
if S('coat_walk_toward'): hb(S('coat_walk_toward'), S('narr_run_front') or 0, 96, 0.28)
if S('narr_run_front'): hb(S('narr_run_front'), S('narr_gasp') or 0, 132, 0.3)
if S('foodcourt_tables'):   # screams + clatter
    for i in range(4):
        L = int(0.9 * SR); tt = np.arange(L) / SR; f = rng.uniform(700, 1100); place(sfx, np.sin(2 * np.pi * (f + 80 * np.sin(tt * 30)) * tt) * env(L, 0.05, 0.4) * 0.05, S('foodcourt_tables') + i * 0.7)
    for i in range(8): L = int(0.2 * SR); place(sfx, band(rng.standard_normal(L), 1500, 8000) * np.exp(-np.arange(L) / (0.03 * SR)) * 0.15, S('foodcourt_tables') + rng.uniform(0, 2.5))
if S('shoulder_tap'): L = int(0.12 * SR); place(sfx, band(rng.standard_normal(L), 100, 1200) * np.exp(-np.arange(L) / (0.02 * SR)) * 0.4, S('shoulder_tap') + 0.6)
if S('reflection_coat_reveal'): boom(S('reflection_coat_reveal') + 0.4, 0.8); whoosh(S('reflection_coat_reveal'), 1.0, 0.2)
if S('phone_ring_close'): ring(S('phone_ring_close'), (S('phone_answer') or S('phone_ring_close') + 4) - S('phone_ring_close'))
rr = [s for s in SEG if s['shot'] == 'phone_ring_close']
if len(rr) > 1: ring(rr[1]['t0'], 3)
for s in SEG:
    if s['shot'] == 'narr_phone_close' and 'static' in s['text']: static(s['t0'] + 1.0, s['t1'] - s['t0'], 0.08)
    if s['spk'] == 'PHONE': static(s['t0'], s['t1'] - s['t0'], 0.02)
if S('crowd_head_turn'): whoosh(S('crowd_head_turn'), 0.8, 0.3); boom(S('crowd_head_turn') + 0.3, 0.5)
if S('doors_close'): doors(S('doors_close'), 0.5); chime(S('doors_close') - 1.0)
if S('lights_flicker'):
    for s in [x for x in SEG if x['shot'] == 'lights_flicker']: buzz(s['t0'], 0.25, 0.4); buzz(s['t0'] + 1.0, 0.18, 0.4)
if S('lights_on_empty'): boom(S('lights_on_empty'), 0.6)
if S('speaker_close'): static(S('speaker_close'), 1.2, 0.08)
if S('doors_open_bedroom'): doors(S('doors_open_bedroom'), 0.5)
if S('doors_closing'): doors(S('doors_closing'), 0.6)
if S('coat_grab'): boom(S('coat_grab'), 0.6); hb(S('coat_grab'), (S('blackout') or 0), 140, 0.35)
if S('blackout'): boom(S('blackout'), 1.0)
if S('three_knocks_black'): knock(S('three_knocks_black') + 0.4, 1.0, wood=False)
if S('phone_buzz_bed'): vib(S('phone_buzz_bed'), 0.25)
if S('phone_message_text'): boom(S('phone_message_text') + 0.3, 0.5)
if S('window_knocks'): knock(S('window_knocks') + 0.3, 1.0, wood=False)
if S('end_black'): boom(S('end_black') + 0.2, 0.9)

# ------------------------------------------------------------------ drone score: minor pads that swell with tension
def tension(tt):
    keys = [(0, 0.15), ((S('coat_across_street_push') or 0), 0.45), ((S('whisper_black') or 0), 0.8), ((S('narr_walk_front') or 0), 0.3), ((S('narr_run_front') or 0), 0.75),
            ((S('foodcourt_wide_calm') or 0), 0.2), ((S('reflection_push') or 0), 0.8), ((S('phone_ring_close') or 0), 0.4), ((S('crowd_head_turn') or 0), 0.95),
            ((S('stairs_down') or 0), 0.5), ((S('train_wide') or 0), 0.25), ((S('lights_flicker') or 0), 0.65), ((S('coat_face_close') or 0), 0.85),
            ((S('bedroom_through_doors') or 0), 0.75), ((S('blackout') or 0), 1.0), ((S('bedroom_lights_on') or 0), 0.3), ((S('window_push') or 0), 1.0), (LEN, 0.6)]
    keys = sorted(keys); xs, ys = zip(*keys); return np.interp(tt, xs, ys)
ten = tension(t)
drone = np.zeros(N)
for f, g in ((55, 1), (82.4, 0.6), (110, 0.5), (130.8, 0.35), (164.8, 0.2), (58.3, 0.3)):
    drone += np.sin(2 * np.pi * f * t + 0.3 * np.sin(2 * np.pi * 0.1 * t * g)) * g
drone += band(rng.standard_normal(N), 2000, 6000) * 0.15 * ten ** 2
drone = drone / np.abs(drone).max() * 0.12 * (0.35 + ten)
# ------------------------------------------------------------------ mix: duck ambience + drone under the voice
envv = movavg(np.abs(voice), 0.25 * SR); duck = 1 - np.clip(envv * 3.5, 0, 0.6)
mix = voice * 1.0 + (amb * 0.9 + drone) * duck + sfx
mix = np.tanh(mix * 1.15) * 0.8
mix *= np.clip(t / 1.0, 0, 1) * np.clip((LEN - t) / 1.5, 0, 1)
st = np.stack([mix * (1 + 0.05 * np.sin(2 * np.pi * 0.17 * t)), mix * (1 - 0.05 * np.sin(2 * np.pi * 0.17 * t))], 1)
wav = os.path.join(os.path.dirname(os.path.abspath(a.out)) or '.', 'film_audio.wav')
with wave.open(wav, 'wb') as wf: wf.setnchannels(2); wf.setsampwidth(2); wf.setframerate(SR); wf.writeframes((np.clip(st, -1, 1) * 32000).astype(np.int16).tobytes())
print('audio', wav)
if a.audio_only: raise SystemExit

# ------------------------------------------------------------------ picture: titles, subtitles, grade, letterbox, encode
W, H = map(int, a.size.split('x'))
def font(name, size):
    for p in (os.path.join(HERE, 'fonts', name), '/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf', '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'):
        try: return ImageFont.truetype(p, size)
        except Exception: pass
    return ImageFont.load_default()
FT = font('SpecialElite-Regular.ttf', int(H * 0.075)); FS = font('Inter-SemiBold.ttf', int(H * 0.036)); FC = font('SpecialElite-Regular.ttf', int(H * 0.032))
def overlay(img, tt):
    d = ImageDraw.Draw(img, 'RGBA')
    s = next((x for x in SEG if x['t0'] <= tt < x['t1']), SEG[-1])
    if s['kind'] == 'title':
        u = (tt - s['t0']) / (s['t1'] - s['t0']); al = int(255 * min(1, u * 3, (1 - u) * 3))
        parts = s['text'].split(' · ')
        if len(parts) == 2:
            bb = d.textbbox((0, 0), parts[0], font=FC); d.text(((W - bb[2]) / 2, H * 0.40), parts[0], font=FC, fill=(200, 30, 30, al))
            bb = d.textbbox((0, 0), parts[1], font=FT); d.text(((W - bb[2]) / 2, H * 0.47), parts[1], font=FT, fill=(235, 230, 220, al))
        else:
            bb = d.textbbox((0, 0), s['text'], font=FT); d.text(((W - bb[2]) / 2, H * 0.44), s['text'], font=FT, fill=(235, 230, 220, al))
    elif s['kind'] == 'line' and not a.no_subs:
        txt = s['text']; words = txt.split(); lines, cur = [], ''
        for w_ in words:
            if d.textbbox((0, 0), cur + ' ' + w_, font=FS)[2] > W * 0.7 and cur: lines.append(cur); cur = w_
            else: cur = (cur + ' ' + w_).strip()
        lines.append(cur); y = H * 0.86 - (len(lines) - 1) * H * 0.045
        al = int(255 * min(1, (tt - s['t0']) / 0.15, (s['t1'] - tt) / 0.15))
        col = {'COAT': (255, 120, 110), 'PHONE': (150, 210, 255), 'OLD': (255, 225, 160), 'PA': (200, 200, 200), 'DOUBLE': (255, 120, 110)}.get(s['spk'], (245, 245, 240))
        if s['spk'] != 'N': lines = [('"' if i == 0 else '') + l + ('"' if i == len(lines) - 1 else '') for i, l in enumerate(lines)]
        # word-by-word: words light up as they are spoken (paced by characters over the line's speech time); the rest wait dimmed
        speak = max(0.5, (s['t1'] - s['t0']) - 0.45); done_chars = (tt - s['t0'] - 0.05) / speak * sum(len(w_) + 1 for w_ in words)
        cnt = 0
        for l in lines:
            bb = d.textbbox((0, 0), l, font=FS); x = (W - bb[2]) / 2
            for wi, w_ in enumerate(l.split(' ')):
                tok = w_ + (' ' if wi < len(l.split(' ')) - 1 else ''); lit = cnt + len(w_) * 0.5 <= done_chars; cnt += len(w_) + 1
                a2 = al if lit else int(al * 0.38)
                d.text((x, y), tok, font=FS, fill=col + (a2,), stroke_width=3, stroke_fill=(0, 0, 0, int(a2 * 0.85))); x += d.textlength(tok, font=FS)
            y += H * 0.045
    return img
frames = sorted(glob.glob(os.path.join(a.frames, 'f*.jpg')))
vig = None
K0 = a.start or 0
VOPT = ['-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '17', '-preset', 'slow', '-tune', 'film', '-profile:v', 'high', '-g', '48', '-keyint_min', '48', '-sc_threshold', '0']
if a.start is None: cmd = ['ffmpeg', '-y', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(a.fps), '-i', '-', '-i', wav] + VOPT + ['-c:a', 'aac', '-b:a', '256k', '-shortest', '-movflags', '+faststart', a.out]
else: cmd = ['ffmpeg', '-y', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(a.fps), '-i', '-'] + VOPT + ['-an', a.out]   # chunk: video only
p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
for k, f in enumerate(frames):
    im = Image.open(f).convert('RGB')
    if im.size != (W, H): im = im.resize((W, H), Image.BICUBIC)
    arr = np.asarray(im).astype(np.float32) / 255
    if vig is None: yy, xx = np.mgrid[0:H, 0:W]; vig = (1 - 0.35 * (((xx - W / 2) / W) ** 2 + ((yy - H / 2) / H) ** 2) * 2)[..., None]
    lum = arr.mean(2, keepdims=True); arr = arr * vig + (0.025 * (1 - lum)) * np.array([-0.3, 0.0, 0.5])   # (film grain comes from the render's post pass; a second random layer here only multiplied the bitrate)
    im = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8)); im = overlay(im, (K0 + k) / a.fps)
    p.stdin.write(im.tobytes())
p.stdin.close(); p.wait(); print('done', a.out, len(frames), 'frames')
