// render.js film.html outdir WxH (--times a,b,c | --range start end [stride offset])  — renders window.__frame(t)
const fs = require('fs'), path = require('path'); const { chromium } = require('playwright');
(async () => {
  const [html, out, size] = process.argv.slice(2, 5); const [W, H] = size.split('x').map(Number); const a = process.argv.slice(5);
  fs.mkdirSync(out, { recursive: true });
  const b = await chromium.launch({ args: ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'] });
  const p = await b.newPage({ viewport: { width: W, height: H } }); const errs = [];
  p.on('pageerror', (e) => errs.push(e.message)); p.on('console', (m) => { if (m.type() === 'error') errs.push(m.text()); });
  /* serve the film's folder over http so glTF/HDR assets (assets/...) load with fetch */
  const root = path.dirname(path.resolve(html)), http = require('http');
  const MIME = { '.html': 'text/html', '.js': 'text/javascript', '.json': 'application/json', '.gltf': 'model/gltf+json', '.glb': 'model/gltf-binary', '.bin': 'application/octet-stream', '.jpg': 'image/jpeg', '.png': 'image/png', '.hdr': 'application/octet-stream' };
  const srv = http.createServer((q, r) => { const f = path.join(root, decodeURIComponent(q.url.split('?')[0])); if (!f.startsWith(root) || !fs.existsSync(f) || fs.statSync(f).isDirectory()) { r.statusCode = 404; return r.end(); }
    r.setHeader('Content-Type', MIME[path.extname(f)] || 'application/octet-stream'); fs.createReadStream(f).pipe(r); });
  await new Promise((ok) => srv.listen(0, '127.0.0.1', ok)); const port = srv.address().port;
  await p.goto(`http://127.0.0.1:${port}/` + path.basename(html)); try { await p.waitForFunction(() => typeof window.__frame === 'function', null, { timeout: 180000 }); } catch (e) { console.log('PAGE NEVER READY. errors:\n' + errs.join('\n')); process.exit(1); }
  const film = await p.evaluate(() => window.__film()); const fps = film.fps;
  let times = [], names = [];
  if (a[0] === '--times') { times = a[1].split(',').map(Number); names = times.map((t) => 't' + t.toFixed(2)); }
  else { const s = +a[1], e = Math.min(+a[2], film.len), st = +(a[3] || 1), off = +(a[4] || 0); for (let k = Math.round(s * fps) + off; k < Math.round(e * fps); k += st) { times.push(k / fps); names.push('f' + String(k).padStart(6, '0')); } }
  await p.evaluate(() => window.__preload());
  for (let i = 0; i < 240 && (await p.evaluate(() => window.__texPending())) > 0; i++) await p.waitForTimeout(500);
  await p.evaluate(() => window.__frame(0.01)); await p.waitForTimeout(1500);
  const t0 = Date.now();
  for (let i = 0; i < times.length; i++) {
    const f = path.join(out, names[i] + '.jpg'); if (fs.existsSync(f)) continue;
    const url = await p.evaluate((t) => window.__frame(t), times[i]); fs.writeFileSync(f, Buffer.from(url.split(',')[1], 'base64'));
    if (i % 50 === 0) console.log(names[i], ((Date.now() - t0) / (i + 1) / 1000).toFixed(2) + 's/frame');
  }
  if (errs.length) { console.log('PAGE ERRORS:\n' + [...new Set(errs)].slice(0, 12).join('\n')); }
  fs.writeFileSync(path.join(out, '_film.json'), JSON.stringify(film)); console.log('done', times.length, 'frames', ((Date.now() - t0) / 1000).toFixed(0) + 's, film', film.len.toFixed(1) + 's'); await b.close(); srv.close();
})();
