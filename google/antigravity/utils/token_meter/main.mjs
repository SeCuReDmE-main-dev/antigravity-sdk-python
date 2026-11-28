import { execFileSync, spawn } from 'node:child_process';
import { existsSync, mkdirSync, readFileSync, watchFile, writeFileSync } from 'node:fs';
import { homedir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { Response, SidecarApp } from 'sidecar_sdk';

const here = dirname(fileURLToPath(import.meta.url));
const dataDir = process.env.ANTIGRAVITY_EXECUTABLE_DATA_DIR
  || join(homedir(), '.gemini', 'antigravity', 'sidecar_data', 'agy-studio-token-meter');
mkdirSync(dataDir, { recursive: true });

const stateFile = join(dataDir, 'active_state.json');

function readState() {
  if (!existsSync(stateFile)) {
    return { conversation_id: null, staged_files: [] };
  }
  try {
    return JSON.parse(readFileSync(stateFile, 'utf8'));
  } catch {
    return { conversation_id: null, staged_files: [] };
  }
}

function writeState(next) {
  writeFileSync(stateFile, JSON.stringify(next, null, 2), 'utf8');
}

function findPythonExecutable() {
  const candidates = process.platform === 'win32'
    ? [
        join(homedir(), 'AppData', 'Local', 'Programs', 'Python', 'Python310', 'python.exe'),
        'python',
      ]
    : ['python3', 'python'];
  for (const bin of candidates) {
    try {
      execFileSync(bin, ['--version'], { stdio: 'ignore' });
      return bin;
    } catch {
      // try next
    }
  }
  return 'python';
}

const pyBin = findPythonExecutable();

// Start the non-intrusive HUD Token Pill anchored in Antigravity's bottom input bar
// with a lightweight watchdog + exponential backoff so server restarts never leave the pill dead.
let hudProc = null;
let hudRetryCount = 0;
let hudBackoffTimer = null;
let hudNextAllowedSpawnAt = 0;
let hudSpawnStartedAt = 0;

const HUD_BASE_BACKOFF_MS = 750;
const HUD_MAX_BACKOFF_MS = 12000;
const HUD_WATCHDOG_INTERVAL_MS = 3500;
const HUD_STABLE_UPTIME_MS = 8000;

function scheduleHudRestart(delayMs) {
  if (hudBackoffTimer) {
    return;
  }
  hudBackoffTimer = setTimeout(() => {
    hudBackoffTimer = null;
    ensureHudRunning();
  }, delayMs);
  if (typeof hudBackoffTimer.unref === 'function') {
    hudBackoffTimer.unref();
  }
}

function ensureHudRunning(forceImmediate = false) {
  if (hudProc && exitCodeOf(hudProc) === null) {
    return;
  }
  const now = Date.now();
  if (!forceImmediate && now < hudNextAllowedSpawnAt) {
    scheduleHudRestart(hudNextAllowedSpawnAt - now);
    return;
  }
  try {
    hudSpawnStartedAt = now;
    const child = spawn(pyBin, [join(here, 'hud_pill_overlay.py')], {
      cwd: here,
      env: { ...process.env, ANTIGRAVITY_EXECUTABLE_DATA_DIR: dataDir },
      stdio: 'ignore',
      windowsHide: true,
    });
    hudProc = child;
    child.on('exit', () => {
      if (hudProc === child) {
        hudProc = null;
      }
      const uptime = Date.now() - hudSpawnStartedAt;
      if (uptime >= HUD_STABLE_UPTIME_MS) {
        hudRetryCount = 0;
      } else {
        hudRetryCount = Math.min(hudRetryCount + 1, 6);
      }
      const delay = Math.min(
        HUD_MAX_BACKOFF_MS,
        HUD_BASE_BACKOFF_MS * (2 ** Math.max(0, hudRetryCount - 1)),
      );
      hudNextAllowedSpawnAt = Date.now() + delay;
      scheduleHudRestart(delay);
    });
    child.on('error', () => {
      if (hudProc === child) {
        hudProc = null;
      }
      hudRetryCount = Math.min(hudRetryCount + 1, 6);
      const delay = Math.min(
        HUD_MAX_BACKOFF_MS,
        HUD_BASE_BACKOFF_MS * (2 ** Math.max(0, hudRetryCount - 1)),
      );
      hudNextAllowedSpawnAt = Date.now() + delay;
      scheduleHudRestart(delay);
    });
  } catch {
    hudProc = null;
    hudRetryCount = Math.min(hudRetryCount + 1, 6);
    const delay = Math.min(
      HUD_MAX_BACKOFF_MS,
      HUD_BASE_BACKOFF_MS * (2 ** Math.max(0, hudRetryCount - 1)),
    );
    hudNextAllowedSpawnAt = Date.now() + delay;
    scheduleHudRestart(delay);
  }
}

function exitCodeOf(proc) {
  return proc ? proc.exitCode : 0;
}

ensureHudRunning(true);
const hudWatchdog = setInterval(() => {
  if (hudProc && exitCodeOf(hudProc) === null && (Date.now() - hudSpawnStartedAt) >= HUD_STABLE_UPTIME_MS) {
    hudRetryCount = 0;
  }
  ensureHudRunning();
}, HUD_WATCHDOG_INTERVAL_MS);
if (typeof hudWatchdog.unref === 'function') {
  hudWatchdog.unref();
}

function restartHudOnCodeChange() {
  if (hudProc) {
    try { hudProc.kill(); } catch {}
    hudProc = null;
  }
  hudRetryCount = 0;
  hudNextAllowedSpawnAt = 0;
  ensureHudRunning(true);
}

watchFile(join(here, 'hud_pill_overlay.py'), { interval: 1000 }, restartHudOnCodeChange);
watchFile(join(here, 'multimodal_token_meter.py'), { interval: 1000 }, restartHudOnCodeChange);

const app = new SidecarApp();

app.page('/', () => readFileSync(join(here, 'index.html'), 'utf8'));
app.page('/app.js', () => readFileSync(join(here, 'app.js'), 'utf8'));
app.api(
  '/styles.css',
  () => new Response(readFileSync(join(here, 'styles.css'), 'utf8'), { contentType: 'text/css' }),
  'GET',
);

app.api('/api/tokens', (data) => {
  ensureHudRunning();
  const st = readState();
  const nowMs = Date.now();
  let changed = false;
  let pushedConvId = '';
  if (data.conversationId) {
    pushedConvId = String(data.conversationId);
    st.conversation_id = pushedConvId;
    st.updated_at_ms = nowMs;
    changed = true;
  }
  if (data.accentColor && data.accentColor.startsWith('#') && data.accentColor !== st.accent_color) {
    st.accent_color = data.accentColor;
    changed = true;
  }
  if (changed) {
    writeState(st);
  }
  const extraFiles = Array.isArray(data.stagedFiles) ? data.stagedFiles : (st.staged_files || []);
  const stateConvId = pushedConvId || st.conversation_id || null;
  const stateUpdatedAtMs = pushedConvId ? nowMs : (typeof st.updated_at_ms === 'number' ? st.updated_at_ms : null);
  const pyCode = [
    'import json, sys',
    `sys.path.insert(0, ${JSON.stringify(here)})`,
    'from multimodal_token_meter import MultimodalTokenMeter, resolve_active_conversation_id',
    'm = MultimodalTokenMeter()',
    `resolved_id = resolve_active_conversation_id(${JSON.stringify(stateConvId)}, ${JSON.stringify(stateUpdatedAtMs)}, brain_root=m.brain_root, annotations_root=m.annotations_root)`,
    `bd = m.get_breakdown(conversation_id=resolved_id, extra_files=${JSON.stringify(extraFiles)}, draft_text=${JSON.stringify(data.draftText || '')})`,
    'print(json.dumps(bd.to_dict(), ensure_ascii=False))',
  ].join('; ');
  const out = execFileSync(pyBin, ['-c', pyCode], { encoding: 'utf8' });
  return JSON.parse(out.trim());
});

app.api('/api/stage-file', (data) => {
  const st = readState();
  const files = new Set(st.staged_files || []);
  if (data.filePath) {
    files.add(String(data.filePath).trim());
  }
  if (data.clear) {
    files.clear();
  }
  st.staged_files = Array.from(files);
  writeState(st);
  return { staged_files: st.staged_files };
});

app.run();
