const express = require('express');
const { Client, LocalAuth } = require('whatsapp-web.js');
const qrcode = require('qrcode');
const cron = require('node-cron');
const fs = require('fs');
const path = require('path');

const app = express();
app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

const PORT = process.env.PORT || 3000;
const SCHEDULES_FILE = path.join('/data', 'schedules.json');

// ─── State ─────────────────────────────────────────────────────────────────
let currentQR = null;
let isConnected = false;
let clientInfo = null;
const activeCronJobs = {}; // id → cron task

// ─── Schedules Store ───────────────────────────────────────────────────────
function loadSchedules() {
  if (!fs.existsSync(SCHEDULES_FILE)) return [];
  try { return JSON.parse(fs.readFileSync(SCHEDULES_FILE, 'utf8')); }
  catch { return []; }
}

function saveSchedules(schedules) {
  const dir = path.dirname(SCHEDULES_FILE);
  if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
  fs.writeFileSync(SCHEDULES_FILE, JSON.stringify(schedules, null, 2));
}

function formatPhone(phone) {
  const digits = phone.replace(/\D/g, '');
  if (digits.startsWith('972')) return `${digits}@c.us`;
  if (digits.startsWith('0')) return `972${digits.slice(1)}@c.us`;
  return `972${digits}@c.us`;
}

// ─── WhatsApp Client ───────────────────────────────────────────────────────
const client = new Client({
  authStrategy: new LocalAuth({ dataPath: '/data/.wwebjs_auth' }),
  puppeteer: {
    headless: true,
    args: [
      '--no-sandbox',
      '--disable-setuid-sandbox',
      '--disable-dev-shm-usage',
      '--disable-accelerated-2d-canvas',
      '--no-first-run',
      '--no-zygote',
      '--single-process',
      '--disable-gpu'
    ]
  }
});

client.on('qr', async (qr) => {
  console.log('📱 QR generated');
  currentQR = await qrcode.toDataURL(qr);
  isConnected = false;
});

client.on('ready', () => {
  console.log('✅ WhatsApp connected');
  currentQR = null;
  isConnected = true;
  clientInfo = client.info;
  restoreSchedules();
});

client.on('auth_failure', () => {
  console.error('❌ Auth failed');
  isConnected = false;
});

client.on('disconnected', () => {
  console.log('⚠️  Disconnected');
  isConnected = false;
  clientInfo = null;
});

// ─── Restore cron jobs after restart ──────────────────────────────────────
function restoreSchedules() {
  const schedules = loadSchedules();
  let restored = 0;
  for (const s of schedules) {
    if (s.active && cron.validate(s.cronExpression)) {
      startCronJob(s);
      restored++;
    }
  }
  console.log(`🔄 Restored ${restored} scheduled jobs`);
}

function startCronJob(schedule) {
  if (activeCronJobs[schedule.id]) {
    activeCronJobs[schedule.id].stop();
  }
  const task = cron.schedule(schedule.cronExpression, async () => {
    if (!isConnected) return;
    try {
      const chatId = formatPhone(schedule.phone);
      await client.sendMessage(chatId, schedule.message);
      console.log(`📤 [${schedule.id}] Sent to ${schedule.phone}`);

      const schedules = loadSchedules();
      const s = schedules.find(x => x.id === schedule.id);
      if (s) {
        s.lastSent = new Date().toISOString();
        s.sentCount = (s.sentCount || 0) + 1;
        saveSchedules(schedules);
      }
    } catch (err) {
      console.error(`❌ [${schedule.id}] Failed:`, err.message);
    }
  });
  activeCronJobs[schedule.id] = task;
}

// ─── API Routes ────────────────────────────────────────────────────────────

// GET /api/status
app.get('/api/status', (req, res) => {
  res.json({
    connected: isConnected,
    hasQR: !!currentQR,
    phone: clientInfo?.wid?.user || null,
    name: clientInfo?.pushname || null,
    activeJobs: Object.keys(activeCronJobs).length
  });
});

// GET /api/qr
app.get('/api/qr', (req, res) => {
  if (!currentQR) {
    return res.status(404).json({ error: isConnected ? 'Already connected' : 'QR not ready yet' });
  }
  res.json({ qr: currentQR });
});

// POST /api/send
app.post('/api/send', async (req, res) => {
  if (!isConnected) return res.status(503).json({ error: 'WhatsApp not connected' });
  const { phone, message } = req.body;
  if (!phone || !message) return res.status(400).json({ error: 'Missing phone or message' });
  try {
    const chatId = formatPhone(phone);
    await client.sendMessage(chatId, message);
    res.json({ success: true, sentTo: chatId });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// POST /api/schedule
app.post('/api/schedule', (req, res) => {
  const { id, phone, message, cronExpression } = req.body;
  if (!id || !phone || !message || !cronExpression) {
    return res.status(400).json({ error: 'Missing: id, phone, message, cronExpression' });
  }
  if (!cron.validate(cronExpression)) {
    return res.status(400).json({ error: `Invalid cron expression: "${cronExpression}"` });
  }

  const schedules = loadSchedules();
  const existing = schedules.find(s => s.id === id);

  if (existing) {
    existing.phone = phone;
    existing.message = message;
    existing.cronExpression = cronExpression;
    existing.active = true;
    existing.updatedAt = new Date().toISOString();
    saveSchedules(schedules);
    if (isConnected) startCronJob(existing);
    return res.json({ success: true, schedule: existing, action: 'updated' });
  }

  const newSchedule = {
    id,
    phone: phone.replace(/\D/g, ''),
    message,
    cronExpression,
    active: true,
    sentCount: 0,
    lastSent: null,
    createdAt: new Date().toISOString()
  };

  schedules.push(newSchedule);
  saveSchedules(schedules);
  if (isConnected) startCronJob(newSchedule);
  res.json({ success: true, schedule: newSchedule, action: 'created' });
});

// DELETE /api/schedule/:id
app.delete('/api/schedule/:id', (req, res) => {
  const { id } = req.params;
  const schedules = loadSchedules();
  const idx = schedules.findIndex(s => s.id === id);
  if (idx === -1) return res.status(404).json({ error: `Schedule "${id}" not found` });

  if (activeCronJobs[id]) {
    activeCronJobs[id].stop();
    delete activeCronJobs[id];
  }

  schedules.splice(idx, 1);
  saveSchedules(schedules);
  res.json({ success: true, deleted: id });
});

// GET /api/schedules
app.get('/api/schedules', (req, res) => {
  const schedules = loadSchedules();
  res.json(schedules.map(s => ({ ...s, isRunning: !!activeCronJobs[s.id] })));
});

// ─── Start ─────────────────────────────────────────────────────────────────
app.listen(PORT, () => {
  console.log(`🚀 Server on port ${PORT}`);
  client.initialize();
});
