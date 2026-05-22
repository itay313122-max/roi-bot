const express = require('express');
const { Client, LocalAuth } = require('whatsapp-web.js');
const qrcode = require('qrcode');
const cron = require('node-cron');
const path = require('path');

const app = express();
app.use(express.json());
app.use(express.static('public'));

const PORT = process.env.PORT || 3000;

let qrCodeData = null;
let clientReady = false;
const scheduledJobs = new Map();

const client = new Client({
  authStrategy: new LocalAuth(),
  puppeteer: {
    args: [
      '--no-sandbox',
      '--disable-setuid-sandbox',
      '--disable-dev-shm-usage',
      '--disable-accelerated-2d-canvas',
      '--no-first-run',
      '--no-zygote',
      '--single-process',
      '--disable-gpu',
    ],
  },
});

client.on('qr', async (qr) => {
  console.log('QR received');
  qrCodeData = await qrcode.toDataURL(qr);
  clientReady = false;
});

client.on('ready', () => {
  console.log('WhatsApp client ready');
  clientReady = true;
  qrCodeData = null;
});

client.on('disconnected', (reason) => {
  console.log('Client disconnected:', reason);
  clientReady = false;
});

client.initialize();

// GET /api/status
app.get('/api/status', (req, res) => {
  res.json({ ready: clientReady, hasQr: !!qrCodeData });
});

// GET /api/qr
app.get('/api/qr', (req, res) => {
  if (clientReady) return res.json({ error: 'Already connected' });
  if (!qrCodeData) return res.json({ error: 'QR not ready yet' });
  res.json({ qr: qrCodeData });
});

// POST /api/send — send a message immediately
app.post('/api/send', async (req, res) => {
  if (!clientReady) return res.status(503).json({ error: 'WhatsApp not connected' });
  const { phone, message } = req.body;
  if (!phone || !message) return res.status(400).json({ error: 'phone and message required' });
  try {
    const chatId = phone.replace(/\D/g, '') + '@c.us';
    await client.sendMessage(chatId, message);
    res.json({ success: true });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// POST /api/schedule — schedule a recurring message
app.post('/api/schedule', (req, res) => {
  if (!clientReady) return res.status(503).json({ error: 'WhatsApp not connected' });
  const { id, phone, message, cronExpression } = req.body;
  if (!id || !phone || !message || !cronExpression)
    return res.status(400).json({ error: 'id, phone, message, cronExpression required' });
  if (!cron.validate(cronExpression))
    return res.status(400).json({ error: 'Invalid cron expression' });

  if (scheduledJobs.has(id)) {
    scheduledJobs.get(id).stop();
    scheduledJobs.delete(id);
  }

  const job = cron.schedule(cronExpression, async () => {
    try {
      const chatId = phone.replace(/\D/g, '') + '@c.us';
      await client.sendMessage(chatId, message);
      console.log(`Scheduled message sent to ${phone}`);
    } catch (err) {
      console.error('Failed to send scheduled message:', err.message);
    }
  });

  scheduledJobs.set(id, job);
  res.json({ success: true, id });
});

// DELETE /api/schedule/:id — cancel a scheduled job
app.delete('/api/schedule/:id', (req, res) => {
  const { id } = req.params;
  if (!scheduledJobs.has(id)) return res.status(404).json({ error: 'Job not found' });
  scheduledJobs.get(id).stop();
  scheduledJobs.delete(id);
  res.json({ success: true });
});

// GET /api/schedules — list active jobs
app.get('/api/schedules', (req, res) => {
  res.json({ jobs: [...scheduledJobs.keys()] });
});

app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

app.listen(PORT, () => console.log(`Server running on port ${PORT}`));
