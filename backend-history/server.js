// server.js
import express from "express";
import mongoose from "mongoose";
import cors from "cors";

// ================== CONFIG ==================
const app = express();
const PORT = 3000;

// Middleware
app.use(cors());
app.use(express.json());

// ================== MONGODB ==================
const MONGO_URI = "mongodb://127.0.0.1:27017/chatdb"; // ⚠️ change le nom si tu veux
mongoose
  .connect(MONGO_URI)
  .then(() => console.log("✅ Connected to MongoDB"))
  .catch((err) => console.error("❌ MongoDB error:", err));

// ================== MODELS ==================
const MessageSchema = new mongoose.Schema({
  role: String, // "user" ou "bot"
  content: String,
});

const ConversationSchema = new mongoose.Schema({
  title: String,
  messages: [MessageSchema],
});

const Conversation = mongoose.model("Conversation", ConversationSchema);

// ================== ROUTES ==================

// Récupérer toutes les conversations
app.get("/conversations", async (req, res) => {
  const convos = await Conversation.find();
  res.json(convos);
});
app.delete("/conversations/:id", async (req, res) => {
  try {
    const { id } = req.params;
    await Conversation.findByIdAndDelete(id);
    res.json({ success: true });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});


// Créer une conversation
app.post("/conversations", async (req, res) => {
  const convo = new Conversation({ title: req.body.title, messages: [] });
  await convo.save();
  res.json(convo);
});

// Renommer une conversation
app.put("/conversations/:id", async (req, res) => {
  const convo = await Conversation.findByIdAndUpdate(
    req.params.id,
    { title: req.body.title },
    { new: true }
  );
  res.json(convo);
});

// Ajouter un message
app.post("/conversations/:id/messages", async (req, res) => {
  const convo = await Conversation.findById(req.params.id);
  convo.messages.push({ role: req.body.role, content: req.body.content });
  await convo.save();
  res.json(convo);
});

// Récupérer les messages d’une conversation
app.get("/conversations/:id/messages", async (req, res) => {
  const convo = await Conversation.findById(req.params.id);
  res.json(convo ? convo.messages : []);
});

// ================== START ==================
app.listen(PORT, () => {
  console.log(`🚀 Backend running at http://127.0.0.1:${PORT}`);
});
