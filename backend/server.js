import express from 'express';
import cors from 'cors';
import { GoogleGenAI } from '@google/genai';
import 'dotenv/config';

const app = express();
app.use(cors());
app.use(express.json());

app.post('/api/generate-questions', async (req, res) => {
    try {
        const { prompt } = req.body;
        
        // Initialize Gemini with your environment variable key
        const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });
        
        // Request question content generation using the flash model
        const response = await ai.models.generateContent({
            model: 'gemini-2.5-flash',
            contents: prompt,
        });

        // Format precisely to match your dashboard frontend expectation
        res.json({
            content: [
                { text: response.text }
            ]
        });

    } catch (error) {
        console.error("AI Error:", error);
        res.status(500).json({ error: "Failed to generate questions" });
    }
});

const PORT = 5000;
app.listen(PORT, () => console.log(`🚀 Server running on port ${PORT}`));
import express from 'express';
import cors from 'cors';
import { GoogleGenAI } from '@google/genai';
import 'dotenv/config';

const app = express();

// 1. Industrial Environmental Validation (Fail Early)
if (!process.env.GEMINI_API_KEY) {
    console.error("❌ CRITICAL ERROR: GEMINI_API_KEY is missing from environment variables.");
    process.exit(1); 
}

// Initialize Gemini Client once at startup rather than recreating it on every request
const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });

// 2. Production CORS Policy Configuration
const allowedOrigins = [
    'http://localhost:5173',
    'http://localhost:5174',
    'http://localhost:5175'
    // Add your production frontend URL here when you deploy it!
];

app.use(cors({
    origin: (origin, callback) => {
        // Allow requests with no origin (like mobile apps, curl, or server-to-server calls)
        if (!origin) return callback(null, true);
        if (allowedOrigins.indexOf(origin) !== -1) {
            callback(null, true);
        } else {
            console.warn(`⚠️ Blocked by CORS: ${origin}`);
            callback(new Error('Not allowed by CORS'));
        }
    }
}));

app.use(express.json());

// 3. API Generation Endpoint
app.post('/api/generate-questions', async (req, res) => {
    try {
        const { prompt } = req.body;

        if (!prompt) {
            return res.status(400).json({ error: "Prompt is required in the request body." });
        }

        // Request question content generation using the flash model
        const response = await ai.models.generateContent({
            model: 'gemini-2.5-flash',
            contents: prompt,
        });

        // Format precisely to match your dashboard frontend expectation
        res.json({
            content: [
                { text: response.text }
            ]
        });

    } catch (error) {
        // Differentiate log formatting for production tracking
        console.error("❌ AI Route Error Details:", error.message || error);
        res.status(500).json({ error: "Failed to generate questions due to an internal server error." });
    }
});

// 4. Dynamic Port Allocation (Crucial for Cloud Platforms like Railway)
const PORT = process.env.PORT || 5000;
app.listen(PORT, () => console.log(`🚀 Industrial Server running on port ${PORT}`));
