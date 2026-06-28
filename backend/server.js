import express from 'express';
import cors from 'cors';
import { GoogleGenAI } from '@google/genai';
import 'dotenv/config';
import logger from './logger.js'; // Import our new logger

const app = express();

if (!process.env.GEMINI_API_KEY) {
    logger.error("❌ CRITICAL CONFIG ERROR: GEMINI_API_KEY is missing from environment variables.");
    process.exit(1); 
}

const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });

// ... Keep your existing CORS configuration here ...

app.use(express.json());

// API Generation Endpoint
app.post('/api/generate-questions', async (req, res, next) => {
    try {
        const { prompt } = req.body;

        if (!prompt) {
            return res.status(400).json({ error: "Prompt is required." });
        }

        logger.info(`🤖 API Request received for prompt length: ${prompt.length}`);

        const response = await ai.models.generateContent({
            model: 'gemini-2.5-flash',
            contents: prompt,
        });

        res.json({
            content: [{ text: response.text }]
        });

    } catch (error) {
        // Forward to centralized error handler instead of simple console.log
        next(error); 
    }
});

// Centralized Production Error-Handling Middleware
app.use((err, req, res, next) => {
    logger.error("Internal Server Error Occurred", { 
        message: err.message, 
        stack: err.stack,
        path: req.path 
    });

    res.status(500).json({
        error: "Internal Server Error",
        message: process.env.NODE_ENV === 'production' ? "An unexpected error occurred." : err.message
    });
});

const PORT = process.env.PORT || 5000;
app.listen(PORT, () => logger.info(`🚀 Industrial Server listening on port ${PORT}`));
