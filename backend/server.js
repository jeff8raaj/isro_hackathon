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
