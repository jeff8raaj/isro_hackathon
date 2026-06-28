import { useState } from "react";

const LEVELS = [
  { id: "Beginner", color: "#1D9E75", bg: "#E1F5EE", desc: "Basic syntax & concepts" },
  { id: "Easy", color: "#185FA5", bg: "#E6F1FB", desc: "Simple implementations" },
  { id: "Intermediate", color: "#BA7517", bg: "#FAEEDA", desc: "Multi-step logic" },
  { id: "Hard", color: "#993C1D", bg: "#FAECE7", desc: "Optimization & edge cases" },
  { id: "Advanced", color: "#A32D2D", bg: "#FCEBEB", desc: "Research-level problems" },
];

const LANGS = ["Python", "NumPy", "rasterio", "GDAL", "OpenCV", "scikit-image", "PyTorch", "TensorFlow"];

const EXAMPLES = [
  "NDVI (NIR - Red) / (NIR + Red) measures vegetation health from satellite bands.",
  "Rasterio reads GeoTIFF files and provides CRS metadata and affine transforms.",
  "PCA reduces dimensionality of multi-spectral satellite images for classification.",
];

export default function App() {
  const [theory, setTheory] = useState("");
  const [selectedMode, setSelectedMode] = useState("all");
  const [selectedLangs, setSelectedLangs] = useState(["Python", "NumPy"]);
  const [count, setCount] = useState(2);
  const [loading, setLoading] = useState(false);
  const [questions, setQuestions] = useState([]);
  const [error, setError] = useState("");
  const [revealedHints, setRevealedHints] = useState({});

  const toggleLang = (lang) => {
    setSelectedLangs((prev) =>
      prev.includes(lang)
        ? prev.length > 1 ? prev.filter((l) => l !== lang) : prev
        : [...prev, lang]
    );
  };

  const toggleHint = (idx) =>
    setRevealedHints((p) => ({ ...p, [idx]: !p[idx] }));

  const generate = async () => {
    if (!theory.trim()) {
      setError("Paste some theory content first.");
      return;
    }
    setError("");
    setLoading(true);
    setQuestions([]);

    const levels =
      selectedMode === "all"
        ? LEVELS.map((l) => l.id)
        : [selectedMode.charAt(0).toUpperCase() + selectedMode.slice(1)];

    const prompt = `You are a satellite image processing educator. From the theory below, generate ${count} coding question(s) per level for: ${levels.join(", ")}.
Libraries: ${selectedLangs.join(", ")}

Theory: ${theory}

Rules:
- Questions must be practical, code-focused, based on the theory
- Higher levels = more complexity
- Keep each question under 60 words
- Keep each hint under 25 words
- application: one-line real-world satellite use case for the question (e.g. "Detecting crop stress from Sentinel-2 imagery")

Respond ONLY with valid JSON, no markdown, no extra text:
{"questions":[{"level":"Beginner","question":"...","application":"...","hint":"..."},...]}`;

    try {
      // CHANGED: Routing traffic securely through your local Node proxy on port 5000
      const res = await fetch("http://localhost:5000/api/generate-questions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt }),
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData?.error?.message || `HTTP ${res.status}`);
      }

      const data = await res.json();
      const raw = data.content?.map((c) => c.text || "").join("") || "";
      const clean = raw.replace(/```json|```/g, "").trim();
      let parsed;
      try {
        parsed = JSON.parse(clean);
      } catch {
        const jsonStart = clean.indexOf('{');
        const jsonEnd = clean.lastIndexOf('}');
        if (jsonStart !== -1 && jsonEnd !== -1) {
          parsed = JSON.parse(clean.slice(jsonStart, jsonEnd + 1));
        } else {
          throw new Error("Response was cut off. Try fewer questions per level or shorter theory text.");
        }
      }
      if (!parsed.questions?.length) throw new Error("No questions returned.");
      setQuestions(parsed.questions);
      setRevealedHints({});
    } catch (err) {
      setError(err.message || "Something went wrong. Try again.");
    }

    setLoading(false);
  };

  const levelMeta = (lvl) => LEVELS.find((l) => l.id === lvl) || LEVELS[0];

  return (
    <div style={{ fontFamily: "var(--font-sans, system-ui)", padding: "1.5rem 1rem", maxWidth: 680, margin: "0 auto" }}>

      {/* Header */}
      <div style={{ marginBottom: "1.5rem" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 4 }}>
          <div style={{
            width: 36, height: 36, borderRadius: 10,
            background: "#E6F1FB", display: "flex", alignItems: "center", justifyContent: "center"
          }}>
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#185FA5" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="12" cy="12" r="3"/><path d="M12 2v3M12 19v3M2 12h3M19 12h3M4.22 4.22l2.12 2.12M17.66 17.66l2.12 2.12M4.22 19.78l2.12-2.12M17.66 6.34l2.12-2.12"/>
            </svg>
          </div>
          <div>
            <h1 style={{ fontSize: 18, fontWeight: 500, margin: 0, color: "var(--text-primary, #111)" }}>
              Satellite code question generator
            </h1>
            <p style={{ fontSize: 13, color: "var(--text-secondary, #666)", margin: 0 }}>
              Paste theory → get coding questions across 5 difficulty levels
            </p>
          </div>
        </div>
      </div>

      {/* Difficulty mode */}
      <div style={{ marginBottom: "1.25rem" }}>
        <p style={{ fontSize: 11, fontWeight: 500, color: "var(--text-muted, #999)", textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 8 }}>
          Difficulty mode
        </p>
        <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
          {[{ id: "all", label: "All 5 levels" }, ...LEVELS.map((l) => ({ id: l.id.toLowerCase(), label: l.id }))].map((m) => (
            <button
              key={m.id}
              onClick={() => setSelectedMode(m.id)}
              style={{
                padding: "5px 13px",
                borderRadius: "var(--radius, 8px)",
                border: selectedMode === m.id ? "1.5px solid #185FA5" : "0.5px solid var(--border-strong, #ccc)",
                background: selectedMode === m.id ? "#E6F1FB" : "var(--surface-2, #fff)",
                color: selectedMode === m.id ? "#185FA5" : "var(--text-secondary, #555)",
                fontSize: 13,
                fontWeight: selectedMode === m.id ? 500 : 400,
                cursor: "pointer",
                fontFamily: "inherit",
              }}
            >
              {m.label}
            </button>
          ))}
        </div>
      </div>

      {/* Languages */}
      <div style={{ marginBottom: "1.25rem" }}>
        <p style={{ fontSize: 11, fontWeight: 500, color: "var(--text-muted, #999)", textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 8 }}>
          Language / library focus
        </p>
        <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
          {LANGS.map((lang) => {
            const active = selectedLangs.includes(lang);
            return (
              <button
                key={lang}
                onClick={() => toggleLang(lang)}
                style={{
                  padding: "4px 12px",
                  borderRadius: 20,
                  border: active ? "1.5px solid #1D9E75" : "0.5px solid var(--border, #ddd)",
                  background: active ? "#E1F5EE" : "var(--surface-2, #fff)",
                  color: active ? "#0F6E56" : "var(--text-secondary, #555)",
                  fontSize: 12,
                  fontWeight: active ? 500 : 400,
                  cursor: "pointer",
                  fontFamily: "inherit",
                }}
              >
                {lang}
              </button>
            );
          })}
        </div>
      </div>

      {/* Count + textarea row */}
      <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 8 }}>
        <p style={{ fontSize: 11, fontWeight: 500, color: "var(--text-muted, #999)", textTransform: "uppercase", letterSpacing: "0.06em", margin: 0 }}>
          Theory content
        </p>
        <div style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: 8 }}>
          <span style={{ fontSize: 13, color: "var(--text-secondary, #666)" }}>Questions per level:</span>
          {[1, 2, 3].map((n) => (
            <button
              key={n}
              onClick={() => setCount(n)}
              style={{
                width: 30, height: 30,
                borderRadius: "var(--radius, 8px)",
                border: count === n ? "1.5px solid #185FA5" : "0.5px solid var(--border-strong, #ccc)",
                background: count === n ? "#E6F1FB" : "var(--surface-2, #fff)",
                color: count === n ? "#185FA5" : "var(--text-secondary, #555)",
                fontSize: 13, fontWeight: 500,
                cursor: "pointer", fontFamily: "inherit",
              }}
            >
              {n}
            </button>
          ))}
        </div>
      </div>

      <textarea
        value={theory}
        onChange={(e) => setTheory(e.target.value)}
        placeholder={`Example: "${EXAMPLES[0]}"`}
        style={{
          width: "100%", minHeight: 130, padding: "10px 12px",
          border: "0.5px solid var(--border-strong, #ccc)",
          borderRadius: "var(--radius, 8px)",
          background: "var(--surface-1, #f9f9f9)",
          color: "var(--text-primary, #111)",
          fontSize: 14, fontFamily: "inherit",
          resize: "vertical", lineHeight: 1.6, outline: "none",
          boxSizing: "border-box",
        }}
      />

      {/* Quick examples */}
      <div style={{ display: "flex", gap: 6, flexWrap: "wrap", marginTop: 6, marginBottom: "1.25rem" }}>
        <span style={{ fontSize: 12, color: "var(--text-muted, #aaa)" }}>Try:</span>
        {EXAMPLES.map((ex, i) => (
          <button
            key={i}
            onClick={() => setTheory(ex)}
            style={{
              fontSize: 11, padding: "2px 8px", borderRadius: 20,
              border: "0.5px solid var(--border, #ddd)",
              background: "none", color: "var(--text-accent, #185FA5)",
              cursor: "pointer", fontFamily: "inherit",
            }}
          >
            Example {i + 1}
          </button>
        ))}
      </div>

      {/* Generate button */}
      <button
        onClick={generate}
        disabled={loading}
        style={{
          width: "100%", padding: "10px 16px",
          borderRadius: "var(--radius, 8px)",
          border: "0.5px solid var(--border-strong, #ccc)",
          background: loading ? "var(--surface-1, #f5f5f5)" : "var(--surface-2, #fff)",
          color: loading ? "var(--text-muted, #aaa)" : "var(--text-primary, #111)",
          fontSize: 14, fontWeight: 500,
          cursor: loading ? "not-allowed" : "pointer",
          fontFamily: "inherit",
          display: "flex", alignItems: "center", justifyContent: "center", gap: 8,
        }}
      >
        {loading ? (
          <>
            <span style={{ display: "inline-flex", gap: 4 }}>
              {[0, 1, 2].map((i) => (
                <span key={i} style={{
                  width: 6, height: 6, borderRadius: "50%",
                  background: "var(--text-muted, #bbb)",
                  animation: `bounce 1s ${i * 0.15}s infinite`,
                }} />
              ))}
            </span>
            Generating questions…
          </>
        ) : (
          <>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
              <path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"/>
            </svg>
            Generate questions
          </>
        )}
      </button>

      <style>{`@keyframes bounce{0%,80%,100%{transform:translateY(0)}40%{transform:translateY(-5px)}}`}</style>

      {/* Error */}
      {error && (
        <div style={{
          marginTop: "1rem", padding: "10px 14px",
          background: "#FCEBEB", borderRadius: "var(--radius, 8px)",
          color: "#A32D2D", fontSize: 13, lineHeight: 1.5,
        }}>
          <strong>Something went wrong.</strong> {error}
        </div>
      )}

      {/* Results */}
      {questions.length > 0 && (
        <div style={{ marginTop: "1.5rem" }}>
          <div style={{ height: "0.5px", background: "var(--border, #e5e5e5)", marginBottom: "1.5rem" }} />
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "1rem" }}>
            <p style={{ fontSize: 13, fontWeight: 500, color: "var(--text-secondary, #666)", margin: 0 }}>
              {questions.length} question{questions.length !== 1 ? "s" : ""} generated
            </p>
            <button
              onClick={() => { setQuestions([]); setRevealedHints({}); }}
              style={{
                fontSize: 12, color: "var(--text-muted, #aaa)", background: "none",
                border: "none", cursor: "pointer", fontFamily: "inherit",
              }}
            >
              Clear
            </button>
          </div>

          {questions.map((q, i) => {
            const meta = levelMeta(q.level);
            return (
              <div
                key={i}
                style={{
                  background: "var(--surface-2, #fff)",
                  border: "0.5px solid var(--border, #e5e5e5)",
                  borderRadius: 12,
                  padding: "1rem 1.25rem",
                  marginBottom: "0.875rem",
                }}
              >
                <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 10 }}>
                  <span style={{ fontSize: 12, color: "var(--text-muted, #aaa)", fontWeight: 500 }}>Q{i + 1}</span>
                  <span style={{
                    fontSize: 11, padding: "2px 9px", borderRadius: 20,
                    background: meta.bg, color: meta.color, fontWeight: 500,
                  }}>
                    {q.level}
                  </span>
                  <span style={{ fontSize: 12, color: "var(--text-muted, #bbb)" }}>{meta.desc}</span>
                </div>

                {q.application && (
                  <div style={{
                    display: "flex", alignItems: "flex-start", gap: 7,
                    background: "var(--surface-1, #f6f6f4)",
                    border: "0.5px solid var(--border, #e5e5e5)",
                    borderRadius: "var(--radius, 8px)",
                    padding: "6px 10px", marginBottom: 10,
                  }}>
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#185FA5" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ flexShrink: 0, marginTop: 1 }}>
                      <circle cx="12" cy="12" r="3"/><path d="M12 2v3M12 19v3M2 12h3M19 12h3M4.22 4.22l2.12 2.12M17.66 17.66l2.12 2.12M4.22 19.78l2.12-2.12M17.66 6.34l2.12-2.12"/>
                    </svg>
                    <span style={{ fontSize: 12, color: "#185FA5", lineHeight: 1.5 }}>
                      <span style={{ fontWeight: 500, marginRight: 4 }}>Application:</span>{q.application}
                    </span>
                  </div>
                )}

                <p style={{ fontSize: 14, color: "var(--text-primary, #111)", lineHeight: 1.65, margin: "0 0 10px" }}>
                  {q.question}
                </p>

                <button
                  onClick={() => toggleHint(i)}
                  style={{
                    fontSize: 12, color: "#185FA5", background: "none",
                    border: "none", cursor: "pointer", fontFamily: "inherit", padding: 0,
                  }}
                >
                  {revealedHints[i] ? "Hide hint" : "Show hint"}
                </button>

                {revealedHints[i] && (
                  <div style={{
                    marginTop: 8, padding: "8px 12px",
                    background: "#E6F1FB", borderRadius: "var(--radius, 8px)",
                    fontSize: 13, color: "#185FA5", lineHeight: 1.55,
                  }}>
                    {q.hint}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
